"""E06: security-expert learning-rate / early-stopping sweep: trajectories of SecPO val metrics, expert contribution and
dev-40 behaviour (Gemini 3.5 Flash Lite judge) for lr 1.6e-4 (run A, = E05 re-run) and lr 4e-5 (run B), against the E05
final checkpoint, the E03 LoRA baseline and the base model.

Inputs (OUT = experiments/modal/outputs/e06_se_lr_sweep):
  OUT/se_lr16e-5/metrics.jsonl, OUT/se_lr4e-5/metrics.jsonl             training/val logs (val every 10 steps)
  E05 secpo_se_v1/metrics.jsonl, E03 secpo_v2/metrics.jsonl              reference trajectories
  OUT/dev40/<run>/{generations,judgments_flash}.jsonl                    run in base, lora_v2, e05_final,
                                                                         lr16e-5_step{N}, lr4e-5_step{N}
Outputs: OUT/analysis/{trajectory.json, trajectory.md, trajectory.png, repro_check.json}

Usage: python experiments/modal/experiments/e06_analysis.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "experiments" / "modal" / "outputs" / "e06_se_lr_sweep"
E05 = REPO / "experiments" / "modal" / "outputs" / "e05_security_expert" / "secpo_se_v1" / "metrics.jsonl"
LORA = REPO / "experiments" / "modal" / "outputs" / "e03_olmoe_security_expert" / "secpo_v2" / "metrics.jsonl"
RUNS = {"lr16e-5": "lr 1.6e-4 (A)", "lr4e-5": "lr 4e-5 (B)"}
STEPS = [10, 20, 40, 60, 80]
VAL_KEYS = ["all/loss", "all/secpo_acc", "all/chosen_reward", "all/rejected_reward"]

spec = importlib.util.spec_from_file_location("summ", REPO / "scripts" / "summarize_e05_eval.py")
summ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summ)


def val_rows(p: Path) -> dict[int, dict]:
    return {r["step"]: r for r in map(json.loads, p.open()) if r["type"] == "val"} if p.exists() else {}


def dev(name: str) -> dict | None:
    d = OUT / "dev40" / name
    if not (d / "generations.jsonl").exists() or not (d / "judgments_flash.jsonl").exists():
        return None
    s, per = summ.summarize(summ.load(d))
    if s["judged_rows"] < 2 * s["n"]:
        return None
    degen = sum(1 for g in map(json.loads, (d / "generations.jsonl").open()) if g["repetition_3gram"] >= 0.5)
    return {k: s[k] for k in ("n", "asr_flash", "asr_flash_unclear", "clean_task_success", "task_under_injection",
                              "median_len_clean", "median_len_inj")} | {"degenerate": degen, "per": per}


def main() -> None:
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    vals = {k: val_rows(OUT / f"se_{k}" / "metrics.jsonl") for k in RUNS}
    e05, lora = val_rows(E05), val_rows(LORA)

    # 1. reproduction: run A vs the logged E05 trajectory
    repro = {}
    for s in sorted(e05):
        if s in vals["lr16e-5"]:
            a, b = vals["lr16e-5"][s], e05[s]
            repro[s] = {k: {"e05": b[k], "rerun": a[k], "abs_diff": round(abs(a[k] - b[k]), 5)}
                        for k in VAL_KEYS + ["se_ratio_mean"] if k in a and k in b}
    (OUT / "analysis" / "repro_check.json").write_text(json.dumps(repro, indent=1) + "\n")

    # 2. one row per (run, step) + references
    rows = []
    refs = {"base": ("base OLMoE", None), "lora_v2": ("E03 LoRA (step 80)", lora.get(80)),
            "e05_final": ("E05 SE final (lr 1.6e-4, step 80)", e05.get(80))}
    for key, (label, v) in refs.items():
        rows.append({"run": key, "label": label, "lr": None, "step": None, "val": v, "dev": dev(key)})
    for k, label in RUNS.items():
        for s in STEPS:
            rows.append({"run": f"{k}_step{s}", "label": label, "lr": k, "step": s, "val": vals[k].get(s),
                         "dev": dev(f"{k}_step{s}")})
    # paired comparison to E05 final on dev (same 40 records): flips
    ref = next(r["dev"] for r in rows if r["run"] == "e05_final")
    for r in rows:
        if r["dev"] and ref and r["run"] != "e05_final":
            a, b = ref["per"], r["dev"]["per"]
            r["vs_e05_final"] = {
                "clean_gained": sum(b[i]["task_clean"] == "success" != a[i]["task_clean"] for i in a),
                "clean_lost": sum(a[i]["task_clean"] == "success" != b[i]["task_clean"] for i in a),
                "attack_gained": sum(b[i]["attack"] == "success" != a[i]["attack"] for i in a),
                "attack_lost": sum(a[i]["attack"] == "success" != b[i]["attack"] for i in a)}
    out_rows = [{k: v for k, v in r.items() if k != "dev"} | {"dev": {k: v for k, v in (r["dev"] or {}).items()
                                                                       if k != "per"} or None} for r in rows]
    for r in out_rows:
        if r["val"]:
            r["val"] = {k: r["val"].get(k) for k in VAL_KEYS + ["se_ratio_mean", "se_ratio_max", "se_ratio_by_layer",
                                                                "se_alpha"] if k in r["val"]}
    (OUT / "analysis" / "trajectory.json").write_text(json.dumps(out_rows, indent=1) + "\n")

    # 3. markdown table
    f = lambda x, d=2: "–" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))  # noqa: E731
    md = ["| run | step | val loss | val acc | chosen rew. | rejected rew. | ‖αS‖/‖h_MoE‖ | dev ASR /40 | "
          "dev clean task /40 | dev task under inj. /40 | med. len clean / inj | degenerate /80 | vs E05 final: clean +/−, attack +/− |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        v, d = r["val"] or {}, r["dev"]
        cmp = r.get("vs_e05_final")
        md.append("| " + " | ".join([
            r["label"], f(r["step"]), f(v.get("all/loss")), f(v.get("all/secpo_acc")), f(v.get("all/chosen_reward")),
            f(v.get("all/rejected_reward"), 1),
            "–" if "se_ratio_mean" not in v else f"{100 * v['se_ratio_mean']:.1f}%",
            "–" if not d else f"{d['asr_flash']}" + (f" (+{d['asr_flash_unclear']}?)" if d["asr_flash_unclear"] else ""),
            "–" if not d else str(d["clean_task_success"]), "–" if not d else str(d["task_under_injection"]),
            "–" if not d else f"{d['median_len_clean']:g} / {d['median_len_inj']:g}",
            "–" if not d else str(d["degenerate"]),
            "–" if not cmp else f"+{cmp['clean_gained']}/−{cmp['clean_lost']}, +{cmp['attack_gained']}/−{cmp['attack_lost']}",
        ]) + " |")
    (OUT / "analysis" / "trajectory.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # 4. figure: small multiples, one measure per panel (no dual axes); A = slot 1 blue, B = slot 2 orange,
    #    references in gray. Dev panels use percent of 40 so the three dev measures share a 0-100 scale.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"lr16e-5": "#2a78d6", "lr4e-5": "#eb6834"}
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    panels = [("SecPO val loss", lambda r: (r["val"] or {}).get("all/loss"), "val"),
              ("SecPO val preference accuracy", lambda r: (r["val"] or {}).get("all/secpo_acc"), "val"),
              ("val chosen reward β·Δlog π(y_w)", lambda r: (r["val"] or {}).get("all/chosen_reward"), "val"),
              ("val rejected reward β·Δlog π(y_l)", lambda r: (r["val"] or {}).get("all/rejected_reward"), "val"),
              ("dev-40 attack success (%)", lambda r: r["dev"] and 100 * r["dev"]["asr_flash"] / 40, "dev"),
              ("dev-40 clean task success (%)", lambda r: r["dev"] and 100 * r["dev"]["clean_task_success"] / 40, "dev"),
              ("dev-40 task done under injection (%)",
               lambda r: r["dev"] and 100 * r["dev"]["task_under_injection"] / 40, "dev"),
              ("security-expert size ‖αS‖ / ‖h_MoE‖ (%)",
               lambda r: (r["val"] or {}).get("se_ratio_mean") and 100 * r["val"]["se_ratio_mean"], "val")]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.6), facecolor="#fcfcfb")
    for ax, (title, get, kind) in zip(axes.flat, panels):
        ax.set_facecolor("#fcfcfb")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.tick_params(colors=muted, labelsize=9)
        ax.set_title(title, fontsize=10.5, color=ink, loc="left")
        ax.set_xticks([0] + STEPS)
        ax.set_xlabel("optimizer step", fontsize=9, color=muted)
        for k, label in RUNS.items():
            pts = [(r["step"], get(r)) for r in rows if r["lr"] == k and get(r) is not None]
            if kind == "val" and get({"val": vals[k].get(0), "dev": None}) is not None:
                pts = [(0, get({"val": vals[k][0], "dev": None}))] + pts
            if pts:
                ax.plot(*zip(*pts), color=col[k], lw=2, marker="o", ms=6, mec="#fcfcfb", mew=2, label=label, zorder=3)
        for key, ls, lab in (("lora_v2", "--", "E03 LoRA (final)"), ("base", ":", "base OLMoE")):
            r = next(x for x in rows if x["run"] == key)
            y = get(r)
            if y is not None:
                ax.axhline(y, color=muted, ls=ls, lw=1.2, label=lab, zorder=1)
        r = next(x for x in rows if x["run"] == "e05_final")
        if get(r) is not None:
            ax.plot([80], [get(r)], marker="D", ms=8, color="none", mec=col["lr16e-5"], mew=1.8,
                    label="E05 final (original run)", zorder=4)
    h, l = axes.flat[0].get_legend_handles_labels()
    for ax in axes.flat[1:]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=10, labelcolor=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "analysis" / "trajectory.png", dpi=130)
    print(f"-> {OUT / 'analysis'}")


if __name__ == "__main__":
    main()
