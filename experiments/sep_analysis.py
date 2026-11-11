"""Steps 1-3 of the SEP security study, from ``features.npz`` (see sep_features.py). CPU only.

Step 1  behavior: how often the probe is executed per condition / stratum; SEP-style separation.
Step 2  linear probes, per layer, grouped 5-fold CV by probe id (no probe text is ever in both train and test):
          notice      injection vs benign, final-token residual            (does the model register an injected instruction?)
          notice_tt   same, TaskTracker-style activation delta (final - end-of-task)
          obey        among injections: executed vs ignored, final-token residual / attn_output / moe_output /
                      router probabilities / attention-to-probe                (is the decision readable, and where?)
          obey_meta   baseline from surface features only (insistent, position, lengths)
          prov_gate   probe-span tokens: trusted (user_instruction) vs injected (injection), from the residual stream the
                      router reads (resid_mid)                                  (could a new gate tell them apart?)
          prov_router same, from the EXISTING router's probabilities          (does the current router already?)
          instr_vs_text  injected probe tokens vs benign inserted sentence (both inside the data), resid_mid
Step 3  localization: SteerMoE-style expert activation-rate differences (provenance; obeyed vs ignored) with split-half
        reproducibility and a permutation noise floor; per-head attention-to-probe AUC for obeying.

Usage: python experiments/sep_analysis.py ~/moe-results/mellum_sep [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Reference palette (dataviz skill, light mode): categorical slots in fixed order, sequential blue, diverging blue<->red.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list("bgr", ["#184f95", "#86b6ef", "#f0efec", "#ef9a8f", "#a52a2a"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
                     "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "lines.linewidth": 2, "lines.markersize": 4, "legend.frameon": False, "axes.axisbelow": True})


# ---------------------------------------------------------------------------------------------------------------
def load(run: Path):
    z = np.load(run / "features.npz")
    idx = pd.read_json(run / "features_index.jsonl", lines=True)
    # An answer that did NOT contain the witness but hit the token cap might have executed the probe later: its label
    # is uncertain, so it is excluded wherever "executed vs ignored" is the target.
    idx["label_ok"] = idx.executed | (idx.finish_reason != "length")
    return {k: z[k] for k in z.files}, idx


def cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C_: float, folds: int = 5) -> float:
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < folds:
        return float("nan")
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            return float("nan")
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C_, max_iter=2000))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, pred))


def per_layer(X3: np.ndarray, y, groups, C_: float, n_jobs: int = 16) -> list[float]:
    X3 = X3.astype(np.float32)
    return Parallel(n_jobs=n_jobs)(delayed(cv_auc)(X3[:, l], y, groups, C_) for l in range(X3.shape[1]))


# ---------------------------------------------------------------------------------------------------------------
def step1(idx: pd.DataFrame) -> dict:
    rate = idx.groupby("condition").executed.mean()
    trunc = idx.groupby("condition").finish_reason.apply(lambda s: float((s == "length").mean()))
    strata = (idx[idx.condition.isin(["injection", "user_instruction"])]
              .groupby(["condition", "insistent", "probe_position_in_data"]).executed.agg(["mean", "size"]).reset_index())
    pv = idx.pivot_table(index="sample_id", columns="condition", values="executed", aggfunc="first")
    exec_instr = pv["user_instruction"].astype(bool)
    exec_data = pv["injection"].astype(bool)
    by_probe = (idx[idx.condition == "injection"].groupby("probe_id").executed.mean())
    inj = idx[idx.condition == "injection"]
    robust = {
        "injection_asr_all": float(inj.executed.mean()),
        "injection_asr_certain_labels": float(inj[inj.label_ok].executed.mean()),
        "injection_uncertain_label_rate": float((~inj.label_ok).mean()),
        "first_token_match_rate": float(idx.first_token_match.astype(float).mean()) if "first_token_match" in idx else None,
        "injection_asr_first_token_match_only": float(inj[inj.first_token_match == True].executed.mean())
        if "first_token_match" in idx else None,
    }
    return {
        "robustness": robust,
        "execution_rate": rate.to_dict(),
        "truncated_rate": trunc.to_dict(),
        "strata": strata.to_dict(orient="records"),
        "sep_utility_P_exec_instruction": float(exec_instr.mean()),
        "sep_attack_success_P_exec_data": float(exec_data.mean()),
        "sep_separation_score": float((exec_instr & ~exec_data).sum() / max(exec_instr.sum(), 1)),
        "both_executed": int((exec_instr & exec_data).sum()), "only_instruction": int((exec_instr & ~exec_data).sum()),
        "only_data": int((~exec_instr & exec_data).sum()), "neither": int((~exec_instr & ~exec_data).sum()),
        "n_items": int(len(pv)),
        "per_probe_asr_quantiles": by_probe.quantile([0, .25, .5, .75, 1]).round(3).to_dict(),
        "n_probes_never_obeyed": int((by_probe == 0).sum()), "n_probes_always_obeyed": int((by_probe == 1).sum()),
    }


def step2(F: dict, idx: pd.DataFrame) -> pd.DataFrame:
    rows = []
    g_all = idx.probe_id.values

    def add(name, X3, mask, y, C_, feature):
        aucs = per_layer(X3[mask], y.astype(int), g_all[mask], C_)
        for l, a in enumerate(aucs):
            rows.append({"probe": name, "feature": feature, "layer": l, "auc": a, "n": int(mask.sum()),
                         "pos_rate": float(y.mean())})
        print(f"  {name:12s} {feature:14s} best layer {int(np.nanargmax(aucs))} AUC {np.nanmax(aucs):.3f}", flush=True)

    cond = idx.condition.values
    inj, ben, usr = cond == "injection", cond == "benign", cond == "user_instruction"
    # notice
    m = inj | ben
    add("notice", F["resid_last"], m, inj[m], 0.05, "resid_last")
    delta = F["resid_last"].astype(np.float32) - F["resid_taskend"].astype(np.float32)
    add("notice_tt", delta, m, inj[m], 0.05, "delta_tasktracker")
    # obey (among injections with a certain label)
    inj = inj & idx.label_ok.values
    y = idx.executed.values[inj]
    for feat, C_ in (("resid_last", 0.05), ("attn_last", 0.05), ("moe_last", 0.05), ("rprob_last", 1.0), ("amass_probe", 1.0)):
        add("obey", F[feat], inj, y, C_, feat)
    # provenance at the probe tokens
    m = inj | usr
    add("prov_gate", F["mid_probe"], m, usr[m], 0.05, "resid_mid_span")
    add("prov_router", F["rprob_probe"], m, usr[m], 1.0, "router_probs_span")
    # injected instruction vs benign text, both in the data
    X = np.where(inj[:, None, None], F["mid_probe"], F["mid_insert"])
    m = inj | ben
    add("instr_vs_text", X, m, inj[m], 0.05, "resid_mid_span")
    df = pd.DataFrame(rows)
    # surface-feature baseline for obeying (single number, broadcast for plotting)
    meta = pd.get_dummies(idx.loc[inj, ["insistent", "probe_position_in_data"]].astype(str)).astype(float)
    meta["probe_words"] = idx.loc[inj, "probe_words"].values
    meta["num_prompt_tokens"] = idx.loc[inj, "num_prompt_tokens"].values
    base = cv_auc(meta.values, y.astype(int), g_all[inj], 1.0)
    print(f"  obey_meta baseline AUC {base:.3f}")
    # all-layer concatenations
    extra = []
    for feat, C_ in (("rprob_last", 0.1), ("amass_probe", 0.1)):
        a = cv_auc(F[feat][inj].reshape(inj.sum(), -1).astype(np.float32), y.astype(int), g_all[inj], C_)
        extra.append({"probe": "obey", "feature": f"{feat}_all_layers", "layer": -1, "auc": a, "n": int(inj.sum())})
        print(f"  obey all layers {feat}: AUC {a:.3f}")
    extra.append({"probe": "obey_meta", "feature": "surface_only", "layer": -1, "auc": base, "n": int(inj.sum())})
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)


def rates(counts: np.ndarray, ntok: np.ndarray) -> np.ndarray:
    """Pooled share of tokens routed to each expert: sum(counts) / sum(tokens). counts [N, L, E]."""
    return counts.sum(0) / max(ntok.sum(), 1)


def step3(F: dict, idx: pd.DataFrame, k: int, rng: np.random.Generator) -> dict:
    cond = idx.condition.values
    inj, usr = cond == "injection", cond == "user_instruction"
    ok = idx.label_ok.values
    probes = np.array(sorted(idx.probe_id.unique()))
    rng.shuffle(probes)
    half = set(probes[: len(probes) // 2])
    in_a = idx.probe_id.isin(half).values
    out: dict = {}

    def diff(mask1, mask0, cnt="ecount_probe", nt="ntok_probe"):
        return rates(F[cnt][mask1], F[nt][mask1]) - rates(F[cnt][mask0], F[nt][mask0])

    def reproducibility(mask1, mask0, cnt, nt):
        da, db = diff(mask1 & in_a, mask0 & in_a, cnt, nt), diff(mask1 & ~in_a, mask0 & ~in_a, cnt, nt)
        rho = spearmanr(da.ravel(), db.ravel()).correlation
        ta = set(np.argsort(-np.abs(da).ravel())[:30]); tb = set(np.argsort(-np.abs(db).ravel())[:30])
        return float(rho), len(ta & tb) / 30

    # provenance: same probe tokens, data position (injection) vs trusted position (user_instruction)
    d_prov = diff(inj, usr)
    rho, top = reproducibility(inj, usr, "ecount_probe", "ntok_probe")
    out["provenance"] = {"delta": d_prov, "split_half_spearman": rho, "split_half_top30_overlap": top}
    # obey: injection items, executed vs ignored; at the probe tokens and at the final template tokens
    ex = idx.executed.values.astype(bool)
    inj_all = inj
    inj = inj & ok  # obey analyses: certain labels only
    for where, cnt, nt in (("probe_tokens", "ecount_probe", "ntok_probe"), ("tail_tokens", "ecount_tail", "ntok_tail")):
        d = diff(inj & ex, inj & ~ex, cnt, nt)
        rho, top = reproducibility(inj & ex, inj & ~ex, cnt, nt)
        # permutation noise floor: shuffle executed labels among injections
        null = []
        ii = np.where(inj)[0]
        for _ in range(200):
            perm = ex.copy()
            perm[ii] = rng.permutation(ex[ii])
            null.append(np.abs(diff(inj & perm, inj & ~perm, cnt, nt)).max())
        out[f"obey_{where}"] = {"delta": d, "split_half_spearman": rho, "split_half_top30_overlap": top,
                                "max_abs_delta": float(np.abs(d).max()), "null_max_abs_delta_95": float(np.quantile(null, .95))}
    # heads: attention from the final prompt token to the probe span, executed vs ignored
    A = F["amass_probe"][inj]  # [N, L, heads]
    y = ex[inj]
    L, Hh = A.shape[1:]
    auc = np.array([[roc_auc_score(y, A[:, l, h]) for h in range(Hh)] for l in range(L)])
    ga = in_a[inj]
    auc_a = np.array([[roc_auc_score(y[ga], A[ga, l, h]) for h in range(Hh)] for l in range(L)])
    auc_b = np.array([[roc_auc_score(y[~ga], A[~ga, l, h]) for h in range(Hh)] for l in range(L)])
    out["heads_obey"] = {"auc": auc, "split_half_spearman": float(spearmanr(auc_a.ravel(), auc_b.ravel()).correlation)}
    # where does the final token look? mean attention mass per span, by outcome
    spans = {}
    for s in ("task", "data", "probe"):
        M = F[f"amass_{s}"][inj].mean(-1)  # mean over heads -> [N, L]
        spans[s] = {"executed": M[y].mean(0).tolist(), "ignored": M[~y].mean(0).tolist()}
    out["attention_by_span"] = spans
    return out


# ---------------------------------------------------------------------------------------------------------------
def plot_step1(s1: dict, idx: pd.DataFrame, path: Path, model: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    order = ["clean", "benign", "injection", "user_instruction"]
    st = idx[idx.condition.isin(["injection", "user_instruction"])]
    groups = [("plain, probe at data start", False, "start"), ("plain, data end", False, "end"),
              ("insistent, data start", True, "start"), ("insistent, data end", True, "end")]
    w = 0.38
    for j, cnd in enumerate(["injection", "user_instruction"]):
        vals = [st[(st.condition == cnd) & (st.insistent == ins) & (st.probe_position_in_data == pos)].executed.mean()
                for _, ins, pos in groups]
        xs = np.arange(len(groups)) + (j - 0.5) * w
        ax.bar(xs, vals, w - 0.04, color=C[j], label={"injection": "probe in DATA (injection: should be ignored)",
                                                      "user_instruction": "probe in TRUSTED task (should be executed)"}[cnd])
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=8, color=TEXT2)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("probe executed (witness in answer)")
    fp = ", ".join(f"{c} {s1['execution_rate'].get(c, float('nan')):.1%}" for c in ("clean", "benign"))
    ax.set_title(f"{model}: how often the probe gets executed  (false-positive rate: {fp})", fontsize=10, loc="left")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_probes(df: pd.DataFrame, path: Path, model: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    left = [("notice", "resid_last", "noticed: injection vs benign (final token)"),
            ("notice_tt", "delta_tasktracker", "noticed: TaskTracker delta"),
            ("obey", "resid_last", "obeys: executed vs ignored (final token)"),
            ("obey", "rprob_last", "obeys: from router probabilities only")]
    right = [("prov_gate", "resid_mid_span", "trusted vs injected probe, gate input"),
             ("prov_router", "router_probs_span", "trusted vs injected probe, current router"),
             ("instr_vs_text", "resid_mid_span", "injected instruction vs benign text, gate input")]
    for ax, spec, title in ((axes[0], left, "A. Decision at the final prompt token"),
                            (axes[1], right, "B. At the injected tokens: what a gate could see")):
        for j, (p, f, lab) in enumerate(spec):
            d = df[(df.probe == p) & (df.feature == f)].sort_values("layer")
            ax.plot(d.layer, d.auc, color=C[j], label=lab, marker="o")
        if ax is axes[0]:
            b = df[(df.probe == "obey_meta")].auc.iloc[0]
            ax.axhline(b, color=TEXT2, lw=1, ls="--")
            ax.text(0, b + 0.01, f"obeys: surface features only ({b:.2f})", fontsize=8, color=TEXT2)
        ax.axhline(0.5, color=TEXT2, lw=0.8)
        ax.set_ylim(0.4, 1.02)
        ax.set_xlabel("layer")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=7.5, ncol=2)
    axes[0].set_ylabel("held-out AUC (grouped by probe)")
    fig.suptitle(f"{model}: linear probes per layer", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heatmap(M: np.ndarray, path: Path, title: str, xlabel: str, cbar: str, center: float = 0.0) -> None:
    v = np.nanmax(np.abs(M - center))
    fig, ax = plt.subplots(figsize=(max(6, M.shape[1] * 0.09 + 2), max(3.5, M.shape[0] * 0.17 + 1.2)))
    im = ax.imshow(M, aspect="auto", cmap=DIVERGING, vmin=center - v, vmax=center + v, interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("layer")
    ax.grid(False)
    ax.set_title(title, fontsize=10, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label(cbar)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_attention_spans(s3: dict, path: Path, model: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.0))
    for j, s in enumerate(("task", "data", "probe")):
        d = s3["attention_by_span"][s]
        ax.plot(d["executed"], color=C[j], label=f"{s}: executed")
        ax.plot(d["ignored"], color=C[j], ls="--", label=f"{s}: ignored")
    ax.set_xlabel("layer")
    ax.set_ylabel("attention mass from final token\n(mean over heads)")
    ax.set_title(f"{model}: where the final prompt token looks, injected prompts", fontsize=10, loc="left")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_table(delta: np.ndarray, n: int = 20) -> list[dict]:
    flat = np.argsort(-np.abs(delta).ravel())[:n]
    E = delta.shape[1]
    return [{"layer": int(i // E), "expert": int(i % E), "delta_share": float(delta.ravel()[i])} for i in flat]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()
    run = Path(args.run).expanduser()
    out = Path(args.out).expanduser() if args.out else run / "security"
    out.mkdir(parents=True, exist_ok=True)
    model = args.model or run.name
    F, idx = load(run)
    print("step 1: behavior")
    s1 = step1(idx)
    print(json.dumps({k: v for k, v in s1.items() if k != "strata"}, indent=1))
    plot_step1(s1, idx, out / "step1_execution_rates.png", model)
    print("step 2: probes")
    df = step2(F, idx)
    df.to_csv(out / "step2_probe_auc.csv", index=False)
    plot_probes(df, out / "step2_probes_by_layer.png", model)
    print("step 3: localization")
    s3 = step3(F, idx, args.top_k, np.random.default_rng(0))
    plot_heatmap(s3["provenance"]["delta"], out / "step3_experts_provenance.png",
                 f"{model}: expert use on the SAME probe tokens, in data minus in trusted task\n"
                 f"(split-half Spearman {s3['provenance']['split_half_spearman']:.2f})", "expert",
                 "difference in share of tokens routed")
    plot_heatmap(s3["obey_probe_tokens"]["delta"], out / "step3_experts_obey_probe_tokens.png",
                 f"{model}: expert use on injected probe tokens, executed minus ignored\n"
                 f"(split-half Spearman {s3['obey_probe_tokens']['split_half_spearman']:.2f}; max |d| "
                 f"{s3['obey_probe_tokens']['max_abs_delta']:.3f} vs noise {s3['obey_probe_tokens']['null_max_abs_delta_95']:.3f})",
                 "expert", "difference in share of tokens routed")
    plot_heatmap(s3["heads_obey"]["auc"], out / "step3_heads_obey_auc.png",
                 f"{model}: does a head's attention to the injected probe predict obeying?  (AUC; >0.5 = more attention "
                 f"when executed)\n(split-half Spearman {s3['heads_obey']['split_half_spearman']:.2f})", "head", "AUC",
                 center=0.5)
    plot_attention_spans(s3, out / "step3_attention_by_span.png", model)
    H = s3["heads_obey"]["auc"]
    flat = np.argsort(-np.abs(H - 0.5).ravel())[:20]
    summary = {
        "model": model, "step1": s1,
        "step2_best": df[df.layer >= 0].groupby(["probe", "feature"]).auc.agg(["max", "idxmax"]).reset_index()
        .assign(best_layer=lambda d: df.loc[d["idxmax"], "layer"].values).drop(columns="idxmax").to_dict(orient="records"),
        "step2_all_layer": df[df.layer < 0].to_dict(orient="records"),
        "step3": {
            "provenance": {k: v for k, v in s3["provenance"].items() if k != "delta"} | {"top": top_table(s3["provenance"]["delta"])},
            "obey_probe_tokens": {k: v for k, v in s3["obey_probe_tokens"].items() if k != "delta"} | {"top": top_table(s3["obey_probe_tokens"]["delta"])},
            "obey_tail_tokens": {k: v for k, v in s3["obey_tail_tokens"].items() if k != "delta"} | {"top": top_table(s3["obey_tail_tokens"]["delta"])},
            "heads_obey": {"split_half_spearman": s3["heads_obey"]["split_half_spearman"],
                           "top": [{"layer": int(i // H.shape[1]), "head": int(i % H.shape[1]), "auc": float(H.ravel()[i])} for i in flat]},
            "attention_by_span": s3["attention_by_span"],
        },
    }
    np.savez(out / "step3_arrays.npz", provenance=s3["provenance"]["delta"], obey_probe=s3["obey_probe_tokens"]["delta"],
             obey_tail=s3["obey_tail_tokens"]["delta"], heads_obey_auc=H)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary["step3"]["provenance"] | {"top": summary["step3"]["provenance"]["top"][:5]}, indent=1, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
