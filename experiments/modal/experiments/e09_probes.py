"""E09 experiments 1 + 2 (laptop, CPU): layer-wise linear probes on the final-prompt-token residual stream, cross-model
probe transfer, and probe-direction cosine similarity, for base / LoRA (secpo_v2) / E05 / routed (E08 main step 80).

Inputs: ACTS/<model>/resid_last.npz from e09_capture_resid.py, and each model's saved E07/E08 generations + judgments.

Probe tasks (rows are AF-208 record x condition; 208 x 7 = 1,456 per model):
    detect        all rows; y = an injection is present (any of the six attacks) vs clean
    obey_own      injected rows; y = this model obeyed the injection (E07 "obeyed": strict judge minus verbatim echoes)
    obey_base     injected rows; y = the BASE model obeyed this prompt ("susceptibility": same labels for every model,
                  so every model has 608 positives; it asks whether a model still encodes what made the base obey)

Protocol, identical for every model:
    split         by record (the underlying task), so no task's injection variants leak across parts:
                  60 / 20 / 20 % train / val / test, fixed seed, saved to artifacts/splits/af208_e09_records.json
    probe         standardise (train mean / std) -> L2 logistic regression, class_weight balanced;
                  C chosen on val AUROC from C_GRID; fitted on train only (val stays clean for E09 exp. 3)
    metrics       test AUROC and AUPRC with prevalence; 95% CI from a cluster bootstrap over test records
    insufficient  flagged (no metric reported as a finding) if train has < MIN_POS_TRAIN or test < MIN_POS_TEST
                  positives (or negatives)
    baselines     obedience: attack-type-only logistic regression, and the probe's mean within-attack AUROC
                  (does the probe know more than which attack it is?); layer 0 (embedding of the final token, which is
                  the same token for every prompt) is a built-in chance control
    held-out      (a) leave one attack family out (SF / ignore / completion, each with its two variants);
                  (b) train on the four non-"before" attacks, test on the two "before" attacks
    transfer      the base model's probe (frozen) scored on another model's test activations, vs that model's own probe.
                  The probe's score is w . (x - mu_base) / sd_base + b; AUROC does not depend on the offset, so only the
                  direction w / sd_base matters
    cosine        between models' probe directions in raw residual space (w / sd), per layer; null = random 2048-d
                  directions (|cos| < 2 / sqrt(H) at ~95%); ceiling = split-half cosine of the same model's probe

Outputs in --out: results.json (everything), summary.md, figs/*.png

Usage: python e09_probes.py --acts ACTS --out OUT [--models base,lora,e05,routed] [--boot 1000] [--jobs 8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from summarize_e07 import ATTACKS, load as load_labels  # noqa: E402

E7 = REPO / "experiments/modal/outputs/e07_af208_six_attacks"
RUNS = {"base": E7 / "base", "lora": E7 / "lora", "e05": E7 / "e05",
        "routed": REPO / "experiments/modal/outputs/e08_routed_expert/gen/re_main_s80"}
LABEL = {"base": "base", "lora": "LoRA (SecPO)", "e05": "E05 always-on expert", "routed": "E08 routed expert"}
FAMILIES = {"SF": ["straightforward", "straightforward_before"], "ignore": ["ignore", "ignore_before"],
            "completion": ["completion", "completion_ignore"]}
BEFORE = ["straightforward_before", "ignore_before"]
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
MIN_POS_TRAIN, MIN_POS_TEST = 20, 10
SPLIT = REPO / "experiments/modal/artifacts/splits/af208_e09_records.json"
DATA = REPO / "data/alpacafarm_secalign_208x6.jsonl"


# ------------------------------------------------------------------------------------------------------ data

def make_split(record_ids, seed: int = 0, fracs=(0.6, 0.2, 0.2)) -> dict[str, str]:
    ids = sorted(set(record_ids))
    rng = np.random.default_rng(seed)
    perm = [ids[i] for i in rng.permutation(len(ids))]
    a, b = round(fracs[0] * len(ids)), round((fracs[0] + fracs[1]) * len(ids))
    return {r: ("train" if i < a else "val" if i < b else "test") for i, r in enumerate(perm)}


def get_split(record_ids) -> dict[str, str]:
    data_sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if SPLIT.exists():
        s = json.loads(SPLIT.read_text())
        assert s["data_sha256"] == data_sha and set(s["assign"]) == set(record_ids), "stale split"
        return s["assign"]
    assign = make_split(record_ids)
    SPLIT.parent.mkdir(parents=True, exist_ok=True)
    SPLIT.write_text(json.dumps({"name": "af208_e09_records", "unit": "AF-208 record (task); all 7 conditions of a "
                                 "record share its part", "seed": 0, "fractions": [0.6, 0.2, 0.2],
                                 "data": str(DATA.relative_to(REPO)), "data_sha256": data_sha,
                                 "counts": {p: sum(v == p for v in assign.values()) for p in ("train", "val", "test")},
                                 "assign": dict(sorted(assign.items()))}, indent=1) + "\n")
    return assign


def load_model(name: str, acts: Path, keys: list | None = None):
    """Activations [N, L+1, H] in canonical row order (sorted (id, condition)) and the label table."""
    z = np.load(acts / name / "resid_last.npz")
    order = {(i, c): n for n, (i, c) in enumerate(zip(z["ids"].tolist(), z["conditions"].tolist()))}
    keys = keys or sorted(order)
    X = z["resid"][[order[k] for k in keys]]
    return keys, X, load_labels(RUNS[name])


# ----------------------------------------------------------------------------------------------------- probes

def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    from scipy.stats import rankdata

    pos = y.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, s) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def boot_ci(y, s, groups, n_boot: int, seed: int = 0):
    """95% CI of AUROC and AUPRC, resampling test records (clusters) with replacement."""
    if n_boot <= 0 or not 0 < y.sum() < len(y):
        return None
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.nonzero(inv == g)[0] for g in range(len(uniq))]
    a, p = [], []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        yy, ss = y[idx], s[idx]
        if 0 < yy.sum() < len(yy):
            a.append(auroc(yy, ss))
            p.append(auprc(yy, ss))
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]  # noqa: E731
    return {"auroc": q(a), "auprc": q(p), "n_valid": len(a)}


def fit_probe(Xtr, ytr, Xva, yva, grid=C_GRID, fixed_C=None):
    """Standardise on train, L2 logistic regression; C by val AUROC. Returns dict with raw-space direction."""
    from sklearn.linear_model import LogisticRegression

    if not (0 < ytr.sum() < len(ytr)):
        return None
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Ztr, Zva = (Xtr - mu) / sd, (Xva - mu) / sd
    best = None
    for C in ([fixed_C] if fixed_C else grid):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, tol=1e-5).fit(Ztr, ytr)
        va = auroc(yva, Zva @ clf.coef_[0]) if 0 < yva.sum() < len(yva) else float("nan")
        if best is None or (not np.isnan(va) and (np.isnan(best[0]) or va > best[0])):
            best = (va, C, clf)
    va, C, clf = best
    w = clf.coef_[0]
    d = w / sd
    return {"mu": mu, "sd": sd, "w": w, "b": float(clf.intercept_[0]), "C": C, "val_auroc": va,
            "dir": d / np.linalg.norm(d) if np.linalg.norm(d) > 0 else d}


def score(p, X):
    return ((X - p["mu"]) / p["sd"]) @ p["w"] + p["b"]


def evaluate(p, X, y, groups, n_boot, seed=0) -> dict:
    s = score(p, X)
    out = {"auroc": round(auroc(y, s), 4), "auprc": round(auprc(y, s), 4), "prevalence": round(float(y.mean()), 4),
           "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
    out["ci"] = boot_ci(y, s, groups, n_boot, seed)
    return out


def sufficient(ytr, yte) -> bool:
    return min(ytr.sum(), len(ytr) - ytr.sum()) >= MIN_POS_TRAIN and min(yte.sum(), len(yte) - yte.sum()) >= MIN_POS_TEST


# --------------------------------------------------------------------------------------------------- analysis

def task_rows(task: str, keys, labels, base_labels):
    """Row mask and labels for one task over the canonical keys."""
    cond = np.array([k[1] for k in keys])
    if task == "detect":
        return np.ones(len(keys), bool), (cond != "clean").astype(int)
    inj = cond != "clean"
    src = labels if task == "obey_own" else base_labels
    y = np.array([int(src[k]["obeyed"]) if k[1] != "clean" else 0 for k in keys])
    return inj, y


def analyse_model(name, keys, X, labels, base_labels, part, n_boot, out_probes: dict) -> dict:
    rec = np.array([k[0] for k in keys])
    cond = np.array([k[1] for k in keys])
    pt = np.array([part[r] for r in rec])
    L1 = X.shape[1]
    res = {}
    for task in ("detect", "obey_own", "obey_base"):
        mask, y = task_rows(task, keys, labels, base_labels)
        tr, va, te = (mask & (pt == p) for p in ("train", "val", "test"))
        ok = sufficient(y[tr], y[te])
        rows = []
        for l in range(L1):
            p = fit_probe(X[tr, l], y[tr], X[va, l], y[va])
            if p is None:
                rows.append(None)
                continue
            out_probes[(name, task, l)] = p
            r = {"layer": l, "C": p["C"], "val_auroc": round(p["val_auroc"], 4)}
            r |= evaluate(p, X[te, l], y[te], rec[te], n_boot if ok else 0, seed=l)
            if task != "detect":  # within-attack AUROC: does the probe know more than the attack type?
                s = score(p, X[te, l])
                per = [auroc(y[te][cond[te] == a], s[cond[te] == a]) for a in ATTACKS]
                per = [v for v in per if not np.isnan(v)]
                r["within_attack_auroc"] = round(float(np.mean(per)), 4) if per else None
            rows.append(r)
        res[task] = {"sufficient": bool(ok), "n_pos_train": int(y[tr].sum()), "n_pos_val": int(y[va].sum()),
                     "n_pos_test": int(y[te].sum()), "n_train": int(tr.sum()), "n_test": int(te.sum()), "layers": rows}
        if task != "detect":  # attack-type-only baseline
            from sklearn.linear_model import LogisticRegression

            oh = lambda m: np.stack([(cond[m] == a).astype(float) for a in ATTACKS], 1)  # noqa: E731
            if 0 < y[tr].sum() < tr.sum():
                clf = LogisticRegression(class_weight="balanced").fit(oh(tr), y[tr])
                s = clf.decision_function(oh(te))
                res[task]["attack_type_baseline"] = {"auroc": round(auroc(y[te], s), 4), "auprc": round(auprc(y[te], s), 4)}
    return res


def held_out(name, keys, X, labels, base_labels, part, layers) -> dict:
    """Generalisation to unseen injection types (detect, obey_base, and obey_own where it has enough positives)."""
    rec = np.array([k[0] for k in keys])
    cond = np.array([k[1] for k in keys])
    pt = np.array([part[r] for r in rec])
    splits = {f"hold_out_{f}": (set(ATTACKS) - set(a), set(a)) for f, a in FAMILIES.items()}
    splits["after_to_before"] = (set(ATTACKS) - set(BEFORE), set(BEFORE))
    res = {}
    for task in ("detect", "obey_base", "obey_own"):
        mask, y = task_rows(task, keys, labels, base_labels)
        for sname, (train_att, test_att) in splits.items():
            keep_tr = np.isin(cond, list(train_att) + (["clean"] if task == "detect" else []))
            keep_te = np.isin(cond, list(test_att) + (["clean"] if task == "detect" else []))
            tr, va, te = mask & keep_tr & (pt == "train"), mask & keep_tr & (pt == "val"), mask & keep_te & (pt == "test")
            if not sufficient(y[tr], y[te]):
                res.setdefault(task, {})[sname] = {"sufficient": False, "n_pos_train": int(y[tr].sum()),
                                                  "n_pos_test": int(y[te].sum())}
                continue
            rows = []
            for l in layers:
                p = fit_probe(X[tr, l], y[tr], X[va, l], y[va])
                r = evaluate(p, X[te, l], y[te], rec[te], 0)
                rows.append({"layer": l, "auroc": r["auroc"], "auprc": r["auprc"]})
            res.setdefault(task, {})[sname] = {"sufficient": True, "n_pos_test": int(y[te].sum()), "layers": rows}
    return res


def split_half_cosine(X, y, tr_mask, va_mask, rec, l, C, seed=0) -> float:
    """Cosine between probes trained on two disjoint halves of the training records (same C): a reliability ceiling."""
    rng = np.random.default_rng(seed)
    recs = np.unique(rec[tr_mask])
    half = set(rng.permutation(recs)[: len(recs) // 2])
    a = tr_mask & np.isin(rec, list(half))
    b = tr_mask & ~np.isin(rec, list(half))
    pa = fit_probe(X[a, l], y[a], X[va_mask, l], y[va_mask], fixed_C=C)
    pb = fit_probe(X[b, l], y[b], X[va_mask, l], y[va_mask], fixed_C=C)
    return float(pa["dir"] @ pb["dir"]) if pa and pb else float("nan")


# ------------------------------------------------------------------------------------------------------- plots

COLORS = {"base": "#2a78d6", "lora": "#eb6834", "e05": "#1baf7a", "routed": "#eda100"}  # validated (dataviz)
MARKERS = {"base": "o", "lora": "s", "e05": "^", "routed": "D"}


def _style(ax, title, ylabel, ylim=(0.4, 1.01)):
    ax.set_title(title, fontsize=10, loc="left", color="#222")
    ax.set_xlabel("residual index (0 = embedding, i = after layer i)", fontsize=8, color="#555")
    ax.set_ylabel(ylabel, fontsize=8, color="#555")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, color="#e6e6e3", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbb")
    ax.tick_params(labelsize=7, colors="#555")
    ax.set_xticks(range(0, 17, 2))


def plot_all(R: dict, models: list[str], figs: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs.mkdir(parents=True, exist_ok=True)
    made = []

    ends: dict = {}

    def line(ax, m, xs, ys, lo=None, hi=None, dashed=False, label=None):
        ax.plot(xs, ys, color=COLORS[m], lw=2, ls="--" if dashed else "-", marker=MARKERS[m], ms=4,
                label=label or LABEL[m])
        if lo is not None:
            ax.fill_between(xs, lo, hi, color=COLORS[m], alpha=0.12, lw=0)
        ends.setdefault(ax, []).append([ys[-1], xs[-1], label or LABEL[m]])

    def place_labels():
        """Direct end labels, nudged apart so that none overlap (min gap 4.5% of the y-range)."""
        for ax, items in ends.items():
            lo, hi = ax.get_ylim()
            gap = 0.045 * (hi - lo)
            items.sort()
            for i in range(1, len(items)):
                items[i][0] = max(items[i][0], items[i - 1][0] + gap)
            for y, x, text in items:
                ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", fontsize=7, color="#333", va="center")
        ends.clear()

    # fig 1: layer-wise probe performance
    titles = {"detect": "Injection detection (AUROC, test records)", "obey_base": "Susceptibility: base model obeyed (AUROC)",
              "obey_own": "Obedience: this model obeyed (AUROC)"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor="#fcfcfb")
    for ax, task in zip(axes, ("detect", "obey_base", "obey_own")):
        _style(ax, titles[task], "AUROC")
        ax.axhline(0.5, color="#999", lw=0.8, ls=":")
        for m in models:
            t = R["probes"][m][task]
            if not t["sufficient"]:
                ax.text(0.02, 0.97 - 0.05 * models.index(m), f"{LABEL[m]}: {t['n_pos_test']} test positives (insufficient)",
                        va="top",
                        transform=ax.transAxes, fontsize=7, color="#666")
                continue
            rows = [r for r in t["layers"] if r]
            xs = [r["layer"] for r in rows]
            line(ax, m, xs, [r["auroc"] for r in rows], [r["ci"]["auroc"][0] for r in rows],
                 [r["ci"]["auroc"][1] for r in rows])
            if task != "detect" and "attack_type_baseline" in t:
                ax.axhline(t["attack_type_baseline"]["auroc"], color=COLORS[m], lw=0.8, ls="--")
        ax.set_xlim(-0.5, 21)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.text(0.01, 0.005, "Bands: 95% cluster-bootstrap CI over test records. Dashed horizontal: attack-type-only baseline. "
             "Dotted: chance.", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    place_labels()
    fig.savefig(figs / "fig1_layerwise_auroc.png", dpi=160)
    plt.close(fig)
    made.append("fig1_layerwise_auroc.png")

    # fig 1b: AUPRC for obedience tasks with prevalence
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor="#fcfcfb")
    for ax, task in zip(axes, ("obey_base", "obey_own")):
        _style(ax, titles[task].replace("AUROC", "AUPRC"), "AUPRC", ylim=(0, 1.01))
        for m in models:
            t = R["probes"][m][task]
            if not t["sufficient"]:
                continue
            rows = [r for r in t["layers"] if r]
            xs = [r["layer"] for r in rows]
            line(ax, m, xs, [r["auprc"] for r in rows], [r["ci"]["auprc"][0] for r in rows], [r["ci"]["auprc"][1] for r in rows])
            ax.axhline(rows[0]["prevalence"], color=COLORS[m], lw=0.8, ls=":")
        ax.set_xlim(-0.5, 21)
    fig.text(0.01, 0.005, "Dotted horizontal: class prevalence (AUPRC of a random scorer).", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    place_labels()
    fig.savefig(figs / "fig1b_obedience_auprc.png", dpi=160)
    plt.close(fig)
    made.append("fig1b_obedience_auprc.png")

    # fig 2: transfer of the base probe vs each model's own probe
    others = [m for m in models if m != "base"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor="#fcfcfb")
    for ax, task in zip(axes, ("detect", "obey_base")):
        _style(ax, f"{'Detection' if task == 'detect' else 'Susceptibility'}: base probe (dashed) vs own probe (solid)", "AUROC")
        ax.axhline(0.5, color="#999", lw=0.8, ls=":")
        for m in others:
            tr = R["transfer"][m][task]
            xs = [r["layer"] for r in tr]
            line(ax, m, xs, [r["own_auroc"] for r in tr])
            ax.plot(xs, [r["base_probe_auroc"] for r in tr], color=COLORS[m], lw=2, ls="--", marker=MARKERS[m], ms=4,
                    markerfacecolor="white")
        ax.set_xlim(-0.5, 21)
    fig.tight_layout()
    place_labels()
    fig.savefig(figs / "fig2_transfer.png", dpi=160)
    plt.close(fig)
    made.append("fig2_transfer.png")

    # fig 3: cosine between base and each model's probe direction, with split-half ceiling and null band
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor="#fcfcfb")
    for ax, task in zip(axes, ("detect", "obey_base")):
        _style(ax, f"cos(base probe, model probe): {'detection' if task == 'detect' else 'susceptibility'}", "cosine",
               ylim=(-0.1, 1.01))
        null = R["cosine"]["null_95"]
        ax.axhspan(-null, null, color="#999", alpha=0.15, lw=0)
        for m in others:
            cs = R["cosine"][task][m]
            xs = [r["layer"] for r in cs]
            line(ax, m, xs, [r["cos_with_base"] for r in cs])
        base_half = R["cosine"][task]["base_split_half"]
        ax.plot([r["layer"] for r in base_half], [r["cos"] for r in base_half], color="#555", lw=1.2, ls=":",
                label="base split-half ceiling")
        ax.set_xlim(-0.5, 21)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.08))
    fig.text(0.01, 0.005, "Grey band: |cos| of random directions (95%). Dotted line: cosine between two base probes trained on "
             "disjoint halves of the training records.", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    place_labels()
    fig.savefig(figs / "fig3_cosine.png", dpi=160)
    plt.close(fig)
    made.append("fig3_cosine.png")
    return made


# --------------------------------------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="base,lora,e05,routed")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--plots-only", action="store_true", help="redraw figs from OUT/results.json")
    args = ap.parse_args()
    t0 = time.time()
    acts, out = Path(args.acts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = args.models.split(",")
    if args.plots_only:
        R = json.loads((out / "results.json").read_text())
        print(plot_all(R, R["models"], out / "figs"))
        return
    assert models[0] == "base", "base must come first (it is the transfer source)"

    keys, Xb, lab_b = load_model("base", acts)
    part = get_split([k[0] for k in keys])
    data = {"base": (Xb, lab_b)}
    for m in models[1:]:
        _, X, lab = load_model(m, acts, keys)
        data[m] = (X, lab)
    L1 = Xb.shape[1]
    H = Xb.shape[2]
    rec = np.array([k[0] for k in keys])
    pt = np.array([part[r] for r in rec])

    R: dict = {"models": models, "split": json.loads(SPLIT.read_text())["counts"], "probes": {}, "held_out": {},
               "transfer": {}, "cosine": {"null_95": round(2 / np.sqrt(H), 4)}, "config": {
                   "C_grid": C_GRID, "min_pos_train": MIN_POS_TRAIN, "min_pos_test": MIN_POS_TEST, "bootstrap": args.boot,
                   "label": "E07 'obeyed' (strict judge minus verbatim echoes)", "position": "last prompt token"}}
    probes: dict = {}
    for m in models:
        X, lab = data[m]
        print(f"[{m}] probes ...", flush=True)
        R["probes"][m] = analyse_model(m, keys, X, lab, lab_b, part, args.boot, probes)
        best = {t: max((r for r in R["probes"][m][t]["layers"] if r), key=lambda r: r["val_auroc"])["layer"]
                for t in ("detect", "obey_base")}
        layers = sorted({best["detect"], best["obey_base"], L1 // 2, L1 - 1})
        R["held_out"][m] = {"layers": layers, **held_out(m, keys, X, lab, lab_b, part, layers)}

    # experiment 2: transfer + cosine
    for task in ("detect", "obey_base", "obey_own"):
        R["cosine"][task] = {}
        mask, yb = task_rows(task, keys, lab_b, lab_b)
        trm, vam = mask & (pt == "train"), mask & (pt == "val")
        R["cosine"][task]["base_split_half"] = [
            {"layer": l, "cos": round(split_half_cosine(Xb, yb, trm, vam, rec, l, probes[("base", task, l)]["C"]), 4)}
            for l in range(L1) if ("base", task, l) in probes]
    for m in models[1:]:
        X, lab = data[m]
        R["transfer"][m] = {}
        for task in ("detect", "obey_base", "obey_own"):
            mask, y = task_rows(task, keys, lab, lab_b)
            te = mask & (pt == "test")
            ok = R["probes"][m][task]["sufficient"]
            rows, cos = [], []
            for l in range(L1):
                pb, pm = probes.get(("base", task, l)), probes.get((m, task, l))
                if pb is None:
                    continue
                r = {"layer": l, "base_probe_auroc": round(auroc(y[te], score(pb, X[te, l])), 4),
                     "base_probe_auprc": round(auprc(y[te], score(pb, X[te, l])), 4),
                     "base_probe_on_base_auroc": R["probes"]["base"][task]["layers"][l]["auroc"]}
                if pm is not None:
                    r["own_auroc"] = R["probes"][m][task]["layers"][l]["auroc"]
                    c = round(float(pb["dir"] @ pm["dir"]), 4)
                    r["cos_with_base"] = c
                    cos.append({"layer": l, "cos_with_base": c})
                rows.append(r)
            R["transfer"][m][task] = rows if ok or task != "obey_own" else {"insufficient": True, "rows": rows}
            R["cosine"][task][m] = cos
    for m in models:  # detection vs susceptibility direction, within a model
        R["cosine"].setdefault("detect_vs_obey_base", {})[m] = [
            {"layer": l, "cos": round(float(probes[(m, "detect", l)]["dir"] @ probes[(m, "obey_base", l)]["dir"]), 4)}
            for l in range(L1) if (m, "detect", l) in probes and (m, "obey_base", l) in probes]

    # directions for experiment 3 (raw space, unit norm), saved next to the results
    np.savez(out / "probe_directions.npz", **{f"{m}__{t}__L{l}": p["dir"].astype(np.float32)
                                              for (m, t, l), p in probes.items()})
    R["figures"] = plot_all(R, models, out / "figs")
    R["seconds"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(R, indent=1) + "\n")
    (out / "summary.md").write_text(summary_md(R, models))
    print(f"done in {R['seconds']} s -> {out}", flush=True)


def summary_md(R, models) -> str:
    md = ["# E09 probes: summary", "", f"Split (records): {R['split']}. Labels: {R['config']['label']}.", ""]
    for task, title in (("detect", "Injection detection"), ("obey_base", "Susceptibility (base obeyed)"),
                        ("obey_own", "Obedience (own labels)")):
        md += [f"## {title}", "", "| model | positives train/val/test | best layer (val) | test AUROC [95% CI] | "
               "test AUPRC (prevalence) | attack-type baseline |", "|---|---|---|---|---|---|"]
        for m in models:
            t = R["probes"][m][task]
            pos = f"{t['n_pos_train']}/{t['n_pos_val']}/{t['n_pos_test']}"
            if not t["sufficient"]:
                md.append(f"| {m} | {pos} | – | insufficient positives | – | – |")
                continue
            r = max((x for x in t["layers"] if x), key=lambda x: x["val_auroc"])
            ci = r["ci"]["auroc"] if r["ci"] else ["–", "–"]
            bl = t.get("attack_type_baseline", {}).get("auroc", "–")
            md.append(f"| {m} | {pos} | {r['layer']} | {r['auroc']} [{ci[0]}, {ci[1]}] | {r['auprc']} ({r['prevalence']}) | {bl} |")
        md.append("")
    md += ["## Transfer (base probe on each model, at the base probe's best val layer)", "",
           "| model | task | layer | base probe AUROC | own probe AUROC | cos(base, own) |", "|---|---|---|---|---|---|"]
    for m in models[1:]:
        for task in ("detect", "obey_base"):
            lb = max((x for x in R["probes"]["base"][task]["layers"] if x), key=lambda x: x["val_auroc"])["layer"]
            r = next(x for x in R["transfer"][m][task] if x["layer"] == lb)
            md.append(f"| {m} | {task} | {lb} | {r['base_probe_auroc']} | {r.get('own_auroc', '–')} | {r.get('cos_with_base', '–')} |")
    md += ["", f"Random-direction |cos| (95%): < {R['cosine']['null_95']}.", ""]
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    main()
