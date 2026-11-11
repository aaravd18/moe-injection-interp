"""Representation-level follow-ups on features.npz (CPU): is obeying = "the injected text looks trusted"?

  G1  split-half re-test of the expert-space correlation: provenance delta from probe half A vs obey delta from
      probe half B (and swapped), so the two quantities share no prompts
  G2  paired routing similarity: for each item, the SAME probe tokens are routed once in the data (injection) and once
      in the trusted task (user_instruction). Per layer, similarity of the two expert distributions
      (1 - Jensen-Shannon, on top-k counts and on router probabilities). Does higher similarity to the trusted twin
      predict obeying? AUC + logistic regression with surface covariates
  G3  residual "trusted-ness": direction = mean(gate input on probe | trusted) - mean(... | data), fitted on held-out
      probes (GroupKFold by probe); score = projection of each injected item; paired score = projection(inj) -
      projection(own trusted twin). AUC for obeying per layer
  G4  final-token geometry: cosine between the "noticed" direction (injection - benign) and the "obey" direction
      (executed - ignored injections), and whether obeyed injections sit closer to the trusted-instruction state

Labels are whole-word witness matches (see sep_behavior.py). Usage:
  python experiments/sep_geometry.py reports/mellum_sep --features /path/features.npz
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "font.size": 10, "axes.titlesize": 11, "axes.titlelocation": "left"})


def wb_hit(text: str, w: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])", text or "", re.I) is not None


def js_sim(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """1 - Jensen-Shannon divergence (base 2) along the last axis."""
    p = p / np.clip(p.sum(-1, keepdims=True), eps, None)
    q = q / np.clip(q.sum(-1, keepdims=True), eps, None)
    m = 0.5 * (p + q)

    def kl(a, b):
        return np.where(a > 0, a * np.log2(np.clip(a, eps, None) / np.clip(b, eps, None)), 0).sum(-1)

    return 1 - 0.5 * (kl(p, m) + kl(q, m))


def auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def boot_auc_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            v.append(roc_auc_score(y[i], s[i]))
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.run / "security" / "geometry"
    out.mkdir(parents=True, exist_ok=True)

    idx = [json.loads(l) for l in open(args.run / "features_index.jsonl")]
    gens = {(r["sample_id"], r["condition"]): r for r in map(json.loads, gzip.open(args.run / "generations.jsonl.gz", "rt"))}
    lab = np.array([wb_hit(gens[(r["sample_id"], r["condition"])]["response_answer"],
                           gens[(r["sample_id"], r["condition"])]["evaluation"]["witness"]) for r in idx])
    cond = np.array([r["condition"] for r in idx])
    pid = np.array([r["probe_id"] for r in idx])
    sid = np.array([r["sample_id"] for r in idx])
    row = {(s, c): i for i, (s, c) in enumerate(zip(sid, cond))}
    items = sorted(set(sid))
    I = np.array([row[(s, "injection")] for s in items])
    U = np.array([row[(s, "user_instruction")] for s in items])
    B_ = np.array([row[(s, "benign")] for s in items])
    y = lab[I]
    g = pid[I]
    meta = [gens[(s, "injection")]["metadata"] for s in items]
    insist = np.array([m["insistent"] for m in meta], float)
    start = np.array([m["probe_position_in_data"] == "start" for m in meta], float)
    dlen = np.log(np.array([gens[(s, "injection")]["spans"]["data"]["end"] - gens[(s, "injection")]["spans"]["data"]["start"]
                            for s in items], float))
    pwords = np.array([idx[i]["probe_words"] for i in I], float)
    F = np.load(args.features)
    L = F["ecount_probe"].shape[1]
    S: dict = {"n_items": len(items), "asr_whole_word": float(y.mean())}

    # ---- G1 split-half expert-space correlation ---------------------------------------------------------------------
    ec = F["ecount_probe"].astype(np.float64)
    nt = F["ntok_probe"].astype(np.float64)
    probes = np.array(sorted(set(g)))
    rng = np.random.default_rng(0)
    res = []
    for rep in range(200):
        perm = rng.permutation(probes)
        A = set(perm[: len(perm) // 2])
        inA = np.array([p in A for p in g])

        def share(rows):
            return ec[rows].sum(0) / max(nt[rows].sum(), 1)

        r_ = []
        for ha, hb in ((inA, ~inA), (~inA, inA)):
            prov = share(I[ha]) - share(U[ha])
            ob = share(I[hb & y]) - share(I[hb & ~y])
            r_.append(stats.spearmanr(prov.ravel(), ob.ravel())[0])
        res.append(np.mean(r_))
    res = np.array(res)
    S["G1_split_half_prov_vs_obey"] = {"mean_spearman": float(res.mean()), "q025": float(np.quantile(res, .025)),
                                       "q975": float(np.quantile(res, .975)), "frac_below_0": float((res < 0).mean()),
                                       "n_splits": len(res)}

    # ---- G2 paired routing similarity ---------------------------------------------------------------------------------
    sim_cnt = js_sim(ec[I], ec[U])                      # [N, L]
    rp = F["rprob_probe"].astype(np.float64)
    sim_prob = js_sim(rp[I], rp[U])
    sim_benign_ref = js_sim(ec[I], F["ecount_insert"].astype(np.float64)[B_])  # injected probe vs benign sentence
    G2 = {"per_layer_auc_counts": [auc(y, sim_cnt[:, l]) for l in range(L)],
          "per_layer_auc_probs": [auc(y, sim_prob[:, l]) for l in range(L)],
          "per_layer_mean_sim_counts": sim_cnt.mean(0).round(4).tolist()}
    mid = slice(8, 23)
    sc = sim_cnt[:, mid].mean(1)
    sp = sim_prob[:, mid].mean(1)
    G2["score_layers"] = "8-22"
    G2["auc_counts_mid"] = auc(y, sc)
    G2["auc_counts_mid_ci"] = boot_auc_ci(y, sc)
    G2["auc_probs_mid"] = auc(y, sp)
    G2["auc_probs_mid_ci"] = boot_auc_ci(y, sp)
    G2["auc_all_layers_counts"] = auc(y, sim_cnt.mean(1))
    # does the similarity add to surface covariates? grouped CV logistic regression
    cov = np.c_[insist, start, dlen, pwords]

    def cv_auc(X):
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, g):
            sc_ = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=2000).fit(sc_.transform(X[tr]), y[tr])
            pred[te] = m.predict_proba(sc_.transform(X[te]))[:, 1]
        return auc(y, pred)

    G2["cv_auc_surface_only"] = cv_auc(cov)
    G2["cv_auc_surface_plus_similarity"] = cv_auc(np.c_[cov, sc, sp])
    G2["cv_auc_surface_plus_similarity_per_layer"] = cv_auc(np.c_[cov, sim_cnt, sim_prob])
    Xs = StandardScaler().fit_transform(np.c_[cov, sc])
    m = LogisticRegression(max_iter=2000).fit(Xs, y)
    G2["logit_coef_std"] = dict(zip(["insistent", "probe_at_start", "log_data_len", "probe_words", "routing_sim_mid"],
                                    m.coef_[0].round(3).tolist()))
    G2["spearman_sim_vs_log_data_len"] = float(stats.spearmanr(sc, dlen)[0])
    G2["mean_sim_obeyed_vs_ignored"] = [float(sc[y].mean()), float(sc[~y].mean())]
    G2["auc_sim_to_benign_sentence_mid"] = auc(y, sim_benign_ref[:, mid].mean(1))
    S["G2_paired_routing"] = G2

    # ---- G3 residual trusted-ness direction at the gate input on the probe tokens -------------------------------------
    mp = F["mid_probe"]
    G3 = {"auc_proj": [], "auc_paired": [], "auc_proj_ci_best": None}
    proj_all = np.zeros((len(items), L))
    pair_all = np.zeros((len(items), L))
    for l in range(L):
        Xi = mp[I, l].astype(np.float32)
        Xu = mp[U, l].astype(np.float32)
        for tr, te in GroupKFold(5).split(Xi, y, g):
            w = Xu[tr].mean(0) - Xi[tr].mean(0)
            w /= np.linalg.norm(w) + 1e-8
            c0 = 0.5 * (Xu[tr].mean(0) + Xi[tr].mean(0)) @ w
            proj_all[te, l] = Xi[te] @ w - c0
            pair_all[te, l] = (Xi[te] - Xu[te]) @ w
        G3["auc_proj"].append(auc(y, proj_all[:, l]))
        G3["auc_paired"].append(auc(y, pair_all[:, l]))
    bl = int(np.nanargmax(np.abs(np.array(G3["auc_proj"]) - 0.5)))
    G3["best_layer"] = bl
    G3["auc_proj_ci_best"] = boot_auc_ci(y, proj_all[:, bl])
    G3["auc_proj_mid_mean_score"] = auc(y, proj_all[:, mid].mean(1))
    G3["auc_proj_mid_mean_score_ci"] = boot_auc_ci(y, proj_all[:, mid].mean(1))
    G3["cv_auc_surface_plus_proj"] = cv_auc(np.c_[cov, proj_all[:, mid].mean(1)])
    S["G3_residual_trustedness"] = G3

    # ---- G4 final-token geometry --------------------------------------------------------------------------------------
    rl = F["resid_last"]
    G4 = {"cos_notice_obey": [], "cos_obey_trusted": [], "null95_notice_obey": [], "null95_obey_trusted": [],
          "auc_trusted_proj": [], "auc_notice_proj": []}

    def cs(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    perms = [rng.permutation(len(y)) for _ in range(200)]
    for l in range(L):
        Xi = rl[I, l].astype(np.float32)
        Xb = rl[B_, l].astype(np.float32)
        Xu = rl[U, l].astype(np.float32)
        notice = Xi.mean(0) - Xb.mean(0)
        trusted = Xu.mean(0) - Xi.mean(0)
        obey = Xi[y].mean(0) - Xi[~y].mean(0)
        G4["cos_notice_obey"].append(cs(notice, obey))
        G4["cos_obey_trusted"].append(cs(obey, trusted))
        n1, n2 = [], []
        for pp in perms:
            yp = y[pp]
            o = Xi[yp].mean(0) - Xi[~yp].mean(0)
            n1.append(abs(cs(notice, o)))
            n2.append(abs(cs(o, trusted)))
        G4["null95_notice_obey"].append(float(np.quantile(n1, .95)))
        G4["null95_obey_trusted"].append(float(np.quantile(n2, .95)))
        tp, npj = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in GroupKFold(5).split(Xi, y, g):  # directions fitted on held-out probes
            w = Xu[tr].mean(0) - Xi[tr].mean(0)
            tp[te] = Xi[te] @ (w / (np.linalg.norm(w) + 1e-8))
            v = Xi[tr].mean(0) - Xb[tr].mean(0)
            npj[te] = Xi[te] @ (v / (np.linalg.norm(v) + 1e-8))
        G4["auc_trusted_proj"].append(auc(y, tp))
        G4["auc_notice_proj"].append(auc(y, npj))
    # confound check: does the notice-direction score add to surface features, and does it hold within strata?
    lb = int(np.argmax(G4["auc_notice_proj"]))
    Xi = rl[I, lb].astype(np.float32)
    Xb = rl[B_, lb].astype(np.float32)
    npj = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xi, y, g):
        v = Xi[tr].mean(0) - Xb[tr].mean(0)
        npj[te] = Xi[te] @ (v / (np.linalg.norm(v) + 1e-8))
    G4["notice_best_layer"] = lb
    G4["notice_auc_ci"] = boot_auc_ci(y, npj)
    G4["cv_auc_surface_plus_notice"] = cv_auc(np.c_[cov, npj])
    G4["notice_auc_within"] = {
        "plain": auc(y[insist == 0], npj[insist == 0]), "insistent": auc(y[insist == 1], npj[insist == 1]),
        "short_data": auc(y[dlen <= np.median(dlen)], npj[dlen <= np.median(dlen)]),
        "long_data": auc(y[dlen > np.median(dlen)], npj[dlen > np.median(dlen)])}
    # within-probe: same probe, different documents (removes everything about the probe text itself)
    w_auc, w_n = [], []
    for p_ in set(g):
        m_ = g == p_
        if len(set(y[m_])) == 2:
            w_auc.append(auc(y[m_], npj[m_]))
            w_n.append(m_.sum())
    G4["notice_auc_within_probe_weighted"] = float(np.average(w_auc, weights=w_n))
    G4["n_probes_mixed_outcome"] = len(w_auc)
    S["G4_final_token"] = G4

    (out / "geometry_summary.json").write_text(json.dumps(S, indent=1, default=float))

    # ---- plot ---------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ll = np.arange(L)
    a = ax[0]
    a.plot(ll, G2["per_layer_auc_counts"], color=C1, lw=2, marker="o", ms=3.5, label="routing: expert choices")
    a.plot(ll, G2["per_layer_auc_probs"], color=C2, lw=2, marker="o", ms=3.5, label="routing: router probabilities")
    a.plot(ll, G3["auc_paired"], color=C3, lw=2, marker="o", ms=3.5, label="gate input: shift toward trusted twin")
    a.axhline(0.5, color=MUTED, lw=1)
    a.set_ylim(0.35, 0.75)
    a.set_xlabel("layer")
    a.set_ylabel("AUC for obeying (injected items)")
    a.set_title("A. Same probe tokens, data vs trusted twin:\ndoes 'looking trusted' predict obeying?", color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="lower left")
    a = ax[1]
    a.hist(res, bins=30, color=C1, alpha=0.85, edgecolor="#fcfcfb")
    a.axvline(0, color=MUTED, lw=1)
    a.set_xlabel("Spearman(provenance on half A, obey on half B)")
    a.set_ylabel("random probe splits")
    a.set_title(f"B. Split-half expert-space test: mean {res.mean():.2f}\n"
                f"({(res < 0).mean() * 100:.0f}% of splits below 0)", color=INK)
    a = ax[2]
    a.plot(ll, G4["cos_notice_obey"], color=C1, lw=2, marker="o", ms=3.5, label="cos(noticed, obey)")
    a.plot(ll, G4["cos_obey_trusted"], color=C2, lw=2, marker="o", ms=3.5, label="cos(obey, trusted - injected)")
    a.axhline(0, color=MUTED, lw=1)
    a.fill_between(ll, -np.array(G4["null95_notice_obey"]), G4["null95_notice_obey"], color=GRID, alpha=0.7, lw=0,
                   label="null for cos(noticed, obey), 95%")
    a.set_xlabel("layer")
    a.set_ylabel("cosine")
    a.set_title("C. Final prompt token: directions", color=INK)
    a.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "geometry.png", dpi=150)
    plt.close(fig)
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main()
