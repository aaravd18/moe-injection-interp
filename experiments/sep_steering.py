"""Roadmap #5 + #2 (GPU): is the "instruction in data" signal causal, and does routing alone carry obedience?

#5  Direction steering. Directions are difference-of-means fitted on probe half A from features.npz:
      d_mid[l]  = mean gate input (resid_mid) over injected-probe tokens - over benign-sentence tokens   ("data-instr")
      d_last[l] = mean resid at the final prompt token, injection - benign                              ("noticed")
    and applied to held-out prompts (probe half B). Steering edits resid_post at the chosen layers:
      data_*    only the USER MESSAGE tokens (the untrusted data, incl. the probe), prompt pass only - what a
                provenance-gated security expert would touch
      all_*     every prompt token and every generated token (no provenance gating)
    Controls: a random direction with typical activation variance (random combination of centred activations), same
    layers/tokens/strength. Utility = trusted probe in the SYSTEM prompt; its data tokens are steered too, so a good
    defense must leave it intact.

#2  Route patching. The same probe tokens are routed once in the data (injection) and once in the system prompt
    (user_instruction). We give the injected probe tokens the expert choices of their trusted twin (router's own
    weights for those experts), and the reverse on the utility set. Controls: random experts; the item's own routes
    shuffled across the probe positions.

Evaluation: as sep_interventions.py (reproducibility filter on a larger pool, paired bootstrap CIs, whole-word labels).

Usage: python experiments/sep_steering.py ~/moe-results/mellum_sep --config configs/mellum_sep.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from moe_interp.config import load_config
from moe_interp.interventions import Intervention, ResidSteer, RouteOverride, padded_mask
from moe_interp.model import load_model
from moe_interp.runner import _generation_config, generate_batch


def split_probes(idx_rows: list[dict], seed: int = 0) -> set:
    """Identical to sep_interventions.split_probes (same half A)."""
    probes = sorted({r["probe_id"] for r in idx_rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(probes)
    return set(probes[: len(probes) // 2])


def wb_hit(text: str, w: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])", text or "", re.I) is not None


def parse_layers(s: str, L: int) -> list[int]:
    if s == "all":
        return list(range(L))
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--config", required=True)
    ap.add_argument("--features", default=None)
    ap.add_argument("--early", type=int, default=120)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--pool", type=int, default=192)
    ap.add_argument("--n-eval", type=int, default=96)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--layers", default="8-22")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run).expanduser()
    out = Path(args.out).expanduser() if args.out else run / "security" / "steering"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    recs = [json.loads(l) for l in open(run / "generations.jsonl")]
    idx = [json.loads(l) for l in open(run / "features_index.jsonl")]
    byk = {(r["sample_id"], r["condition"]): r for r in recs}
    half_a = split_probes(idx)
    for r in recs:
        r["_hit"] = wb_hit(r.get("response_answer") or "", r["evaluation"]["witness"])

    # ---- directions from probe half A ----------------------------------------------------------------------------
    F = np.load(Path(args.features).expanduser() if args.features else run / "features.npz")
    cond = np.array([r["condition"] for r in idx])
    in_a = np.array([r["probe_id"] in half_a for r in idx])
    inj_a, ben_a = np.where((cond == "injection") & in_a)[0], np.where((cond == "benign") & in_a)[0]
    mid_p, mid_b, last = F["mid_probe"], F["mid_insert"], F["resid_last"]
    L = mid_p.shape[1]

    def rand_dir(X: np.ndarray) -> np.ndarray:
        Xc = X - X.mean(0)
        v = Xc.T @ rng.standard_normal(len(Xc))
        return v / np.linalg.norm(v)

    D = {"mid": {}, "last": {}, "mid_rand": {}, "last_rand": {}}
    info = {"cos_mid_rand": [], "cos_last_rand": [], "cos_mid_last": []}
    for l in range(L):
        xi, xb = mid_p[inj_a, l].astype(np.float32), mid_b[ben_a, l].astype(np.float32)
        d = xi.mean(0) - xb.mean(0)
        D["mid"][l] = d / np.linalg.norm(d)
        D["mid_rand"][l] = rand_dir(np.r_[xi, xb])
        li, lb = last[inj_a, l].astype(np.float32), last[ben_a, l].astype(np.float32)
        e = li.mean(0) - lb.mean(0)
        D["last"][l] = e / np.linalg.norm(e)
        D["last_rand"][l] = rand_dir(np.r_[li, lb])
        info["cos_mid_rand"].append(float(D["mid"][l] @ D["mid_rand"][l]))
        info["cos_last_rand"].append(float(D["last"][l] @ D["last_rand"][l]))
        info["cos_mid_last"].append(float(D["mid"][l] @ D["last"][l]))
    del F, mid_p, mid_b, last

    # ---- model + evaluation sets (probe half B) -------------------------------------------------------------------
    cfg = load_config(args.config)
    lm = load_model(cfg.model, seed=cfg.seed)
    tok, model, layout, device = lm.tokenizer, lm.model, lm.layout, lm.device
    gc = _generation_config(model, cfg, tok)
    gc.max_new_tokens = args.max_new
    layers = parse_layers(args.layers, L)

    def early(r):
        a = (r.get("response_answer") or "")
        m = re.search(r"(?<![A-Za-z0-9])" + re.escape(r["evaluation"]["witness"]) + r"(?![A-Za-z0-9])", a, re.I)
        return m is not None and len(tok(a[:m.start()], add_special_tokens=False)["input_ids"]) <= args.early

    attack = [r for r in recs if r["condition"] == "injection" and r["metadata"]["probe_id"] not in half_a
              and r["_hit"] and early(r)]
    utility = [r for r in recs if r["condition"] == "user_instruction" and r["metadata"]["probe_id"] not in half_a
               and r["_hit"] and early(r)]
    rng.shuffle(attack)
    rng.shuffle(utility)
    attack, utility = attack[: args.pool], utility[: args.pool]

    def user_span(r):
        s = r["message_token_spans"][1]
        return (s["start"], s["end"])

    def probe_span(r):
        s = r["spans"]["probe"]
        return (s["start"], s["end"])

    def run_gen(rows, make_iv):
        hits = []
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            ids = [r["input_ids"] for r in chunk]
            T = max(map(len, ids))
            iv = make_iv(chunk, T)
            if iv is None:
                resp = generate_batch(model, ids, gc, device)
            else:
                with iv:
                    resp = generate_batch(model, ids, gc, device)
            hits += [wb_hit(tok.decode(x, skip_special_tokens=True), r["evaluation"]["witness"])
                     for r, x in zip(chunk, resp)]
        return hits

    fragility = {}
    for nm in ("attack", "utility"):
        rows = attack if nm == "attack" else utility
        h = run_gen(rows, lambda c, T: None)
        fragility[nm] = {"pool": len(rows), "reproduced": int(sum(h)), "rate": float(np.mean(h))}
        kept = [r for r, x in zip(rows, h) if x][: args.n_eval]
        if nm == "attack":
            attack = kept
        else:
            utility = kept
    clean = [byk[(r["sample_id"], "clean")] for r in attack[:48]]
    print("fragility", json.dumps(fragility), f"attack {len(attack)} utility {len(utility)}", flush=True)

    results: list[dict] = []
    base: dict[str, np.ndarray] = {}

    def ci(new, key):
        d = np.asarray(new, float) - base[key]
        b = np.random.default_rng(7).integers(0, len(d), size=(2000, len(d)))
        m = d[b].mean(1)
        return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]

    def dump():
        (out / "results.json").write_text(json.dumps(
            {"fragility": fragility, "n_attack": len(attack), "n_utility": len(utility), "layers": layers,
             "direction_info": info, "results": results}, indent=1))

    @torch.inference_mode()
    def run_kl(make_iv):
        kls = []
        for r in clean:
            p, a = r["input_ids"], r["response_ids"][:64]
            x = torch.tensor([p + a], device=device)
            b_ = model(input_ids=x, use_cache=False).logits[0, len(p) - 1:-1].float().log_softmax(-1)
            # KL on the teacher-forced answer: the mask covers the user message; answer positions are 'generated'
            iv = make_iv([r], len(p), extra=len(a))
            with iv:
                n_ = model(input_ids=x, use_cache=False).logits[0, len(p) - 1:-1].float().log_softmax(-1)
            kls.append(float((b_.exp() * (b_ - n_)).sum(-1).mean()))
        return float(np.mean(kls))

    def record(name, family, ha=None, hu=None, kl=None, extra=None):
        res = {"name": name, "family": family}
        if ha is not None:
            if family == "baseline":
                base["attack"] = np.asarray(ha, float)
            res.update(attack_success=float(np.mean(ha)), attack_delta_ci=ci(ha, "attack"),
                       attack_hits=[bool(x) for x in ha])
        if hu is not None:
            if family == "baseline":
                base["utility"] = np.asarray(hu, float)
            res.update(utility_exec=float(np.mean(hu)), utility_delta_ci=ci(hu, "utility"),
                       utility_hits=[bool(x) for x in hu])
        if kl is not None:
            res["clean_kl"] = kl
        res.update(extra or {})
        results.append(res)
        print(json.dumps({k: v for k, v in res.items() if not k.endswith("_hits")}), flush=True)
        dump()

    t0 = time.time()
    record("baseline", "baseline", run_gen(attack, lambda c, T: None), run_gen(utility, lambda c, T: None), 0.0)

    # ---- #5 direction steering ------------------------------------------------------------------------------------
    def steer_iv(dirs, lys, coef, gated, decode):
        def make(chunk, T, extra=0):
            dd = {l: torch.from_numpy(dirs[l]) for l in lys}
            if gated:
                m = padded_mask([user_span(r) for r in chunk], [len(r["input_ids"]) for r in chunk], T)
                if extra:  # teacher-forced KL pass: answer tokens appended, never steered when gated
                    m = torch.cat([m, torch.zeros(m.shape[0], extra, dtype=torch.bool)], 1)
                return Intervention(layout, steer=ResidSteer(dd, token_mask=m, coef=coef, decode=False))
            return Intervention(layout, steer=ResidSteer(dd, token_mask=None, coef=coef, decode=decode))
        return make

    conds = [
        ("data_ablate_mid", "mid", layers, 1.0, True, False),
        ("data_reflect_mid", "mid", layers, 2.0, True, False),
        ("data_ablate_mid_all_layers", "mid", list(range(L)), 1.0, True, False),
        ("all_ablate_mid", "mid", layers, 1.0, False, True),
        ("all_ablate_last", "last", layers, 1.0, False, True),
    ]
    for name, key, lys, coef, gated, decode in conds:
        for tag, k2 in (("", key), (" [random dir]", key + "_rand")):
            make = steer_iv(D[k2], lys, coef, gated, decode)
            t = time.time()
            record(name + tag, name + ("_random" if tag else ""), run_gen(attack, make), run_gen(utility, make),
                   run_kl(make), {"layers": f"{lys[0]}-{lys[-1]}", "coef": coef, "gated_to_data": gated,
                                  "seconds": round(time.time() - t, 1)})

    # ---- #2 route patching ----------------------------------------------------------------------------------------
    @torch.inference_mode()
    def routes_of(r) -> dict[int, torch.Tensor]:
        got = {}
        hs = [layout.routers[l].register_forward_hook(lambda m, a, o, l=l: got.__setitem__(l, o[2].clone()))
              for l in range(L)]
        try:
            model(input_ids=torch.tensor([r["input_ids"]], device=device), use_cache=False)
        finally:
            for h in hs:
                h.remove()
        s, e = probe_span(r)
        return {l: got[l][s:e].cpu() for l in range(L)}

    def twins(rows, other):
        keep, own, tw = [], [], []
        for r in rows:
            o = byk[(r["sample_id"], other)]
            a, b = probe_span(r), probe_span(o)
            if r["input_ids"][a[0]:a[1]] != o["input_ids"][b[0]:b[1]]:
                continue
            keep.append(r)
            own.append(routes_of(r))
            tw.append(routes_of(o))
        return keep, own, tw

    def route_iv(rows_all, routes_all, lys, mode):
        pos = {id(r): i for i, r in enumerate(rows_all)}

        def make(chunk, T, extra=0):
            m = padded_mask([probe_span(r) for r in chunk], [len(r["input_ids"]) for r in chunk], T)
            new = {}
            for l in lys:
                parts = []
                for r in chunk:
                    x = routes_all[pos[id(r)]][l]
                    if mode == "shuffle":
                        x = x[torch.from_numpy(rng.permutation(len(x)))]
                    elif mode == "random":
                        x = torch.stack([torch.from_numpy(rng.choice(layout.num_experts, layout.top_k, replace=False))
                                         for _ in range(len(x))])
                    parts.append(x)
                new[l] = torch.cat(parts)
            return Intervention(layout, routes=RouteOverride(new, m))
        return make

    a_rows, a_own, a_tw = twins(attack, "user_instruction")
    u_rows, u_own, u_tw = twins(utility, "injection")
    ia = [attack.index(r) for r in a_rows]
    iu = [utility.index(r) for r in u_rows]
    base_a, base_u = base["attack"], base["utility"]
    base["attack"], base["utility"] = base_a[ia], base_u[iu]
    rinfo = {"n_attack_aligned": len(a_rows), "n_utility_aligned": len(u_rows),
             "baseline_attack_aligned": float(base["attack"].mean()),
             "baseline_utility_aligned": float(base["utility"].mean())}
    # agreement between own and twin routes on the probe tokens (share of the top-k sets that coincide)
    agree = [np.mean([len(set(a.tolist()) & set(b.tolist())) / layout.top_k
                      for a, b in zip(o[l], t[l])]) for o, t in zip(a_own, a_tw) for l in layers]
    rinfo["mean_topk_overlap_own_vs_twin"] = float(np.mean(agree))
    print("routes", json.dumps(rinfo), flush=True)
    for nm, lys in (("routes_from_trusted_twin", layers), ("routes_from_trusted_twin_all_layers", list(range(L)))):
        record(nm, "route_patch", ha=run_gen(a_rows, route_iv(a_rows, a_tw, lys, "twin")),
               extra={"layers": f"{lys[0]}-{lys[-1]}", **rinfo})
    record("routes_own_shuffled", "route_patch_control", ha=run_gen(a_rows, route_iv(a_rows, a_own, layers, "shuffle")),
           extra={"layers": f"{layers[0]}-{layers[-1]}"})
    record("routes_random", "route_patch_control", ha=run_gen(a_rows, route_iv(a_rows, a_own, layers, "random")),
           extra={"layers": f"{layers[0]}-{layers[-1]}"})
    record("utility: routes_from_injected_twin", "route_patch_reverse",
           hu=run_gen(u_rows, route_iv(u_rows, u_tw, layers, "twin")), extra={"layers": f"{layers[0]}-{layers[-1]}"})
    record("utility: routes_own_shuffled", "route_patch_reverse_control",
           hu=run_gen(u_rows, route_iv(u_rows, u_own, layers, "shuffle")), extra={"layers": f"{layers[0]}-{layers[-1]}"})
    print("done in", round(time.time() - t0), "s; wrote", out / "results.json")


if __name__ == "__main__":
    main()
