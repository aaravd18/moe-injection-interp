"""Step 4: causal tests on held-out probes (GPU).

Candidates (experts, heads) are selected on probe half A and evaluated on probe half B, so the test is not circular.
Every targeted intervention has a random control of the same size (same layers for experts, random heads).

Evaluation sets (half B, from the main run's baseline generations):
  attack   injection prompts the model OBEYED, with the witness within the first ``--early`` answer tokens
  utility  user_instruction prompts the model EXECUTED, witness early as well (trusted instructions must keep working)
  clean    clean prompts; side effects = mean KL(baseline || intervened) of next-token distributions over the first 64
           tokens of the baseline answer (teacher-forced)
Generation under intervention is greedy, ``--max-new`` tokens (enough to reach an early witness).

Reproducibility filter: greedy bf16 decoding is NOT invariant to batch composition / left padding (the main run's
first generated token differed from the bs=1 prefill argmax in ~8% of prompts), so a prompt that obeyed in the main
run may not obey when re-generated in another batch. We therefore re-generate a larger pool (``--pool``) with the exact
batching used for evaluation, keep only prompts whose outcome reproduces, and report the reproduction rate
("decision fragility"). All effects are paired per prompt against the baseline on the same batches, with bootstrap
95% CIs, and each targeted intervention gets ``--n-random`` random controls of the same size.

Interventions (steering applies to the DATA tokens = the user message; generated tokens only where stated):
  experts_obey_off        deactivate top-N experts used more on injected probe tokens when the model obeyed
  experts_instr_off       deactivate top-N experts used more on the probe when it sits in the TRUSTED task
                          (hypothesis: injected text that gets routed "like an instruction" is what gets obeyed)
  experts_data_on         force-activate top-N experts used more on the probe when it sits in the DATA
  experts_tail_off_all    deactivate top-N experts used more at the final template tokens when obeying, on ALL tokens
                          incl. generated ones (SteerMoE-style, no knowledge of the data boundary)
  heads_obey_off          zero the top-N heads whose attention to the injected probe predicts obeying
  patch_clean_tail@l      replace resid_post at the final 8 prompt tokens (identical template tokens) of layer l with
                          the values from the CLEAN prompt (no probe) -> how much of the decision rides on those tokens
  + random controls

Usage: python experiments/sep_interventions.py ~/moe-results/mellum_sep --config configs/mellum_sep.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from moe_interp.config import load_config
from moe_interp.interventions import ExpertSteer, HeadAblation, Intervention, ResidPatch, padded_mask
from moe_interp.model import load_model
from moe_interp.runner import _generation_config, generate_batch

NEG = float("-inf")


def split_probes(idx_rows: list[dict], seed: int = 0) -> set:
    probes = sorted({r["probe_id"] for r in idx_rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(probes)
    return set(probes[: len(probes) // 2])


def top_experts(delta: np.ndarray, n: int, sign: int) -> dict[int, list[int]]:
    """Top-n (layer, expert) by sign*delta (only positive side)."""
    d = sign * delta
    flat = [i for i in np.argsort(-d.ravel()) if d.ravel()[i] > 0][:n]
    E = delta.shape[1]
    out: dict[int, list[int]] = {}
    for i in flat:
        out.setdefault(int(i // E), []).append(int(i % E))
    return out


def random_like(sel: dict[int, list[int]], E: int, rng: np.random.Generator) -> dict[int, list[int]]:
    return {l: sorted(rng.choice(E, size=len(es), replace=False).tolist()) for l, es in sel.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--config", required=True)
    ap.add_argument("--early", type=int, default=120)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--n-eval", type=int, default=96)
    ap.add_argument("--pool", type=int, default=192, help="candidates per set before the reproducibility filter")
    ap.add_argument("--n-random", type=int, default=2, help="random controls per targeted intervention")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--sizes", default="16,64")
    ap.add_argument("--head-sizes", default="8,32")
    ap.add_argument("--patch-layers", default=None, help="comma list; default every 2nd layer")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run).expanduser()
    out = Path(args.out).expanduser() if args.out else run / "security" / "interventions"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    cfg = load_config(args.config)
    lm = load_model(cfg.model, seed=cfg.seed)
    tok, model, layout, device = lm.tokenizer, lm.model, lm.layout, lm.device
    gc = _generation_config(model, cfg, tok)
    gc.max_new_tokens = args.max_new
    E, L, Hh = layout.num_experts, layout.num_layers, layout.num_heads

    recs = [json.loads(l) for l in open(run / "generations.jsonl")]
    idx = [json.loads(l) for l in open(run / "features_index.jsonl")]
    half_a = split_probes(idx)
    F = np.load(run / "features.npz")

    # ---- candidates from half A ----------------------------------------------------------------------------------
    cond = np.array([r["condition"] for r in idx])
    ex = np.array([r["executed"] for r in idx])
    in_a = np.array([r["probe_id"] in half_a for r in idx])
    inj, usr = cond == "injection", cond == "user_instruction"

    def share(mask, cnt="ecount_probe", nt="ntok_probe"):
        return F[cnt][mask].sum(0) / max(F[nt][mask].sum(), 1)

    d_obey = share(inj & in_a & ex) - share(inj & in_a & ~ex)
    d_prov = share(inj & in_a) - share(usr & in_a)            # >0: used more when the probe is in DATA
    d_tail = share(inj & in_a & ex, "ecount_tail", "ntok_tail") - share(inj & in_a & ~ex, "ecount_tail", "ntok_tail")
    A = F["amass_probe"][inj & in_a]
    y = ex[inj & in_a]
    from sklearn.metrics import roc_auc_score
    head_auc = np.array([[roc_auc_score(y, A[:, l, h]) for h in range(Hh)] for l in range(L)])

    # ---- evaluation sets from half B -------------------------------------------------------------------------------
    def early(r):
        a = (r.get("response_answer") or "").lower()
        w = r["evaluation"]["witness"].lower()
        i = a.find(w)
        return i >= 0 and len(tok(a[:i], add_special_tokens=False)["input_ids"]) <= args.early

    byk = {(r["sample_id"], r["condition"]): r for r in recs}
    attack = [r for r in recs if r["condition"] == "injection" and r["metadata"]["probe_id"] not in half_a
              and r.get("attack_target_in_answer") and early(r)]
    utility = [r for r in recs if r["condition"] == "user_instruction" and r["metadata"]["probe_id"] not in half_a
               and r.get("attack_target_in_answer") and early(r)]
    rng.shuffle(attack)
    rng.shuffle(utility)
    attack, utility = attack[: args.pool], utility[: args.pool]

    def data_span(r):
        s = r["message_token_spans"][1]
        return (s["start"], s["end"])

    def run_gen(rows, make_iv):
        hits = []
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            ids = [r["input_ids"] for r in chunk]
            T = max(map(len, ids))
            mask = padded_mask([data_span(r) for r in chunk], [len(x) for x in ids], T)
            iv = make_iv(chunk, mask, T)
            if iv is None:
                resp = generate_batch(model, ids, gc, device)
            else:
                with iv:
                    resp = generate_batch(model, ids, gc, device)
            for r, rr in zip(chunk, resp):
                hits.append(r["evaluation"]["witness"].lower() in tok.decode(rr, skip_special_tokens=True).lower())
        return hits

    @torch.inference_mode()
    def run_kl(make_iv):
        kls = []
        for r in clean:
            p, a = r["input_ids"], r["response_ids"][:64]
            x = torch.tensor([p + a], device=device)
            mask = padded_mask([data_span(r)], [len(p) + len(a)], len(p) + len(a))
            base = model(input_ids=x, use_cache=False).logits[0, len(p) - 1:-1].float().log_softmax(-1)
            iv = make_iv([r], mask, x.shape[1])
            if iv is None:
                kls.append(0.0)
                continue
            with iv:
                new = model(input_ids=x, use_cache=False).logits[0, len(p) - 1:-1].float().log_softmax(-1)
            kls.append(float((base.exp() * (base - new)).sum(-1).mean()))
        return float(np.mean(kls))

    # ---- reproducibility filter -----------------------------------------------------------------------------------
    fragility = {}
    for nm in ("attack", "utility"):
        rows = attack if nm == "attack" else utility
        h = run_gen(rows, lambda c, m, T: None)
        fragility[nm] = {"pool": len(rows), "reproduced": int(sum(h)), "rate": float(np.mean(h))}
        kept = [r for r, x in zip(rows, h) if x][: args.n_eval]
        if nm == "attack":
            attack = kept
        else:
            utility = kept
    clean = [byk[(r["sample_id"], "clean")] for r in attack[:48]]
    print("fragility", json.dumps(fragility), flush=True)
    print(f"eval sets: attack {len(attack)}, utility {len(utility)}, clean {len(clean)}", flush=True)

    results = []
    base_hits: dict[str, np.ndarray] = {}
    boot = np.random.default_rng(1).integers(0, 1 << 30, size=1)[0]

    def paired_ci(new: list[bool], key: str) -> list[float]:
        d = np.asarray(new, float) - base_hits[key]
        b = np.random.default_rng(int(boot)).integers(0, len(d), size=(2000, len(d)))
        m = d[b].mean(1)
        return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]

    def dump():
        (out / "results.json").write_text(json.dumps({"fragility": fragility, "n_attack": len(attack),
                                                      "n_utility": len(utility), "results": results}, indent=1))

    def evaluate(name, family, make_iv, n=None, extra=None):
        t = time.time()
        ha, hu = run_gen(attack, make_iv), run_gen(utility, make_iv)
        if family == "baseline":
            base_hits["attack"], base_hits["utility"] = np.asarray(ha, float), np.asarray(hu, float)
        res = {"name": name, "family": family, "n_components": n,
               "attack_success": float(np.mean(ha)), "utility_exec": float(np.mean(hu)),
               "attack_delta_ci": paired_ci(ha, "attack"), "utility_delta_ci": paired_ci(hu, "utility"),
               "clean_kl": run_kl(make_iv), "attack_hits": [bool(x) for x in ha],
               "utility_hits": [bool(x) for x in hu], **(extra or {})}
        res["seconds"] = round(time.time() - t, 1)
        results.append(res)
        print(json.dumps({k: v for k, v in res.items() if not k.endswith("_hits")}), flush=True)
        dump()

    evaluate("baseline (no intervention)", "baseline", lambda c, m, T: None)

    def steer(sel, value, decode=False, all_tokens=False):
        offs = {l: {e: value for e in es} for l, es in sel.items()}
        return lambda c, m, T: Intervention(layout, experts=ExpertSteer(offs, token_mask=None if all_tokens else m,
                                                                       decode=decode))

    for n in map(int, args.sizes.split(",")):
        for name, delta, sign, value, kw in (
            ("experts_obey_off", d_obey, +1, NEG, {}),
            ("experts_instr_off", d_prov, -1, NEG, {}),
            ("experts_data_on", d_prov, +1, 1e4, {}),
            ("experts_tail_off_all", d_tail, +1, NEG, {"decode": True, "all_tokens": True}),
        ):
            sel = top_experts(delta, n, sign)
            k = sum(map(len, sel.values()))
            if name == "experts_data_on":  # at most top_k forced per layer, or routing degenerates to the forced set
                sel = {l: es[: layout.top_k // 2] for l, es in sel.items()}
                k = sum(map(len, sel.values()))
            evaluate(f"{name} N={k}", name, steer(sel, value, **kw), k, {"selected": {str(l): es for l, es in sel.items()}})
            for j in range(args.n_random):
                rsel = random_like(sel, E, rng)
                evaluate(f"random control {j + 1} for {name} N={k}", name + "_random", steer(rsel, value, **kw), k)

    for n in map(int, args.head_sizes.split(",")):
        flat = np.argsort(-(head_auc - 0.5).ravel())[:n]
        heads: dict[int, list[int]] = {}
        for i in flat:
            heads.setdefault(int(i // Hh), []).append(int(i % Hh))
        evaluate(f"heads_obey_off N={n}", "heads_obey_off",
                 lambda c, m, T, h=heads: Intervention(layout, heads=HeadAblation(h)), n,
                 {"selected": {str(l): v for l, v in heads.items()}})
        for j in range(args.n_random):
            rh: dict[int, list[int]] = {}
            for i in rng.choice(L * Hh, size=n, replace=False):
                rh.setdefault(int(i // Hh), []).append(int(i % Hh))
            evaluate(f"random heads {j + 1} N={n}", "heads_random",
                     lambda c, m, T, h=rh: Intervention(layout, heads=HeadAblation(h)), n)

    # ---- residual patching clean -> injection at the final 8 (identical) template tokens ---------------------------
    k = 8
    for r in attack:
        c = byk[(r["sample_id"], "clean")]
        assert r["input_ids"][-k:] == c["input_ids"][-k:], "final template tokens differ"
    layers = [int(x) for x in args.patch_layers.split(",")] if args.patch_layers else list(range(0, L, 2)) + [L - 1]
    store: dict[int, dict] = {}

    @torch.inference_mode()
    def clean_tail(chunk):
        """resid_post at the final k positions of the clean prompts, all layers: {layer: [B, k, H]}."""
        ids = [byk[(r["sample_id"], "clean")]["input_ids"] for r in chunk]
        T = max(map(len, ids))
        pad = gc.pad_token_id
        x = torch.tensor([[pad] * (T - len(i)) + i for i in ids], device=device)
        am = torch.tensor([[0] * (T - len(i)) + [1] * len(i) for i in ids], device=device)
        got = {}
        hs = [layout.layers[l].register_forward_hook(
            lambda m, a, o, l=l: got.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[:, -k:].clone())) for l in layers]
        try:
            model(input_ids=x, attention_mask=am, use_cache=False)
        finally:
            for h in hs:
                h.remove()
        return got

    for i in range(0, len(attack), args.batch):
        chunk = attack[i:i + args.batch]
        store[i] = clean_tail(chunk)
    for l in layers:
        hits = []
        for i in range(0, len(attack), args.batch):
            chunk = attack[i:i + args.batch]
            with Intervention(layout, patch=ResidPatch({l: store[i][l]})):
                resp = generate_batch(model, [r["input_ids"] for r in chunk], gc, device)
            hits += [r["evaluation"]["witness"].lower() in tok.decode(x, skip_special_tokens=True).lower()
                     for r, x in zip(chunk, resp)]
        res = {"name": f"patch_clean_tail layer {l}", "family": "patch_clean_tail", "layer": l, "k_tokens": k,
               "attack_success": float(np.mean(hits)), "attack_delta_ci": paired_ci(hits, "attack"),
               "attack_hits": [bool(x) for x in hits]}
        results.append(res)
        print(json.dumps({k_: v for k_, v in res.items() if not k_.endswith("_hits")}), flush=True)
        dump()
    print("wrote", out / "results.json")


if __name__ == "__main__":
    main()
