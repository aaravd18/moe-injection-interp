"""E04 phase 1 capture: prompt-only expert routing of base OLMoE vs the SecPO v2 adapter on AlpacaFarm-SecAlign 208.

Four conditions, one teacher-forced prefill pass per prompt, no generated tokens:

    base  x clean      base  x injection      secpo x clean      secpo x injection

Everything that could make the two checkpoints' passes differ other than the adapter is held fixed:
- one process, one copy of the base weights. The base runs first; then the adapter is attached unmerged with
  PeftModel.from_pretrained (the same forward pass as training and as e03_generate.py evaluation);
- dtype float32 for both (the adapter was trained in fp32; bf16 rounds away most of the expert-LoRA delta, and bf16
  OLMoE routing depends on batch shape). TF32 matmuls are disabled;
- batch size 1, no padding, no KV cache, identical token ids for both models (asserted), records in file order.

Routing comes from moe_interp.tracing.TraceRecorder (router-module hooks, checked against output_router_logits and
against softmax(logits)[ids]). Per token and layer we save the top-8 expert ids, the dispatched weights (raw softmax
probabilities, OLMoE has norm_topk_prob = False) and the full fp32 router logits [64].

At the last prompt position we also save the next-token distribution summary (top-20 tokens, and the probability mass
on first tokens that can begin "hacked", see hacked_token_ids), so the prompt-only routing can be tied to what the model is about to say.

Checks written to checks.json (the run aborts if a hard check fails):
- repeat: the first --check-n prompts of every (model, condition) are run again and must reproduce bit for bit;
- adapter_off: with the adapter attached but disabled (peft.disable_adapter), the first --check-n prompts must
  reproduce the base capture (this is what makes "base" and "secpo" differ only by the adapter);
- adapter_on: the adapter must change routing somewhere, and its sha256 is recorded (and compared with --expect-sha).

Outputs (--out):
    {base,secpo}_{clean,injection}.npz   offsets [n+1], input_ids [T], topk_ids [T, L, k] uint8,
                                         topk_weights [T, L, k] f32, router_logits [T, L, E] f32,
                                         next_top_ids [n, 20], next_top_logp [n, 20], p_hack_start [n]
    examples.jsonl                       per record x condition: id, n_tokens, token spans (task, data, probe,
                                         user, suffix), top-20 next tokens decoded, per model
    checks.json, run_info.json

Usage:
    python experiments/modal/experiments/e04_routing_capture.py --adapter ADAPTER_DIR --out OUT_DIR
    python experiments/modal/experiments/e04_routing_capture.py --tiny --random-adapter --limit 4 --out /tmp/x  # CPU
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e03_generate import OLMOE  # noqa: E402  (pinned model id + revision)

MODELS = ("base", "secpo")
CONDITIONS = ("clean", "injection")
SPAN_NAMES = ("task", "data", "probe")
TOP_NEXT = 20
SECPO_V2_SHA = "eb1472ff576e3508119a086de01cfb1161888e9d2ce6c28734c45f5f5bf6c4f0"


# ------------------------------------------------------------------------------------------------------- prompts

def render_all(tok, samples, conditions=CONDITIONS) -> dict[str, list[dict]]:
    """Render every sample/condition once. Returns cond -> list of {id, input_ids, spans} in file order.

    Spans are token [start, end) ranges: task (system content), data (user data before the probe), probe (injected
    instruction, injection only), user (whole user message content) and suffix (everything after the user content:
    the template's "\\n<|assistant|>\\n", whose last token produces the first answer token).
    """
    from moe_interp.prompts import render_prompt

    out: dict[str, list[dict]] = {c: [] for c in conditions}
    for s in samples:
        for c in conditions:
            v = s.variants[c]
            rp = render_prompt(tok, v.messages, True, v.spans, {})
            S = len(rp.input_ids)
            user = rp.message_token_spans[-1]
            if v.messages[-1]["role"] != "user" or user is None:
                raise ValueError(f"{s.id}/{c}: expected the data in a final user message")
            spans = {k: {"start": t.start, "end": t.end, "exact": t.exact} for k, t in rp.spans.items()}
            spans["user"] = {"start": user.start, "end": user.end, "exact": user.exact}
            spans["suffix"] = {"start": user.end, "end": S, "exact": True}
            missing = {"task", "data"} - set(spans) | ({"probe"} - set(spans) if c == "injection" else set())
            if missing:
                raise ValueError(f"{s.id}/{c}: spans {sorted(missing)} not annotated")
            out[c].append({"id": s.id, "input_ids": rp.input_ids, "spans": spans})
    return out


def hacked_token_ids(tok) -> list[int]:
    """Vocabulary entries that can start an answer beginning with "hacked": OLMoE splits "Hacked!" into "H" + "acked",
    so this is the bare letter h/H (with or without a leading space) plus "hack"-prefix tokens of ≥ 3 letters, e.g.
    " Hack", " hacked". Bare "H" also starts a few other words, so p_hack_start is a proxy, not a label."""
    ids = []
    for t in range(len(tok)):
        s = tok.decode([t]).strip().lower()
        if s == "h" or (len(s) >= 3 and ("hacked".startswith(s) or s.startswith("hacked"))):
            ids.append(t)
    return ids


# ------------------------------------------------------------------------------------------------------- forward

def traced_prefill(model, layout, input_ids: list[int], device, hack_ids):
    """One batch-1 prefill pass with router hooks. Returns routing arrays [S, L, ·] and the last-position readout."""
    import torch

    from moe_interp.tracing import TraceRecorder, TraceSpec

    ids = torch.tensor([input_ids], device=device)
    S = ids.shape[1]
    spec = TraceSpec(activation_types=[], activation_layers=[], positions="last", save_router_logits=True)
    with torch.inference_mode():
        with TraceRecorder(layout, spec, S) as rec:
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_router_logits=True,
                        use_cache=False, logits_to_keep=1)
        routing = rec.finalize_routing(out.router_logits)
        logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        top = torch.topk(logp, TOP_NEXT)
        p_hack = float(logp[hack_ids].exp().sum()) if hack_ids else float("nan")
    return {
        "topk_ids": routing.topk_ids.permute(1, 0, 2).numpy().astype(np.uint8),          # [S, L, k]
        "topk_weights": routing.topk_weights.permute(1, 0, 2).numpy().astype(np.float32),
        "router_logits": routing.router_logits.permute(1, 0, 2).numpy().astype(np.float32),  # [S, L, E]
        "next_top_ids": top.indices.cpu().numpy().astype(np.int32),
        "next_top_logp": top.values.cpu().numpy().astype(np.float32),
        "p_hack_start": p_hack,
        "checks": routing.checks,
    }


def run_condition(model, layout, prompts: list[dict], device, hack_ids, label: str, log_every: int = 25) -> dict:
    t0, recs = time.time(), []
    for i, p in enumerate(prompts):
        recs.append(traced_prefill(model, layout, p["input_ids"], device, hack_ids))
        if (i + 1) % log_every == 0 or i + 1 == len(prompts):
            print(f"{label}: {i + 1}/{len(prompts)}  ({time.time() - t0:.0f}s)", flush=True)
    lens = [len(p["input_ids"]) for p in prompts]
    return {
        "offsets": np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        "input_ids": np.concatenate([np.asarray(p["input_ids"], dtype=np.int32) for p in prompts]),
        **{k: np.concatenate([r[k] for r in recs]) for k in ("topk_ids", "topk_weights", "router_logits")},
        **{k: np.stack([r[k] for r in recs]) for k in ("next_top_ids", "next_top_logp")},
        "p_hack_start": np.asarray([r["p_hack_start"] for r in recs], dtype=np.float32),
        "seconds": time.time() - t0,
        "max_logit_check_err": max(r["checks"]["max_abs_err_logits_vs_output_router_logits"] for r in recs),
    }


def compare(a: dict, b: dict, n: int) -> dict:
    """Max abs router-logit difference and count of (token, layer) top-k sets that differ, over the first n prompts."""
    T = int(a["offsets"][n])
    if not np.array_equal(a["input_ids"][:T], b["input_ids"][:T]):
        raise RuntimeError("compare: different token ids")
    sa, sb = np.sort(a["topk_ids"][:T], -1), np.sort(b["topk_ids"][:T], -1)
    return {"n_prompts": n, "n_tokens": T,
            "max_abs_router_logit_diff": float(np.abs(a["router_logits"][:T] - b["router_logits"][:T]).max()),
            "topk_sets_differ": int((sa != sb).any(-1).sum()), "n_token_layers": int(sa.shape[0] * sa.shape[1])}


def head(cap: dict, n: int) -> dict:
    """The first n prompts of a capture, as a capture."""
    T = int(cap["offsets"][n])
    return {"offsets": cap["offsets"][: n + 1], "input_ids": cap["input_ids"][:T],
            **{k: cap[k][:T] for k in ("topk_ids", "topk_weights", "router_logits")}}


# ------------------------------------------------------------------------------------------------------- driver

def capture(model, layout, tok, prompts: dict[str, list[dict]], device, adapter: str | None,
            random_adapter: bool, check_n: int, tol: float) -> tuple[dict, dict, object]:
    """Run base then secpo on all conditions, with the repeat / adapter-off / adapter-on checks.

    Returns (captures[(model, cond)], checks, peft_model)."""
    from peft import PeftModel

    hack_ids = hacked_token_ids(tok)
    caps, checks = {}, {"hacked_first_token_ids": len(hack_ids)}
    for c, ps in prompts.items():
        caps[("base", c)] = run_condition(model, layout, ps, device, hack_ids, f"base/{c}")
    n = {c: min(check_n, len(ps)) for c, ps in prompts.items()}

    def repeat(m, mdl, lay, c):
        again = run_condition(mdl, lay, prompts[c][: n[c]], device, hack_ids, f"{m}/{c} repeat")
        r = compare(head(caps[(m, c)], n[c]), again, n[c])
        r["ok"] = r["max_abs_router_logit_diff"] == 0.0 and r["topk_sets_differ"] == 0
        return r

    checks["repeat"] = {f"base/{c}": repeat("base", model, layout, c) for c in prompts}

    if random_adapter:  # CPU pipeline check only: a non-zero random adapter of the same shape as the real one
        peft_model = _random_adapter(model, layout)
    else:
        peft_model = PeftModel.from_pretrained(model, adapter).eval()
    pm_base = peft_model.get_base_model()
    from moe_interp.model import get_moe_layout

    play = get_moe_layout(pm_base)
    if any(a is not b for a, b in zip(play.routers, layout.routers)):
        raise RuntimeError("PEFT replaced the router modules; the router must stay frozen and untouched")
    trainable_router = [nm for nm, _ in peft_model.named_parameters() if ".mlp.gate." in nm and "lora" in nm]
    if trainable_router:
        raise RuntimeError(f"adapter has router LoRA weights: {trainable_router[:3]}")

    with peft_model.disable_adapter():
        off = {c: run_condition(peft_model, play, prompts[c][: n[c]], device, hack_ids, f"adapter-off/{c}")
               for c in prompts}
    checks["adapter_off"] = {}
    for c in prompts:
        r = compare(head(caps[("base", c)], n[c]), off[c], n[c])
        r["ok"] = r["max_abs_router_logit_diff"] <= tol and r["topk_sets_differ"] == 0
        checks["adapter_off"][c] = r

    for c, ps in prompts.items():
        caps[("secpo", c)] = run_condition(peft_model, play, ps, device, hack_ids, f"secpo/{c}")
    checks["repeat"].update({f"secpo/{c}": repeat("secpo", peft_model, play, c) for c in prompts})
    checks["adapter_on"] = {}
    for c, ps in prompts.items():
        r = compare(caps[("base", c)], caps[("secpo", c)], len(ps))
        r["ok"] = r["max_abs_router_logit_diff"] > 0.0
        checks["adapter_on"][c] = r
    checks["max_logit_check_err"] = max(v["max_logit_check_err"] for v in caps.values())
    checks["all_ok"] = all(r["ok"] for grp in ("repeat", "adapter_off", "adapter_on") for r in checks[grp].values())
    return caps, checks, peft_model


def _random_adapter(model, layout):
    import torch
    from peft import LoraConfig, get_peft_model

    experts = getattr(layout.moe_blocks[0], "experts", None)
    fused = isinstance(getattr(experts, "gate_up_proj", None), torch.nn.Parameter)
    attn = r".*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    kw = ({"target_modules": attn, "target_parameters": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]}
          if fused else {"target_modules": f"({attn})|(.*\\.mlp\\.experts\\.\\d+\\.(gate_proj|up_proj|down_proj))"})
    torch.manual_seed(0)
    cfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, init_lora_weights=False, **kw)
    return get_peft_model(model, cfg).eval()


def save(out: Path, caps: dict, prompts: dict, tok) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for (m, c), cap in caps.items():
        np.savez(out / f"{m}_{c}.npz", **{k: v for k, v in cap.items() if isinstance(v, np.ndarray)})
    with (out / "examples.jsonl").open("w") as f:
        for c, ps in prompts.items():
            for i, p in enumerate(ps):
                row = {"id": p["id"], "condition": c, "index": i, "n_tokens": len(p["input_ids"]), "spans": p["spans"],
                       "last_token": tok.decode([p["input_ids"][-1]])}
                for m in MODELS:
                    cap = caps[(m, c)]
                    row[m] = {"next_top": [[tok.decode([int(t)]), round(float(lp), 4)]
                                           for t, lp in zip(cap["next_top_ids"][i][:5], cap["next_top_logp"][i][:5])],
                              "next_id": int(cap["next_top_ids"][i][0]),
                              "p_hack_start": float(cap["p_hack_start"][i])}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(REPO / "data" / "alpacafarm_secalign_208.jsonl"))
    ap.add_argument("--adapter", default=None, help="SecPO PEFT adapter dir (secpo_v2/adapter)")
    ap.add_argument("--expect-sha", default=SECPO_V2_SHA, help="sha256 of adapter_model.safetensors ('' to skip)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--limit", type=int, default=None, help="first N records only (debugging)")
    ap.add_argument("--check-n", type=int, default=8, help="prompts per (model, condition) for repeat/adapter-off checks")
    ap.add_argument("--adapter-off-tol", type=float, default=0.0,
                    help="max router-logit diff allowed between base and adapter-disabled passes")
    ap.add_argument("--tiny", action="store_true", help="tiny random OLMoE on CPU (pipeline check; output is noise)")
    ap.add_argument("--random-adapter", action="store_true", help="random non-zero LoRA instead of --adapter (CPU check)")
    args = ap.parse_args()
    if not args.adapter and not args.random_adapter:
        raise SystemExit("--adapter is required (or --random-adapter for a pipeline check)")

    import peft
    import torch
    import transformers

    from moe_interp.config import config_from_dict
    from moe_interp.dataset import load_dataset
    from moe_interp.model import load_model

    try:  # no TF32 anywhere: fp32 means fp32
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
    except AttributeError:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    raw = Path(args.data).read_bytes()
    samples = load_dataset(args.data)[: args.limit] if args.limit else load_dataset(args.data)
    model_cfg = {**OLMOE, "dtype": args.dtype,
                 **({"tiny_random": True, "device": "cpu", "dtype": "float32"} if args.tiny else {})}
    cfg = config_from_dict({"model": model_cfg, "trace": {"enabled": False}})
    t0 = time.time()
    lm = load_model(cfg.model)
    load_s = time.time() - t0
    prompts = render_all(lm.tokenizer, samples)
    adapter_sha = None
    if args.adapter and not args.random_adapter:
        adapter_sha = hashlib.sha256((Path(args.adapter) / "adapter_model.safetensors").read_bytes()).hexdigest()
        if args.expect_sha and adapter_sha != args.expect_sha:
            raise SystemExit(f"adapter sha256 {adapter_sha} != expected {args.expect_sha}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t1 = time.time()
    caps, checks, _ = capture(lm.model, lm.layout, lm.tokenizer, prompts, lm.device, args.adapter,
                              args.random_adapter, args.check_n, args.adapter_off_tol)
    out = Path(args.out)
    save(out, caps, prompts, lm.tokenizer)
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    info = {
        "step": "e04_routing_capture", "model": lm.info["model_id"], "requested_revision": lm.info["requested_revision"],
        "resolved_commit_hash": lm.info["resolved_commit_hash"], "dtype": lm.info["dtype"], "device": lm.info["device"],
        "tiny_random": args.tiny, "adapter": args.adapter, "adapter_sha256": adapter_sha,
        "random_adapter": args.random_adapter, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
        "attn_implementation": lm.info["attn_implementation"], "experts_implementation": lm.info["experts_implementation"],
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "data": args.data, "data_sha256": hashlib.sha256(raw).hexdigest(), "n_records": len(samples),
        "conditions": list(CONDITIONS), "models": list(MODELS), "batch_size": 1, "padding": None, "use_cache": False,
        "n_tokens": {c: int(sum(len(p["input_ids"]) for p in ps)) for c, ps in prompts.items()},
        "architecture": lm.info["architecture"],
        "load_seconds": round(load_s, 1), "capture_seconds": round(time.time() - t1, 1),
        "seconds_per_condition": {f"{m}/{c}": round(v["seconds"], 1) for (m, c), v in caps.items()},
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
        "checks_all_ok": checks["all_ok"],
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: info[k] for k in ("capture_seconds", "peak_gpu_mem_gb", "n_tokens", "checks_all_ok")}))
    if not checks["all_ok"]:
        raise SystemExit(f"CHECKS FAILED, see {out / 'checks.json'}")


if __name__ == "__main__":
    main()
