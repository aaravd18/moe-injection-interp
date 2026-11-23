"""E07: generations for AlpacaFarm-208 under the six static attacks (GPU pod; judging happens later, locally).

One process per model (base | lora | e05 | a40 | b80), all in fp32, greedy, 512 new tokens, same chat template as E03/E05.
Uses moe_interp.slot_generate (grouped_mm, slot refill, preallocated cache) in place of `model.generate`.

Before the 1,248 prompts, each model replays --controls prompts of its own saved AF-208 run and must reproduce the
first 16 tokens on enough of them. That checks, in under a minute, that the weights are the right ones and that the
faster decoder has not changed the model. A failed control exits with code 3 and generates nothing else.

Outputs in --out/<model>/:
  controls.jsonl, controls.json   the replay check
  generations.partial.jsonl       appended as each response finishes; a rerun resumes from it
  generations.jsonl, run_info.json   written once every prompt is done

Exit codes: 0 done, 3 control failed, 75 out of GPU memory (rerun with a smaller --cache-gb; it resumes).

Usage:
    python e07_generate.py --model a40 --model-dir /dev/shm/olmoe_b89a7c4 --ckpt-root CKPT --ref-root REF --out OUT
    python e07_generate.py ... --check-only --device cpu   # load and verify weights and prompts; no forward pass
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

MODEL_ID = "allenai/OLMoE-1B-7B-0125-Instruct"
REVISION = "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e"
ATTACKS = ("straightforward", "straightforward_before", "ignore", "ignore_before", "completion", "completion_ignore")
WEIGHTS = {  # model -> (checkpoint file, sha256 of the file that produced the saved E03/E05/E06 results)
    "base": None,
    "lora": ("lora/adapter_model.safetensors", "eb1472ff576e3508119a086de01cfb1161888e9d2ce6c28734c45f5f5bf6c4f0"),
    "e05": ("e05/security_expert.safetensors", "ed9b3c196083e69a23282aa5b8b0e59fb7d5152ab2f978ab050407d694d77367"),
    "a40": ("a40/security_expert.safetensors", "5607b816d9943449be7e37af7362c4bec63b303a920db55b8e3490d3fa9b7bd8"),
    "b80": ("b80/security_expert.safetensors", "0564ced807be83923116721167cebb689a3ed6e863a8db371a2fee0fdcb90354"),
}
# Share of controls whose first 16 tokens must match. The saved base run was bf16 on an L40S, hence its lower bar.
CONTROL_MIN = {"base": 0.65, "lora": 0.8, "e05": 0.8, "a40": 0.8, "b80": 0.8}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def merge_expert_lora(model) -> int:
    """Materialise PEFT's expert-parameter LoRA once instead of on every forward.

    PEFT's ParamWrapper forward computes W + delta and runs the experts on it; merge() stores that same W + delta.
    The attention LoRA layers stay unmerged, exactly as in the saved evaluation."""
    from peft.tuners.lora.layer import ParamWrapper

    wrappers = [m for m in model.modules() if isinstance(m, ParamWrapper)]
    for w in wrappers:
        w.merge()
    return len(wrappers)


def build_model(args):
    """(OlmoeForCausalLM with the variant applied, tokenizer, layout, checkpoint sha256)."""
    import torch

    from moe_interp.config import config_from_dict
    from moe_interp.model import load_model

    spec = {"model_id": MODEL_ID, "revision": REVISION, "dtype": "float32", "device": args.device}
    if args.tiny:
        spec |= {"tiny_random": True, "device": "cpu", "local_path": args.model_dir}
    else:
        spec |= {"local_path": args.model_dir, "experts_implementation": "grouped_mm"}
    lm = load_model(config_from_dict({"model": spec}).model)
    model, got = lm.model, None
    if args.checkpoint:  # unregistered model: --checkpoint routed_expert:DIR (E08)
        kind, d = args.checkpoint.split(":", 1)
        if kind != "routed_expert":
            raise SystemExit(f"unknown checkpoint kind {kind}")
        from moe_interp.routed_expert import WEIGHTS as RE_WEIGHTS, load_routed_experts

        got = sha256(Path(d) / RE_WEIGHTS)
        load_routed_experts(model, lm.layout, d)
    elif WEIGHTS.get(args.model) and not args.tiny:
        rel, want = WEIGHTS[args.model]
        path = Path(args.ckpt_root) / rel
        got = sha256(path)
        if got != want:
            raise SystemExit(f"{path}: sha256 {got}, expected {want}")
        if args.model == "lora":
            from peft import PeftModel

            peft_model = PeftModel.from_pretrained(model, str(path.parent)).eval()
            n = merge_expert_lora(peft_model)
            if n != 2 * lm.layout.num_layers:
                raise SystemExit(f"merged {n} expert LoRA wrappers, expected {2 * lm.layout.num_layers}")
            model = peft_model.get_base_model()
        else:
            from moe_interp.security_expert import load_security_experts

            load_security_experts(model, lm.layout, path.parent)
    model.requires_grad_(False)
    model.eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise SystemExit("every parameter must be fp32")
    return model, lm.tokenizer, lm.info, got


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def read_partial(path: Path) -> list[dict]:
    """Rows of an interrupted run. An unfinished last line is cut off so that appending stays valid."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    good = raw[: raw.rfind(b"\n") + 1]
    if good != raw:
        path.write_bytes(good)
    return [json.loads(x) for x in good.decode().splitlines()]


def repetition(text: str, n: int = 3) -> float:
    w = text.split()
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 0.0 if not g else round(1 - len(set(g)) / len(g), 4)


def schedule(jobs, slots_for, long_prompt: int, expected: dict):
    """Two tiers, so that a few very long prompts do not size every slot's cache. Within a tier, rows expected to
    give long answers go first, so the batch is not left waiting on one straggler at the end."""
    tiers = ([j for j in jobs if len(j.ids) <= long_prompt], [j for j in jobs if len(j.ids) > long_prompt])
    for tier in tiers:
        if tier:
            tier.sort(key=lambda j: (-expected.get(j.key[0], 0), -len(j.ids)))
            yield tier, slots_for(max(len(j.ids) for j in tier))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help=f"one of {list(WEIGHTS)}, or any name with --checkpoint")
    ap.add_argument("--checkpoint", help="for a model not in WEIGHTS: routed_expert:DIR (sha256 recorded, not pinned)")
    ap.add_argument("--data", default=str(REPO / "data" / "alpacafarm_secalign_208x6.jsonl"))
    ap.add_argument("--conditions", default=",".join(ATTACKS))
    ap.add_argument("--model-dir")
    ap.add_argument("--ckpt-root")
    ap.add_argument("--ref-root", help="REF/<model>.jsonl = that model's saved AF-208 generations")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--cache-gb", type=float, default=26.0)
    ap.add_argument("--max-slots", type=int, default=128)
    ap.add_argument("--long-prompt", type=int, default=384)
    ap.add_argument("--controls", type=int, default=32)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="tiny random OLMoE on CPU (pipeline check; output is noise)")
    args = ap.parse_args()
    if args.model not in WEIGHTS and not args.checkpoint:
        raise SystemExit(f"--model {args.model} is not registered; pass --checkpoint")

    import torch
    import transformers

    from moe_interp.prompts import render_prompt
    from moe_interp.slot_generate import Job, generate_slots, slot_bytes

    t0 = time.time()
    data_path = Path(args.data)
    rows = read_jsonl(data_path)[: args.limit]
    model, tok, info, ckpt_sha = build_model(args)
    conds = args.conditions.split(",")
    rendered = {(r["id"], c): render_prompt(tok, r["variants"][c]["messages"], True, None, {})
                for c in conds for r in rows}
    ids_sha = hashlib.sha256(json.dumps([x.input_ids for x in rendered.values()]).encode()).hexdigest()
    longest = max(len(x.input_ids) for x in rendered.values())
    if not args.tiny and longest + args.max_new_tokens > model.config.max_position_embeddings:
        raise SystemExit(f"{longest} + {args.max_new_tokens} tokens exceed the model's position limit")
    impl = getattr(model.config, "_experts_implementation", None)
    load_s = time.time() - t0
    if args.check_only:
        print(json.dumps({"check": "ok", "model": args.model, "n_prompts": len(rendered), "longest_prompt": longest,
                          "prompt_ids_sha256": ids_sha, "checkpoint_sha256": ckpt_sha, "experts": impl,
                          "parameters": sum(p.numel() for p in model.parameters()), "seconds": round(load_s, 1)}))
        return
    if not args.tiny and impl != "grouped_mm":
        raise SystemExit(f"experts implementation is {impl}, expected grouped_mm")

    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    gc = model.generation_config
    eos = gc.eos_token_id if gc.eos_token_id is not None else tok.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = gc.pad_token_id if gc.pad_token_id is not None else tok.pad_token_id

    def slots_for(prompt_len: int) -> int:
        per = slot_bytes(model.config, prompt_len + args.max_new_tokens, torch.float32)
        return max(1, min(args.max_slots, int(args.cache_gb * 1e9 // per)))

    def generate(jobs, on_done, expected) -> list[dict]:
        stats, t = [], time.time()

        def progress(s):
            if s["steps"] % 100 == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                print(f"  {args.model}: step {s['steps']} active {s['active']} queued {s['queued']} "
                      f"{s['tokens'] / (time.time() - t):.0f} tok/s peak {mem:.1f} GB", flush=True)

        for tier, slots in schedule(jobs, slots_for, args.long_prompt, expected):
            print(f"{args.model}: {len(tier)} prompts in {slots} slots", flush=True)
            t = time.time()
            s = generate_slots(model, tier, slots=slots, max_new_tokens=args.max_new_tokens, eos_ids=eos,
                               pad_id=pad, on_done=on_done, on_step=progress)
            stats.append(s | {"prompts": len(tier), "slots": slots, "seconds": round(time.time() - t, 1)})
        return stats

    try:
        if torch.cuda.is_available():
            # A hard ceiling turns an overrun into a catchable error instead of taking the pod down.
            torch.cuda.set_per_process_memory_fraction(0.94)
        ref_file = Path(args.ref_root) / f"{args.model}.jsonl" if args.ref_root else None
        ref = read_jsonl(ref_file) if ref_file and ref_file.exists() else []  # no saved run -> no control
        ref = [r for r in ref if r["condition"] == "injection"]
        expected = {r["id"]: r["n_response_tokens"] for r in ref}
        control_file = out / "controls.json"
        passed = control_file.exists() and json.loads(control_file.read_text())["passed"]
        if args.controls and ref and not passed:
            pick = ref[:: max(1, len(ref) // args.controls)][: args.controls]
            jobs = [Job((r["id"], "injection"), tok(r["prompt_text"], add_special_tokens=False)["input_ids"])
                    for r in pick]
            done: list = []
            stats = generate(jobs, done.append, expected)
            old, checks = {r["id"]: r["response"] for r in pick}, []
            for j in done:
                new = tok.decode(j.out, skip_special_tokens=True)
                head = tok.decode(j.out[:16], skip_special_tokens=True).rstrip("�")
                was = old[j.key[0]]
                checks.append({"id": j.key[0], "prefix16_match": bool(was.startswith(head) if head else not was),
                               "exact_match": new == was, "response": new, "saved": was})
            (out / "controls.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in checks))
            share = sum(c["prefix16_match"] for c in checks) / len(checks)
            result = {"n": len(checks), "prefix16": share, "exact": sum(c["exact_match"] for c in checks) / len(checks),
                      "required": CONTROL_MIN.get(args.model, 0.8), "passed": share >= CONTROL_MIN.get(args.model, 0.8), "stats": stats}
            control_file.write_text(json.dumps(result, indent=2) + "\n")
            print(f"{args.model} controls: {json.dumps(result)}", flush=True)
            if not result["passed"]:
                sys.exit(3)

        partial = out / "generations.partial.jsonl"
        prior = read_partial(partial)
        for r in prior:
            key = (r["id"], r["condition"])
            if key not in rendered or r["prompt_text"] != rendered[key].text:
                raise SystemExit(f"{partial} was written for different prompts; refusing to mix")
        have = {(r["id"], r["condition"]) for r in prior}
        jobs = [Job(k, x.input_ids) for k, x in rendered.items() if k not in have]
        print(f"{args.model}: {len(jobs)} to generate, {len(have)} already done", flush=True)
        t1 = time.time()
        with partial.open("a") as f:
            def save(j) -> None:
                text = tok.decode(j.out, skip_special_tokens=True)
                f.write(json.dumps({
                    "id": j.key[0], "condition": j.key[1], "prompt_text": rendered[j.key].text,
                    "n_prompt_tokens": len(j.ids), "response": text, "response_ids": j.out,
                    "n_response_tokens": len(j.out), "finish_reason": "stop" if j.out[-1] in eos else "length",
                    "repetition_3gram": repetition(text)}, ensure_ascii=False) + "\n")
                f.flush()

            stats = generate(jobs, save, expected)
            os.fsync(f.fileno())
    except torch.cuda.OutOfMemoryError as e:
        print(f"{args.model}: OOM at --cache-gb {args.cache_gb}: {str(e).splitlines()[0]}", flush=True)
        sys.exit(75)

    by_key = {(r["id"], r["condition"]): r for r in read_partial(partial)}
    if set(by_key) != set(rendered):
        raise SystemExit("generation finished with missing rows")
    final, tmp = out / "generations.jsonl", out / "generations.jsonl.tmp"
    tmp.write_text("".join(json.dumps(by_key[k], ensure_ascii=False) + "\n" for k in rendered))
    tmp.replace(final)
    run = {
        "step": "e07_generate", "variant": args.model, "model": MODEL_ID, "revision": REVISION,
        "dtype": "float32", "checkpoint": args.checkpoint, "checkpoint_sha256": ckpt_sha, "tiny_random": args.tiny,
        "data": "data/" + data_path.name, "data_sha256": sha256(data_path), "prompt_ids_sha256": ids_sha,
        "conditions": conds, "n_records": len(rows), "n_generations": len(by_key),
        "max_new_tokens": args.max_new_tokens, "do_sample": False, "eos_ids": sorted(eos),
        "chat_template": "OLMoE tokenizer template; system = instruction, user = data; add_generation_prompt",
        "engine": "moe_interp.slot_generate", "experts_implementation": impl,
        "attn_implementation": info["attn_implementation"], "cache_gb": args.cache_gb, "tiers": stats,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "transformers": transformers.__version__, "load_seconds": round(load_s, 1),
        "generate_seconds": round(time.time() - t1, 1), "generations_sha256": sha256(final),
    }
    (out / "run_info.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"{args.model}: DONE {len(by_key)} generations in {run['generate_seconds']} s", flush=True)


if __name__ == "__main__":
    main()
