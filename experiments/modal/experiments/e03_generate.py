"""E03 step 1: generate clean and injected responses with the FROZEN OLMoE-1B-7B-0125-Instruct.

Plain Python (no Modal): run it on the SSH H100, or locally with --tiny for a pipeline check.

For every record of data/secpo_olmoe_v1/selected.jsonl, both variants are rendered with OLMoE's own chat template
through moe_interp.prompts.render_prompt (system = trusted instruction, user = data; generation prompt
"<|assistant|>\\n"), then greedy-decoded with moe_interp.runner.generate_batch (left padding; batched == single-prompt
output is covered by tests/test_batch_generation.py). Clean prompts are batched together, then injected prompts, in
file order, so a rerun with the same settings reproduces the same batches.

Output (--out): generations.jsonl (one row per id x condition) and run_info.json (model, versions, settings, timing).

Usage:
    python experiments/modal/experiments/e03_generate.py --subset pilot --out OUT_DIR              # 20 x 2 prompts
    python experiments/modal/experiments/e03_generate.py --subset all --out OUT_DIR                # 500 x 2 prompts
    python experiments/modal/experiments/e03_generate.py --tiny --limit 2 --out /tmp/x             # CPU check, noise
    # Step 3 evaluation: any dataset with clean/injection variants (e.g. data/alpacafarm_secalign_208.jsonl), optionally
    # restricted to one side of the grouped split, optionally with a SecPO LoRA adapter (applied unmerged, i.e. the
    # same forward pass as training):
    python experiments/modal/experiments/e03_generate.py --subset all --split val --adapter RUN/adapter --out OUT_DIR
    # E05: always-active security experts instead of a LoRA adapter (fp32-trained -> evaluate with --dtype float32):
    python experiments/modal/experiments/e03_generate.py --data data/alpacafarm_secalign_208.jsonl --subset all \
        --dtype float32 --security-expert RUN/security_expert --out OUT_DIR
    # SecPO's rejected side, y_l = f_undef(z'): the seed task alone (metadata.seed_task, the source rejected_input,
    # whitespace-stripped) as one user message, no system message (paper Algorithm 1, line 13):
    python experiments/modal/experiments/e03_generate.py --subset all --conditions seed_task --out OUT_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

OLMOE = {"model_id": "allenai/OLMoE-1B-7B-0125-Instruct", "revision": "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e",
         "dtype": "bfloat16", "device": "auto"}
CONDITIONS = ("clean", "injection")
ALL_CONDITIONS = CONDITIONS + ("seed_task",)


def messages(r: dict, cond: str) -> list[dict]:
    """clean / injection: the record's variants. seed_task: SecPO's z' alone (no trusted task, no clean data)."""
    if cond == "seed_task":
        return [{"role": "user", "content": r["metadata"]["seed_task"].strip()}]
    return r["variants"][cond]["messages"]


def repetition(text: str, n: int = 3) -> float:
    """Share of repeated word 3-grams (0 = none). A degeneration hint, not a judgement."""
    w = text.split()
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 0.0 if not g else round(1 - len(set(g)) / len(g), 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(REPO / "data" / "secpo_olmoe_v1" / "selected.jsonl"))
    ap.add_argument("--subset", choices=["pilot", "all"], default="pilot")
    ap.add_argument("--limit", type=int, default=None, help="first N records only (debugging)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--conditions", default=",".join(CONDITIONS), help=f"comma-separated, from {ALL_CONDITIONS}")
    ap.add_argument("--shard", default=None, help="K/N: keep the K-th of N contiguous chunks (parallel runs; merge after)")
    ap.add_argument("--dtype", default=None, help="override model dtype (default bfloat16). Use float32 for adapters "
                    "trained in float32: PEFT adds the expert LoRA delta into the fused weight in the model dtype")
    ap.add_argument("--tiny", action="store_true", help="tiny random OLMoE on CPU (pipeline check; output is noise)")
    ap.add_argument("--adapter", default=None, help="PEFT LoRA adapter dir (e.g. from e03_secpo_train.py); unmerged")
    ap.add_argument("--security-expert", default=None,
                    help="dir from e03_secpo_train.py --method security_expert (moe_interp/security_expert.py)")
    ap.add_argument("--split", choices=["train", "val"], default=None, help="keep only this side of --split-file")
    ap.add_argument("--split-file", default=str(REPO / "experiments" / "modal" / "artifacts" / "splits" /
                                                "secpo_olmoe_v1_grouped_v1.json"))
    args = ap.parse_args()
    conditions = tuple(c for c in args.conditions.split(",") if c)
    if not conditions or set(conditions) - set(ALL_CONDITIONS):
        raise SystemExit(f"--conditions must be a non-empty subset of {ALL_CONDITIONS}")

    import torch
    import transformers

    from moe_interp.config import config_from_dict
    from moe_interp.model import load_model
    from moe_interp.prompts import render_prompt
    from moe_interp.runner import _eos_ids, _generation_config, generate_batch

    raw = Path(args.data).read_bytes()
    rows = [json.loads(l) for l in raw.decode().splitlines()]
    if args.subset == "pilot":
        rows = [r for r in rows if r["metadata"]["pilot"]]
    if args.split:
        keep = set(json.loads(Path(args.split_file).read_text())["ids"][args.split])
        rows = [r for r in rows if r["id"] in keep]
    rows = rows[: args.limit] if args.limit else rows
    if args.shard:
        k, n = map(int, args.shard.split("/"))
        size = -(-len(rows) // n)
        rows = rows[k * size: (k + 1) * size]
    model_cfg = {**OLMOE, **({"tiny_random": True, "device": "cpu", "dtype": "float32"} if args.tiny else {})}
    if args.dtype:
        model_cfg["dtype"] = args.dtype
    cfg = config_from_dict({"model": model_cfg, "prompt": {"add_generation_prompt": True},
                            "generation": {"max_new_tokens": args.max_new_tokens, "do_sample": False},
                            "trace": {"enabled": False}})
    t0 = time.time()
    lm = load_model(cfg.model)
    load_s = time.time() - t0
    adapter_sha = None
    if args.adapter and args.security_expert:
        raise SystemExit("--adapter and --security-expert are mutually exclusive")
    if args.security_expert:
        from moe_interp.security_expert import WEIGHTS, load_security_experts

        load_security_experts(lm.model, lm.layout, args.security_expert)
        lm.model.eval()
        adapter_sha = hashlib.sha256((Path(args.security_expert) / WEIGHTS).read_bytes()).hexdigest()
    if args.adapter:
        from peft import PeftModel

        lm.model = PeftModel.from_pretrained(lm.model, args.adapter).eval()
        adapter_sha = hashlib.sha256((Path(args.adapter) / "adapter_model.safetensors").read_bytes()).hexdigest()
    tok, gc = lm.tokenizer, _generation_config(lm.model, cfg, lm.tokenizer)
    eos = _eos_ids(gc)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs, t1 = [], time.time()
    partial = (out / "generations.partial.jsonl").open("w")  # streamed per batch (judge can start early)
    for cond in conditions:
        rendered = [render_prompt(tok, messages(r, cond), True, None, {}) for r in rows]
        for b in range(0, len(rows), args.batch_size):
            chunk = rendered[b: b + args.batch_size]
            ids_list = generate_batch(lm.model, [x.input_ids for x in chunk], gc, lm.device)
            for r, x, ids in zip(rows[b: b + args.batch_size], chunk, ids_list):
                text = tok.decode(ids, skip_special_tokens=True)
                recs.append({"id": r["id"], "condition": cond, "prompt_text": x.text,
                             "n_prompt_tokens": len(x.input_ids), "response": text, "n_response_tokens": len(ids),
                             "finish_reason": "stop" if ids and ids[-1] in eos else "length",
                             "repetition_3gram": repetition(text)})
                partial.write(json.dumps(recs[-1], ensure_ascii=False) + "\n")
            partial.flush()
            print(f"{cond}: {min(b + args.batch_size, len(rows))}/{len(rows)}  ({time.time() - t1:.0f}s)", flush=True)
    partial.close()
    with (out / "generations.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = {
        "step": "e03_generate", "model": lm.info["model_id"], "requested_revision": lm.info["requested_revision"],
        "resolved_commit_hash": lm.info["resolved_commit_hash"], "dtype": lm.info["dtype"], "device": lm.info["device"],
        "tiny_random": args.tiny, "adapter": args.adapter, "security_expert": args.security_expert, "adapter_sha256": adapter_sha,
        "split": args.split, "split_file": args.split_file if args.split else None, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "data": args.data, "data_sha256": hashlib.sha256(raw).hexdigest(), "subset": args.subset, "shard": args.shard,
        "n_records": len(rows), "n_generations": len(recs), "conditions": list(conditions),
        "max_new_tokens": args.max_new_tokens, "do_sample": False, "batch_size": args.batch_size,
        "chat_template": "OLMoE tokenizer template; clean/injection: system = instruction, user = data; seed_task: "
                         "user = metadata.seed_task.strip(), no system; add_generation_prompt",
        "load_seconds": round(load_s, 1), "generate_seconds": round(time.time() - t1, 1),
        "finished_with_eos": sum(r["finish_reason"] == "stop" for r in recs),
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
