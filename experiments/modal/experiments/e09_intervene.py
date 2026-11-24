"""E09 experiment 3 (GPU): generation with a fixed vector added to the residual stream after one decoder layer.

For each config, at decoder layer `layer` (resid index layer + 1 in e09_capture_resid.py terms), every position of
every forward pass (prompt prefill and each generated token) gets

    h  <-  h + delta,      delta = norm * v,     v a unit vector from --vectors (probe direction or random)

`norm` is an absolute L2 norm, so a probe direction, a random direction and the detection direction can be compared
at matched magnitude. A config with "vector": null is the unmodified model (the control row of the sweep).

The model, prompts, chat template, decoding (greedy, fp32, 512 new tokens) and output format are E07's
(e07_generate.build_model, moe_interp.slot_generate), so the E07 judges and scripts/summarize_e07.py run unchanged on
OUT/<config>/generations.jsonl.

Usage (pod):
    python e09_intervene.py --model base --model-dir DIR --configs CONFIGS.json --vectors VEC.npz \
        --records RECORDS.json --out OUT [--ckpt-root CKPT | --checkpoint routed_expert:DIR]
CONFIGS.json = [{"name": "obey_L8_-4", "layer": 7, "vector": "base__obey_base__L8", "norm": 12.3}, ...]
RECORDS.json = {"ids": [...]}  (record ids; all 7 conditions of each are generated)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from e07_generate import ATTACKS, build_model, read_jsonl, read_partial, repetition, sha256  # noqa: E402

CONDS = ["clean", *ATTACKS]


class AddVector:
    def __init__(self, layer_module, delta):
        self.m, self.delta, self.h = layer_module, delta, None

    def __enter__(self):
        def hook(_m, _a, out):
            if isinstance(out, tuple):
                return (out[0] + self.delta.to(out[0].dtype), *out[1:])
            return out + self.delta.to(out.dtype)

        if self.delta is not None:
            self.h = self.m.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self.h is not None:
            self.h.remove()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--ckpt-root")
    ap.add_argument("--model-dir")
    ap.add_argument("--data", default=str(REPO / "data" / "alpacafarm_secalign_208x6.jsonl"))
    ap.add_argument("--configs", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--cache-gb", type=float, default=10.0)
    ap.add_argument("--max-slots", type=int, default=128)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()

    import torch

    from moe_interp.model import get_moe_layout
    from moe_interp.prompts import render_prompt
    from moe_interp.slot_generate import Job, generate_slots, slot_bytes

    torch.backends.cuda.matmul.allow_tf32 = False
    model, tok, info, ckpt_sha = build_model(args)
    layers = get_moe_layout(model).layers
    vecs = np.load(args.vectors)
    configs = json.loads(Path(args.configs).read_text())
    keep = set(json.loads(Path(args.records).read_text())["ids"])
    rows = [r for r in read_jsonl(Path(args.data)) if r["id"] in keep]
    rendered = {(r["id"], c): render_prompt(tok, r["variants"][c]["messages"], True, None, {}) for c in CONDS for r in rows}
    gc = model.generation_config
    eos = gc.eos_token_id if gc.eos_token_id is not None else tok.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = gc.pad_token_id if gc.pad_token_id is not None else tok.pad_token_id
    longest = max(len(x.input_ids) for x in rendered.values())
    per = slot_bytes(model.config, longest + args.max_new_tokens, torch.float32)
    slots = max(1, min(args.max_slots, int(args.cache_gb * 1e9 // per)))
    if torch.cuda.is_available() and not args.tiny:
        torch.cuda.set_per_process_memory_fraction(0.94)

    for cfg in configs:
        out = Path(args.out) / f"{args.model}__{cfg['name']}"
        final = out / "generations.jsonl"
        if final.exists():
            print(f"{out.name}: done already", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        delta = None
        if cfg.get("vector"):
            v = torch.tensor(vecs[cfg["vector"]], dtype=torch.float32)
            delta = (cfg["norm"] * v / v.norm()).to(model.device)
        partial = out / "generations.partial.jsonl"
        have = {(r["id"], r["condition"]) for r in read_partial(partial)}
        jobs = [Job(k, x.input_ids) for k, x in rendered.items() if k not in have]
        t = time.time()
        with partial.open("a") as f, AddVector(layers[cfg["layer"]] if delta is not None else None, delta):
            def save(j) -> None:
                text = tok.decode(j.out, skip_special_tokens=True)
                f.write(json.dumps({
                    "id": j.key[0], "condition": j.key[1], "prompt_text": rendered[j.key].text,
                    "n_prompt_tokens": len(j.ids), "response": text, "response_ids": j.out,
                    "n_response_tokens": len(j.out), "finish_reason": "stop" if j.out and j.out[-1] in eos else "length",
                    "repetition_3gram": repetition(text)}, ensure_ascii=False) + "\n")
                f.flush()

            generate_slots(model, jobs, slots=slots, max_new_tokens=args.max_new_tokens, eos_ids=eos, pad_id=pad,
                           on_done=save)
            os.fsync(f.fileno())
        by_key = {(r["id"], r["condition"]): r for r in read_partial(partial)}
        assert set(by_key) == set(rendered), "missing rows"
        final.write_text("".join(json.dumps(by_key[k], ensure_ascii=False) + "\n" for k in rendered))
        (out / "run_info.json").write_text(json.dumps({
            "step": "e09_intervene", "model": args.model, "config": cfg, "checkpoint_sha256": ckpt_sha,
            "vectors_sha256": sha256(Path(args.vectors)), "data_sha256": sha256(Path(args.data)),
            "n_records": len(rows), "n_generations": len(by_key), "max_new_tokens": args.max_new_tokens,
            "do_sample": False, "dtype": "float32", "slots": slots, "seconds": round(time.time() - t, 1),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and not args.tiny else None,
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
            "generations_sha256": sha256(final)}, indent=1) + "\n")
        print(f"{out.name}: {len(by_key)} generations in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
