"""Step 2/3 input: extract compact per-variant features from a SEP run's traces (runs where the traces live).

Reads only the slices it needs from each trace (safetensors ``safe_open``), so the ~100 GB of traces never has to
be loaded at once. Output: one ``features.npz`` + ``features_index.jsonl`` with, per (sample, condition):

* ``resid_last``    [L, H]  resid_post at the final prompt token (the position that emits the first answer token)
* ``resid_taskend`` [L, H]  resid_post at the last token of the trusted task (system message), i.e. *before* any data
                            (TaskTracker-style "activation delta" = resid_last - resid_taskend is computed later)
* ``attn_last``     [L, H]  attn_output at the final prompt token
* ``moe_last``      [L, H]  moe_output at the final prompt token
* ``rprob_last``    [L, E]  router softmax at the final prompt token
* ``ecount_<span>`` [L, E]  how many tokens of that span were routed to each expert (top-k counts), for spans
                            probe / insert / data / task / tail (the final ``--tail`` tokens: template + answer header)
* ``ntok_<span>``   []      span length in tokens
* ``amass_<span>``  [L, heads]  attention mass from the final prompt token onto the span (sum over the span's keys)
* ``mid_<span>``    [L, H]  mean over the span's tokens of resid_mid = resid_post - moe_output (the residual stream
                            the router reads, before its RMSNorm) -- for probe / insert spans. Token-level provenance
                            probe: can a router-style gate tell an injected instruction from a trusted one?
* ``rprob_<span>``  [L, E]  mean router softmax over the span's tokens (probe / insert)

Usage (on the Spark): python experiments/sep_features.py ~/moe-results/mellum_sep --tail 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

SPANS = ("probe", "insert", "data", "task")


def span_range(record: dict, name: str) -> tuple[int, int] | None:
    s = record["spans"].get(name)
    return (s["start"], s["end"]) if s else None


def extract(run: Path, record: dict, tail: int) -> dict[str, np.ndarray]:
    d = run / record["trace_dir"]
    meta = json.loads((d / "trace.json").read_text())
    S = meta["num_tokens"]
    L = meta["routing"]["num_layers"]
    out: dict[str, np.ndarray] = {}
    ranges = {k: span_range(record, k) for k in SPANS}
    ranges["tail"] = (S - tail, S)
    task_end = ranges["task"][1] - 1
    with safe_open(str(d / "activations.safetensors"), "pt") as f:
        def rows(kind: str, pos: int) -> np.ndarray:
            return np.stack([f.get_slice(f"{kind}.layer{l:03d}")[pos:pos + 1][0].float().numpy() for l in range(L)])
        out["resid_last"] = rows("resid_post", S - 1).astype(np.float16)
        out["resid_taskend"] = rows("resid_post", task_end).astype(np.float16)
        out["attn_last"] = rows("attn_output", S - 1).astype(np.float16)
        out["moe_last"] = rows("moe_output", S - 1).astype(np.float16)
        for name in ("probe", "insert"):
            r = ranges[name]
            if r is None:
                continue
            a, b = r
            mid = [(f.get_slice(f"resid_post.layer{l:03d}")[a:b].float() - f.get_slice(f"moe_output.layer{l:03d}")[a:b].float())
                   .mean(0).numpy() for l in range(L)]
            out[f"mid_{name}"] = np.stack(mid).astype(np.float16)
    with safe_open(str(d / "routing.safetensors"), "pt") as f:
        ids = f.get_tensor("topk_ids")  # [L, S, k]
        out["rprob_last"] = f.get_slice("router_probs")[:, S - 1:S, :][:, 0].float().numpy().astype(np.float32)
        for name in ("probe", "insert"):
            r = ranges[name]
            if r is not None:
                out[f"rprob_{name}"] = f.get_slice("router_probs")[:, r[0]:r[1], :].float().mean(1).numpy().astype(np.float32)
    E = meta["routing"]["num_experts"]
    for name, r in ranges.items():
        if r is None:
            continue
        a, b = r
        c = torch.zeros(L, E, dtype=torch.int32)
        c.scatter_add_(1, ids[:, a:b, :].reshape(L, -1), torch.ones(L, (b - a) * ids.shape[-1], dtype=torch.int32))
        out[f"ecount_{name}"] = c.numpy().astype(np.int16)
        out[f"ntok_{name}"] = np.array(b - a)
    with safe_open(str(d / "attention.safetensors"), "pt") as f:
        q = f.get_tensor("query_positions").tolist()
        qi = q.index(S - 1)
        A = np.stack([f.get_slice(f"attn_pattern.layer{l:03d}")[:, qi:qi + 1, :][:, 0].float().numpy() for l in range(L)])  # [L, heads, S]
    for name, r in ranges.items():
        if r is None or name == "tail":
            continue
        out[f"amass_{name}"] = A[:, :, r[0]:r[1]].sum(-1).astype(np.float32)
    out["amass_self_tail"] = A[:, :, S - tail:S].sum(-1).astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tail", type=int, default=8)
    args = ap.parse_args()
    run = Path(args.run).expanduser()
    recs = [json.loads(l) for l in open(run / "generations.jsonl")]
    feats: dict[str, list] = {}
    index = []
    for i, r in enumerate(recs):
        f = extract(run, r, args.tail)
        for k, v in f.items():
            feats.setdefault(k, [None] * len(recs))[i] = v
        index.append({
            "i": i, "sample_id": r["sample_id"], "condition": r["condition"],
            "executed": bool(r.get("attack_target_in_answer")), "finish_reason": r.get("finish_reason"),
            "first_token_match": r.get("first_token_matches_prefill"),
            "num_prompt_tokens": r["num_prompt_tokens"], **{k: r["metadata"][k] for k in
            ("probe_id", "subtask", "task_type", "probe_position_in_data", "insistent", "probe_words")},
        })
        if i % 200 == 0:
            print(f"{i}/{len(recs)}", flush=True)
    arrays = {}
    for k, vals in feats.items():
        present = np.array([v is not None for v in vals])
        shape = next(v for v in vals if v is not None).shape
        dtype = next(v for v in vals if v is not None).dtype
        arr = np.zeros((len(vals), *shape), dtype=dtype)
        for i, v in enumerate(vals):
            if v is not None:
                arr[i] = v
        arrays[k] = arr
        arrays[f"has_{k}"] = present
    np.savez_compressed(run / "features.npz", **arrays)
    with open(run / "features_index.jsonl", "w") as fo:
        for row in index:
            fo.write(json.dumps(row) + "\n")
    print("wrote", run / "features.npz", {k: v.shape for k, v in arrays.items() if not k.startswith("has_")})


if __name__ == "__main__":
    main()
