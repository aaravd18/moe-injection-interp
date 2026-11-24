"""E09 capture: residual stream at the final prompt token, all layers, for one model on AF-208 x (clean + 6 attacks).

One teacher-forced prefill per prompt: batch 1, no padding, no KV cache, fp32, TF32 off. The prompts are the exact
token ids of the saved E07/E08 generations (rendered with the same chat template, and checked against each saved
row's prompt_text and n_prompt_tokens), so every activation belongs to a prompt whose response and judge labels exist.

Saved per prompt (resid index i = number of decoder layers applied; i = 0 is the token embedding, i = L the last
layer's output before the final norm):
    resid [N, L+1, H] float32         residual stream at the last prompt position
    next_argmax [N], ref_first [N]    greedy next token here vs the first token of the saved response
    ref_first_logp [N], top2_margin [N]   log p of the saved first token; logit gap between the top two tokens
    p_hack_start [N]                  probability mass on tokens that can start "hacked" (e04 hacked_token_ids)
    ids [N], conditions [N]           record id and condition per row, in the order of the saved generations file

Checks (checks.json; a failed hard check exits non-zero and saves nothing):
    prompts_match_saved       every rendered prompt equals the saved prompt_text / n_prompt_tokens   (hard)
    repeat_bit_identical      the first --check-n prompts, run again, give bit-identical activations  (hard)
    first_token_agreement     share of prompts whose greedy next token equals the saved first token
                              (hard if below --min-agree; near-ties across GPUs are expected to flip a few)

Usage:
    python e09_capture_resid.py --model base --model-dir DIR --ref GEN.jsonl --out OUT
    python e09_capture_resid.py --model lora|e05 --ckpt-root CKPT --model-dir DIR --ref GEN.jsonl --out OUT
    python e09_capture_resid.py --model routed --checkpoint routed_expert:DIR --model-dir DIR --ref GEN.jsonl --out OUT
    python e09_capture_resid.py --model base --tiny --limit 3 --ref GEN.jsonl --out /tmp/x      # CPU pipeline check
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from e07_generate import build_model, read_jsonl, sha256  # noqa: E402


def hacked_token_ids(tok) -> list[int]:
    """Same rule as e04_routing_capture.hacked_token_ids (copied to keep this script's imports small)."""
    ids = []
    for t in range(len(tok)):
        s = tok.decode([t]).strip().lower()
        if s == "h" or (len(s) >= 3 and ("hacked".startswith(s) or s.startswith("hacked"))):
            ids.append(t)
    return ids


class ResidCapture:
    """Hooks: pre-hook on layer 0 (embedding output = resid 0) and a forward hook on every decoder layer
    (resid l+1). Keeps only the last position."""

    def __init__(self, layers):
        self.layers, self.buf, self.handles = layers, {}, []

    @staticmethod
    def _last(x):
        x = x[0] if isinstance(x, tuple) else x
        return x[0, -1].detach().float().cpu()

    def __enter__(self):
        def pre(_m, args, kwargs):
            self.buf[0] = self._last(args[0] if args else kwargs["hidden_states"])

        self.handles.append(self.layers[0].register_forward_pre_hook(pre, with_kwargs=True))
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(
                lambda _m, _a, out, i=i: self.buf.__setitem__(i + 1, self._last(out))))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()

    def take(self):
        n = len(self.layers) + 1
        assert sorted(self.buf) == list(range(n)), sorted(self.buf)
        out = np.stack([self.buf[i].numpy() for i in range(n)])
        self.buf = {}
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", help="routed_expert:DIR for the E08 routed expert")
    ap.add_argument("--ckpt-root", help="dir with lora/adapter_model.safetensors and e05/security_expert.safetensors")
    ap.add_argument("--model-dir")
    ap.add_argument("--ref", required=True, help="the model's saved generations.jsonl (E07 / E08)")
    ap.add_argument("--data", default=str(REPO / "data" / "alpacafarm_secalign_208x6.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check-n", type=int, default=8)
    ap.add_argument("--min-agree", type=float, default=0.9)
    ap.add_argument("--limit", type=int, help="first N saved rows only (debugging)")
    ap.add_argument("--tiny", action="store_true", help="tiny random OLMoE on CPU (pipeline check; output is noise)")
    args = ap.parse_args()

    import torch

    from moe_interp.prompts import render_prompt

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    ref_rows = read_jsonl(Path(args.ref))[: args.limit]
    data = {r["id"]: r for r in read_jsonl(Path(args.data))}
    model, tok, info, ckpt_sha = build_model(args)
    from moe_interp.model import get_moe_layout

    layers = get_moe_layout(model).layers  # the decoder layers of the model as built (LoRA / experts attached)
    hack = torch.tensor(hacked_token_ids(tok), device=model.device)

    checks: dict = {}
    prompts, bad = [], []
    for r in ref_rows:
        ids = render_prompt(tok, data[r["id"]]["variants"][r["condition"]]["messages"], True, None, {}).input_ids
        if (tok.decode(ids) != r["prompt_text"] or len(ids) != r["n_prompt_tokens"]):
            bad.append((r["id"], r["condition"]))
        prompts.append(ids)
    checks["prompts_match_saved"] = {"ok": not bad, "n": len(prompts), "mismatched": bad[:10]}
    if bad:
        (out / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")
        raise SystemExit(f"{len(bad)} rendered prompts differ from the saved ones")
    load_s = time.time() - t0

    @torch.inference_mode()
    def forward(ids):
        x = torch.tensor([ids], device=model.device)
        with ResidCapture(layers) as cap:
            logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1).logits
        lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
        return cap.take(), lp

    N, L, H = len(prompts), len(layers), model.config.hidden_size
    resid = np.zeros((N, L + 1, H), dtype=np.float32)
    argmax = np.zeros(N, dtype=np.int64)
    first = np.array([r["response_ids"][0] if r["response_ids"] else -1 for r in ref_rows], dtype=np.int64)
    first_lp, margin, p_hack = (np.zeros(N, dtype=np.float32) for _ in range(3))
    t1 = time.time()
    for n, ids in enumerate(prompts):
        resid[n], lp = forward(ids)
        top2 = torch.topk(lp, 2).values
        argmax[n] = int(lp.argmax())
        first_lp[n] = float(lp[first[n]]) if first[n] >= 0 else float("nan")
        margin[n] = float(top2[0] - top2[1])
        p_hack[n] = float(lp[hack].exp().sum()) if len(hack) else float("nan")
        if (n + 1) % 200 == 0:
            print(f"{args.model}: {n + 1}/{N} prompts, {time.time() - t1:.0f} s", flush=True)
    fwd_s = time.time() - t1

    k = min(args.check_n, N)
    same = all(np.array_equal(forward(prompts[i])[0], resid[i]) for i in range(k))
    checks["repeat_bit_identical"] = {"ok": same, "n": k}
    agree = float((argmax == first).mean())
    flips = np.nonzero(argmax != first)[0]
    checks["first_token_agreement"] = {
        "ok": args.tiny or agree >= args.min_agree, "rate": round(agree, 4), "n_disagree": int(len(flips)),
        "median_top2_margin_of_disagreements": float(np.median(margin[flips])) if len(flips) else None,
        "examples": [{"id": ref_rows[i]["id"], "condition": ref_rows[i]["condition"], "argmax": tok.decode([int(argmax[i])]),
                      "saved": tok.decode([int(first[i])]), "top2_margin": round(float(margin[i]), 4)} for i in flips[:10]]}
    (out / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    print(json.dumps(checks), flush=True)
    if not all(c["ok"] for c in checks.values()):
        raise SystemExit("a hard check failed; nothing saved")

    path = out / "resid_last.npz"
    np.savez(path, resid=resid, next_argmax=argmax, ref_first=first, ref_first_logp=first_lp, top2_margin=margin,
             p_hack_start=p_hack, ids=np.array([r["id"] for r in ref_rows]),
             conditions=np.array([r["condition"] for r in ref_rows]))
    info_out = {"model": args.model, "checkpoint_sha256": ckpt_sha, "ref": str(args.ref), "ref_sha256": sha256(Path(args.ref)),
                "data_sha256": sha256(Path(args.data)), "n_prompts": N, "n_layers": L, "hidden": H, "dtype": "float32",
                "tf32": False, "batch": 1, "position": "last prompt token", "resid_index": "0 = embedding, i = output of layer i-1",
                "npz_sha256": sha256(path), "load_seconds": round(load_s, 1), "forward_seconds": round(fwd_s, 1),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and not args.tiny else None,
                "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                "tiny_random": args.tiny, "model_info": info}
    (out / "run_info.json").write_text(json.dumps(info_out, indent=1, default=str) + "\n")
    print(f"{args.model}: saved {path} ({N} x {L + 1} x {H}) in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
