"""Shared Modal infrastructure for the experiments in this folder.

Everything a GPU experiment needs that is not specific to one experiment lives here:
  image            container with the pinned torch / transformers stack + the repo's `moe_interp` package
  hf_cache         persistent Modal volume holding the Hugging Face cache (Mellum weights download once, ~25 GB)
  MELLUM           model config dict (pinned revision, bf16, eager experts) in `moe_interp.config.ModelConfig` keys
  generate_variants   render chat prompts with the real template, batch-generate, split thinking/answer

Experiment scripts import this module, build their own `modal.App(image=image)`, and mount `hf_cache` at
HF_CACHE_DIR. `generate_variants` is plain Python, so it also runs locally on the tiny random Mellum for a quick
pipeline check (see README.md).
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

import modal

# Locally (modal run / python), make the repo's `moe_interp` importable so add_local_python_source can ship it.
# Doesn't rely on the editable install. In the container this file sits at /root/modal_common.py and moe_interp is
# already at /root/moe_interp.
_here = Path(__file__).resolve()
if len(_here.parents) > 3 and (_here.parents[3] / "moe_interp").is_dir():
    sys.path.insert(0, str(_here.parents[3]))

GPU = "H100"
HF_CACHE_DIR = "/root/.cache/huggingface"

MELLUM: dict[str, Any] = {
    "model_id": "JetBrains/Mellum2.1-12B-A2.5B-Thinking",
    "revision": "92ddae9fc7665e9f801d141d2e5a6b2caf2460c4",
    "dtype": "bfloat16",
    "experts_implementation": "eager",
}

# Same direct pins as the verified H100 / Spark runs (experiments/mellum_pilot_v1/requirements.txt).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.1",
        "transformers==5.22.1",
        "accelerate==1.15.0",
        "safetensors==0.8.0",
        "numpy==2.5.4",
        "pandas==3.1.1",
        "PyYAML==6.0.3",
        "huggingface-hub==2.4.0",
    )
    .env({"HF_HOME": HF_CACHE_DIR, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("moe_interp", "modal_common")
)

hf_cache = modal.Volume.from_name("moe-interp-hf-cache", create_if_missing=True)


def wb_hit(text: str | None, word: str) -> bool:
    """Whole-word, case-insensitive witness match (the label used in sep_behavior / sep_steering)."""
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])", text or "", re.I) is not None


def generate_variants(
    jobs: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    thinking: bool = False,
    max_new_tokens: int = 1024,
    batch_size: int = 32,
    progress_every: int = 20,
) -> dict[str, Any]:
    """Greedy-generate a response for every job.

    jobs: [{"key": any JSON-able id, "messages": [{"role", "content"}, ...]}, ...]
    Returns {"results": [{key, prompt_text, n_prompt_tokens, response_full, response_thinking, response_answer,
    n_response_tokens, finish_reason}, ...], "info": {...}} in the order of `jobs`.
    """
    import torch
    import transformers

    from moe_interp.config import config_from_dict
    from moe_interp.model import load_model
    from moe_interp.prompts import render_prompt
    from moe_interp.runner import _eos_ids, _generation_config, generate_batch, split_response

    cfg = config_from_dict({
        "model": model_cfg,
        "prompt": {"add_generation_prompt": True, "chat_template_kwargs": {} if thinking else {"enable_thinking": False}},
        "generation": {"max_new_tokens": max_new_tokens, "do_sample": False, "batch_size": batch_size},
        "trace": {"enabled": False},
    })
    t0 = time.time()
    lm = load_model(cfg.model)
    t_load = time.time() - t0
    tok = lm.tokenizer
    gc = _generation_config(lm.model, cfg, tok)
    eos = _eos_ids(gc)

    # Progress: every decoding step is one forward call of the top-level model; print every `progress_every` steps.
    step = {"n": 0}

    def _tick(*_):
        step["n"] += 1
        if progress_every and step["n"] % progress_every == 0:
            print(f"    step {step['n']}/{max_new_tokens}  ({time.time() - t1:.0f}s)", flush=True)

    lm.model.register_forward_hook(_tick)

    rendered = [render_prompt(tok, j["messages"], True, None, cfg.prompt.chat_template_kwargs) for j in jobs]
    results: list[dict[str, Any]] = []
    t1 = time.time()
    for b in range(0, len(jobs), batch_size):
        chunk = rendered[b: b + batch_size]
        step["n"] = 0
        print(f"  batch {b // batch_size + 1}/{-(-len(jobs) // batch_size)}: {len(chunk)} prompts", flush=True)
        outs = generate_batch(lm.model, [r.input_ids for r in chunk], gc, lm.device)
        for j, r, ids in zip(jobs[b: b + batch_size], chunk, outs):
            text = tok.decode(ids, skip_special_tokens=True)
            think, answer = split_response(tok, ids, text)
            results.append({
                "key": j["key"],
                "prompt_text": r.text,
                "n_prompt_tokens": len(r.input_ids),
                "response_full": text,
                "response_thinking": think,
                "response_answer": answer,
                "n_response_tokens": len(ids),
                "finish_reason": "stop" if ids and ids[-1] in eos else "length",
            })
        print(f"  generated {min(b + batch_size, len(jobs))}/{len(jobs)}  ({time.time() - t1:.0f}s)", flush=True)

    info = {
        "model": lm.info["model_id"],
        "requested_revision": lm.info["requested_revision"],
        "resolved_commit_hash": lm.info["resolved_commit_hash"],
        "dtype": lm.info["dtype"],
        "experts_implementation": lm.info["experts_implementation"],
        "device": lm.info["device"],
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "thinking": thinking,
        "chat_template_kwargs": cfg.prompt.chat_template_kwargs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "batch_size": batch_size,
        "load_seconds": round(t_load, 1),
        "generate_seconds": round(time.time() - t1, 1),
    }
    return {"results": results, "info": info}
