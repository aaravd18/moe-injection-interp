"""Batched (left-padded) generation must reproduce batch-size-1 generation on the tiny models (fp32)."""

import pytest
import torch

from moe_interp.model import LoadedModel, build_tiny_random_mellum, get_moe_layout
from moe_interp.prompts import render_prompt
from moe_interp.runner import _generation_config, generate_batch
from moe_interp.config import ExperimentConfig

from conftest import make_tiny_loaded

TEXTS = ["Hi.", "Doc: the bridge opened in 1932.\n\nQuestion: when?", "Summarize: rain today, maybe snow tomorrow."]


def _mellum(tok):
    m = build_tiny_random_mellum(len(tok), sliding_window=8, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id)
    return LoadedModel(m, tok, get_moe_layout(m), torch.device("cpu"), torch.float32, {})


@pytest.mark.parametrize("which", ["olmoe", "mellum"])
def test_batched_equals_single(which, toy_tokenizer):
    lm = make_tiny_loaded(toy_tokenizer) if which == "olmoe" else _mellum(toy_tokenizer)
    cfg = ExperimentConfig()
    cfg.generation.max_new_tokens = 12
    gc = _generation_config(lm.model, cfg, toy_tokenizer)
    ids = [render_prompt(toy_tokenizer, [{"role": "user", "content": t}]).input_ids for t in TEXTS]
    single = [generate_batch(lm.model, [x], gc, lm.device)[0] for x in ids]
    batched = generate_batch(lm.model, ids, gc, lm.device)
    assert batched == single
    assert len({len(x) for x in ids}) > 1  # padding actually exercised
