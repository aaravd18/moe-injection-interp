"""Qwen3-MoE (used for the SEP replication) passes every extraction check on a tiny random model."""

import torch

from moe_interp.model import LoadedModel, build_tiny_random_qwen3_moe, get_moe_layout
from moe_interp.verify import verify_model


def test_qwen3_moe_verify(toy_tokenizer):
    tok = toy_tokenizer
    m = build_tiny_random_qwen3_moe(len(tok), pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                                    bos_token_id=tok.bos_token_id)
    lm = LoadedModel(m, tok, get_moe_layout(m), torch.device("cpu"), torch.float32, {})
    assert lm.layout.describe()["model_type"] == "qwen3_moe"
    report = verify_model(lm)
    assert report["passed"], {k: c for k, c in report["checks"].items() if not c["passed"]}
