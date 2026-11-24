"""E09 experiment 3: the intervention hook adds exactly delta to one decoder layer's output (tiny OLMoE, CPU)."""

import sys
from pathlib import Path

import torch

from conftest import make_tiny_loaded

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/modal/experiments"))
from e09_intervene import AddVector  # noqa: E402


def _layer_outputs(lm, ids):
    outs = {}
    hs = [l.register_forward_hook(lambda _m, _a, o, i=i: outs.__setitem__(i, (o[0] if isinstance(o, tuple) else o).clone()))
          for i, l in enumerate(lm.layout.layers)]
    with torch.no_grad():
        logits = lm.model(input_ids=ids).logits
    for h in hs:
        h.remove()
    return outs, logits


def test_hook_adds_delta_to_one_layer_only(toy_tokenizer):
    lm = make_tiny_loaded(toy_tokenizer)
    ids = torch.randint(0, 250, (2, 11))
    base, logits0 = _layer_outputs(lm, ids)
    delta = torch.randn(lm.model.config.hidden_size)
    k = 1
    seen = {}
    pre = lm.layout.layers[k + 1].register_forward_pre_hook(
        lambda _m, args, kwargs: seen.__setitem__("x", (args[0] if args else kwargs["hidden_states"]).clone()),
        with_kwargs=True)
    with AddVector(lm.layout.layers[k], delta):
        with torch.no_grad():
            logits1 = lm.model(input_ids=ids).logits
    pre.remove()
    torch.testing.assert_close(seen["x"], base[k] + delta)  # the next layer receives exactly output_k + delta
    assert not torch.equal(logits1, logits0)
    _, logits2 = _layer_outputs(lm, ids)
    assert torch.equal(logits2, logits0)  # the hook is removed on exit


def test_no_vector_is_a_no_op(toy_tokenizer):
    lm = make_tiny_loaded(toy_tokenizer)
    ids = torch.randint(0, 250, (1, 7))
    _, a = _layer_outputs(lm, ids)
    with AddVector(None, None):
        _, b = _layer_outputs(lm, ids)
    assert torch.equal(a, b)
