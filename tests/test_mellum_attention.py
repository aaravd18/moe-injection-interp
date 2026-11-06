"""Mellum support and attention capture, against a tiny *real* MellumForCausalLM (random weights).

The tiny model has grouped-query attention (4 query / 2 KV heads), alternating sliding-window (window 8) and full
attention layers, and norm_topk_prob=True, so it exercises everything that differs from OLMoE.
"""

import pytest
import torch

from moe_interp.config import ConfigError, config_from_dict
from moe_interp.model import LoadedModel, build_tiny_random_mellum, get_moe_layout
from moe_interp.prompts import char_span_to_token_span, render_prompt
from moe_interp.runner import split_thinking
from moe_interp.storage import load_trace, save_trace
from moe_interp.tracing import TraceSpec, get_attn_implementation, run_traced_forward
from moe_interp.verify import verify_model

from conftest import make_tiny_loaded

ALL_TYPES = ["resid_pre", "attn_input", "attn_z", "attn_output", "resid_mid", "router_input", "moe_output", "resid_post"]
WINDOW = 8


@pytest.fixture(scope="module")
def mellum(toy_tokenizer):
    tok = toy_tokenizer
    m = build_tiny_random_mellum(len(tok), sliding_window=WINDOW, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id)
    # Transformers' capture_outputs installs its own (persistent) recorder hooks on the first forward call;
    # warm up so hook-count comparisons below only see hooks added by this package.
    with torch.inference_mode():
        m(input_ids=torch.tensor([[1, 2, 3]]), output_router_logits=True, use_cache=False)
    return LoadedModel(m, tok, get_moe_layout(m), torch.device("cpu"), torch.float32, {})


def _ids(tok, text="Doc: the bridge opened in 1932.\n\nQuestion: when?"):
    return torch.tensor([render_prompt(tok, [{"role": "user", "content": text}]).input_ids])


def _spec(L, **kw):
    base = dict(activation_types=ALL_TYPES, activation_layers=list(range(L)), positions="all",
                save_router_logits=True, save_attention=True, attention_layers=list(range(L)),
                attention_query_positions="all", attention_dtype=torch.float32)
    base.update(kw)
    return TraceSpec(**base)


def _all_hooks(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def test_layout(mellum):
    d = mellum.layout.describe()
    assert d["model_type"] == "mellum"
    assert (d["num_attention_heads"], d["num_key_value_heads"], d["head_dim"]) == (4, 2, 16)
    assert d["norm_topk_prob"] is True and d["sliding_window"] == WINDOW
    assert d["layer_types"] == ["sliding_attention", "full_attention"] * 2


def test_attention_patterns_shape_mask_and_restore(mellum, toy_tokenizer):
    lm = mellum
    ids = _ids(toy_tokenizer)
    S, L, H = ids.shape[1], lm.layout.num_layers, lm.layout.num_heads
    assert S > 2 * WINDOW
    before = get_attn_implementation(lm.model)
    assert before != "eager"
    n_hooks = _all_hooks(lm.model)
    tr = run_traced_forward(lm.model, lm.layout, ids, _spec(L))
    assert get_attn_implementation(lm.model) == before, "eager switch must be undone"
    assert _all_hooks(lm.model) == n_hooks
    att = tr.attention
    assert att.attn_implementation == "eager" and att.query_positions == list(range(S))
    q, k = torch.arange(S)[:, None], torch.arange(S)[None, :]
    for layer, A in att.patterns.items():
        assert A.shape == (H, S, S) and A.dtype == torch.float32
        torch.testing.assert_close(A.sum(-1), torch.ones(H, S))
        assert (A[:, (k > q).expand(S, S)] == 0).all(), "future keys must be exactly 0"
        far = (q - k) >= WINDOW
        if lm.layout.layer_types[layer] == "sliding_attention":
            assert (A[:, far.expand(S, S)] == 0).all(), "keys outside the sliding window must be exactly 0"
        else:
            assert (A[:, far.expand(S, S)] > 0).all(), "full-attention layers see every past key"
    # renormalised top-k: dispatched weights sum to 1
    torch.testing.assert_close(tr.routing.topk_weights.sum(-1), torch.ones(L, S))


def test_attention_query_subset_and_layer_subset(mellum, toy_tokenizer):
    lm = mellum
    ids = _ids(toy_tokenizer)
    S = ids.shape[1]
    full = run_traced_forward(lm.model, lm.layout, ids, _spec(lm.layout.num_layers))
    sub = run_traced_forward(lm.model, lm.layout, ids, _spec(
        lm.layout.num_layers, activation_types=[], attention_layers=[1, 3], attention_query_positions="last_k:3"))
    assert sorted(sub.attention.patterns) == [1, 3]
    assert sub.attention.query_positions == [S - 3, S - 2, S - 1]
    torch.testing.assert_close(sub.attention.patterns[3], full.attention.patterns[3][:, S - 3:])


def test_attention_identities(mellum, toy_tokenizer):
    lm = mellum
    tr = run_traced_forward(lm.model, lm.layout, _ids(toy_tokenizer), _spec(lm.layout.num_layers))
    a = tr.activations.tensors
    for layer in range(lm.layout.num_layers):
        attn = lm.layout.attentions[layer]
        assert a[("attn_z", layer)].shape[-1] == lm.layout.num_heads * lm.layout.head_dim
        torch.testing.assert_close(a[("resid_pre", layer)] + a[("attn_output", layer)], a[("resid_mid", layer)])
        torch.testing.assert_close(lm.layout.input_norms[layer](a[("resid_pre", layer)]), a[("attn_input", layer)])
        with torch.no_grad():
            torch.testing.assert_close(attn.o_proj(a[("attn_z", layer)]), a[("attn_output", layer)])


@pytest.mark.parametrize("which", ["mellum", "olmoe"])
def test_verify_passes_with_attention_checks(which, mellum, toy_tokenizer):
    lm = mellum if which == "mellum" else make_tiny_loaded(toy_tokenizer)
    report = verify_model(lm)
    failed = {k: c for k, c in report["checks"].items() if not c["passed"]}
    assert report["passed"], failed
    assert report["checks"]["manual_attn_z_from_patterns_rel_err"]["passed"]
    assert report["checks"]["attention_pattern_masked_entries_max"]["value"] == 0.0


def test_patterns_require_eager_semantics(mellum, toy_tokenizer):
    """The pattern hook refuses an attention module that returned no weights instead of saving nothing."""
    lm = mellum
    spec = _spec(lm.layout.num_layers, activation_types=[])
    from moe_interp.tracing import RoutingVerificationError, TraceRecorder

    ids = _ids(toy_tokenizer)
    n_hooks = _all_hooks(lm.model)
    with pytest.raises(RoutingVerificationError, match="eager"):
        with TraceRecorder(lm.layout, spec, ids.shape[1]):
            with torch.inference_mode():
                lm.model(input_ids=ids, use_cache=False)  # model is in sdpa here: weights are None
    assert _all_hooks(lm.model) == n_hooks


def test_attention_storage_roundtrip(mellum, toy_tokenizer, tmp_path):
    lm = mellum
    ids = _ids(toy_tokenizer)
    tr = run_traced_forward(lm.model, lm.layout, ids, _spec(lm.layout.num_layers))
    d = save_trace(tmp_path / "t", sample_id="s", condition="c", input_ids=ids[0].tolist(), routing=tr.routing,
                   activations=tr.activations, attention=tr.attention)
    t = load_trace(d)
    assert t.query_positions == tr.attention.query_positions
    for layer, A in tr.attention.patterns.items():
        torch.testing.assert_close(t.attention[layer], A)
        info = t.meta["attention"]["tensors"][f"attn_pattern.layer{layer:03d}"]
        assert info["dims"] == ["head", "query_position", "key_position"]
        assert info["layer_type"] == lm.layout.layer_types[layer]
    assert load_trace(d, load_attention=False).attention == {}


def test_split_thinking():
    assert split_thinking("<think>\nhmm\n</think>\n\nParis.") == ("hmm", "Paris.")
    assert split_thinking("Paris.") == (None, "Paris.")
    assert split_thinking("<think>\nstill thinking when cut off") == ("still thinking when cut off", "")


def test_zero_width_token_inside_span():
    # Mellum emits a standalone "Ġ" before digits whose offset is zero-width: (50, 50).
    offsets = [(38, 46), (47, 49), (50, 50), (50, 51), (51, 52)]
    ts = char_span_to_token_span(offsets, 47, 52)
    assert (ts.start, ts.end) == (1, 5)


def test_attention_config_validation():
    cfg = config_from_dict({"trace": {"attention": {"save_patterns": True, "layers": [0, -1], "dtype": "float32"}},
                            "prompt": {"chat_template_kwargs": {"enable_thinking": False}}})
    assert cfg.trace.attention.layers == [0, -1]
    with pytest.raises(ConfigError):
        config_from_dict({"trace": {"attention": {"dtype": "int8"}}})
    with pytest.raises(ConfigError):
        config_from_dict({"trace": {"attention": {"layers": "some"}}})
