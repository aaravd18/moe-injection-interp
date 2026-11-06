"""Routing / activation extraction against a tiny *real* OlmoeForCausalLM (random weights)."""

import pytest
import torch

from moe_interp.model import transformers_major_version
from moe_interp.prompts import render_prompt
from moe_interp.tracing import (RoutingVerificationError, TraceRecorder, TraceSpec, resolve_layers,
                                resolve_positions, run_traced_forward)
from moe_interp.verify import manual_moe_output, verify_model

from conftest import make_tiny_loaded

ALL_TYPES = ["resid_pre", "resid_mid", "router_input", "moe_output", "resid_post"]


def _ids(tok, text="Doc: hello there.\n\nQuestion: why?"):
    rp = render_prompt(tok, [{"role": "user", "content": text}])
    return torch.tensor([rp.input_ids])


def _all_hooks(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def test_resolve_positions_and_layers():
    assert resolve_positions("last", 5) == [4]
    assert resolve_positions("all", 3) == [0, 1, 2]
    assert resolve_positions("last_k:2", 5) == [3, 4]
    assert resolve_positions("last_k:9", 3) == [0, 1, 2]
    assert resolve_positions([0, -1, -1], 5) == [0, 4]
    with pytest.raises(ValueError):
        resolve_positions([7], 5)
    with pytest.raises(ValueError):
        resolve_positions("middle", 5)
    assert resolve_layers([0, -1], 16) == [0, 15]
    assert resolve_layers("all", 3) == [0, 1, 2]
    with pytest.raises(ValueError):
        resolve_layers([16], 16)


def test_routing_shapes_semantics_and_consistency(tiny_loaded, toy_tokenizer):
    lm = tiny_loaded
    ids = _ids(toy_tokenizer)
    S = ids.shape[1]
    spec = TraceSpec(["resid_post"], [0], "last", save_router_logits=True, save_router_probs=True)
    tr = run_traced_forward(lm.model, lm.layout, ids, spec)
    r = tr.routing
    L, E, k = lm.layout.num_layers, lm.layout.num_experts, lm.layout.top_k
    assert (L, E, k) == (3, 8, 2)
    assert r.topk_ids.shape == (L, S, k) and r.topk_ids.dtype == torch.int64
    assert r.topk_weights.shape == (L, S, k) and r.topk_weights.dtype == torch.float32
    assert r.router_logits.shape == (L, S, E) and r.router_probs.shape == (L, S, E)
    # transformers 5.x routers return (logits, weights, ids); 4.x routers are nn.Linear returning logits only.
    assert r.source == ("router_module_output" if transformers_major_version() >= 5 else "recomputed_from_logits")
    assert r.checks["max_abs_err_logits_vs_output_router_logits"] == 0.0
    # Dispatched weights = softmax(logits)[ids], NOT renormalised (norm_topk_prob=False).
    probs = torch.softmax(r.router_logits, -1)
    torch.testing.assert_close(r.router_probs, probs)
    torch.testing.assert_close(r.topk_weights, probs.gather(-1, r.topk_ids))
    assert (r.topk_weights.sum(-1) < 1).all()
    # ids are the true top-k, in descending order of probability.
    torch.testing.assert_close(r.topk_ids, probs.topk(k, -1).indices)
    assert (r.topk_weights[..., :-1] >= r.topk_weights[..., 1:]).all()
    # Matches output_router_logits from the model itself.
    with torch.inference_mode():
        out = lm.model(ids, output_router_logits=True, use_cache=False)
    for layer in range(L):
        torch.testing.assert_close(out.router_logits[layer].float(), r.router_logits[layer])


def test_norm_topk_prob_true_is_respected(toy_tokenizer):
    lm = make_tiny_loaded(toy_tokenizer, norm_topk_prob=True)
    tr = run_traced_forward(lm.model, lm.layout, _ids(toy_tokenizer), TraceSpec([], [], "last"))
    torch.testing.assert_close(tr.routing.topk_weights.sum(-1), torch.ones(tr.routing.topk_weights.shape[:2]))
    assert tr.routing.norm_topk_prob


def test_dispatched_weights_reproduce_moe_output(tiny_loaded, toy_tokenizer):
    """Recomputing the MoE output by hand from the captured ids/weights reproduces the model's MoE output."""
    lm = tiny_loaded
    ids = _ids(toy_tokenizer)
    spec = TraceSpec(ALL_TYPES, [0, 1, 2], "all")
    tr = run_traced_forward(lm.model, lm.layout, ids, spec)
    a = tr.activations.tensors
    for layer in range(3):
        manual = manual_moe_output(lm.layout.moe_blocks[layer], a[("router_input", layer)],
                                   tr.routing.topk_ids[layer], tr.routing.topk_weights[layer])
        torch.testing.assert_close(manual, a[("moe_output", layer)], rtol=1e-4, atol=1e-5)
        # A wrong weighting (e.g. renormalised) must NOT reproduce it -> the check has teeth.
        w = tr.routing.topk_weights[layer]
        wrong = manual_moe_output(lm.layout.moe_blocks[layer], a[("router_input", layer)],
                                  tr.routing.topk_ids[layer], w / w.sum(-1, keepdim=True))
        assert not torch.allclose(wrong, a[("moe_output", layer)], rtol=1e-3, atol=1e-4)


def test_activation_semantics_and_shapes(tiny_loaded, toy_tokenizer):
    lm = tiny_loaded
    ids = _ids(toy_tokenizer)
    S, H = ids.shape[1], lm.layout.hidden_size
    tr = run_traced_forward(lm.model, lm.layout, ids, TraceSpec(ALL_TYPES, [0, 1, 2], "all"))
    a = tr.activations.tensors
    assert tr.activations.positions == list(range(S))
    with torch.inference_mode():
        hs = lm.model(ids, output_hidden_states=True, use_cache=False).hidden_states
    for layer in range(3):
        for t in ALL_TYPES:
            assert a[(t, layer)].shape == (S, H) and a[(t, layer)].device.type == "cpu"
        torch.testing.assert_close(a[("resid_mid", layer)] + a[("moe_output", layer)], a[("resid_post", layer)])
        with torch.inference_mode():
            torch.testing.assert_close(lm.layout.post_attention_norms[layer](a[("resid_mid", layer)]),
                                       a[("router_input", layer)])
        torch.testing.assert_close(a[("resid_pre", layer)], hs[layer][0])
        if layer < 2:
            torch.testing.assert_close(a[("resid_pre", layer + 1)], a[("resid_post", layer)])
    # Embedding output = first layer input.
    with torch.inference_mode():
        torch.testing.assert_close(a[("resid_pre", 0)], lm.model.model.embed_tokens(ids)[0])


def test_last_position_matches_all_positions(tiny_loaded, toy_tokenizer):
    lm = tiny_loaded
    ids = _ids(toy_tokenizer)
    full = run_traced_forward(lm.model, lm.layout, ids, TraceSpec(["resid_post"], [2], "all"))
    last = run_traced_forward(lm.model, lm.layout, ids, TraceSpec(["resid_post"], [2], "last"))
    assert last.activations.positions == [ids.shape[1] - 1]
    torch.testing.assert_close(last.activations.tensors[("resid_post", 2)][0],
                               full.activations.tensors[("resid_post", 2)][-1])
    torch.testing.assert_close(last.routing.topk_ids, full.routing.topk_ids)


def test_hooks_removed_after_run_and_after_errors(tiny_loaded, toy_tokenizer):
    lm = tiny_loaded
    before = _all_hooks(lm.model)
    ids = _ids(toy_tokenizer)
    run_traced_forward(lm.model, lm.layout, ids, TraceSpec(ALL_TYPES, [0, 1, 2], "all"))
    assert _all_hooks(lm.model) == before
    rec = TraceRecorder(lm.layout, TraceSpec(ALL_TYPES, [0, 1, 2], "all"), ids.shape[1])
    with pytest.raises(RuntimeError, match="boom"):
        with rec:
            assert _all_hooks(lm.model) > before
            raise RuntimeError("boom")
    assert not rec.active and _all_hooks(lm.model) == before
    # Wrong declared seq_len -> shape check fires, and hooks are still cleaned up.
    with pytest.raises(RoutingVerificationError):
        with TraceRecorder(lm.layout, TraceSpec(["resid_post"], [0], "last"), ids.shape[1] + 1):
            with torch.inference_mode():
                lm.model(ids, use_cache=False)
    assert _all_hooks(lm.model) == before


def test_batch_size_must_be_one(tiny_loaded, toy_tokenizer):
    ids = _ids(toy_tokenizer).repeat(2, 1)
    with pytest.raises(ValueError, match=r"\[1, seq_len\]"):
        run_traced_forward(tiny_loaded.model, tiny_loaded.layout, ids, TraceSpec([], [], "last"))


def test_traces_deterministic(tiny_loaded, toy_tokenizer):
    ids = _ids(toy_tokenizer)
    spec = TraceSpec(["moe_output"], [1], "all")
    a = run_traced_forward(tiny_loaded.model, tiny_loaded.layout, ids, spec)
    b = run_traced_forward(tiny_loaded.model, tiny_loaded.layout, ids, spec)
    assert torch.equal(a.routing.topk_ids, b.routing.topk_ids)
    assert torch.equal(a.activations.tensors[("moe_output", 1)], b.activations.tensors[("moe_output", 1)])


def test_verify_model_passes_on_tiny_model(tiny_loaded):
    report = verify_model(tiny_loaded, n_tokens=3, max_new_tokens=3)
    failed = {k: v for k, v in report["checks"].items() if not v["passed"]}
    assert report["passed"], failed
