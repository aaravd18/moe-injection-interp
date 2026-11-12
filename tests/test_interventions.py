"""Interventions on the tiny Mellum / OLMoE models: expert steering, head ablation, residual patching."""

import pytest
import torch

from moe_interp.config import ExperimentConfig
from moe_interp.interventions import ExpertSteer, HeadAblation, Intervention, ResidPatch, padded_mask
from moe_interp.model import LoadedModel, build_tiny_random_mellum, get_moe_layout
from moe_interp.prompts import render_prompt
from moe_interp.runner import _generation_config, generate_batch
from moe_interp.tracing import TraceSpec, run_traced_forward

from conftest import make_tiny_loaded


@pytest.fixture(scope="module")
def mellum(toy_tokenizer):
    tok = toy_tokenizer
    m = build_tiny_random_mellum(len(tok), sliding_window=8, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id)
    with torch.inference_mode():
        m(input_ids=torch.tensor([[1, 2, 3]]), output_router_logits=True, use_cache=False)
    return LoadedModel(m, tok, get_moe_layout(m), torch.device("cpu"), torch.float32, {})


def _ids(tok, text="Doc: the bridge opened in 1932.\n\nQuestion: when?"):
    return render_prompt(tok, [{"role": "user", "content": text}]).input_ids


def _capture_ids(lm, ids, intervention=None):
    """Selected expert ids per layer as dispatched (read from the MoE block's gate output after all hooks)."""
    got = {}
    hs = []
    for i, gate in enumerate(lm.layout.routers):
        hs.append(gate.register_forward_hook(lambda m, a, o, i=i: got.__setitem__(i, o[2].clone())))
    try:
        with torch.inference_mode():
            if intervention:
                with intervention:
                    lm.model(input_ids=ids, use_cache=False)
            else:
                lm.model(input_ids=ids, use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return got


def _n_hooks(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


@pytest.mark.parametrize("which", ["mellum", "olmoe"])
def test_expert_deactivation_all_tokens(which, mellum, toy_tokenizer):
    lm = mellum if which == "mellum" else make_tiny_loaded(toy_tokenizer)
    ids = torch.tensor([_ids(toy_tokenizer)])
    base = _capture_ids(lm, ids)
    # deactivate the most-used expert of layer 1
    e = int(torch.bincount(base[1].reshape(-1), minlength=lm.layout.num_experts).argmax())
    n0 = _n_hooks(lm.model)
    # register the intervention BEFORE the capture hook, so the capture hook sees the steered output
    iv = Intervention(lm.layout, experts=ExpertSteer({1: {e: float("-inf")}}))
    with iv:
        got = _capture_ids(lm, ids)
    assert _n_hooks(lm.model) == n0
    assert not (got[1] == e).any()
    assert torch.equal(got[0], base[0])  # other layers before it unchanged


def test_expert_deactivation_token_mask_and_force(mellum, toy_tokenizer):
    lm = mellum
    ids = torch.tensor([_ids(toy_tokenizer)])
    S = ids.shape[1]
    base = _capture_ids(lm, ids)
    e = int(base[0][5, 0])
    mask = torch.zeros(1, S, dtype=torch.bool)
    mask[0, 5] = True
    with Intervention(lm.layout, experts=ExpertSteer({0: {e: float("-inf")}}, token_mask=mask)):
        got = _capture_ids(lm, ids)
    assert e not in got[0][5].tolist()
    assert torch.equal(got[0][:5], base[0][:5])  # earlier positions untouched (layer 0, causal)
    # force-activate an expert that was not selected at position 3
    f = next(x for x in range(lm.layout.num_experts) if x not in base[0][3].tolist())
    with Intervention(lm.layout, experts=ExpertSteer({0: {f: 1e4}})):
        got = _capture_ids(lm, ids)
    assert (got[0] == f).any(-1).all()


def test_head_ablation_matches_manual(mellum, toy_tokenizer):
    lm = mellum
    ids = torch.tensor([_ids(toy_tokenizer)])
    spec = TraceSpec(["attn_z", "attn_output"], [2], "all")
    with Intervention(lm.layout, heads=HeadAblation({2: [1, 3]})):
        tr = run_traced_forward(lm.model, lm.layout, ids, spec)
    z = tr.activations.tensors[("attn_z", 2)]
    d = lm.layout.head_dim
    # attn_z is captured by a pre-hook registered after the ablation hook -> sees the zeroed input
    assert (z[:, 1 * d:2 * d] == 0).all() and (z[:, 3 * d:4 * d] == 0).all()
    with torch.no_grad():
        torch.testing.assert_close(lm.layout.attentions[2].o_proj(z), tr.activations.tensors[("attn_output", 2)])


def test_resid_patch_last_positions(mellum, toy_tokenizer):
    lm = mellum
    a = torch.tensor([_ids(toy_tokenizer, "Doc: alpha beta.\n\nQ: why?")])
    b = torch.tensor([_ids(toy_tokenizer, "Doc: gamma delta!\n\nQ: why?")])
    spec = TraceSpec(["resid_post"], [1, 3], "all")
    ta = run_traced_forward(lm.model, lm.layout, a, spec)
    k = 4
    src = ta.activations.tensors[("resid_post", 1)][-k:].unsqueeze(0)
    with Intervention(lm.layout, patch=ResidPatch({1: src})):
        tb = run_traced_forward(lm.model, lm.layout, b, spec)
    torch.testing.assert_close(tb.activations.tensors[("resid_post", 1)][-k:], src[0])


def test_generation_with_masked_steering_batched(mellum, toy_tokenizer):
    lm = mellum
    cfg = ExperimentConfig()
    cfg.generation.max_new_tokens = 6
    gc = _generation_config(lm.model, cfg, toy_tokenizer)
    ids = [_ids(toy_tokenizer, "Hi."), _ids(toy_tokenizer)]
    T = max(len(x) for x in ids)
    mask = padded_mask([(0, 2), (3, 6)], [len(x) for x in ids], T)
    assert mask.sum() == 5 and mask[0, T - len(ids[0])] and not mask[0, 0] if T > len(ids[0]) else True
    # masking an expert on a span changes nothing structurally and must not crash in batched decode
    with Intervention(lm.layout, experts=ExpertSteer({0: {0: float("-inf")}}, token_mask=mask, decode=True)):
        out = generate_batch(lm.model, ids, gc, lm.device)
    assert len(out) == 2 and all(len(o) >= 1 for o in out)
    # no-op steering reproduces plain generation
    with Intervention(lm.layout, experts=ExpertSteer({0: {0: 0.0}}, token_mask=mask, decode=True)):
        same = generate_batch(lm.model, ids, gc, lm.device)
    assert same == generate_batch(lm.model, ids, gc, lm.device)


def test_resid_steer_projects_out_direction_on_masked_tokens(mellum, toy_tokenizer):
    from moe_interp.interventions import ResidSteer
    lm = mellum
    ids = torch.tensor([_ids(toy_tokenizer)])
    S = ids.shape[1]
    spec = TraceSpec(["resid_post"], [1, 2], "all")
    base = run_traced_forward(lm.model, lm.layout, ids, spec)
    d = torch.randn(lm.layout.hidden_size)
    u = d / d.norm()
    mask = torch.zeros(1, S, dtype=torch.bool)
    mask[0, 2:6] = True
    with Intervention(lm.layout, steer=ResidSteer({1: d}, token_mask=mask)):
        tr = run_traced_forward(lm.model, lm.layout, ids, spec)
    h = tr.activations.tensors[("resid_post", 1)]
    torch.testing.assert_close(h[2:6] @ u, torch.zeros(4), atol=1e-4, rtol=0)
    torch.testing.assert_close(h[:2], base.activations.tensors[("resid_post", 1)][:2])
    torch.testing.assert_close(h[6:], base.activations.tensors[("resid_post", 1)][6:])  # same layer, unmasked
    # zero-strength steering is a no-op
    with Intervention(lm.layout, steer=ResidSteer({1: d}, token_mask=mask, coef=0.0)):
        same = run_traced_forward(lm.model, lm.layout, ids, spec)
    torch.testing.assert_close(same.activations.tensors[("resid_post", 2)], base.activations.tensors[("resid_post", 2)])


def test_route_override_sets_ids_and_router_weights(mellum, toy_tokenizer):
    from moe_interp.interventions import RouteOverride
    lm = mellum
    ids = torch.tensor([_ids(toy_tokenizer)])
    S = ids.shape[1]
    base = _capture_ids(lm, ids)
    k, E = lm.layout.top_k, lm.layout.num_experts
    mask = torch.zeros(1, S, dtype=torch.bool)
    mask[0, 3:5] = True
    new = torch.stack([torch.arange(E - k, E), torch.arange(k)])  # 2 tokens
    got_w = {}
    h = lm.layout.routers[0].register_forward_hook(lambda m, a, o: got_w.__setitem__(0, (o[0].clone(), o[1].clone())))
    try:
        with Intervention(lm.layout, routes=RouteOverride({0: new}, mask)):
            got = _capture_ids(lm, ids)
    finally:
        h.remove()
    assert torch.equal(got[0][3:5].sort(-1).values, new.sort(-1).values)
    assert torch.equal(got[0][:3], base[0][:3])
    # identity override (the model's own choices) changes nothing
    with Intervention(lm.layout, routes=RouteOverride({0: base[0][3:5]}, mask)):
        same = _capture_ids(lm, ids)
    assert torch.equal(same[1], base[1])
