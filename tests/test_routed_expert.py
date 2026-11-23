"""E08 routed security expert on tiny OLMoE (CPU, fp32): it must be exactly a 65-expert top-k MoE."""

import pytest
import torch
from torch.nn import functional as F

from moe_interp import routed_expert as RE
from moe_interp.config import ExperimentConfig
from moe_interp.runner import _generation_config, generate_batch
from moe_interp.slot_generate import Job, generate_slots

from conftest import make_tiny_loaded
from moe_interp.model import transformers_major_version

if transformers_major_version() < 5:  # fused experts + router module (transformers 5.x), as for E05's security expert
    pytest.skip("routed experts need transformers >= 5", allow_module_level=True)


def _batches(n=3, seed=0, vocab=250):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        lens = torch.randint(5, 40, (4,), generator=g).tolist()
        L = max(lens)
        ids = torch.zeros(4, L, dtype=torch.long)
        att = torch.zeros(4, L, dtype=torch.long)
        for i, k in enumerate(lens):  # left padding, as in training
            ids[i, L - k:] = torch.randint(0, vocab, (k,), generator=g)
            att[i, L - k:] = 1
        out.append((ids, att))
    return out


def _attach(lm, train_expert=True, copy_from=None, noise=0.0):
    copy_from = copy_from or [l % lm.layout.num_experts for l in range(lm.layout.num_layers)]
    RE.add_routed_experts(lm.model, lm.layout, copy_from, train_expert=train_expert, row_noise_std=noise)
    return copy_from


def _naive_block(block, x):
    """A from-scratch 65-expert MoE with the same weights: router [65, H] (+ offset on the new row), experts stacked."""
    W = torch.cat([block.gate.weight, block.re_row])
    logits = x @ W.T
    logits[:, -1] += block.re_bias
    p = F.softmax(logits.float(), dim=-1)
    v, i = p.topk(block.gate.top_k, dim=-1)
    if block.gate.norm_topk_prob:
        v = v / v.sum(-1, keepdim=True)
    gu = torch.cat([block.experts.gate_up_proj, block.re_gate_up[None]])
    dn = torch.cat([block.experts.down_proj, block.re_down[None]])
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for s in range(i.shape[1]):
            e = i[t, s]
            g, u = (gu[e] @ x[t]).chunk(2)
            out[t] += v[t, s] * (dn[e] @ (block.experts.act_fn(g) * u))
    return out


@pytest.mark.parametrize("bias", [-50.0, 0.0, 1.5, 50.0])
@pytest.mark.parametrize("norm_topk", [False, True])
def test_block_is_exactly_a_65_expert_moe(toy_tokenizer, bias, norm_topk):
    lm = make_tiny_loaded(toy_tokenizer, norm_topk_prob=norm_topk)
    _attach(lm)
    torch.manual_seed(0)
    for b in RE.blocks(lm.model):
        b.re_bias.fill_(bias)
        with torch.no_grad():  # make the new expert differ from its source, so a wrong index would show
            b.re_gate_up.add_(torch.randn_like(b.re_gate_up) * 0.1)
            b.re_down.add_(torch.randn_like(b.re_down) * 0.1)
    x = torch.randn(1, 37, lm.model.config.hidden_size)
    for b in RE.blocks(lm.model):
        got = b(x)[0]
        want = _naive_block(b, x[0])
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    sel = (x[0] @ torch.cat([b.gate.weight, b.re_row]).T)
    assert sel.shape[-1] == lm.layout.num_experts + 1


def test_disabled_is_the_exact_base_model(toy_tokenizer):
    lm = make_tiny_loaded(toy_tokenizer)
    ids, att = _batches(1)[0]
    with torch.no_grad():
        base = lm.model(input_ids=ids, attention_mask=att).logits.clone()
    _attach(lm)
    for b in RE.blocks(lm.model):
        b.re_bias.fill_(3.0)
    with torch.no_grad():
        on = lm.model(input_ids=ids, attention_mask=att).logits
        with RE.routed_experts_disabled(lm.model):
            off = lm.model(input_ids=ids, attention_mask=att).logits
    assert torch.equal(off, base)
    assert not torch.equal(on, base)


def test_copy_init_and_frozen_set(toy_tokenizer):
    lm = make_tiny_loaded(toy_tokenizer)
    src = _attach(lm)
    for b, j in zip(RE.blocks(lm.model), src):
        assert torch.equal(b.re_row[0], b.gate.weight[j])
        assert torch.equal(b.re_gate_up, b.experts.gate_up_proj[j])
        assert torch.equal(b.re_down, b.experts.down_proj[j])
    rep = RE.param_report(lm.model)
    H, I, L = lm.model.config.hidden_size, lm.model.config.intermediate_size, lm.layout.num_layers
    assert rep["trainable"] == L * (H + 2 * I * H + H * I)
    assert not rep["non_routed_expert_trainable"] and not rep["original_router_trainable"]


def test_gradients_reach_only_the_new_row_and_expert(toy_tokenizer):
    lm = make_tiny_loaded(toy_tokenizer)
    _attach(lm)
    RE.calibrate_bias(lm.model, _batches(), 0.3)
    lm.model.train()
    ids, att = _batches(1, seed=5)[0]
    out = lm.model(input_ids=ids, attention_mask=att).logits
    (out.float().pow(2).mean()).backward()
    for n, p in lm.model.named_parameters():
        if n.endswith(("re_row", "re_gate_up", "re_down")):
            assert p.grad is not None and p.grad.abs().sum() > 0, n
        else:
            assert p.grad is None, n


def test_calibration_hits_the_target_rate_layer_by_layer(toy_tokenizer):
    lm = make_tiny_loaded(toy_tokenizer, num_layers=4)
    _attach(lm, noise=1e-3)
    cal = _batches(4, seed=1)
    for target in (0.05, 0.2, 0.5):
        RE.calibrate_bias(lm.model, cal, target)
        with RE.record_routing(lm.model) as rec, torch.no_grad():
            for ids, att in cal:
                lm.model(input_ids=ids, attention_mask=att)
        rates = rec.summary()["rate_by_layer"]
        assert all(abs(r - target) < 0.01 for r in rates), (target, rates)


def test_routing_frequency_and_recorder_ignore_padding(toy_tokenizer):
    lm = make_tiny_loaded(toy_tokenizer)
    cal = _batches(2, seed=2)
    freq = RE.routing_frequency(lm.model, lm.layout, cal)
    k = lm.model.config.num_experts_per_tok
    assert all(abs(f.sum().item() - k) < 1e-9 for f in freq)  # each real token picks k experts
    _attach(lm, copy_from=[int(f.argmax()) for f in freq])
    with RE.record_routing(lm.model) as rec, torch.no_grad():
        for ids, att in cal:
            lm.model(input_ids=ids, attention_mask=att)
    assert rec.summary()["tokens"] == sum(int(a.sum()) for _, a in cal)


def test_save_load_round_trip(toy_tokenizer, tmp_path):
    lm = make_tiny_loaded(toy_tokenizer)
    _attach(lm, noise=0.01)
    RE.calibrate_bias(lm.model, _batches(), 0.25)
    with torch.no_grad():
        for b in RE.blocks(lm.model):
            b.re_down.mul_(1.7)
    ids, att = _batches(1, seed=9)[0]
    with torch.no_grad():
        want = lm.model(input_ids=ids, attention_mask=att).logits
    RE.save_routed_experts(lm.model, tmp_path / "re")
    fresh = make_tiny_loaded(toy_tokenizer)
    cfg = RE.load_routed_experts(fresh.model, fresh.layout, tmp_path / "re")
    with torch.no_grad():
        got = fresh.model(input_ids=ids, attention_mask=att).logits
    assert torch.equal(got, want) and len(cfg["bias"]) == fresh.layout.num_layers


def test_slot_generation_matches_single_prompt_generation(toy_tokenizer):
    lm = make_tiny_loaded(toy_tokenizer)
    _attach(lm)
    RE.calibrate_bias(lm.model, _batches(), 0.3)
    lm.model.eval()
    g = torch.Generator().manual_seed(4)
    prompts = [torch.randint(0, 250, (k,), generator=g).tolist() for k in (3, 17, 9, 30, 1, 12, 25)]
    cfg = ExperimentConfig()
    cfg.generation.max_new_tokens = 15
    gc = _generation_config(lm.model, cfg, toy_tokenizer)
    want = [generate_batch(lm.model, [p], gc, lm.device)[0] for p in prompts]
    done = {}
    generate_slots(lm.model, [Job(i, p) for i, p in enumerate(prompts)], slots=3, max_new_tokens=15,
                   eos_ids={toy_tokenizer.eos_token_id}, pad_id=0, on_done=lambda j: done.__setitem__(j.key, j.out))
    assert [done[i] for i in range(len(prompts))] == want


def test_attaching_twice_is_refused(toy_tokenizer):
    lm = make_tiny_loaded(toy_tokenizer)
    _attach(lm)
    with pytest.raises(ValueError):
        _attach(lm)
