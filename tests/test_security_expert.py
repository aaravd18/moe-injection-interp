"""E05: always-active security expert (moe_interp/security_expert.py) on a tiny random OLMoE. Checks exact equivalence
with the base at init, that S_l reproduces one OLMoE expert's computation, the trainable set, the zero-init gradient
pattern, contribution recording, save/load, generation, and the full security-expert smoke routine of
e03_secpo_train.py. Nothing downloads."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from conftest import make_tiny_loaded
from moe_interp import security_expert as SE

ROOT = Path(__file__).resolve().parent.parent
PAIRS_V2 = ROOT / "data" / "secpo_olmoe_v2" / "pairs.jsonl"


def _tr():
    p = ROOT / "experiments" / "modal" / "experiments" / "e03_secpo_train.py"
    spec = importlib.util.spec_from_file_location("e03_secpo_train", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _batch(lm, n=3, T=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, lm.model.config.vocab_size, (n, T), generator=g)
    att = torch.ones_like(ids)
    att[1, 9:] = 0  # some right padding
    return ids, att


def test_init_is_bit_identical_and_router_untouched(toy_tokenizer):
    lm = make_tiny_loaded(toy_tokenizer)
    ids, att = _batch(lm)
    routers = []
    hooks = [r.register_forward_hook(lambda _m, _i, o: routers.append((o[2].clone(), o[1].clone())))
             for r in lm.layout.routers]
    with torch.no_grad():
        base = lm.model(input_ids=ids, attention_mask=att).logits
    n = len(routers)
    SE.add_security_experts(lm.model, lm.layout, alpha_init=0.1, seed=1)
    with torch.no_grad():
        new = lm.model(input_ids=ids, attention_mask=att).logits
        with SE.security_experts_disabled(lm.model):
            off = lm.model(input_ids=ids, attention_mask=att).logits
    for h in hooks:
        h.remove()
    assert torch.equal(base, new) and torch.equal(base, off)
    for (a, wa), (b, wb) in zip(routers[:n], routers[n: 2 * n]):
        assert torch.equal(a, b) and torch.equal(wa, wb)
    # module names of the original model are unchanged; the class is a subclass of the original block
    names = {n for n, _ in lm.model.named_parameters()}
    assert "model.layers.0.mlp.gate.weight" in names and "model.layers.0.mlp.experts.gate_up_proj" in names
    assert "model.layers.0.mlp.security_expert.down_proj" in names
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
    assert all(isinstance(b, OlmoeSparseMoeBlock) for b in lm.layout.moe_blocks)


def test_expert_matches_olmoe_expert_computation(toy_tokenizer):
    lm = make_tiny_loaded(toy_tokenizer)
    ses = SE.add_security_experts(lm.model, lm.layout, alpha_init=0.7)
    exp = lm.layout.moe_blocks[0].experts
    se = ses[0]
    assert se.gate_up_proj.shape == exp.gate_up_proj.shape[1:] and se.down_proj.shape == exp.down_proj.shape[1:]
    with torch.no_grad():
        se.gate_up_proj.copy_(exp.gate_up_proj[3])
        se.down_proj.copy_(exp.down_proj[3])
        x = torch.randn(5, lm.model.config.hidden_size)
        # OlmoeExperts.forward with every token routed to expert 3 at weight 1
        ref = exp(x, torch.full((5, 1), 3), torch.ones(5, 1))
        assert torch.allclose(se(x), 0.7 * ref, atol=1e-6)
        # and the block output is h_MoE + alpha * S
        blk = lm.layout.moe_blocks[0]
        xb = x.view(1, 5, -1)
        with SE.security_experts_disabled(lm.model):
            moe = blk(xb)
        assert torch.allclose(blk(xb), moe + 0.7 * ref.view(1, 5, -1), atol=1e-6)


def test_only_security_experts_trainable_and_grad_pattern(toy_tokenizer):
    lm = make_tiny_loaded(toy_tokenizer)
    SE.add_security_experts(lm.model, lm.layout, alpha_init=0.1)
    rep = SE.param_report(lm.model)
    cfg = lm.model.config
    L, H, I = cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size
    assert rep["trainable"] == L * (3 * H * I + 1) and not rep["non_security_expert_trainable"]
    assert not rep["router_trainable"] and rep["n_security_experts"] == L
    assert rep["trainable"] + rep["frozen"] == sum(p.numel() for p in lm.model.parameters())

    ids, att = _batch(lm)
    named = [(n, p) for n, p in lm.model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([p for _, p in named], lr=1e-2)
    frozen0 = {n: p.detach().clone() for n, p in lm.model.named_parameters() if not p.requires_grad}
    for step in range(2):
        lm.model(input_ids=ids, attention_mask=att).logits.float().pow(2).mean().backward()
        g = {n: (p.grad is not None and p.grad.abs().sum().item() > 0) for n, p in named}
        if step == 0:  # zero-init down_proj: only down_proj gets a gradient
            assert all(v for n, v in g.items() if n.endswith("down_proj"))
            assert not any(v for n, v in g.items() if n.endswith(("gate_up_proj", "alpha")))
        else:
            assert all(g.values())
        assert all(p.grad is None for n, p in lm.model.named_parameters() if not p.requires_grad)
        opt.step()
        opt.zero_grad(set_to_none=True)
    assert all(torch.equal(frozen0[n], p) for n, p in lm.model.named_parameters() if not p.requires_grad)


def test_contributions_save_load_and_generation(toy_tokenizer, tmp_path):
    lm = make_tiny_loaded(toy_tokenizer)
    ses = SE.add_security_experts(lm.model, lm.layout, alpha_init=0.1)
    ids, att = _batch(lm)
    with torch.no_grad(), SE.record_contributions(lm.model) as rec:
        lm.model(input_ids=ids, attention_mask=att)
    s = rec.summary()
    assert s["tokens"] == int(att.sum()) and s["ratio_mean"] == 0.0
    with torch.no_grad():
        for se in ses:
            se.down_proj.normal_(0, 0.1)
            se.alpha.fill_(0.3)
        with SE.record_contributions(lm.model) as rec:
            want = lm.model(input_ids=ids, attention_mask=att).logits
    assert rec.summary()["ratio_mean"] > 0 and all(b.security_expert_recorder is None for b in lm.layout.moe_blocks)
    SE.save_security_experts(lm.model, tmp_path / "se", {"alpha_init": 0.1})

    fresh = make_tiny_loaded(toy_tokenizer)
    cfg = SE.load_security_experts(fresh.model, fresh.layout, tmp_path / "se")
    assert cfg["alpha_init"] == 0.1 and SE.alphas(fresh.model) == [0.3] * len(ses)
    with torch.no_grad():
        assert torch.allclose(fresh.model(input_ids=ids, attention_mask=att).logits, want, atol=1e-6)
    with pytest.raises(ValueError):
        SE.add_security_experts(fresh.model, fresh.layout)

    # cached generation == uncached greedy (the expert is per-token, so KV caching is unaffected)
    p = ids[:1, :6]
    with torch.no_grad():
        gen = fresh.model.generate(p, max_new_tokens=5, do_sample=False, pad_token_id=0)
        cur = p
        for _ in range(5):
            nxt = fresh.model(input_ids=cur, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
            cur = torch.cat([cur, nxt], 1)
    assert torch.equal(gen, cur)


@pytest.mark.skipif(not PAIRS_V2.exists(), reason="needs data/secpo_olmoe_v2/pairs.jsonl")
def test_security_expert_smoke_routine_on_tiny_olmoe(toy_tokenizer, tmp_path):
    tr = _tr()
    rows = [json.loads(l) for l in PAIRS_V2.open()]
    short = sorted(rows, key=lambda r: len(r["preference"]["chosen"]) + len(r["preference"]["rejected"]))[:4]
    lm = make_tiny_loaded(toy_tokenizer)
    args = argparse.Namespace(micro_batch=2, eval_batch=4, max_len=100_000, beta=0.1, lr=1e-3, tiny=True)
    pad = toy_tokenizer.pad_token_id
    exs = tr.tokenize_pairs(toy_tokenizer, short, args.max_len)
    cache = tr.compute_cache(lm.model, exs, args, pad, lm.device, tr.nullcontext)
    with torch.no_grad():
        rc, rr, _ = tr.pair_logps(lm.model, exs[:2], "ref_prompt", args.max_len, pad, lm.device, tr.nullcontext)
        bc, br, _ = tr.pair_logps(lm.model, exs[:2], "policy_prompt", args.max_len, pad, lm.device, tr.nullcontext)
    SE.add_security_experts(lm.model, lm.layout, alpha_init=0.1, init_std=0.2)
    lm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rep = SE.param_report(lm.model)
    res = tr.smoke_security_expert(lm.model, lm.layout, exs, cache, args, pad, lm.device, tr.nullcontext, rep,
                                   {"rc": rc, "rr": rr, "bc": bc, "br": br}, tmp_path)
    assert res["all_ok"], {k: v for k, v in res["checks"].items() if not v["ok"]}
