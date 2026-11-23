"""Slot-refill generation must give what one-prompt-at-a-time greedy generate gives (tiny OLMoE, fp32, CPU)."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from moe_interp.config import ExperimentConfig
from moe_interp.runner import _generation_config, generate_batch
from moe_interp.slot_generate import Job, generate_slots

from conftest import make_tiny_loaded

ROOT = Path(__file__).resolve().parent.parent
MAX_NEW = 24


def _prompts(n=23):
    g = torch.Generator().manual_seed(3)
    lens = torch.randint(1, 70, (n,), generator=g).tolist()
    return [torch.randint(0, 250, (k,), generator=g).tolist() for k in lens]


def _single(lm, model, prompts, eos):
    cfg = ExperimentConfig()
    cfg.generation.max_new_tokens = MAX_NEW
    gc = _generation_config(model, cfg, lm.tokenizer)
    gc.eos_token_id = sorted(eos)
    return [generate_batch(model, [p], gc, lm.device)[0] for p in prompts]


def _slots(model, prompts, eos, slots, **kw):
    done = []
    generate_slots(model, [Job(i, p) for i, p in enumerate(prompts)], slots=slots, max_new_tokens=MAX_NEW,
                   eos_ids=eos, pad_id=0, on_done=done.append, **kw)
    assert sorted(j.key for j in done) == list(range(len(prompts)))  # each exactly once
    return [j.out for j in sorted(done, key=lambda j: j.key)]


def _early_stop_tokens(lm, model, prompts):
    """EOS ids taken from the model's own outputs, so that rows finish at many different steps."""
    free = _single(lm, model, prompts, {lm.tokenizer.eos_token_id})
    return {lm.tokenizer.eos_token_id} | {o[min(k % 17, len(o) - 1)] for k, o in enumerate(free) if k % 3 == 0}


def _add_expert(lm):
    from moe_interp.security_expert import add_security_experts

    for se in add_security_experts(lm.model, lm.layout, alpha_init=0.5):
        torch.nn.init.normal_(se.down_proj, std=0.1)
    lm.model.eval()


@pytest.mark.parametrize("variant", ["base", "security_expert"])
@pytest.mark.parametrize("slots", [1, 2, 5, 23, 64])
def test_slots_equal_single(variant, slots, toy_tokenizer):
    lm = make_tiny_loaded(toy_tokenizer)
    if variant == "security_expert":
        _add_expert(lm)
    prompts = _prompts()
    eos = _early_stop_tokens(lm, lm.model, prompts)
    want = _single(lm, lm.model, prompts, eos)
    assert len({len(o) for o in want}) > 2  # compaction and refill are exercised
    assert _slots(lm.model, prompts, eos, slots) == want


def test_dirty_memory_does_not_leak_into_the_cache(toy_tokenizer, monkeypatch):
    lm = make_tiny_loaded(toy_tokenizer)
    prompts = _prompts()
    eos = _early_stop_tokens(lm, lm.model, prompts)
    want = _single(lm, lm.model, prompts, eos)
    empty = torch.empty

    def dirty(*a, **k):
        x = empty(*a, **k)
        return x.fill_(float("nan")) if x.is_floating_point() else x

    monkeypatch.setattr(torch, "empty", dirty)
    assert _slots(lm.model, prompts, eos, 5) == want


def test_small_prefill_bounds_still_complete(toy_tokenizer):
    lm = make_tiny_loaded(toy_tokenizer)
    prompts = _prompts()
    eos = _early_stop_tokens(lm, lm.model, prompts)
    got = _slots(lm.model, prompts, eos, 8, max_prefill_tokens=10, max_prefill_attn_cells=10)
    assert got == _single(lm, lm.model, prompts, eos)


def test_merged_expert_lora_is_bit_identical(toy_tokenizer):
    peft = pytest.importorskip("peft")
    spec = importlib.util.spec_from_file_location(
        "e07_generate", ROOT / "experiments" / "modal" / "experiments" / "e07_generate.py")
    e07 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e07)

    lm = make_tiny_loaded(toy_tokenizer)
    cfg = peft.LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=2.0, lora_dropout=0.0,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          target_parameters=["mlp.experts.down_proj", "mlp.experts.gate_up_proj"])
    model = peft.get_peft_model(copy.deepcopy(lm.model), cfg).eval()
    torch.manual_seed(1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.normal_(0, 0.05)
    prompts = _prompts()
    x = torch.tensor([prompts[0]])
    unmerged = model(input_ids=x).logits.detach().clone()
    eos = _early_stop_tokens(lm, model, prompts)
    want = _single(lm, model, prompts, eos)

    assert e07.merge_expert_lora(model) == 2 * lm.layout.num_layers
    assert torch.equal(model(input_ids=x).logits, unmerged)
    assert _slots(model.get_base_model(), prompts, eos, 5) == want
