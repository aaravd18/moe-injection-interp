"""Offline checks for E03 SecPO training: the loss matches the official SecPO code, the grouped split has no leakage,
and the full smoke routine (LoRA on attention + experts, frozen router, cached reference, one optimizer step) passes on
a tiny random OLMoE. Nothing downloads; the LoRA tests need `peft` and are skipped without it."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from conftest import make_tiny_loaded

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ROOT / "data" / "secpo_olmoe_v1" / "pairs.jsonl"
SPLIT = ROOT / "experiments" / "modal" / "artifacts" / "splits" / "secpo_olmoe_v1_grouped_v1.json"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tr():
    return load(ROOT / "experiments" / "modal" / "experiments" / "e03_secpo_train.py", "e03_secpo_train")


@pytest.fixture(scope="module")
def pairs():
    return [json.loads(l) for l in PAIRS.open()]


# ------------------------------------------------------------------------------------------------------------ loss

def official_sequence_logps(logits, labels):  # SecPO src/secpo/loss.py, verbatim math
    sl, lab = logits[:, :-1, :], labels[:, 1:]
    mask = lab.ne(-100)
    lp = F.log_softmax(sl, dim=-1).gather(-1, lab.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    return lp.masked_fill(~mask, 0.0).sum(-1)


def test_sequence_logps_matches_official(tr):
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 9, 17, generator=g)
    labels = torch.randint(0, 17, (3, 9), generator=g)
    labels[:, :4] = -100
    labels[1, 7:] = -100
    assert torch.allclose(tr.sequence_logps(logits, labels), official_sequence_logps(logits, labels), atol=1e-5)
    with pytest.raises(ValueError):
        tr.sequence_logps(logits, torch.full_like(labels, -100))


def test_secpo_loss_matches_formula_and_reduces_to_dpo(tr):
    pc, pr, rc, rr = (torch.tensor(v) for v in ([-10.0, -3.0], [-4.0, -9.0], [-8.0, -5.0], [-6.0, -2.0]))
    o = tr.secpo_loss(pc, pr, rc, rr, 0.1)
    want = [-math.log(1 / (1 + math.exp(-0.1 * ((a - b) - (c - d))))) for a, b, c, d in zip(pc, pr, rc, rr)]
    assert torch.allclose(o["losses"], torch.tensor(want), atol=1e-6)
    assert torch.allclose(o["chosen_rewards"], 0.1 * (pc - rc)) and torch.allclose(o["rejected_rewards"], 0.1 * (pr - rr))
    # policy == reference -> loss is log 2 (DPO at init)
    assert torch.allclose(tr.secpo_loss(pc, pr, pc, pr, 0.1)["losses"], torch.full((2,), math.log(2)))


def test_build_input_and_labels_follows_official(tr):
    ids, lab = tr.build_input_and_labels([1, 2, 3, 4], [7, 8], 5)  # prompt left-truncated to fit
    assert ids == [2, 3, 4, 7, 8] and lab == [-100, -100, -100, 7, 8]
    with pytest.raises(ValueError):
        tr.build_input_and_labels([1], [7, 8, 9], 3)  # responses are never truncated


def test_response_ids_eos_only_when_generation_ended(tr, toy_tokenizer):
    eos = toy_tokenizer.eos_token_id
    assert tr.response_ids(toy_tokenizer, "ok", True)[-1] == eos
    assert tr.response_ids(toy_tokenizer, "ok", False)[-1] != eos


# ----------------------------------------------------------------------------------------------------------- split

def test_split_is_grouped_and_consistent(pairs):
    split = json.loads(SPLIT.read_text())
    recs = [json.loads(l) for l in (ROOT / split["data"]).open()]
    assert split["data_sha256"] == hashlib.sha256((ROOT / split["data"]).read_bytes()).hexdigest()
    tr_ids, va_ids = set(split["ids"]["train"]), set(split["ids"]["val"])
    assert not tr_ids & va_ids and tr_ids | va_ids == {r["id"] for r in recs}
    pair_ids = {r["id"] for r in pairs}
    assert set(split["pair_ids"]["train"]) == tr_ids & pair_ids and set(split["pair_ids"]["val"]) == va_ids & pair_ids
    assert len(split["pair_ids"]["train"]) == 556 and len(split["pair_ids"]["val"]) == 98
    sp = load(ROOT / "scripts" / "split_secpo_olmoe.py", "split_secpo_olmoe")
    side = {r["id"]: ("val" if r["id"] in va_ids else "train") for r in recs}
    for g in sp.group_records(recs):  # every group lies on one side
        assert len({side[recs[i]["id"]] for i in g}) == 1


def test_all_pairs_fit_max_len_without_truncation(tr, pairs, toy_tokenizer):
    # Byte-level toy tokenizer gives far more tokens than OLMoE's; check structure, not the 1024 budget.
    exs = tr.tokenize_pairs(toy_tokenizer, pairs[:20], 100_000)
    assert not any(e["truncated_prompt"] for e in exs)
    for e in exs:
        assert len(e["policy_prompt"]) > len(e["ref_prompt"])  # the injection is only in the policy prompt
        assert toy_tokenizer.decode(e["policy_prompt"]).endswith("<|assistant|>\n")


# ------------------------------------------------------------------------------------------------- LoRA + smoke

def _args(**kw):
    base = dict(micro_batch=2, eval_batch=4, max_len=100_000, beta=0.1, lr=1e-3, tiny=True)
    return argparse.Namespace(**{**base, **kw})


def test_smoke_routine_passes_on_tiny_olmoe(tr, pairs, toy_tokenizer, tmp_path):
    pytest.importorskip("peft")
    short = sorted(pairs, key=lambda r: len(r["preference"]["chosen"]) + len(r["preference"]["rejected"]))[:4]
    lm = make_tiny_loaded(toy_tokenizer)
    args = _args()
    exs = tr.tokenize_pairs(toy_tokenizer, short, args.max_len)
    cache = tr.compute_cache(lm.model, exs, args, toy_tokenizer.pad_token_id, lm.device, tr.nullcontext)
    with torch.no_grad():
        rc, rr, _ = tr.pair_logps(lm.model, exs[:2], "ref_prompt", args.max_len, toy_tokenizer.pad_token_id,
                                  lm.device, tr.nullcontext)
        bc, br, _ = tr.pair_logps(lm.model, exs[:2], "policy_prompt", args.max_len, toy_tokenizer.pad_token_id,
                                  lm.device, tr.nullcontext)
    pre = {"rc": rc, "rr": rr, "bc": bc, "br": br}
    policy, _, kind = tr.attach_lora(lm.model, lm.layout, 4, 1.0, True)
    rep = tr.param_report(policy)
    assert rep["trainable_by_group"]["experts"] > 0 and rep["trainable_by_group"]["attention"] > 0
    res = tr.smoke(policy, exs, cache, args, toy_tokenizer.pad_token_id, lm.device, tr.nullcontext, rep, pre)
    assert res["all_ok"], {k: v for k, v in res["checks"].items() if not v["ok"]}

    # The saved adapter reproduces the trained policy on a fresh copy of the same base model.
    from peft import PeftModel

    policy.eval()
    with torch.no_grad():
        want = tr.pair_logps(policy, exs[:2], "policy_prompt", args.max_len, toy_tokenizer.pad_token_id,
                             lm.device, tr.nullcontext)
    policy.save_pretrained(tmp_path / "adapter")
    fresh = PeftModel.from_pretrained(make_tiny_loaded(toy_tokenizer).model, tmp_path / "adapter").eval()
    with torch.no_grad():
        got = tr.pair_logps(fresh, exs[:2], "policy_prompt", args.max_len, toy_tokenizer.pad_token_id,
                            lm.device, tr.nullcontext)
    assert torch.allclose(want[0], got[0], atol=1e-4) and torch.allclose(want[1], got[1], atol=1e-4)


# ------------------------------------------------------------------------------------------------ v2 (paper y_l)

PAIRS_V2 = ROOT / "data" / "secpo_olmoe_v2" / "pairs.jsonl"


@pytest.mark.skipif(not PAIRS_V2.exists(), reason="build with scripts/build_secpo_olmoe_v2.py")
def test_v2_pairs_are_unfiltered_paper_construction(tr, toy_tokenizer):
    v2 = [json.loads(l) for l in PAIRS_V2.open()]
    sel = [json.loads(l)["id"] for l in (ROOT / "data" / "secpo_olmoe_v1" / "selected.jsonl").open()]
    assert [r["id"] for r in v2] == sel  # every selected record is a pair, nothing filtered
    split = json.loads(SPLIT.read_text())
    assert set(split["ids"]["train"]) | set(split["ids"]["val"]) == set(sel)
    for r in v2[:50]:
        g = r["olmoe"]["seed_task"]
        # y_l comes from the seed task alone: no trusted instruction, no clean data, no attack wrapper
        assert g["prompt_text"].endswith(f"<|user|>\n{r['metadata']['seed_task'].strip()}\n<|assistant|>\n")
        assert "<|system|>" not in g["prompt_text"]
        assert r["preference"]["rejected"] == g["response"]
        assert r["preference"]["chosen"] == r["olmoe"]["clean"]["response"]
    exs = tr.tokenize_pairs(toy_tokenizer, v2[:20], 100_000)
    for r, e in zip(v2[:20], exs):
        assert len(e["policy_prompt"]) > len(e["ref_prompt"])
        assert e["strict"]  # no judge labels in v2
        stop = r["olmoe"]["seed_task"]["finish_reason"] == "stop"
        assert e["rejected"] == tr.response_ids(toy_tokenizer, r["preference"]["rejected"], stop)
