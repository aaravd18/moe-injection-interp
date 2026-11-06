import json

import pytest
import torch

from moe_interp.storage import (append_jsonl, load_trace, read_jsonl, safe_name, save_trace, trace_dir)
from moe_interp.tracing import ActivationTrace, RoutingTrace


def _routing(L=2, S=5, k=2, E=4, logits=True):
    g = torch.Generator().manual_seed(0)
    lg = torch.randn(L, S, E, generator=g)
    w, ids = torch.softmax(lg, -1).topk(k, -1)
    return RoutingTrace(ids, w, lg if logits else None, None, "router_module_output", "bfloat16", False, E,
                        {"max_abs_err_weights_vs_recomputed": 0.0})


def test_safe_name_deterministic_and_collision_free():
    assert safe_name("task_001") == "task_001"
    assert safe_name("clean") == "clean"
    a, b = safe_name("a/b"), safe_name("a_b")
    assert a != b and "/" not in a and a == safe_name("a/b")
    assert safe_name("inj v2 (polite)") != safe_name("inj_v2__polite_")
    assert safe_name("..") != ".."
    assert all(c.isalnum() or c in "._-" for c in safe_name("ünïcødé/../x"))
    with pytest.raises(ValueError):
        safe_name("")


def test_round_trip(tmp_path):
    r = _routing()
    acts = ActivationTrace([0, 4], {("resid_post", 1): torch.randn(2, 8), ("moe_output", 0): torch.randn(2, 8)},
                           "bfloat16")
    d = trace_dir(tmp_path, "s/1", "inj v2")
    save_trace(d, sample_id="s/1", condition="inj v2", input_ids=[5, 6, 7, 8, 9], routing=r, activations=acts,
               extra={"spans": {"q": {"start": 1, "end": 3}}})
    t = load_trace(d)
    assert torch.equal(t.topk_ids, r.topk_ids) and torch.equal(t.topk_weights, r.topk_weights)
    assert torch.equal(t.router_logits, r.router_logits) and t.router_probs is None
    assert t.input_ids.tolist() == [5, 6, 7, 8, 9] and t.positions == [0, 4]
    for key, v in acts.tensors.items():
        assert torch.equal(t.activations[key], v)
    assert t.spans == {"q": {"start": 1, "end": 3}}
    meta = json.loads((d / "trace.json").read_text())
    assert meta["routing"]["tensors"]["topk_ids"]["shape"] == [2, 5, 2]
    assert meta["routing"]["tensors"]["topk_ids"]["dims"] == ["layer", "token", "k"]
    assert meta["routing"]["num_experts"] == 4
    info = meta["activations"]["tensors"]["resid_post.layer001"]
    assert info["layer"] == 1 and info["shape"] == [2, 8] and "meaning" in info
    assert t.activation_layers("resid_post") == [1]
    t2 = load_trace(d, load_activations=False)
    assert t2.activations == {}


def test_bf16_activations_round_trip(tmp_path):
    x = torch.randn(1, 4).to(torch.bfloat16)
    save_trace(tmp_path, sample_id="a", condition="b", input_ids=[1, 2, 3, 4, 5], routing=_routing(logits=False),
               activations=ActivationTrace([4], {("resid_post", 0): x}, "bfloat16"))
    t = load_trace(tmp_path)
    assert t.activations[("resid_post", 0)].dtype == torch.bfloat16 and torch.equal(t.activations[("resid_post", 0)], x)


def test_length_mismatch_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_trace(tmp_path, sample_id="a", condition="b", input_ids=[1, 2], routing=_routing(), activations=None)


def test_jsonl(tmp_path):
    p = tmp_path / "g.jsonl"
    append_jsonl(p, {"a": 1, "t": "ü"})
    append_jsonl(p, {"a": 2})
    assert [r["a"] for r in read_jsonl(p)] == [1, 2]
