"""Metrics on hand-built traces with known answers (no model)."""

import json

import numpy as np
import pytest
import torch

from moe_interp.analysis import AnalysisSettings, Experiment, compute_metrics, run_analysis
from moe_interp.plotting import make_all_plots
from moe_interp.storage import append_jsonl, save_trace, trace_dir
from moe_interp.tracing import ActivationTrace, RoutingTrace

E, K, L, H = 4, 2, 2, 3


def _write(root, sid, cond, input_ids, ids, acts_last):
    ids = torch.tensor(ids, dtype=torch.int64)  # [L, S, K]
    r = RoutingTrace(ids, torch.full(ids.shape, 0.4), None, None, "router_module_output", "float32", False, E, {})
    S = len(input_ids)
    acts = ActivationTrace([S - 1], {("resid_post", layer): torch.tensor([acts_last[layer]], dtype=torch.float32)
                                     for layer in range(L)}, "float32")
    d = trace_dir(root, sid, cond)
    save_trace(d, sample_id=sid, condition=cond, input_ids=input_ids, routing=r, activations=acts)
    append_jsonl(root / "generations.jsonl", {"sample_id": sid, "condition": cond, "trace_dir": str(d.relative_to(root)),
                                              "num_prompt_tokens": S, "response_text": "x",
                                              "attack_target_in_response": cond == "injection"})


@pytest.fixture
def synthetic(tmp_path):
    # clean:     [1, 2, | 7, 8]      -> suffix [7, 8] shared by all
    # injection: [1, 9, 9, | 7, 8]
    # benign:    [1, 3, | 7, 8]
    same = [[0, 1], [0, 1]]
    _write(tmp_path, "s1", "clean", [1, 2, 7, 8],
           [[[0, 1], [0, 1], [0, 1], [0, 1]], [[2, 3], [2, 3], [2, 3], [2, 3]]], [[1, 0, 0], [1, 0, 0]])
    _write(tmp_path, "s1", "injection", [1, 9, 9, 7, 8],
           [[[3, 2], [3, 2], [3, 2], [0, 2], [2, 3]], [same[0]] * 5], [[0, 1, 0], [1, 0, 0]])
    _write(tmp_path, "s1", "benign", [1, 3, 7, 8],
           [[[0, 1], [0, 1], [0, 1], [0, 1]], [[2, 3], [2, 3], [2, 3], [2, 3]]], [[1, 0, 0], [2, 0, 0]])
    return tmp_path


def test_overlap_jsd_and_alignment(synthetic):
    t = compute_metrics(Experiment.open(synthetic), AnalysisSettings())
    al = t["alignment"].set_index("condition")
    assert al.loc["injection", "n_aligned"] == 2 and al.loc["injection", "ok"]
    assert al.loc["injection", "a_start"] == 2 and al.loc["injection", "b_start"] == 3
    ov = t["overlap"].set_index(["condition", "layer"])["overlap"]
    # layer 0 aligned: clean {0,1},{0,1} vs injection {0,2},{2,3} -> (1/2 + 0)/2
    assert ov[("injection", 0)] == pytest.approx(0.25)
    # layer 1: clean {2,3} vs injection {0,1} -> 0
    assert ov[("injection", 1)] == pytest.approx(0.0)
    assert ov[("benign", 0)] == 1.0 and ov[("benign", 1)] == 1.0
    jsd = t["jsd"].set_index(["condition", "layer"])["jsd"]
    assert jsd[("injection", 1)] == pytest.approx(1.0)   # disjoint expert sets
    assert jsd[("benign", 0)] == pytest.approx(0.0)


def test_activation_similarity_and_control_delta(synthetic):
    t = compute_metrics(Experiment.open(synthetic), AnalysisSettings())
    act = t["activation_similarity"].set_index(["condition", "layer"])
    assert act.loc[("injection", 0), "cosine"] == pytest.approx(0.0)
    assert act.loc[("benign", 1), "cosine"] == pytest.approx(1.0)
    assert act.loc[("benign", 1), "rel_l2"] == pytest.approx(1.0)   # ||[1,0,0]-[2,0,0]|| / ||[1,0,0]||
    d = t["control_delta"]
    ov = d[(d.metric == "routing_overlap_distance")].set_index("layer")
    assert ov.loc[0, "dist_treatment"] == pytest.approx(0.75) and ov.loc[0, "dist_control"] == 0.0
    assert ov.loc[0, "delta"] == pytest.approx(0.75) and ov.loc[1, "delta"] == pytest.approx(1.0)
    cos = d[(d.metric == "act_cosine_distance") & (d.act_type == "resid_post")].set_index("layer")
    assert cos.loc[0, "delta"] == pytest.approx(1.0) and cos.loc[1, "delta"] == pytest.approx(0.0)
    assert set(d["treatment"]) == {"injection"} and set(d["control"]) == {"benign"}


def test_unalignable_pairs_reported_not_used(synthetic):
    t = compute_metrics(Experiment.open(synthetic), AnalysisSettings(alignment="last_n", last_n=3))
    al = t["alignment"]
    assert not al["ok"].any() and al["reason"].str.contains("differ").all()
    assert t["overlap"].empty and t["control_delta"].empty


def test_common_prefix_sanity(synthetic):
    t = compute_metrics(Experiment.open(synthetic), AnalysisSettings(alignment="common_prefix"))
    assert (t["alignment"]["n_aligned"] == 1).all()


def test_run_analysis_writes_files_and_plots(synthetic):
    run_analysis(synthetic, AnalysisSettings())
    m = synthetic / "metrics"
    for name in ("overlap", "overlap_summary", "jsd", "jsd_pooled", "expert_freq", "activation_similarity",
                 "control_delta", "control_delta_summary", "alignment", "generation_summary"):
        assert (m / f"{name}.csv").exists(), name
    s = json.loads((m / "summary.json").read_text())
    assert s["num_samples"] == 1 and s["num_unaligned_pairs"] == 0
    written = make_all_plots(synthetic, save_pdf=True)
    names = {p.name for p in written}
    assert {"overlap_by_layer.png", "overlap_by_layer.pdf", "jsd_by_layer.png", "control_delta.png",
            "expert_freq_heatmaps.png", "activation_cosine_by_layer.png"} <= names


def test_missing_inputs_fail_loudly(tmp_path, synthetic):
    with pytest.raises(FileNotFoundError):
        Experiment.open(tmp_path / "nothing")
    from moe_interp.analysis import load_metrics
    with pytest.raises(FileNotFoundError):
        load_metrics(synthetic)  # analyze not run yet
    with pytest.raises(ValueError, match="Reference"):
        compute_metrics(Experiment.open(synthetic), AnalysisSettings(reference="nope"))
