"""End-to-end on CPU: dataset -> runner (tiny random OLMoE) -> saved traces -> analysis -> plots."""

import json

import pytest
import torch

from moe_interp.analysis import AnalysisSettings, run_analysis
from moe_interp.config import apply_overrides, load_config
from moe_interp.plotting import make_all_plots
from moe_interp.prompts import _as_id_list
from moe_interp.runner import ExperimentExistsError, run_experiment
from moe_interp.storage import load_trace, read_jsonl


@pytest.fixture
def cfg(example_jsonl):
    return apply_overrides(load_config(None), [
        f"data.path={example_jsonl}", "generation.max_new_tokens=4", "trace.activations.layers=all",
        "trace.activations.positions=all", "trace.routing.save_router_logits=true",
    ])


def test_end_to_end(tmp_path, cfg, tiny_loaded):
    out = run_experiment(cfg, tmp_path / "exp", loaded=tiny_loaded)
    recs = list(read_jsonl(out / "generations.jsonl"))
    assert [(r["sample_id"], r["condition"]) for r in recs][:3] == [("t1", "clean"), ("t1", "benign"), ("t1", "injection")]
    assert len(recs) == 6
    tok = tiny_loaded.tokenizer
    for r in recs:
        assert r["prompt_text"] == tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=False)
        assert r["input_ids"] == _as_id_list(tok.apply_chat_template(r["messages"], add_generation_prompt=True))
        assert tok(r["prompt_text"], add_special_tokens=False)["input_ids"] == r["input_ids"]
        assert len(r["response_ids"]) == r["num_generated_tokens"] <= 4
        assert r["finish_reason"] in ("eos", "length")
        assert r["first_token_matches_prefill"] is True  # fp32 CPU: cached and uncached paths agree
        t = load_trace(out / r["trace_dir"])
        assert t.input_ids.tolist() == r["input_ids"]
        S = len(r["input_ids"])
        assert t.topk_ids.shape == (3, S, 2) and t.router_logits.shape == (3, S, 8)
        assert t.positions == list(range(S))
        assert t.activations[("resid_post", 2)].shape == (S, 32)
    t1 = [r for r in recs if r["sample_id"] == "t1"]
    assert "attack_target_in_response" in t1[0]
    assert t1[0]["spans"]["q"]["end"] - t1[0]["spans"]["q"]["start"] == len("Question: what color?")
    # slash in sample id -> one safe directory level, still loadable
    slash = next(r for r in recs if r["sample_id"] == "t2/with slash")
    assert slash["trace_dir"].count("/") == 2 and (out / slash["trace_dir"] / "trace.json").exists()
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["status"] == "complete" and meta["dataset"]["sha256"]
    assert meta["environment"]["torch"] == torch.__version__
    assert (out / "config.yaml").exists()

    tables = run_analysis(out, AnalysisSettings(reference="clean", control="benign"))
    assert tables["alignment"]["ok"].all()
    assert not tables["overlap_summary"].empty and not tables["control_delta"].empty
    span_tables = run_analysis(out, AnalysisSettings(alignment="span", span="q"))
    sa = span_tables["alignment"]
    assert sa[sa.sample_id == "t1"]["ok"].all() and not sa[sa.sample_id == "t2/with slash"]["ok"].any()
    prefix = run_analysis(out, AnalysisSettings(alignment="common_prefix"))
    # Identical prefixes -> identical routing under causal attention.
    assert (prefix["overlap"]["overlap"] == 1.0).all()
    act = prefix["activation_similarity"]
    assert (act["cosine"] > 0.9999).all()
    assert make_all_plots(out)


def test_existing_output_and_resume(tmp_path, cfg, tiny_loaded):
    out = tmp_path / "exp"
    run_experiment(apply_overrides(cfg, ["data.limit=1"]), out, loaded=tiny_loaded)
    with pytest.raises(ExperimentExistsError):
        run_experiment(cfg, out, loaded=tiny_loaded)
    run_experiment(apply_overrides(cfg, ["data.limit=null"]), out, resume=True, loaded=tiny_loaded)
    recs = list(read_jsonl(out / "generations.jsonl"))
    assert len(recs) == 6 and len({(r["sample_id"], r["condition"]) for r in recs}) == 6
    with pytest.raises(ExperimentExistsError, match="differ"):
        run_experiment(apply_overrides(cfg, ["generation.max_new_tokens=9"]), out, resume=True, loaded=tiny_loaded)


def test_trace_only_and_generate_only(tmp_path, cfg, tiny_loaded):
    a = run_experiment(apply_overrides(cfg, ["generation.enabled=false"]), tmp_path / "a", loaded=tiny_loaded)
    r = next(read_jsonl(a / "generations.jsonl"))
    assert "trace_dir" in r and "response_text" not in r
    b = run_experiment(apply_overrides(cfg, ["trace.enabled=false"]), tmp_path / "b", loaded=tiny_loaded)
    r = next(read_jsonl(b / "generations.jsonl"))
    assert "trace_dir" not in r and "response_text" in r
