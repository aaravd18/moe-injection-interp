"""Offline checks for the E03 OLMoE preference-set pipeline: selection invariants, judge parsing, pair rules."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SEL = ROOT / "data" / "secpo_olmoe_v1"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def build():
    return load(ROOT / "scripts" / "build_secpo_olmoe.py", "build_secpo_olmoe")


@pytest.fixture(scope="module")
def judge():
    return load(ROOT / "experiments" / "modal" / "experiments" / "e03_judge.py", "e03_judge")


def test_selection_invariants():
    rows = [json.loads(l) for l in (SEL / "selected.jsonl").open()]
    sel = json.loads((SEL / "selection.json").read_text())
    m = [r["metadata"] for r in rows]
    n = sel["n"]
    assert len(rows) == n == 1500 and [r["id"] for r in rows] == sel["ids"]
    assert sum(x["position"] == "start" for x in m) == n // 2
    assert not any(x["completion_style"] for x in m)
    assert all(x["clean_reconstruction"] == "chosen_data" for x in m)
    assert len({x["instruction"] for x in m}) == len({x["seed_task"] for x in m}) == n
    assert sorted(r["id"] for r in rows if r["metadata"]["pilot"]) == sel["pilot_ids"] and len(sel["pilot_ids"]) == 20
    assert not set(sel["ids"]) & set(sel["heldout_completion_style_ids"])


def test_judge_to_record(judge):
    r = judge.to_record("task", {"task_verdict": {"type": "choice", "choice": "correct", "confidence": 0.9,
                                                  "probabilities": {"correct": 0.9, "incorrect": 0.1}}})
    assert r["verdict"] == "correct" and r["confidence"] == 0.9
    a = judge.to_record("attack", {"injected_task_done": {"type": "noul", "noul": 0.7, "confidence": 0.7},
                                   "trusted_task_done": {"type": "noul", "noul": 0.2, "confidence": 0.8}})
    assert a["injected_task_done"] is True and a["trusted_task_done"] is False
    assert a["p_injected_task_done"] == 0.7 and a["confidence"] == pytest.approx(0.7)
    jobs = judge.build_jobs({"x": {"metadata": {"instruction": "I", "clean_data": "D", "injection": "INJ"},
                                   "variants": {"injection": {"messages": [{}, {"content": "D INJ"}]}}}},
                            [{"id": "x", "condition": "clean", "response": "R"},
                             {"id": "x", "condition": "injection", "response": "R2"}])
    assert jobs[0]["kind"] == "task" and "<response>\nR\n" in jobs[0]["input"]
    assert jobs[1]["kind"] == "attack" and "<injected_text>\nINJ\n" in jobs[1]["input"]


def gen(resp="A fine answer.", finish="stop", rep=0.0, **judge):
    return {"response": resp, "finish_reason": finish, "repetition_3gram": rep, "judge": judge}


def test_pair_rules(build):
    good = gen(verdict="correct")
    follow = gen(injected_task_done=True, trusted_task_done=False)
    assert build.classify(good, follow)["is_pair"]
    assert build.classify(good, gen(injected_task_done=True, trusted_task_done=True))["injection_outcome"] == "attack_and_task"
    for bad in (gen(verdict="partially_correct"), gen(finish="length", verdict="correct"),
                gen(rep=0.8, verdict="correct"), gen(resp="  ", verdict="correct"), gen(parse_error=True)):
        lab = build.classify(bad, follow)
        assert not lab["chosen_valid"] and not lab["is_pair"] and lab["not_pair_reasons"]
    lab = build.classify(good, gen(injected_task_done=False, trusted_task_done=True))
    assert lab["injection_outcome"] == "resisted" and not lab["is_pair"]
    assert build.classify(good, gen(parse_error=True))["injection_outcome"] == "judge_parse_error"


def test_borderline_flag(build):
    lab = build.classify(gen(verdict="correct", probabilities={"correct": 0.6, "incorrect": 0.4}),
                         gen(injected_task_done=True, trusted_task_done=False, p_injected_task_done=0.55))
    assert lab["is_pair"] and lab["borderline"] == ["chosen", "attack"]
    lab = build.classify(gen(verdict="correct", probabilities={"correct": 0.9}),
                         gen(injected_task_done=True, trusted_task_done=False, p_injected_task_done=0.95))
    assert lab["borderline"] == []
