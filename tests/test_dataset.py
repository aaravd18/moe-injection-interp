import json

import pytest

from moe_interp.dataset import DatasetValidationError, filter_samples, load_dataset, summarize_dataset


def write(tmp_path, rows):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    return p


def good(id_="a", **variants):
    variants = variants or {"clean": {"messages": [{"role": "user", "content": "hi"}]},
                            "attack": {"messages": [{"role": "user", "content": "hi!"}]}}
    return {"id": id_, "metadata": {"task_type": "qa"}, "variants": variants}


def test_loads_examples_and_preserves_order_and_content(example_jsonl):
    samples = load_dataset(example_jsonl)
    assert [s.id for s in samples] == ["t1", "t2/with slash"]
    assert samples[0].conditions == ["clean", "benign", "injection"]
    assert samples[1].variants["clean"].messages[0] == {"role": "system", "content": "Be brief."}
    assert samples[0].evaluation["attack_target"] == "HIPPO"
    s = summarize_dataset(samples)
    assert s["num_variants"] == 6 and s["conditions"]["injection"] == 2


def test_arbitrary_condition_names_and_pairs(tmp_path):
    row = good(variants=None)
    row["variants"] = {"control-A": {"messages": [{"role": "user", "content": "x"}]},
                       "inj v2 (polite)": {"messages": [{"role": "user", "content": "y"}]}}
    s = load_dataset(write(tmp_path, [row]))[0]
    assert s.conditions == ["control-A", "inj v2 (polite)"]


def test_span_text_and_offsets(tmp_path):
    row = good()
    row["variants"]["clean"] = {"messages": [{"role": "user", "content": "Doc. Question?"}],
                                "spans": {"q": {"message": 0, "text": "Question?"}, "d": {"message": 0, "start": 0, "end": 4}}}
    s = load_dataset(write(tmp_path, [row]))[0]
    sp = s.variants["clean"].spans
    assert (sp["q"].start, sp["q"].end) == (5, 14)
    assert (sp["d"].start, sp["d"].end) == (0, 4)


@pytest.mark.parametrize("row,needle", [
    ({"metadata": {}, "variants": {"c": {"messages": [{"role": "user", "content": "x"}]}}}, "'id'"),
    ({"id": "a", "variants": {}}, "'variants'"),
    ({"id": "a", "variants": {"c": {"messages": []}}}, "'messages'"),
    ({"id": "a", "variants": {"c": {"messages": [{"role": "user"}]}}}, "'content'"),
    ({"id": "a", "variants": {"c": {"messages": [{"content": "x"}]}}}, "'role'"),
    ({"id": "a", "variants": {"c": {"messages": [{"role": "user", "content": "x", "name": "n"}]}}}, "unknown key"),
    ({"id": "a", "variants": {"c": {"messages": [{"role": "user", "content": "x"}], "prompt": 1}}}, "unknown key"),
    ({"id": "a", "variants": {"c": {"messages": [{"role": "user", "content": "x"}],
                                    "spans": {"s": {"message": 0, "text": "zzz"}}}}}, "not found"),
    ({"id": "a", "variants": {"c": {"messages": [{"role": "user", "content": "aa aa"}],
                                    "spans": {"s": {"message": 0, "text": "aa"}}}}}, "occurs 2 times"),
    ({"id": "a", "variants": {"c": {"messages": [{"role": "user", "content": "x"}],
                                    "spans": {"s": {"message": 3, "text": "x"}}}}}, "'message'"),
    ({"id": "a", "metadata": [], "variants": {"c": {"messages": [{"role": "user", "content": "x"}]}}}, "'metadata'"),
])
def test_malformed_rows_are_reported(tmp_path, row, needle):
    with pytest.raises(DatasetValidationError) as e:
        load_dataset(write(tmp_path, [row]))
    assert needle in str(e.value)
    assert "line 1" in str(e.value)


def test_all_problems_reported_with_line_numbers(tmp_path):
    with pytest.raises(DatasetValidationError) as e:
        load_dataset(write(tmp_path, [good("a"), "{not json", good("a"), {"id": ""}]))
    msg = str(e.value)
    assert "line 2: invalid JSON" in msg and "duplicate id 'a'" in msg and "line 4" in msg
    assert len(e.value.problems) >= 3


def test_blank_and_comment_lines_ignored(tmp_path):
    p = write(tmp_path, ["// comment", "", json.dumps(good("a"))])
    assert len(load_dataset(p)) == 1


def test_filters(example_jsonl):
    samples = load_dataset(example_jsonl)
    assert [s.id for s in filter_samples(samples, task_types=["qa"])] == ["t1"]
    assert [s.id for s in filter_samples(samples, sample_ids=["t2/with slash"])] == ["t2/with slash"]
    f = filter_samples(samples, conditions=["clean", "injection"])
    assert all(s.conditions == ["clean", "injection"] for s in f)
    assert samples[0].conditions == ["clean", "benign", "injection"]  # originals untouched
    assert len(filter_samples(samples, limit=1)) == 1
    with pytest.raises(KeyError):
        filter_samples(samples, sample_ids=["nope"])
    with pytest.raises(KeyError):
        filter_samples(samples, conditions=["typo"])


def test_example_file_in_repo_is_valid():
    from pathlib import Path
    samples = load_dataset(Path(__file__).parent.parent / "data" / "examples.jsonl")
    assert len(samples) >= 3
