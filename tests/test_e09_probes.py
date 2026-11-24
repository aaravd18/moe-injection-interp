"""E09 probe analysis on synthetic activations (real E07/E08 labels, planted directions): CPU, seconds."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments/modal/experiments"))
import e09_probes as P  # noqa: E402

pytestmark = pytest.mark.skipif(not all((d / "judgments_strict.jsonl").exists() for d in P.RUNS.values()),
                                reason="needs the saved E07/E08 judgments")


def _write_fake(acts: Path, H=24, L1=4, seed=0):
    """Per model: layer 0 constant (as the real final-token embedding), layers 1+ carry a planted detection direction
    and a planted base-susceptibility direction (shared across models, weaker in later models), plus noise."""
    rng = np.random.default_rng(seed)
    d_det, d_obey = rng.standard_normal(H), rng.standard_normal(H)
    base = P.load_labels(P.RUNS["base"])
    keys = sorted(base)
    perm = rng.permutation(len(keys))  # files are not in canonical order
    for i, m in enumerate(P.RUNS):
        X = rng.standard_normal((len(keys), L1, H)).astype(np.float32)
        X[:, 0] = 1.0
        for n, k in enumerate(keys):
            inj = k[1] != "clean"
            for l in range(1, L1):
                X[n, l] += 2.0 * inj * d_det + (1.5 / (1 + i)) * (inj and base[k]["obeyed"]) * d_obey
        (acts / m).mkdir(parents=True)
        np.savez(acts / m / "resid_last.npz", resid=X[perm], ids=np.array([keys[j][0] for j in perm]),
                 conditions=np.array([keys[j][1] for j in perm]))
    return d_det, d_obey


def test_split_is_by_record_and_deterministic():
    ids = [f"r{i}" for i in range(208)]
    a, b = P.make_split(ids), P.make_split(list(reversed(ids)))
    assert a == b
    counts = {p: sum(v == p for v in a.values()) for p in ("train", "val", "test")}
    assert counts == {"train": 125, "val": 41, "test": 42}


def test_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.standard_normal(300) + y, 1)  # ties on purpose
    assert abs(P.auroc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_full_pipeline_on_planted_directions(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "SPLIT", tmp_path / "split.json")
    acts, out = tmp_path / "acts", tmp_path / "out"
    d_det, d_obey = _write_fake(acts)
    monkeypatch.setattr(sys, "argv", ["e09_probes.py", "--acts", str(acts), "--out", str(out), "--boot", "30"])
    P.main()
    R = json.loads((out / "results.json").read_text())
    split = json.loads((tmp_path / "split.json").read_text())
    assert split["counts"] == {"train": 125, "val": 41, "test": 42}

    det = R["probes"]["base"]["detect"]["layers"]
    assert abs(det[0]["auroc"] - 0.5) < 1e-9          # constant layer 0 = chance
    assert all(r["auroc"] > 0.95 for r in det[1:])     # planted detection direction found
    ob = R["probes"]["base"]["obey_base"]
    assert ob["sufficient"] and ob["layers"][2]["auroc"] > 0.8
    # own-label obedience: E05 (3 positives) and routed (9) must be flagged, not reported
    assert not R["probes"]["e05"]["obey_own"]["sufficient"]
    assert not R["probes"]["routed"]["obey_own"]["sufficient"]
    # transfer: the shared detection direction transfers; cosine with base well above the random-direction null
    t = R["transfer"]["lora"]["detect"]
    assert t[2]["base_probe_auroc"] > 0.95 and t[2]["cos_with_base"] > R["cosine"]["null_95"]
    # recovered raw-space direction aligns with the planted one
    dirs = np.load(out / "probe_directions.npz")
    v = dirs["base__detect__L2"]
    assert abs(v @ d_det / np.linalg.norm(d_det)) > 0.7
    # held-out families evaluated for detection and susceptibility
    assert R["held_out"]["base"]["detect"]["hold_out_SF"]["sufficient"]
    assert R["held_out"]["base"]["obey_base"]["after_to_before"]["sufficient"]
    for f in R["figures"]:
        assert (out / "figs" / f).stat().st_size > 10_000
    assert "insufficient positives" in (out / "summary.md").read_text()
