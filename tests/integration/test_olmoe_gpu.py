"""Integration test against the REAL OLMoE-1B-7B-0125-Instruct (~14 GB download, ~14 GB GPU memory).

Skipped unless explicitly enabled:

    MOE_INTERP_RUN_INTEGRATION=1 pytest tests/integration -v -s

Optional environment variables:
    MOE_INTERP_MODEL_ID / MOE_INTERP_LOCAL_PATH   which weights to load
    MOE_INTERP_DEVICE                              default "cuda" (set "cpu" only on a machine with >= 32 GB RAM)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("MOE_INTERP_RUN_INTEGRATION") != "1",
                       reason="set MOE_INTERP_RUN_INTEGRATION=1 to load the real OLMoE model"),
]

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def loaded():
    from moe_interp.config import load_config
    from moe_interp.model import load_model

    cfg = load_config(REPO / "configs" / "olmoe.yaml")
    cfg.model.device = os.environ.get("MOE_INTERP_DEVICE", "cuda")
    if cfg.model.device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if os.environ.get("MOE_INTERP_MODEL_ID"):
        cfg.model.model_id = os.environ["MOE_INTERP_MODEL_ID"]
        cfg.model.revision = None
    if os.environ.get("MOE_INTERP_LOCAL_PATH"):
        cfg.model.local_path = os.environ["MOE_INTERP_LOCAL_PATH"]
    return load_model(cfg.model)


def test_architecture_matches_config(loaded):
    c, lay = loaded.model.config, loaded.layout
    assert lay.num_layers == c.num_hidden_layers
    assert lay.num_experts == c.num_experts
    assert lay.top_k == c.num_experts_per_tok
    print("\narchitecture:", lay.describe(), "dtype:", loaded.dtype)
    # Published OLMoE-1B-7B-0125 configuration (informational; the code never hardcodes these).
    if "OLMoE-1B-7B" in loaded.info["model_id"] and not loaded.info.get("local_path"):
        assert (lay.num_layers, lay.num_experts, lay.top_k) == (16, 64, 8)


def test_verify_model(loaded):
    from moe_interp.verify import verify_model

    report = verify_model(loaded, n_tokens=4, max_new_tokens=16)
    for name, c in report["checks"].items():
        print(f"{'PASS' if c['passed'] else 'FAIL'} {name}: {c['value']}")
    print("generated:", repr(report["generated_text"]))
    failed = {k: v for k, v in report["checks"].items() if not v["passed"]}
    # first-token agreement can legitimately flip on a bf16 near-tie; report it but don't fail on it alone
    failed.pop("first_token_matches_prefill_argmax", None)
    assert not failed, failed


def test_end_to_end_run_and_analysis(loaded, tmp_path):
    from moe_interp.analysis import AnalysisSettings, run_analysis
    from moe_interp.config import apply_overrides, load_config
    from moe_interp.plotting import make_all_plots
    from moe_interp.runner import run_experiment
    from moe_interp.storage import load_trace, read_jsonl

    cfg = apply_overrides(load_config(REPO / "configs" / "olmoe.yaml"), [
        f"data.path={REPO / 'data' / 'examples.jsonl'}", "generation.max_new_tokens=24",
        "trace.routing.save_router_logits=true",
    ])
    out = run_experiment(cfg, tmp_path / "olmoe_smoke", loaded=loaded)
    recs = list(read_jsonl(out / "generations.jsonl"))
    assert recs
    L, E, k, H = loaded.layout.num_layers, loaded.layout.num_experts, loaded.layout.top_k, loaded.layout.hidden_size
    for r in recs:
        assert r["num_generated_tokens"] > 0 and isinstance(r["response_text"], str)
        t = load_trace(out / r["trace_dir"])
        S = len(r["input_ids"])
        assert t.topk_ids.shape == (L, S, k) and t.router_logits.shape == (L, S, E)
        assert int(t.topk_ids.min()) >= 0 and int(t.topk_ids.max()) < E
        assert t.positions == [S - 1]
        for (kind, layer), x in t.activations.items():
            assert x.shape == (1, H) and torch.isfinite(x).all(), (kind, layer)
        print(f"{r['sample_id']}/{r['condition']}: {r['response_text'][:80]!r}")
    tables = run_analysis(out, AnalysisSettings())
    assert tables["alignment"]["ok"].all()
    assert not tables["overlap_summary"].empty
    prefix = run_analysis(out, AnalysisSettings(alignment="common_prefix"))
    # identical prefix -> identical computation; allow rare bf16 kernel nondeterminism
    assert prefix["overlap"]["overlap"].mean() > 0.99
    assert make_all_plots(out)
