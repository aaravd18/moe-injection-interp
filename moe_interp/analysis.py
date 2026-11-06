"""Compute comparison metrics from a saved experiment directory, with no model needed.

For each sample, every condition is compared against ``reference``, but only on token positions aligned by
:mod:`moe_interp.alignment`. Pairs that cannot be aligned are listed in ``alignment.csv`` with a reason and
excluded from all metrics.

Output tables are written to ``<results>/metrics/``:

* ``alignment.csv``: one row per (sample, reference, condition) with ok / n_aligned / reason
* ``overlap.csv``: mean top-k overlap |A∩B|/k per (sample, pair, layer)
* ``jsd.csv``: per-sample JSD between expert-selection distributions on the aligned tokens
* ``jsd_pooled.csv``: JSD after pooling aligned-token counts across all samples (less small-sample bias)
* ``expert_freq.csv``: per (condition, layer, expert) selection frequency over ALL prompt tokens, pooled
* ``expert_freq_aligned.csv``: per (pair, layer, expert) frequencies on aligned tokens, pooled
* ``activation_similarity.csv``: cosine / relative-L2 per (sample, pair, type, layer) on aligned captured positions
* ``control_delta.csv``: delta = d(ref, treatment) - d(ref, control) on the SAME reference tokens
* ``*_summary.csv``: mean / std / sem / n across samples
* ``generation_summary.csv``: per-condition generation stats (attack-target substring hit rate, lengths)
* ``summary.json``: settings and warnings
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import metrics as M
from .alignment import Alignment, align_tokens, intersect_on_reference
from .storage import LoadedTrace, load_trace, read_jsonl, write_json

log = logging.getLogger(__name__)


@dataclass
class AnalysisSettings:
    reference: str = "clean"
    control: str | None = "benign"
    alignment: str = "common_suffix"
    last_n: int = 8
    span: str | None = None
    min_aligned_tokens: int = 1


@dataclass
class Experiment:
    """Index of a results directory. Traces are loaded on demand."""

    root: Path
    records: list[dict[str, Any]]
    num_experts: int | None = None
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def open(cls, root: str | Path) -> "Experiment":
        root = Path(root)
        gen = root / "generations.jsonl"
        if not gen.exists():
            raise FileNotFoundError(f"{gen} not found; is {root} an experiment output directory?")
        records = list(read_jsonl(gen))
        exp = cls(root=root, records=records)
        for r in records:
            if "trace_dir" in r:
                meta_path = root / r["trace_dir"] / "trace.json"
                if meta_path.exists():
                    import json
                    with meta_path.open() as f:
                        exp.num_experts = int(json.load(f)["routing"]["num_experts"])
                    break
        return exp

    def samples(self) -> dict[str, dict[str, dict[str, Any]]]:
        """sample_id -> condition -> generation record (insertion order preserved)."""
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for r in self.records:
            out.setdefault(r["sample_id"], {})[r["condition"]] = r
        return out

    def conditions(self) -> list[str]:
        seen: dict[str, None] = {}
        for r in self.records:
            seen.setdefault(r["condition"], None)
        return list(seen)

    def load(self, record: dict[str, Any], activations: bool = True, attention: bool = False) -> LoadedTrace | None:
        if "trace_dir" not in record:
            return None
        d = self.root / record["trace_dir"]
        if not (d / "trace.json").exists():
            self.warnings.append(f"Missing trace for {record['sample_id']}/{record['condition']} at {d}")
            return None
        return load_trace(d, load_activations=activations, load_attention=attention)


def _align(tr_a: LoadedTrace, tr_b: LoadedTrace, s: AnalysisSettings) -> Alignment:
    return align_tokens(
        tr_a.input_ids.numpy(), tr_b.input_ids.numpy(), s.alignment, n=s.last_n, span_name=s.span,
        spans_a=tr_a.spans, spans_b=tr_b.spans, min_length=s.min_aligned_tokens,
    )


def _capture_index(trace: LoadedTrace, positions: np.ndarray) -> np.ndarray:
    """Row index into ``trace.activations[...]`` for each token position, or -1 if that position was not captured."""
    lookup = {p: i for i, p in enumerate(trace.positions)}
    return np.array([lookup.get(int(p), -1) for p in positions], dtype=np.int64)


def _act_pairs(tr_a: LoadedTrace, tr_b: LoadedTrace) -> list[tuple[str, int]]:
    return sorted(set(tr_a.activations) & set(tr_b.activations))


def _route_ids(trace: LoadedTrace, positions: np.ndarray) -> np.ndarray:
    """[L, N, k] selected expert ids at the given token positions."""
    return trace.topk_ids.numpy()[:, positions, :]


def _summarize(df: pd.DataFrame, group: list[str], values: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    agg = df.groupby(group, dropna=False)[values].agg(["mean", "std", "count"])
    agg.columns = [f"{v}_{stat}" for v, stat in agg.columns]
    for v in values:
        agg[f"{v}_sem"] = agg[f"{v}_std"] / np.sqrt(agg[f"{v}_count"].clip(lower=1))
    return agg.reset_index()


def compute_metrics(exp: Experiment, settings: AnalysisSettings) -> dict[str, pd.DataFrame]:
    """Compute every metric table. See the module docstring for definitions."""
    s = settings
    E = exp.num_experts
    if E is None:
        raise ValueError(f"No traces found under {exp.root}; was the experiment run with tracing enabled?")
    rows: dict[str, list[dict[str, Any]]] = {k: [] for k in
                                             ("alignment", "overlap", "jsd", "act", "delta")}
    freq_all: dict[str, np.ndarray] = {}
    tokens_all: dict[str, int] = {}
    pooled: dict[str, list[np.ndarray]] = {}  # condition -> [ref_counts, cond_counts]
    pooled_tokens: dict[str, int] = {}

    conds_all = exp.conditions()
    if s.reference not in conds_all:
        raise ValueError(f"Reference condition {s.reference!r} not in results (conditions: {conds_all})")
    if s.control is not None and s.control not in conds_all:
        exp.warnings.append(f"Control condition {s.control!r} not in results; skipping control-adjusted deltas")

    for sid, by_cond in exp.samples().items():
        traces = {c: exp.load(r) for c, r in by_cond.items()}
        traces = {c: t for c, t in traces.items() if t is not None}
        for c, t in traces.items():
            counts = M.expert_counts(t.topk_ids.numpy(), E)
            freq_all[c] = freq_all.get(c, 0) + counts
            tokens_all[c] = tokens_all.get(c, 0) + int(t.topk_ids.shape[1])
        ref = traces.get(s.reference)
        if ref is None:
            exp.warnings.append(f"{sid}: no {s.reference!r} trace; sample skipped")
            continue
        aligns: dict[str, Alignment] = {}
        for c, tr in traces.items():
            if c == s.reference:
                continue
            al = _align(ref, tr, s)
            aligns[c] = al
            rows["alignment"].append({"sample_id": sid, "reference": s.reference, "condition": c,
                                      "ref_len": len(ref.input_ids), "cond_len": len(tr.input_ids), **al.summary()})
            if not al.ok:
                continue
            ra, rb = _route_ids(ref, al.positions_a), _route_ids(tr, al.positions_b)
            ov = M.topk_overlap(ra, rb)  # [L, N]
            ca, cb = M.expert_counts(ra, E), M.expert_counts(rb, E)
            jsd = M.js_divergence(ca, cb)  # [L]
            p = pooled.setdefault(c, [np.zeros_like(ca), np.zeros_like(cb)])
            p[0] += ca
            p[1] += cb
            pooled_tokens[c] = pooled_tokens.get(c, 0) + al.n
            for layer in range(ov.shape[0]):
                rows["overlap"].append({"sample_id": sid, "reference": s.reference, "condition": c, "layer": layer,
                                        "n_tokens": al.n, "overlap": float(ov[layer].mean())})
                rows["jsd"].append({"sample_id": sid, "reference": s.reference, "condition": c, "layer": layer,
                                    "n_tokens": al.n, "jsd": float(jsd[layer])})
            ia, ib = _capture_index(ref, al.positions_a), _capture_index(tr, al.positions_b)
            both = (ia >= 0) & (ib >= 0)
            if both.any():
                ia, ib = ia[both], ib[both]
                for kind, layer in _act_pairs(ref, tr):
                    xa = ref.activations[(kind, layer)].float().numpy()[ia]
                    xb = tr.activations[(kind, layer)].float().numpy()[ib]
                    rows["act"].append({
                        "sample_id": sid, "reference": s.reference, "condition": c, "act_type": kind, "layer": layer,
                        "n_positions": int(both.sum()),
                        "cosine": float(np.nanmean(M.cosine_similarity(xa, xb))),
                        "rel_l2": float(np.nanmean(M.relative_l2(xa, xb))),
                    })

        # Control-adjusted deltas on the same reference tokens.
        ctrl = s.control
        if ctrl is None or ctrl not in traces or ctrl == s.reference or not aligns.get(ctrl, Alignment(False, "")).ok:
            continue
        for treat, al_t in aligns.items():
            if treat == ctrl or not al_t.ok:
                continue
            p_ref, p_t, p_c = intersect_on_reference(al_t, aligns[ctrl])
            if len(p_ref) < s.min_aligned_tokens:
                exp.warnings.append(f"{sid}: no reference tokens aligned to both {treat!r} and {ctrl!r}")
                continue
            r_ids, t_ids, c_ids = _route_ids(ref, p_ref), _route_ids(traces[treat], p_t), _route_ids(traces[ctrl], p_c)
            d_t = 1 - M.topk_overlap(r_ids, t_ids).mean(-1)
            d_c = 1 - M.topk_overlap(r_ids, c_ids).mean(-1)
            cr = M.expert_counts(r_ids, E)
            j_t = M.js_divergence(cr, M.expert_counts(t_ids, E))
            j_c = M.js_divergence(cr, M.expert_counts(c_ids, E))
            base = {"sample_id": sid, "reference": s.reference, "treatment": treat, "control": ctrl}
            for layer in range(len(d_t)):
                for metric, a, b in (("routing_overlap_distance", d_t, d_c), ("routing_jsd", j_t, j_c)):
                    rows["delta"].append({**base, "metric": metric, "act_type": "", "layer": layer,
                                          "n_tokens": len(p_ref), "dist_treatment": float(a[layer]),
                                          "dist_control": float(b[layer]), "delta": float(a[layer] - b[layer])})
            tt, tc = traces[treat], traces[ctrl]
            ir, it, ic = _capture_index(ref, p_ref), _capture_index(tt, p_t), _capture_index(tc, p_c)
            keep = (ir >= 0) & (it >= 0) & (ic >= 0)
            if not keep.any():
                continue
            ir, it, ic = ir[keep], it[keep], ic[keep]
            for kind, layer in sorted(set(ref.activations) & set(tt.activations) & set(tc.activations)):
                xr = ref.activations[(kind, layer)].float().numpy()[ir]
                xt = tt.activations[(kind, layer)].float().numpy()[it]
                xc = tc.activations[(kind, layer)].float().numpy()[ic]
                for metric, fn in (("act_cosine_distance", lambda a, b: 1 - M.cosine_similarity(a, b)),
                                   ("act_rel_l2", M.relative_l2)):
                    a, b = float(np.nanmean(fn(xr, xt))), float(np.nanmean(fn(xr, xc)))
                    rows["delta"].append({**base, "metric": metric, "act_type": kind, "layer": layer,
                                          "n_tokens": int(keep.sum()), "dist_treatment": a, "dist_control": b,
                                          "delta": a - b})

    out: dict[str, pd.DataFrame] = {}
    out["alignment"] = pd.DataFrame(rows["alignment"])
    out["overlap"] = pd.DataFrame(rows["overlap"])
    out["jsd"] = pd.DataFrame(rows["jsd"])
    out["activation_similarity"] = pd.DataFrame(rows["act"])
    out["control_delta"] = pd.DataFrame(rows["delta"])
    out["overlap_summary"] = _summarize(out["overlap"], ["reference", "condition", "layer"], ["overlap"])
    out["jsd_summary"] = _summarize(out["jsd"], ["reference", "condition", "layer"], ["jsd"])
    out["activation_similarity_summary"] = _summarize(
        out["activation_similarity"], ["reference", "condition", "act_type", "layer"], ["cosine", "rel_l2"])
    out["control_delta_summary"] = _summarize(
        out["control_delta"], ["reference", "treatment", "control", "metric", "act_type", "layer"],
        ["dist_treatment", "dist_control", "delta"])

    jp = []
    fa = []
    for c, (cr, cc) in pooled.items():
        j = M.js_divergence(cr, cc)
        fr, fc = M.selection_distribution(cr), M.selection_distribution(cc)
        for layer in range(len(j)):
            jp.append({"reference": s.reference, "condition": c, "layer": layer, "n_tokens": pooled_tokens[c],
                       "jsd": float(j[layer])})
            for e in range(E):
                fa.append({"reference": s.reference, "condition": c, "layer": layer, "expert": e,
                           "freq_reference": fr[layer, e], "freq_condition": fc[layer, e]})
    out["jsd_pooled"] = pd.DataFrame(jp)
    out["expert_freq_aligned"] = pd.DataFrame(fa)
    fr_rows = []
    for c, counts in freq_all.items():
        dist = M.selection_distribution(counts)
        for layer in range(counts.shape[0]):
            for e in range(E):
                fr_rows.append({"condition": c, "layer": layer, "expert": e, "count": int(counts[layer, e]),
                                "n_tokens": tokens_all[c], "freq": dist[layer, e]})
    out["expert_freq"] = pd.DataFrame(fr_rows)
    out["generation_summary"] = generation_summary(exp.records)
    return out


def generation_summary(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "condition": r["condition"],
            "num_prompt_tokens": r.get("num_prompt_tokens"),
            "num_generated_tokens": r.get("num_generated_tokens"),
            "finished_with_eos": None if "finish_reason" not in r else r["finish_reason"] == "eos",
            "attack_target_in_response": r.get("attack_target_in_response"),
            "attack_target_in_answer": r.get("attack_target_in_answer"),
            "first_token_matches_prefill": r.get("first_token_matches_prefill"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = df.groupby("condition", sort=False).agg(
        n=("condition", "size"),
        mean_prompt_tokens=("num_prompt_tokens", "mean"),
        mean_generated_tokens=("num_generated_tokens", "mean"),
        eos_rate=("finished_with_eos", lambda x: x.dropna().astype(float).mean() if x.notna().any() else np.nan),
        attack_target_hit_rate=("attack_target_in_response",
                                lambda x: x.dropna().astype(float).mean() if x.notna().any() else np.nan),
        attack_target_answer_hit_rate=("attack_target_in_answer",
                                       lambda x: x.dropna().astype(float).mean() if x.notna().any() else np.nan),
        first_token_match_rate=("first_token_matches_prefill",
                                lambda x: x.dropna().astype(float).mean() if x.notna().any() else np.nan),
    )
    return agg.reset_index()


def run_analysis(results_dir: str | Path, settings: AnalysisSettings) -> dict[str, pd.DataFrame]:
    """Compute metrics and write them to ``<results_dir>/metrics``."""
    exp = Experiment.open(results_dir)
    tables = compute_metrics(exp, settings)
    mdir = Path(results_dir) / "metrics"
    mdir.mkdir(exist_ok=True)
    for name, df in tables.items():
        df.to_csv(mdir / f"{name}.csv", index=False)
    al = tables["alignment"]
    n_fail = int((~al["ok"]).sum()) if not al.empty else 0
    if n_fail:
        exp.warnings.append(f"{n_fail} of {len(al)} pairs could not be aligned with '{settings.alignment}'; "
                            "see metrics/alignment.csv")
    write_json(mdir / "summary.json", {
        "settings": settings.__dict__,
        "conditions": exp.conditions(),
        "num_samples": len(exp.samples()),
        "num_pairs": len(al),
        "num_unaligned_pairs": n_fail,
        "warnings": exp.warnings,
    })
    for w in exp.warnings:
        log.warning(w)
    return tables


def load_metrics(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load every CSV in ``<results_dir>/metrics``. Missing directory -> FileNotFoundError (never fabricated)."""
    mdir = Path(results_dir) / "metrics"
    if not mdir.exists():
        raise FileNotFoundError(f"{mdir} does not exist; run `analyze` first")
    out = {}
    for p in sorted(mdir.glob("*.csv")):
        try:
            out[p.stem] = pd.read_csv(p, keep_default_na=True)
        except pd.errors.EmptyDataError:
            out[p.stem] = pd.DataFrame()
    return out
