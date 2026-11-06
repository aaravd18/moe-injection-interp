"""Plots built from the metric tables in ``<results>/metrics``. No metric is computed here.

Each plot function takes DataFrames and returns a matplotlib Figure (or None if the input is empty), so the same
functions work in the notebook. ``make_all_plots`` writes PNGs (and optionally PDFs) to ``<results>/plots``.
Missing or empty tables are reported and skipped. Nothing is ever fabricated.

Colour: each condition keeps a fixed categorical colour (assigned by first appearance), so a condition has the same
colour in every figure. Heatmaps use a single-hue sequential ramp; difference heatmaps use a blue-grey-red
diverging ramp centred on zero.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

log = logging.getLogger(__name__)

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#a32b2b"])

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
    "lines.linewidth": 2.0, "lines.markersize": 5, "font.family": "sans-serif",
}


def condition_colors(conditions: list[str]) -> dict[str, str]:
    """Fixed colour per condition, in order of first appearance. Beyond 8 conditions, colours repeat with a warning."""
    if len(conditions) > len(CATEGORICAL):
        log.warning("More than %d conditions; colours will repeat. Consider plotting subsets.", len(CATEGORICAL))
    return {c: CATEGORICAL[i % len(CATEGORICAL)] for i, c in enumerate(conditions)}


def _layer_lines(ax, df: pd.DataFrame, value: str, colors: dict[str, str], group: str = "condition",
                 band: bool = True) -> None:
    for cond, sub in df.groupby(group, sort=False):
        sub = sub.sort_values("layer")
        x, y = sub["layer"].to_numpy(), sub[f"{value}_mean"].to_numpy()
        c = colors.get(cond, INK_2)
        ax.plot(x, y, color=c, marker="o", label=str(cond))
        sem = sub.get(f"{value}_sem")
        if band and sem is not None and sem.notna().any():
            ax.fill_between(x, y - sem.fillna(0).to_numpy(), y + sem.fillna(0).to_numpy(), color=c, alpha=0.15, lw=0)
    ax.set_xlabel("layer")
    if df["layer"].nunique() <= 20:
        ax.set_xticks(sorted(df["layer"].unique()))


def _caption(fig: plt.Figure, text: str) -> None:
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=7.5, color=MUTED)


def _n_note(df: pd.DataFrame, count_col: str) -> str:
    if count_col not in df or df.empty:
        return ""
    n = df[count_col]
    return f"n = {int(n.min())} samples" if n.min() == n.max() else f"n = {int(n.min())}–{int(n.max())} samples"


def plot_overlap(summary: pd.DataFrame, colors: dict[str, str], title_suffix: str = "") -> plt.Figure | None:
    if summary is None or summary.empty:
        return None
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        _layer_lines(ax, summary, "overlap", colors)
        ref = summary["reference"].iloc[0]
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(f"top-k overlap with '{ref}'  |A∩B|/k")
        ax.set_title(f"Expert top-k overlap on aligned tokens{title_suffix}")
        ax.legend(title="condition", loc="lower left")
        _caption(fig, f"mean ± s.e.m. across samples; {_n_note(summary, 'overlap_count')}")
    return fig


def plot_jsd(summary: pd.DataFrame, pooled: pd.DataFrame | None, colors: dict[str, str],
             title_suffix: str = "") -> plt.Figure | None:
    if summary is None or summary.empty:
        return None
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        _layer_lines(ax, summary, "jsd", colors)
        if pooled is not None and not pooled.empty:
            for cond, sub in pooled.groupby("condition", sort=False):
                sub = sub.sort_values("layer")
                ax.plot(sub["layer"], sub["jsd"], color=colors.get(cond, INK_2), ls="--", lw=1.2, marker=None,
                        label=f"{cond} (pooled)")
        ref = summary["reference"].iloc[0]
        ax.set_ylim(bottom=0)
        ax.set_ylabel(f"JSD vs '{ref}' (bits)")
        ax.set_title(f"Routing-distribution divergence per layer{title_suffix}")
        ax.legend(loc="upper left", ncol=2)
        _caption(fig, "solid: per-sample JSD, mean ± s.e.m.; dashed: JSD of counts pooled over samples. "
                      "Per-sample JSD is biased upward for few tokens.")
    return fig


def plot_activation_similarity(summary: pd.DataFrame, colors: dict[str, str], value: str = "cosine",
                               title_suffix: str = "") -> plt.Figure | None:
    if summary is None or summary.empty:
        return None
    types = list(dict.fromkeys(summary["act_type"]))
    with plt.rc_context(STYLE):
        ncol = min(len(types), 3)
        nrow = int(np.ceil(len(types) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow), squeeze=False, sharey=True)
        for ax, t in zip(axes.flat, types):
            _layer_lines(ax, summary[summary["act_type"] == t], value, colors)
            ax.set_title(t)
        for ax in list(axes.flat)[len(types):]:
            ax.set_visible(False)
        ref = summary["reference"].iloc[0]
        label = "cosine similarity" if value == "cosine" else "relative L2  ||a−b||/||a||"
        for ax in axes[:, 0]:
            ax.set_ylabel(f"{label} vs '{ref}'")
        axes.flat[0].legend(title="condition")
        fig.suptitle(f"Activation {label} on aligned captured positions{title_suffix}", fontsize=11)
        fig.tight_layout()
    return fig


def plot_control_delta(summary: pd.DataFrame, colors: dict[str, str], title_suffix: str = "") -> plt.Figure | None:
    """One panel per metric: d(ref, treatment) and d(ref, control) by layer, plus delta = their difference."""
    if summary is None or summary.empty:
        return None
    df = summary.copy()
    df["act_type"] = df["act_type"].fillna("")
    df["panel"] = np.where(df["act_type"] == "", df["metric"], df["metric"] + " · " + df["act_type"])
    panels = list(dict.fromkeys(df["panel"]))
    with plt.rc_context(STYLE):
        ncol = min(len(panels), 3)
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow), squeeze=False)
        for ax, panel in zip(axes.flat, panels):
            sub = df[df["panel"] == panel]
            for treat, s2 in sub.groupby("treatment", sort=False):
                s2 = s2.sort_values("layer")
                x = s2["layer"].to_numpy()
                y = s2["delta_mean"].to_numpy()
                e = s2["delta_sem"].fillna(0).to_numpy()
                c = colors.get(treat, INK_2)
                ax.plot(x, y, color=c, marker="o", label=f"Δ {treat} − {s2['control'].iloc[0]}")
                ax.fill_between(x, y - e, y + e, color=c, alpha=0.15, lw=0)
            ax.axhline(0, color=AXIS, lw=1, zorder=0)
            ax.set_title(panel, fontsize=10)
            ax.set_xlabel("layer")
            if sub["layer"].nunique() <= 20:
                ax.set_xticks(sorted(sub["layer"].unique()))
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("Δ distance (treatment − control)")
        axes.flat[0].legend()
        fig.suptitle(f"Control-adjusted shift vs '{df['reference'].iloc[0]}' (>0: treatment moves further than control)"
                     f"{title_suffix}", fontsize=11)
        fig.tight_layout()
    return fig


def plot_control_comparison(summary: pd.DataFrame, colors: dict[str, str], metric: str = "routing_overlap_distance",
                            act_type: str = "", title_suffix: str = "") -> plt.Figure | None:
    """Raw distances d(ref, treatment) vs d(ref, control) per layer for one metric."""
    if summary is None or summary.empty:
        return None
    df = summary[(summary["metric"] == metric) & (summary["act_type"].fillna("") == act_type)]
    if df.empty:
        return None
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ctrl = df["control"].iloc[0]
        drawn_ctrl = False
        for treat, s2 in df.groupby("treatment", sort=False):
            s2 = s2.sort_values("layer")
            x = s2["layer"].to_numpy()
            for col, name, c, ls in (("dist_treatment", treat, colors.get(treat, INK_2), "-"),
                                     ("dist_control", ctrl, colors.get(ctrl, MUTED), "--")):
                if col == "dist_control" and drawn_ctrl and len(df["treatment"].unique()) > 1:
                    continue
                y, e = s2[f"{col}_mean"].to_numpy(), s2[f"{col}_sem"].fillna(0).to_numpy()
                ax.plot(x, y, color=c, ls=ls, marker="o", label=f"{df['reference'].iloc[0]} → {name}")
                ax.fill_between(x, y - e, y + e, color=c, alpha=0.12, lw=0)
            drawn_ctrl = True
        ax.set_xlabel("layer")
        ax.set_ylabel(metric + (f" · {act_type}" if act_type else ""))
        ax.set_ylim(bottom=0)
        ax.set_title(f"Treatment vs control distance from reference{title_suffix}")
        ax.legend()
        fig.tight_layout()
    return fig


def plot_expert_freq_heatmaps(freq: pd.DataFrame, conditions: list[str], title_suffix: str = "") -> plt.Figure | None:
    """Per-condition [layer × expert] selection frequency over all prompt tokens (shared colour scale)."""
    if freq is None or freq.empty:
        return None
    conds = [c for c in conditions if c in set(freq["condition"])]
    mats = {c: freq[freq["condition"] == c].pivot(index="layer", columns="expert", values="freq").sort_index()
            for c in conds}
    vmax = max(float(m.to_numpy().max()) for m in mats.values())
    n_layers = len(next(iter(mats.values())))
    n_experts = next(iter(mats.values())).shape[1]
    xt = list(range(0, n_experts, 8 if n_experts > 16 else 1))
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(len(conds), 1, figsize=(10, 0.9 + (0.6 + 0.18 * n_layers) * len(conds)),
                                 squeeze=False, sharex=True, layout="constrained")
        for ax, c in zip(axes[:, 0], conds):
            m = mats[c]
            im = ax.imshow(m.to_numpy(), aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=vmax, interpolation="nearest")
            ntok = int(freq.loc[freq["condition"] == c, "n_tokens"].iloc[0])
            ax.set_title(f"{c}  ({ntok} prompt tokens pooled)", fontsize=10, loc="left")
            ax.set_ylabel("layer")
            yt = list(range(0, n_layers, 1 if n_layers <= 16 else 2))
            ax.set_yticks(yt)
            ax.set_yticklabels([m.index[i] for i in yt], fontsize=7)
            ax.set_xticks(xt)
        axes[-1, 0].set_xlabel("expert id")
        cb = fig.colorbar(im, ax=axes[:, 0].tolist(), shrink=0.9, pad=0.01)
        cb.set_label("share of top-k slots")
        fig.suptitle(f"Expert-selection frequency (all prompt tokens){title_suffix}", fontsize=11)
    return fig


def plot_expert_freq_diff(aligned: pd.DataFrame, condition: str, title_suffix: str = "") -> plt.Figure | None:
    """freq(condition) − freq(reference) on aligned tokens, [layer × expert], diverging around 0."""
    if aligned is None or aligned.empty:
        return None
    sub = aligned[aligned["condition"] == condition]
    if sub.empty:
        return None
    sub = sub.assign(diff=sub["freq_condition"] - sub["freq_reference"])
    m = sub.pivot(index="layer", columns="expert", values="diff").sort_index()
    lim = float(np.abs(m.to_numpy()).max()) or 1e-9
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(10, 1.2 + 0.28 * len(m.index)))
        im = ax.imshow(m.to_numpy(), aspect="auto", cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim),
                       interpolation="nearest")
        ax.set_yticks(range(len(m.index)))
        ax.set_yticklabels(m.index, fontsize=7)
        ax.set_xticks(list(range(0, m.shape[1], 8 if m.shape[1] > 16 else 1)))
        ax.set_ylabel("layer")
        ax.set_xlabel("expert id")
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label("Δ share of top-k slots")
        ref = sub["reference"].iloc[0]
        ax.set_title(f"Selection frequency: '{condition}' − '{ref}' on aligned tokens{title_suffix}", fontsize=11)
        fig.tight_layout()
    return fig


def _save(fig: plt.Figure | None, path: Path, pdf: bool, written: list[Path]) -> None:
    if fig is None:
        log.info("Skipping %s (no data)", path.name)
        return
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    written.append(path.with_suffix(".png"))
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        written.append(path.with_suffix(".pdf"))
    plt.close(fig)


def make_all_plots(results_dir: str | Path, save_pdf: bool = False) -> list[Path]:
    """Render every plot whose metric table exists and is non-empty. Returns written paths."""
    from .analysis import load_metrics

    results_dir = Path(results_dir)
    t = load_metrics(results_dir)
    out = results_dir / "plots"
    out.mkdir(exist_ok=True)
    conds: list[str] = []
    for name in ("expert_freq", "alignment", "overlap_summary"):
        df = t.get(name)
        if df is not None and not df.empty:
            for col in ("reference", "condition"):
                if col in df:
                    conds += list(dict.fromkeys(df[col]))
    conds = list(dict.fromkeys(conds))
    colors = condition_colors(conds)
    suffix = ""
    meta = results_dir / "metadata.json"
    if meta.exists():
        import json
        with meta.open() as f:
            m = json.load(f)
        if m.get("model", {}).get("tiny_random"):
            suffix = "\n[TINY RANDOM DEBUG MODEL: not real results]"
    written: list[Path] = []
    _save(plot_overlap(t.get("overlap_summary"), colors, suffix), out / "overlap_by_layer", save_pdf, written)
    _save(plot_jsd(t.get("jsd_summary"), t.get("jsd_pooled"), colors, suffix), out / "jsd_by_layer", save_pdf, written)
    act = t.get("activation_similarity_summary")
    _save(plot_activation_similarity(act, colors, "cosine", suffix), out / "activation_cosine_by_layer", save_pdf, written)
    _save(plot_activation_similarity(act, colors, "rel_l2", suffix), out / "activation_rel_l2_by_layer", save_pdf, written)
    cd = t.get("control_delta_summary")
    _save(plot_control_delta(cd, colors, suffix), out / "control_delta", save_pdf, written)
    _save(plot_control_comparison(cd, colors, "routing_overlap_distance", "", suffix),
          out / "control_comparison_routing", save_pdf, written)
    _save(plot_expert_freq_heatmaps(t.get("expert_freq"), conds, suffix), out / "expert_freq_heatmaps", save_pdf, written)
    al = t.get("expert_freq_aligned")
    if al is not None and not al.empty:
        from .storage import safe_name
        for c in dict.fromkeys(al["condition"]):
            _save(plot_expert_freq_diff(al, c, suffix), out / f"expert_freq_diff_{safe_name(c)}", save_pdf, written)
    return written
