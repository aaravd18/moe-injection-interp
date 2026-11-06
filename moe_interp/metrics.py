"""Pure metric functions on arrays (no I/O, no plotting, no model).

Shape conventions: ``L`` = layers, ``N`` = aligned tokens, ``k`` = top-k, ``E`` = experts, ``H`` = hidden size.
"""

from __future__ import annotations

import numpy as np


def topk_overlap(ids_a: np.ndarray, ids_b: np.ndarray) -> np.ndarray:
    """Fraction of shared experts between two top-k selections: ``|A ∩ B| / k``.

    Args:
        ids_a, ids_b: integer arrays ``[..., k]`` (e.g. ``[L, N, k]``) of selected expert ids for aligned tokens.
            Within a row the ids must be distinct (always true for top-k routing).
    Returns:
        float array ``[...]`` in [0, 1].
    """
    ids_a, ids_b = np.asarray(ids_a), np.asarray(ids_b)
    if ids_a.shape != ids_b.shape:
        raise ValueError(f"Shape mismatch {ids_a.shape} vs {ids_b.shape}; align tokens first")
    k = ids_a.shape[-1]
    shared = (ids_a[..., :, None] == ids_b[..., None, :]).any(-1).sum(-1)
    return shared / k


def expert_counts(ids: np.ndarray, num_experts: int) -> np.ndarray:
    """Count how often each expert is selected.

    Args:
        ids: ``[L, N, k]`` selected expert ids.
    Returns:
        ``[L, E]`` counts; each row sums to ``N * k``.
    """
    ids = np.asarray(ids)
    if ids.ndim != 3:
        raise ValueError(f"Expected [L, N, k], got shape {ids.shape}")
    if ids.size and (ids.min() < 0 or ids.max() >= num_experts):
        raise ValueError(f"Expert ids outside [0, {num_experts})")
    L = ids.shape[0]
    out = np.zeros((L, num_experts), dtype=np.int64)
    for layer in range(L):
        out[layer] = np.bincount(ids[layer].reshape(-1), minlength=num_experts)
    return out


def selection_distribution(counts: np.ndarray) -> np.ndarray:
    """Normalise counts ``[..., E]`` to the fraction of top-k *slots* per expert (rows sum to 1).

    Multiply by ``k`` to get the fraction of *tokens* routed to each expert (rows then sum to k).
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum(-1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("Cannot normalise a distribution with zero total count")
    return counts / total


def js_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> np.ndarray:
    """Jensen-Shannon divergence along the last axis. In [0, 1] for ``base=2``.

    Inputs are non-negative and are normalised internally. Zero probabilities are handled exactly, using the
    convention ``0 * log(0/x) = 0``. No epsilon smoothing is needed, because ``m = (p+q)/2 > 0`` wherever
    ``p > 0`` or ``q > 0``.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch {p.shape} vs {q.shape}")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Distributions must be non-negative")
    p = selection_distribution(p)
    q = selection_distribution(q)
    m = 0.5 * (p + q)

    def _kl(x: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(x > 0, x * np.log(x / np.where(m > 0, m, 1.0)), 0.0)
        return terms.sum(-1)

    jsd = 0.5 * _kl(p) + 0.5 * _kl(q)
    jsd = jsd / np.log(base)
    return np.clip(jsd, 0.0, None)  # clip tiny negative rounding error


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Cosine similarity along the last axis. Returns NaN where either vector has (near-)zero norm."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch {a.shape} vs {b.shape}")
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    denom = na * nb
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > eps, (a * b).sum(-1) / np.where(denom > eps, denom, 1.0), np.nan)


def relative_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """``||a - b|| / ||a||`` along the last axis, with ``a`` the reference. NaN where ``||a||`` ≈ 0."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch {a.shape} vs {b.shape}")
    na = np.linalg.norm(a, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(na > eps, np.linalg.norm(a - b, axis=-1) / np.where(na > eps, na, 1.0), np.nan)


def control_adjusted_delta(dist_treatment: np.ndarray, dist_control: np.ndarray) -> np.ndarray:
    """``delta = distance(ref, treatment) - distance(ref, control)``.

    Positive values mean the treatment (e.g. injection) moves the representation further from the reference than the
    control (e.g. a benign edit) does. Both distances must be computed on the same reference tokens
    (see ``alignment.intersect_on_reference``). Use *distances*, e.g. ``1 - overlap`` or ``1 - cosine``.
    """
    return np.asarray(dist_treatment, dtype=np.float64) - np.asarray(dist_control, dtype=np.float64)
