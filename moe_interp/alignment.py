"""Token alignment between two prompts of possibly different lengths.

Positions are only paired when the *token IDs are identical*. Anything else is
reported as a failed alignment with a reason and is never paired silently.
Original token indices are always kept (``positions_a`` / ``positions_b``).

Methods:
* ``common_suffix``: the longest identical token suffix. This is the main use case: an identical instruction (plus the
  chat template's assistant header) following different document contents. Routing at these positions can still
  differ, because attention sees different earlier context.
* ``common_prefix``: the longest identical token prefix. With causal attention these positions are computed from
  identical inputs, so routing overlap should be ~1.0. This makes it a useful sanity check.
* ``last_n``: exactly the final ``n`` tokens. Fails unless those ``n`` tokens are identical in both prompts.
* ``span``: a named span annotated in the dataset. Fails unless the span's token IDs are identical in both prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

METHODS = ("common_suffix", "common_prefix", "last_n", "span")


@dataclass
class Alignment:
    ok: bool
    method: str
    positions_a: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    positions_b: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    reason: str = ""

    @property
    def n(self) -> int:
        return int(len(self.positions_a))

    def summary(self) -> dict[str, Any]:
        return {"ok": self.ok, "method": self.method, "n_aligned": self.n, "reason": self.reason,
                "a_start": int(self.positions_a[0]) if self.n else None,
                "b_start": int(self.positions_b[0]) if self.n else None}


def _arr(x: Sequence[int]) -> np.ndarray:
    return np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=np.int64).reshape(-1)


def common_suffix_length(a: Sequence[int], b: Sequence[int]) -> int:
    a, b = _arr(a), _arr(b)
    n = min(len(a), len(b))
    if n == 0:
        return 0
    eq = a[len(a) - n:][::-1] == b[len(b) - n:][::-1]
    return int(n if eq.all() else np.argmin(eq))


def common_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    a, b = _arr(a), _arr(b)
    n = min(len(a), len(b))
    eq = a[:n] == b[:n]
    return int(n if eq.all() else np.argmin(eq))


def align_tokens(
    ids_a: Sequence[int],
    ids_b: Sequence[int],
    method: str = "common_suffix",
    *,
    n: int | None = None,
    span_name: str | None = None,
    spans_a: dict[str, dict[str, Any]] | None = None,
    spans_b: dict[str, dict[str, Any]] | None = None,
    min_length: int = 1,
) -> Alignment:
    """Align two token sequences. Returns ``Alignment(ok=False, reason=...)`` when alignment is not reliable.

    Spans are dicts with token ``start``/``end`` (end exclusive), as saved in trace.json.
    """
    a, b = _arr(ids_a), _arr(ids_b)
    if method == "common_suffix":
        m = common_suffix_length(a, b)
        if m < min_length:
            return Alignment(False, method, reason=f"common suffix has {m} tokens (< min_length={min_length})")
        return Alignment(True, method, np.arange(len(a) - m, len(a)), np.arange(len(b) - m, len(b)))
    if method == "common_prefix":
        m = common_prefix_length(a, b)
        if m < min_length:
            return Alignment(False, method, reason=f"common prefix has {m} tokens (< min_length={min_length})")
        return Alignment(True, method, np.arange(m), np.arange(m))
    if method == "last_n":
        if n is None or n < 1:
            raise ValueError("last_n alignment needs n >= 1")
        if len(a) < n or len(b) < n:
            return Alignment(False, method, reason=f"prompt shorter than n={n} (lengths {len(a)}, {len(b)})")
        mism = np.nonzero(a[-n:] != b[-n:])[0]
        if len(mism):
            return Alignment(
                False, method,
                reason=f"last {n} tokens differ ({len(mism)} mismatches; first at offset {int(mism[0]) - n} from the end); "
                       f"common suffix is only {common_suffix_length(a, b)} tokens",
            )
        return Alignment(True, method, np.arange(len(a) - n, len(a)), np.arange(len(b) - n, len(b)))
    if method == "span":
        if not span_name:
            raise ValueError("span alignment needs span_name")
        sa, sb = (spans_a or {}).get(span_name), (spans_b or {}).get(span_name)
        if sa is None or sb is None:
            which = [x for x, s in (("a", sa), ("b", sb)) if s is None]
            return Alignment(False, method, reason=f"span {span_name!r} not annotated in prompt(s) {which}")
        ta, tb = a[sa["start"]:sa["end"]], b[sb["start"]:sb["end"]]
        if len(ta) != len(tb) or not np.array_equal(ta, tb):
            return Alignment(
                False, method,
                reason=f"span {span_name!r} tokenizes differently ({len(ta)} vs {len(tb)} tokens"
                       f"{'' if len(ta) != len(tb) else ', same length but different ids'}); context changed the BPE merges",
            )
        if len(ta) < min_length:
            return Alignment(False, method, reason=f"span {span_name!r} has {len(ta)} tokens (< min_length)")
        return Alignment(True, method, np.arange(sa["start"], sa["end"]), np.arange(sb["start"], sb["end"]))
    raise ValueError(f"Unknown alignment method {method!r}; choose from {METHODS}")


def intersect_on_reference(al_rt: Alignment, al_rc: Alignment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-way alignment through a shared reference.

    Given reference->treatment and reference->control alignments (reference is ``a`` in both), return
    ``(ref_positions, treatment_positions, control_positions)`` restricted to reference positions present in both.
    This guarantees a control-adjusted delta compares the *same reference tokens*.
    """
    if not (al_rt.ok and al_rc.ok):
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    common = np.intersect1d(al_rt.positions_a, al_rc.positions_a)
    t = al_rt.positions_b[np.searchsorted(al_rt.positions_a, common)]
    c = al_rc.positions_b[np.searchsorted(al_rc.positions_a, common)]
    return common, t, c
