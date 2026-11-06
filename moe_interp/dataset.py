"""JSONL dataset loading and validation.

Each line is one base task containing several prompt *conditions* (variants)::

    {
      "id": "task_001",
      "metadata": {"task_type": "summarization", "attack_type": "instruction_override"},
      "variants": {
        "clean":     {"messages": [{"role": "user", "content": "..."}]},
        "injection": {"messages": [...],
                      "spans": {"instruction": {"message": 0, "text": "Summarize this document:"}}}
      },
      "evaluation": {"attack_target": "ORANGE HIPPO"}
    }

Condition names are arbitrary strings. ``spans`` are optional named character
ranges inside one message's content, given either as ``{"message": i, "text": "..."}``
(the text must occur exactly once in that message, or pass ``"occurrence": n``)
or as ``{"message": i, "start": s, "end": e}`` (character offsets into that
message's content, end exclusive). Spans are mapped to token indices later, at
tokenization time (see :mod:`moe_interp.prompts`).

This module does not import torch or transformers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KNOWN_TOP_LEVEL_KEYS = {"id", "metadata", "variants", "evaluation"}


class DatasetValidationError(ValueError):
    """Raised with a list of every problem found, each prefixed by its line number."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        shown = "\n  ".join(problems[:50])
        more = f"\n  ... and {len(problems) - 50} more" if len(problems) > 50 else ""
        super().__init__(f"{len(problems)} dataset problem(s):\n  {shown}{more}")


@dataclass(frozen=True)
class CharSpan:
    """A character range ``[start, end)`` inside ``messages[message].content``."""

    message: int
    start: int
    end: int


@dataclass
class Variant:
    name: str
    messages: list[dict[str, str]]
    """Exactly as given in the file: list of {"role": str, "content": str}."""
    spans: dict[str, CharSpan] = field(default_factory=dict)


@dataclass
class Sample:
    id: str
    variants: dict[str, Variant]
    """Insertion-ordered mapping condition name -> Variant (order as in the file)."""
    metadata: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    line_number: int = 0

    @property
    def task_type(self) -> str | None:
        return self.metadata.get("task_type")

    @property
    def conditions(self) -> list[str]:
        return list(self.variants)


def _parse_span(raw: Any, messages: list[dict], where: str, problems: list[str]) -> CharSpan | None:
    if not isinstance(raw, dict):
        problems.append(f"{where}: span must be an object, got {type(raw).__name__}")
        return None
    msg_idx = raw.get("message")
    if not isinstance(msg_idx, int) or not (0 <= msg_idx < len(messages)):
        problems.append(f"{where}: 'message' must be an int index in [0, {len(messages)}), got {msg_idx!r}")
        return None
    content = messages[msg_idx].get("content")
    if not isinstance(content, str):
        return None  # message problem already reported
    if "text" in raw:
        text = raw["text"]
        if not isinstance(text, str) or not text:
            problems.append(f"{where}: 'text' must be a non-empty string")
            return None
        hits = []
        i = content.find(text)
        while i != -1:
            hits.append(i)
            i = content.find(text, i + 1)
        if not hits:
            problems.append(f"{where}: text {text!r} not found in message {msg_idx}")
            return None
        occ = raw.get("occurrence")
        if occ is None:
            if len(hits) > 1:
                problems.append(
                    f"{where}: text {text!r} occurs {len(hits)} times in message {msg_idx}; add 'occurrence': n"
                )
                return None
            occ = 0
        if not isinstance(occ, int) or not (0 <= occ < len(hits)):
            problems.append(f"{where}: 'occurrence' must be in [0, {len(hits)}), got {occ!r}")
            return None
        return CharSpan(msg_idx, hits[occ], hits[occ] + len(text))
    start, end = raw.get("start"), raw.get("end")
    if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(content)):
        problems.append(
            f"{where}: need 'text' or integer 'start'/'end' with 0 <= start < end <= {len(content)}; "
            f"got start={start!r} end={end!r}"
        )
        return None
    return CharSpan(msg_idx, start, end)


def parse_sample(obj: Any, line_number: int, problems: list[str]) -> Sample | None:
    """Validate one decoded JSON object. Appends human-readable problems and returns None on failure."""
    where = f"line {line_number}"
    n_before = len(problems)
    if not isinstance(obj, dict):
        problems.append(f"{where}: expected a JSON object, got {type(obj).__name__}")
        return None
    sid = obj.get("id")
    if not isinstance(sid, str) or not sid.strip():
        problems.append(f"{where}: 'id' must be a non-empty string, got {sid!r}")
    else:
        where = f"line {line_number} (id={sid!r})"
    metadata = obj.get("metadata", {})
    if not isinstance(metadata, dict):
        problems.append(f"{where}: 'metadata' must be an object")
        metadata = {}
    evaluation = obj.get("evaluation", {})
    if not isinstance(evaluation, dict):
        problems.append(f"{where}: 'evaluation' must be an object")
        evaluation = {}
    raw_variants = obj.get("variants")
    if not isinstance(raw_variants, dict) or not raw_variants:
        problems.append(f"{where}: 'variants' must be a non-empty object mapping condition name -> variant")
        return None
    variants: dict[str, Variant] = {}
    for cond, raw in raw_variants.items():
        vwhere = f"{where} variant {cond!r}"
        if not cond.strip():
            problems.append(f"{where}: condition names must be non-empty")
            continue
        if not isinstance(raw, dict):
            problems.append(f"{vwhere}: must be an object with a 'messages' list")
            continue
        unknown = set(raw) - {"messages", "spans"}
        if unknown:
            problems.append(f"{vwhere}: unknown key(s) {sorted(unknown)} (allowed: messages, spans)")
        messages = raw.get("messages")
        if not isinstance(messages, list) or not messages:
            problems.append(f"{vwhere}: 'messages' must be a non-empty list")
            continue
        ok = True
        for i, m in enumerate(messages):
            if not isinstance(m, dict):
                problems.append(f"{vwhere} message {i}: must be an object")
                ok = False
                continue
            if not isinstance(m.get("role"), str) or not m["role"]:
                problems.append(f"{vwhere} message {i}: 'role' must be a non-empty string")
                ok = False
            if not isinstance(m.get("content"), str):
                problems.append(f"{vwhere} message {i}: 'content' must be a string")
                ok = False
            extra = set(m) - {"role", "content"}
            if extra:
                problems.append(f"{vwhere} message {i}: unknown key(s) {sorted(extra)}")
                ok = False
        if not ok:
            continue
        spans: dict[str, CharSpan] = {}
        raw_spans = raw.get("spans", {})
        if not isinstance(raw_spans, dict):
            problems.append(f"{vwhere}: 'spans' must be an object mapping span name -> span")
        else:
            for sname, sraw in raw_spans.items():
                span = _parse_span(sraw, messages, f"{vwhere} span {sname!r}", problems)
                if span is not None:
                    spans[sname] = span
        variants[cond] = Variant(
            name=cond, messages=[{"role": m["role"], "content": m["content"]} for m in messages], spans=spans
        )
    if len(problems) > n_before:
        return None
    return Sample(id=sid, variants=variants, metadata=metadata, evaluation=evaluation, line_number=line_number)


def load_dataset(path: str | Path) -> list[Sample]:
    """Load and validate a JSONL dataset. Raises DatasetValidationError listing *all* problems.

    Blank lines and lines starting with ``//`` are ignored.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    problems: list[str] = []
    samples: list[Sample] = []
    seen: dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as e:
                problems.append(f"line {line_number}: invalid JSON ({e.msg} at column {e.colno})")
                continue
            sample = parse_sample(obj, line_number, problems)
            if sample is None:
                continue
            if sample.id in seen:
                problems.append(f"line {line_number}: duplicate id {sample.id!r} (first seen on line {seen[sample.id]})")
                continue
            seen[sample.id] = line_number
            samples.append(sample)
    if problems:
        raise DatasetValidationError(problems)
    if not samples:
        raise DatasetValidationError([f"{path}: no samples found"])
    return samples


def dataset_warnings(samples: list[Sample]) -> list[str]:
    """Non-fatal issues worth surfacing (e.g. samples that cannot be compared)."""
    warnings = []
    for s in samples:
        if len(s.variants) < 2:
            warnings.append(f"{s.id}: only one condition ({s.conditions}); it cannot be compared to anything")
        if "task_type" not in s.metadata:
            warnings.append(f"{s.id}: metadata.task_type missing (filtering by task type will exclude it)")
    return warnings


def filter_samples(
    samples: list[Sample],
    sample_ids: list[str] | None = None,
    task_types: list[str] | None = None,
    conditions: list[str] | None = None,
    limit: int | None = None,
) -> list[Sample]:
    """Return filtered copies. ``conditions`` keeps only those variants; samples left with none are dropped.

    Unknown sample IDs raise a KeyError so a typo is not silently ignored.
    """
    out = samples
    if sample_ids:
        known = {s.id for s in samples}
        missing = [i for i in sample_ids if i not in known]
        if missing:
            raise KeyError(f"Sample id(s) not in dataset: {missing}")
        wanted = set(sample_ids)
        out = [s for s in out if s.id in wanted]
    if task_types:
        wanted_t = set(task_types)
        out = [s for s in out if s.task_type in wanted_t]
    if conditions:
        wanted_c = list(conditions)
        all_conds = {c for s in samples for c in s.variants}
        missing = [c for c in wanted_c if c not in all_conds]
        if missing:
            raise KeyError(f"Condition(s) not present in any sample: {missing}. Available: {sorted(all_conds)}")
        filtered = []
        for s in out:
            kept = {c: v for c, v in s.variants.items() if c in wanted_c}
            if kept:
                filtered.append(
                    Sample(id=s.id, variants=kept, metadata=s.metadata, evaluation=s.evaluation,
                           line_number=s.line_number)
                )
        out = filtered
    if limit is not None:
        out = out[:limit]
    return out


def summarize_dataset(samples: list[Sample]) -> dict[str, Any]:
    conds: dict[str, int] = {}
    tasks: dict[str, int] = {}
    for s in samples:
        for c in s.variants:
            conds[c] = conds.get(c, 0) + 1
        t = str(s.task_type)
        tasks[t] = tasks.get(t, 0) + 1
    return {
        "num_samples": len(samples),
        "num_variants": sum(len(s.variants) for s in samples),
        "conditions": conds,
        "task_types": tasks,
        "num_spans": sum(len(v.spans) for s in samples for v in s.variants.values()),
    }
