"""Render chat messages with the model's official chat template and tokenize them.

The rendered text and the token IDs are produced once here and then saved
verbatim; nothing downstream re-tokenizes decoded text.

Steps:
1. ``tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=...)`` -> exact prompt text.
2. ``tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)``. The template already inserts any
   BOS/special tokens, so the tokenizer must not add them again.
3. Cross-check against ``apply_chat_template(tokenize=True)``. A mismatch raises an error instead of being ignored.
4. Locate every message's content inside the rendered text. If a message is missing, the template dropped it (for
   example, OLMoE's template silently ignores a ``tool`` role), and we raise an error.
5. Map annotated character spans to token index ranges using the offset mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dataset import CharSpan


class PromptRenderError(ValueError):
    pass


@dataclass
class TokenSpan:
    """Token indices ``[start, end)`` covering a character range of the rendered prompt."""

    start: int
    end: int
    char_start: int
    char_end: int
    exact: bool
    """True if the first/last token boundaries coincide exactly with the character span
    (False when e.g. a BPE token also covers a preceding space)."""

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "char_start": self.char_start,
                "char_end": self.char_end, "exact": self.exact}


@dataclass
class RenderedPrompt:
    text: str
    input_ids: list[int]
    attention_mask: list[int]
    offsets: list[tuple[int, int]] | None
    message_char_spans: list[tuple[int, int]]
    """For each message, the [start, end) character range of its content in ``text``."""
    message_token_spans: list[TokenSpan | None] = field(default_factory=list)
    spans: dict[str, TokenSpan] = field(default_factory=dict)


def _as_id_list(x: Any) -> list[int]:
    """apply_chat_template(tokenize=True) returns a list in transformers 4.x and a BatchEncoding in 5.x."""
    if hasattr(x, "keys") and "input_ids" in x:
        x = x["input_ids"]
    if hasattr(x, "tolist"):
        x = x.tolist()
    if x and isinstance(x[0], list):
        if len(x) != 1:
            raise PromptRenderError(f"Expected a single sequence, got batch of {len(x)}")
        x = x[0]
    return [int(i) for i in x]


def char_span_to_token_span(offsets: list[tuple[int, int]], char_start: int, char_end: int) -> TokenSpan | None:
    """Return the tokens that overlap ``[char_start, char_end)``. Zero-width tokens are ignored. Returns None if no token overlaps."""
    idx = [i for i, (s, e) in enumerate(offsets) if e > s and e > char_start and s < char_end]
    if not idx:
        return None
    first, last = idx[0], idx[-1]
    # Tokens between first and last that don't overlap must be zero-width (e.g. Mellum's standalone "Ġ" before
    # digits gets offsets (n, n) with trim_offsets); they belong to the span. Anything else means bad offsets.
    gaps = [i for i in range(first, last + 1) if i not in set(idx)]
    if any(offsets[i][1] > offsets[i][0] for i in gaps):
        raise PromptRenderError("Non-contiguous token span; offset mapping looks malformed")
    exact = offsets[first][0] == char_start and offsets[last][1] == char_end
    return TokenSpan(first, last + 1, char_start, char_end, exact)


def render_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
    char_spans: dict[str, CharSpan] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> RenderedPrompt:
    """Render and tokenize one conversation. See the module docstring for the guarantees."""
    if getattr(tokenizer, "chat_template", None) is None:
        raise PromptRenderError("Tokenizer has no chat_template; refusing to guess a prompt format")
    tkw = dict(chat_template_kwargs or {})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, **tkw)
    if not isinstance(text, str):
        raise PromptRenderError(f"apply_chat_template(tokenize=False) returned {type(text).__name__}, expected str")

    is_fast = bool(getattr(tokenizer, "is_fast", False))
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=is_fast)
    input_ids = [int(i) for i in enc["input_ids"]]
    offsets = [tuple(map(int, o)) for o in enc["offset_mapping"]] if is_fast else None

    reference = _as_id_list(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt, **tkw)
    )
    if reference != input_ids:
        n = next((i for i, (a, b) in enumerate(zip(reference, input_ids)) if a != b), min(len(reference), len(input_ids)))
        raise PromptRenderError(
            "Tokenizing the rendered prompt with add_special_tokens=False does not reproduce "
            f"apply_chat_template(tokenize=True): lengths {len(input_ids)} vs {len(reference)}, first difference at "
            f"token {n}. The tokenizer may add special tokens differently from the template; inspect before trusting traces."
        )

    # Locate each message's content in the rendered text, in order.
    message_char_spans: list[tuple[int, int]] = []
    cursor = 0
    for i, m in enumerate(messages):
        content = m["content"]
        pos = text.find(content, cursor)
        if pos == -1:
            raise PromptRenderError(
                f"Message {i} (role={m['role']!r}) does not appear verbatim in the rendered prompt. The chat template "
                "probably dropped or rewrote it (OLMoE's template only renders 'system', 'user', and 'assistant' roles; "
                "Mellum's wraps 'tool' messages). Put tool outputs or retrieved documents inside a supported role's content."
            )
        message_char_spans.append((pos, pos + len(content)))
        cursor = pos + len(content)

    rendered = RenderedPrompt(
        text=text, input_ids=input_ids, attention_mask=[1] * len(input_ids), offsets=offsets,
        message_char_spans=message_char_spans,
    )
    if offsets is not None:
        rendered.message_token_spans = [
            char_span_to_token_span(offsets, s, e) if e > s else None for s, e in message_char_spans
        ]
    if char_spans:
        if offsets is None:
            raise PromptRenderError("Span annotations need a fast tokenizer (offset mapping is unavailable)")
        for name, cs in char_spans.items():
            base = message_char_spans[cs.message][0]
            ts = char_span_to_token_span(offsets, base + cs.start, base + cs.end)
            if ts is None:
                raise PromptRenderError(f"Span {name!r} maps to zero tokens")
            rendered.spans[name] = ts
    return rendered
