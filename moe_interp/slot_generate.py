"""Greedy generation with slot refill and a preallocated KV cache (fast, fixed-memory batched inference).

Why not `model.generate`:
  * it switches the experts to `batched_mm` while decoding, which copies top-k full expert weights per token and layer
    (in fp32 about 100 GB per step at batch 32). Calling the model directly keeps `grouped_mm`, which copies nothing;
  * a batch lasts until its slowest row ends. Here a finished row's slot is refilled at once;
  * DynamicCache reallocates every step. Here each layer's cache is one [slots, heads, width, head_dim] buffer, so
    the memory is known before the first token.

Rows are left-aligned in their slots. Prefill is right-padded under a causal mask (pads follow the prompt, so they
cannot influence it); each decode step writes at the row's own length and attends to positions up to it. The result
equals one-prompt-at-a-time greedy `generate` up to floating point (tests/test_slot_generate.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass
class Job:
    key: Any
    ids: list[int]
    out: list[int] = field(default_factory=list)


class SlotCache:
    """The part of the transformers Cache interface that OLMoE calls, over preallocated buffers."""

    def __init__(self, config, slots: int, width: int, device, dtype):
        heads = config.num_key_value_heads
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        shape = (slots, heads, width, dim)
        layers = range(config.num_hidden_layers)
        # zeros, not empty: masked positions are still multiplied by their (zero) attention weight, and 0 * NaN = NaN
        self.keys = [torch.zeros(shape, device=device, dtype=dtype) for _ in layers]
        self.values = [torch.zeros(shape, device=device, dtype=dtype) for _ in layers]
        self.prefill_at: int | None = None
        self.rows = self.positions = None
        self.span = 0

    def update(self, key, value, layer_idx, *args, **kwargs):
        ks, vs = self.keys[layer_idx], self.values[layer_idx]
        if self.prefill_at is not None:
            a, b, length = self.prefill_at, self.prefill_at + key.shape[0], key.shape[2]
            ks[a:b, :, :length] = key
            vs[a:b, :, :length] = value
            return key, value
        n = key.shape[0]
        ks[self.rows, :, self.positions] = key[:, :, 0]
        vs[self.rows, :, self.positions] = value[:, :, 0]
        return ks[:n, :, : self.span], vs[:n, :, : self.span]

    def get_seq_length(self, *args, **kwargs) -> int:
        return 0

    def move(self, src: int, dst: int, length: int) -> None:
        for buf in (*self.keys, *self.values):
            buf[dst, :, :length] = buf[src, :, :length]


def slot_bytes(config, width: int, dtype) -> int:
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return 2 * config.num_hidden_layers * config.num_key_value_heads * width * dim * torch.empty(0, dtype=dtype).element_size()


def _take(queue: list[Job], free: int, max_tokens: int, max_attn_cells: int) -> list[Job]:
    """The next prompts that fit in the free slots and in the prefill bounds (always at least one)."""
    got, longest = [], 0
    for job in queue[:free]:
        new = max(longest, len(job.ids))
        n = len(got) + 1
        if got and (n * new > max_tokens or n * new * new > max_attn_cells):
            break
        got.append(job)
        longest = new
    del queue[: len(got)]
    return got


@torch.inference_mode()
def generate_slots(model, jobs: list[Job], *, slots: int, max_new_tokens: int, eos_ids: set[int], pad_id: int,
                   on_done: Callable[[Job], None], max_prefill_tokens: int = 12288,
                   max_prefill_attn_cells: int = 6_000_000, on_step: Callable[[dict], None] | None = None) -> dict:
    """Generate for `jobs` in the order given; `on_done(job)` fires as each one finishes (`job.out` includes EOS)."""
    if not jobs:
        return {"steps": 0, "tokens": 0}
    body, head = model.model, model.lm_head
    param = next(model.parameters())
    device, dtype = param.device, head.weight.dtype
    queue = list(jobs)
    slots = min(slots, len(queue))
    cache = SlotCache(model.config, slots, max(len(j.ids) for j in queue) + max_new_tokens, device, dtype)
    active: list[Job] = []
    lens: list[int] = []  # tokens already in each slot's cache
    stats = {"steps": 0, "tokens": 0}

    def retire(finished: list[int]) -> None:
        for i in sorted(finished, reverse=True):  # keep the active rows a prefix of the slots
            on_done(active[i])
            last = len(active) - 1
            if i != last:
                cache.move(last, i, lens[last])
                active[i], lens[i] = active[last], lens[last]
            active.pop()
            lens.pop()

    def accept(start: int, tokens: list[int]) -> None:
        done = []
        for i, t in enumerate(tokens, start):
            active[i].out.append(t)
            if t in eos_ids or len(active[i].out) >= max_new_tokens:
                done.append(i)
        stats["tokens"] += len(tokens)
        retire(done)

    while queue or active:
        while queue and len(active) < slots:
            new = _take(queue, slots - len(active), max_prefill_tokens, max_prefill_attn_cells)
            m, longest = len(new), max(len(j.ids) for j in new)
            ids = torch.tensor([j.ids + [pad_id] * (longest - len(j.ids)) for j in new], device=device)
            mask = torch.ones(longest, longest, dtype=torch.bool, device=device).tril()[None, None].expand(m, 1, -1, -1)
            pos = torch.arange(longest, device=device)[None].expand(m, -1)
            cache.prefill_at = start = len(active)
            hidden = body(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache,
                          use_cache=True).last_hidden_state
            last = torch.tensor([len(j.ids) - 1 for j in new], device=device)
            first = head(hidden[torch.arange(m, device=device), last]).argmax(-1).tolist()
            del hidden
            cache.prefill_at = None
            active.extend(new)
            lens.extend(len(j.ids) for j in new)
            accept(start, first)
        if not active:
            continue

        n, span = len(active), max(lens) + 1
        ids = torch.tensor([[j.out[-1]] for j in active], device=device)
        pos = torch.tensor(lens, device=device)
        mask = (torch.arange(span, device=device)[None] <= pos[:, None])[:, None, None]
        cache.rows, cache.positions, cache.span = torch.arange(n, device=device), pos, span
        hidden = body(input_ids=ids, attention_mask=mask, position_ids=pos[:, None], past_key_values=cache,
                      use_cache=True).last_hidden_state
        tokens = head(hidden[:, -1]).argmax(-1).tolist()
        for i in range(n):
            lens[i] += 1
        stats["steps"] += 1
        accept(0, tokens)
        if on_step:
            on_step(stats | {"active": len(active), "queued": len(queue)})
    return stats
