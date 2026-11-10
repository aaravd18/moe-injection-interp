"""Causal interventions for MoE models (Step 4): expert (de)activation, head ablation, residual patching.

All interventions are forward hooks registered for the duration of a ``with Intervention(...)`` block and removed
on exit (also on exceptions). They work for a single forward pass and for batched, LEFT-padded generation.

Expert steering (router-level)
------------------------------
The router hook recomputes routing from the router's own logits with an additive offset on chosen experts, using the
exact algorithm the model uses (softmax in fp32 -> top-k -> optional renormalisation -> cast), and returns the new
``(logits, weights, ids)``, which the MoE block then dispatches. Offset ``-inf`` = deactivate (the expert can never be
selected), ``+inf``-like large value = force-activate (as in SteerMoE, Fayyaz et al., ICLR 2026). The returned
logits are the *original* ones, so ``output_router_logits`` still reports what the router computed.

Which tokens are steered:
* ``token_mask`` [B, S] bool for the prompt (prefill call). Positions are in the *padded* batch, i.e. include the
  left padding. During decoding (one new token per row), ``decode`` decides: steer generated tokens too or not.
* ``token_mask=None`` steers every prompt token.

Head ablation
-------------
Zeroes chosen heads' slices of ``attn_z`` (the input to ``o_proj``), i.e. removes those heads' contribution to the
residual stream, at every position (prompt and generated).

Residual patching
-----------------
Replaces ``resid_post`` of a layer at the final ``k`` prompt positions with given values (prefill call only). With
left padding the final positions of every row line up, so a [B, k, H] source applies directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .model import MoELayout

NEG = float("-inf")


@dataclass
class ExpertSteer:
    """``offsets[layer] = {expert: logit offset}``; use ``-inf`` to deactivate, e.g. +1e4 to force-activate."""

    offsets: dict[int, dict[int, float]]
    token_mask: torch.Tensor | None = None
    """[B, S] bool over the (padded) prompt; None = all prompt tokens."""
    decode: bool = False
    """Also steer generated tokens."""


@dataclass
class HeadAblation:
    heads: dict[int, list[int]]
    """layer -> heads to zero"""


@dataclass
class ResidPatch:
    values: dict[int, torch.Tensor]
    """layer -> [B, k, H] replacement for resid_post at the final k prompt positions"""


@dataclass
class Intervention:
    layout: MoELayout
    experts: ExpertSteer | None = None
    heads: HeadAblation | None = None
    patch: ResidPatch | None = None
    _handles: list = field(default_factory=list)
    _calls: dict = field(default_factory=dict)

    # ---- router ------------------------------------------------------------------------------------
    def _router_hook(self, layer: int, offs: dict[int, float]):
        L = self.layout
        idx = torch.tensor(list(offs.keys()))
        val = torch.tensor(list(offs.values()), dtype=torch.float32)

        def hook(module, args, output):
            logits = output[0] if isinstance(output, tuple) else output
            n = logits.shape[0]
            first = self._calls.get(("router", layer), 0) == 0
            self._calls[("router", layer)] = self._calls.get(("router", layer), 0) + 1
            st = self.experts
            if first:  # prefill: n = B * S
                rows = None if st.token_mask is None else st.token_mask.reshape(-1).to(logits.device)
                if rows is not None and rows.numel() != n:
                    raise RuntimeError(f"token_mask has {rows.numel()} entries but the prefill routed {n} tokens")
            else:
                if not st.decode:
                    return None
                rows = None
            z = logits.float().clone()
            o = torch.zeros(z.shape[1], device=z.device)
            o[idx.to(z.device)] = val.to(z.device)
            if rows is None:
                z = z + o
            else:
                z[rows] = z[rows] + o
            probs = torch.softmax(z, dim=-1)
            w, ids = torch.topk(probs, L.top_k, dim=-1)
            if L.norm_topk_prob:
                w = w / w.sum(-1, keepdim=True)
            w = w.to(logits.dtype)
            if isinstance(output, tuple):
                return (logits, w, ids)
            raise RuntimeError("Expert steering needs a router module that returns (logits, weights, ids) (transformers >= 5)")

        return hook

    # ---- heads -------------------------------------------------------------------------------------
    def _head_hook(self, layer: int, heads: list[int]):
        d = self.layout.head_dim

        def hook(module, args, kwargs):
            x = args[0] if args else kwargs["input"]
            x = x.clone()
            for h in heads:
                x[..., h * d:(h + 1) * d] = 0
            return ((x,) + tuple(args[1:]), kwargs) if args else (args, {**kwargs, "input": x})

        return hook

    # ---- residual patch ----------------------------------------------------------------------------
    def _patch_hook(self, layer: int, src: torch.Tensor):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] == 1:  # decoding step
                return None
            k = src.shape[1]
            h = h.clone()
            h[:, -k:, :] = src.to(device=h.device, dtype=h.dtype)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

        return hook

    def __enter__(self) -> "Intervention":
        L = self.layout
        try:
            if self.experts:
                for layer, offs in self.experts.offsets.items():
                    if offs:
                        self._handles.append(L.routers[layer].register_forward_hook(self._router_hook(layer, offs)))
            if self.heads:
                for layer, hs in self.heads.heads.items():
                    if hs:
                        self._handles.append(L.attentions[layer].o_proj.register_forward_pre_hook(
                            self._head_hook(layer, hs), with_kwargs=True))
            if self.patch:
                for layer, src in self.patch.values.items():
                    self._handles.append(L.layers[layer].register_forward_hook(self._patch_hook(layer, src)))
        except BaseException:
            self.remove()
            raise
        return self

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._calls.clear()

    def __exit__(self, *exc: Any) -> None:
        self.remove()


def padded_mask(spans: list[tuple[int, int] | None], lengths: list[int], total: int) -> torch.Tensor:
    """[B, total] bool mask that is True on each row's [start, end) span, shifted for LEFT padding."""
    m = torch.zeros(len(lengths), total, dtype=torch.bool)
    for i, (sp, n) in enumerate(zip(spans, lengths)):
        if sp is not None:
            off = total - n
            m[i, off + sp[0]:off + sp[1]] = True
    return m
