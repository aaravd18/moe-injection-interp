"""Instrumented prefill forward pass: expert-routing and hidden-state capture.

Routing semantics, verified against the Transformers OLMoE source (4.x and 5.x)
---------------------------------------------------------------------------------
For each MoE layer the router computes, per token::

    logits  = x @ W_router.T                      # [S, E], pre-softmax, model dtype ("router_logits")
    probs   = softmax(logits.float(), dim=-1)     # [S, E], fp32, sums to 1
    w, ids  = topk(probs, k)                      # [S, k] each, w sorted descending
    if norm_topk_prob: w = w / w.sum(-1)          # OLMoE-1B-7B-0125: norm_topk_prob = False
    w       = w.to(model dtype)                   # these are the weights multiplied into expert outputs

So with ``norm_topk_prob=False``, the **dispatched weights are the raw softmax
probabilities of the selected experts**. They are *not* renormalised and sum
to less than 1. This module always stores the weights exactly as dispatched,
upcast losslessly to fp32, and records which case applies.

Where the values come from:
* transformers >= 5: ``mlp.gate`` (``OlmoeTopKRouter``) returns ``(logits, weights, ids)``. A forward hook
  captures the module's own ids and weights (``source="router_module_output"``). We also recompute them
  from the logits and raise an error if they disagree.
* transformers 4.x: ``mlp.gate`` is an ``nn.Linear`` returning logits only. Ids and weights are recomputed
  from the logits with the same algorithm (``source="recomputed_from_logits"``). Exact ties in top-k could in
  principle break differently than on the GPU. Ties are measure-zero for continuous logits, and
  ``verify.py`` checks the result end to end.

In both cases the hooked logits are compared with ``output_router_logits=True``
(``outputs.router_logits``), which must match.

Activation types (per decoder layer ``l``)::

    resid_pre[l]   = input to decoder layer l (residual stream before attention)
    resid_mid[l]   = resid_pre[l] + attn(...)       (residual stream after attention, before the MoE)
    router_input[l]= post_attention_layernorm(resid_mid[l])   (exact input to the router and experts)
    moe_output[l]  = sum_j w_j * expert_{id_j}(router_input[l])   (MoE block output, before the residual add)
    resid_post[l]  = resid_mid[l] + moe_output[l]   (decoder layer output = resid_pre[l+1])
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .config import ACTIVATION_TYPES, TraceConfig
from .model import MoELayout

ACTIVATION_SEMANTICS = {
    "resid_pre": "Residual stream entering the decoder layer (before attention). Equals resid_post of the previous layer.",
    "resid_mid": "Residual stream after attention, before the MoE block (input to post_attention_layernorm).",
    "router_input": "post_attention_layernorm(resid_mid): the exact tensor fed to the router and the experts.",
    "moe_output": "Output of the sparse MoE block (weighted sum of top-k expert outputs) before the residual add.",
    "resid_post": "Residual stream after the MoE block (decoder layer output) = resid_mid + moe_output.",
}

ROUTING_SEMANTICS = {
    "topk_ids": "Expert indices selected for each token, in descending order of routing probability.",
    "topk_weights": (
        "Weights actually multiplied into each selected expert's output: softmax(router_logits)[topk_ids], "
        "renormalised over the top-k only if norm_topk_prob is true. Stored as fp32 (lossless upcast from the model dtype)."
    ),
    "router_logits": "Raw pre-softmax router logits (x @ W_router.T), fp32 upcast from the model dtype.",
    "router_probs": "softmax(router_logits) over all experts, computed in fp32 (sums to 1 per token).",
}

_TOL = {torch.float32: 1e-5, torch.float16: 2.0**-10, torch.bfloat16: 2.0**-7}


class RoutingVerificationError(RuntimeError):
    pass


@dataclass
class TraceSpec:
    activation_types: list[str]
    activation_layers: list[int]
    """Non-negative, sorted layer indices."""
    positions: str | list[int]
    save_router_logits: bool = False
    save_router_probs: bool = False
    activation_dtype: torch.dtype = torch.float32

    @classmethod
    def from_config(cls, cfg: TraceConfig, num_layers: int) -> "TraceSpec":
        a = cfg.activations
        return cls(
            activation_types=list(a.types),
            activation_layers=resolve_layers(a.layers, num_layers),
            positions=a.positions,
            save_router_logits=cfg.routing.save_router_logits,
            save_router_probs=cfg.routing.save_router_probs,
            activation_dtype={"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[a.dtype],
        )


def resolve_layers(layers: list[int] | str, num_layers: int) -> list[int]:
    if layers == "all":
        return list(range(num_layers))
    out = set()
    for layer in layers:
        idx = layer + num_layers if layer < 0 else layer
        if not 0 <= idx < num_layers:
            raise ValueError(f"Layer index {layer} out of range for a model with {num_layers} layers")
        out.add(idx)
    return sorted(out)


def resolve_positions(positions: str | list[int], seq_len: int) -> list[int]:
    """Turn a position spec into sorted, unique, non-negative token indices."""
    if positions == "last":
        return [seq_len - 1]
    if positions == "all":
        return list(range(seq_len))
    if isinstance(positions, str) and positions.startswith("last_k:"):
        k = int(positions.split(":", 1)[1])
        if k < 1:
            raise ValueError("last_k:N needs N >= 1")
        return list(range(max(0, seq_len - k), seq_len))
    if isinstance(positions, str):
        raise ValueError(f"Unknown positions spec {positions!r}: use last, all, last_k:N, or a list of ints")
    out = set()
    for p in positions:
        idx = p + seq_len if p < 0 else p
        if not 0 <= idx < seq_len:
            raise ValueError(f"Position {p} out of range for a prompt of {seq_len} tokens")
        out.add(idx)
    return sorted(out)


@dataclass
class RoutingTrace:
    topk_ids: torch.Tensor
    """[num_layers, seq_len, top_k] int64"""
    topk_weights: torch.Tensor
    """[num_layers, seq_len, top_k] float32 (dispatched weights)"""
    router_logits: torch.Tensor | None
    """[num_layers, seq_len, num_experts] float32, or None if not requested"""
    router_probs: torch.Tensor | None
    """[num_layers, seq_len, num_experts] float32, or None if not requested"""
    source: str
    weights_dtype: str
    norm_topk_prob: bool
    num_experts: int
    checks: dict[str, float] = field(default_factory=dict)


@dataclass
class ActivationTrace:
    positions: list[int]
    tensors: dict[tuple[str, int], torch.Tensor]
    """(activation_type, layer) -> [len(positions), hidden_size] tensor on CPU"""
    source_dtype: str


@dataclass
class TraceResult:
    routing: RoutingTrace
    activations: ActivationTrace
    next_token_id: int
    """argmax of the prefill logits at the last position (greedy prediction of the first response token)."""


def _first_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and x and isinstance(x[0], torch.Tensor):
        return x[0]
    raise TypeError(f"Expected a tensor or a tuple starting with one, got {type(x).__name__}")


class TraceRecorder:
    """Context manager that registers forward hooks for a single forward pass and removes them on exit.

    Usage::

        with TraceRecorder(layout, spec, seq_len) as rec:
            out = model(input_ids, output_router_logits=True, use_cache=False)
        routing = rec.finalize_routing(out.router_logits)
        acts = rec.activations()

    Batch size must be 1. Hooks move captured data to the CPU immediately, so no GPU tensors are kept.
    """

    def __init__(self, layout: MoELayout, spec: TraceSpec, seq_len: int):
        self.layout = layout
        self.spec = spec
        self.seq_len = seq_len
        self.positions = resolve_positions(spec.positions, seq_len)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._router_out: dict[int, tuple] = {}
        self._acts: dict[tuple[str, int], torch.Tensor] = {}
        self._model_dtype: torch.dtype | None = None
        self._pos_index: torch.Tensor | None = None

    # ---- hook factories -------------------------------------------------------------------------
    def _check_bsh(self, x: torch.Tensor, what: str) -> None:
        if x.dim() != 3 or x.shape[0] != 1 or x.shape[1] != self.seq_len or x.shape[2] != self.layout.hidden_size:
            raise RoutingVerificationError(
                f"{what}: expected shape [1, {self.seq_len}, {self.layout.hidden_size}], got {tuple(x.shape)}"
            )

    def _store(self, kind: str, layer: int, x: torch.Tensor) -> None:
        self._check_bsh(x, f"{kind}[layer {layer}]")
        if self._pos_index is None or self._pos_index.device != x.device:
            self._pos_index = torch.tensor(self.positions, device=x.device)
        self._acts[(kind, layer)] = (
            x[0].index_select(0, self._pos_index).detach().to("cpu", dtype=self.spec.activation_dtype).clone()
        )

    def _router_hook(self, layer: int):
        def hook(module, args, output):
            if isinstance(output, tuple):
                logits, weights, ids = output[0], output[1], output[2]
                self._router_out[layer] = (
                    logits.detach().float().cpu(), weights.detach().float().cpu(), ids.detach().long().cpu(),
                    weights.dtype,
                )
            else:
                self._router_out[layer] = (output.detach().float().cpu(), None, None, output.dtype)
            self._model_dtype = (output[0] if isinstance(output, tuple) else output).dtype

        return hook

    def _pre_hook(self, kind: str, layer: int):
        def hook(module, args, kwargs):
            x = args[0] if args else kwargs.get("hidden_states")
            self._store(kind, layer, x)

        return hook

    def _post_hook(self, kind: str, layer: int):
        def hook(module, args, output):
            self._store(kind, layer, _first_tensor(output))

        return hook

    # ---- context manager ------------------------------------------------------------------------
    def __enter__(self) -> "TraceRecorder":
        if self._handles:
            raise RuntimeError("TraceRecorder is already active")
        L = self.layout
        try:
            for i, router in enumerate(L.routers):
                self._handles.append(router.register_forward_hook(self._router_hook(i)))
            for i in self.spec.activation_layers:
                for kind in self.spec.activation_types:
                    if kind == "resid_pre":
                        h = L.layers[i].register_forward_pre_hook(self._pre_hook(kind, i), with_kwargs=True)
                    elif kind == "resid_mid":
                        h = L.post_attention_norms[i].register_forward_pre_hook(self._pre_hook(kind, i), with_kwargs=True)
                    elif kind == "router_input":
                        h = L.moe_blocks[i].register_forward_pre_hook(self._pre_hook(kind, i), with_kwargs=True)
                    elif kind == "moe_output":
                        h = L.moe_blocks[i].register_forward_hook(self._post_hook(kind, i))
                    elif kind == "resid_post":
                        h = L.layers[i].register_forward_hook(self._post_hook(kind, i))
                    else:
                        raise ValueError(f"Unknown activation type {kind!r}; valid: {ACTIVATION_TYPES}")
                    self._handles.append(h)
        except BaseException:
            self.remove()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @property
    def active(self) -> bool:
        return bool(self._handles)

    # ---- results --------------------------------------------------------------------------------
    def finalize_routing(self, model_router_logits: tuple[torch.Tensor, ...] | None = None) -> RoutingTrace:
        """Stack per-layer routing into [L, S, ·] tensors and verify them. Raises RoutingVerificationError."""
        L = self.layout
        missing = [i for i in range(L.num_layers) if i not in self._router_out]
        if missing:
            raise RoutingVerificationError(f"Router hooks did not fire for layers {missing}")
        S, E, k = self.seq_len, L.num_experts, L.top_k
        ids_all, w_all, logits_all, probs_all = [], [], [], []
        checks = {"max_abs_err_weights_vs_recomputed": 0.0, "max_abs_err_logits_vs_output_router_logits": float("nan")}
        source = None
        weights_dtype = None
        for i in range(L.num_layers):
            logits, weights, ids, wdtype = self._router_out[i]
            weights_dtype = wdtype
            if tuple(logits.shape) != (S, E):
                raise RoutingVerificationError(
                    f"Layer {i}: router logits shape {tuple(logits.shape)}, expected (seq_len={S}, num_experts={E})"
                )
            probs = torch.softmax(logits, dim=-1)
            rec_w, rec_ids = torch.topk(probs, k, dim=-1)
            if L.norm_topk_prob:
                rec_w = rec_w / rec_w.sum(-1, keepdim=True)
            rec_w = rec_w.to(wdtype).float()  # same precision as the dispatched weights
            if weights is None:
                source = "recomputed_from_logits"
                ids, weights = rec_ids, rec_w
            else:
                source = "router_module_output"
                if tuple(ids.shape) != (S, k) or tuple(weights.shape) != (S, k):
                    raise RoutingVerificationError(
                        f"Layer {i}: router returned ids {tuple(ids.shape)} / weights {tuple(weights.shape)}, expected ({S}, {k})"
                    )
                # Check the module's weights against softmax(logits)[ids] (renormalised if norm_topk_prob is set).
                g = probs.gather(-1, ids)
                if L.norm_topk_prob:
                    g = g / g.sum(-1, keepdim=True)
                err = (weights - g).abs()
                tol = _TOL.get(wdtype, 1e-3) * g.abs() + 1e-6
                if bool((err > tol).any()):
                    raise RoutingVerificationError(
                        f"Layer {i}: router module weights deviate from softmax(logits)[ids] "
                        f"(max abs err {err.max().item():.3g}). The routing semantics differ from what this code assumes."
                    )
                checks["max_abs_err_weights_vs_recomputed"] = max(checks["max_abs_err_weights_vs_recomputed"], err.max().item())
            # The selected experts must be a valid top-k set: no unselected expert has a strictly higher probability.
            chosen = probs.gather(-1, ids)
            masked = probs.scatter(-1, ids, float("-inf"))
            violations = int((masked.max(-1).values > chosen.min(-1).values + 1e-6).sum())
            if violations:
                raise RoutingVerificationError(f"Layer {i}: {violations} tokens whose selected experts are not the top-{k}")
            if bool((ids.sort(-1).values.diff(dim=-1) == 0).any()):
                raise RoutingVerificationError(f"Layer {i}: duplicate expert ids within a token's top-k")
            ids_all.append(ids)
            w_all.append(weights)
            if self.spec.save_router_logits:
                logits_all.append(logits)
            if self.spec.save_router_probs:
                probs_all.append(probs)
        if model_router_logits is not None:
            if len(model_router_logits) != L.num_layers:
                raise RoutingVerificationError(
                    f"output_router_logits returned {len(model_router_logits)} layers, expected {L.num_layers}"
                )
            diff = 0.0
            for i, ref in enumerate(model_router_logits):
                ref = ref.detach().float().cpu().reshape(-1, E)
                if ref.shape != self._router_out[i][0].shape:
                    raise RoutingVerificationError(
                        f"Layer {i}: outputs.router_logits shape {tuple(ref.shape)} != hooked {tuple(self._router_out[i][0].shape)}"
                    )
                diff = max(diff, (ref - self._router_out[i][0]).abs().max().item())
            checks["max_abs_err_logits_vs_output_router_logits"] = diff
            if diff > 1e-5:
                raise RoutingVerificationError(f"Hooked router logits differ from outputs.router_logits by {diff:.3g}")
        return RoutingTrace(
            topk_ids=torch.stack(ids_all),
            topk_weights=torch.stack(w_all),
            router_logits=torch.stack(logits_all) if logits_all else None,
            router_probs=torch.stack(probs_all) if probs_all else None,
            source=source or "unknown",
            weights_dtype=str(weights_dtype).replace("torch.", ""),
            norm_topk_prob=L.norm_topk_prob,
            num_experts=E,
            checks=checks,
        )

    def activations(self) -> ActivationTrace:
        expected = {(k, i) for i in self.spec.activation_layers for k in self.spec.activation_types}
        missing = expected - set(self._acts)
        if missing:
            raise RuntimeError(f"Activation hooks did not fire for {sorted(missing)}")
        return ActivationTrace(
            positions=list(self.positions),
            tensors=dict(self._acts),
            source_dtype=str(self._model_dtype).replace("torch.", ""),
        )


def _supports_kwarg(model: nn.Module, name: str) -> bool:
    try:
        return name in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


@torch.inference_mode()
def run_traced_forward(model: nn.Module, layout: MoELayout, input_ids: torch.Tensor, spec: TraceSpec) -> TraceResult:
    """One instrumented prefill pass over the prompt (no generation, no KV cache).

    Args:
        input_ids: [1, seq_len] LongTensor on the model's device.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"run_traced_forward expects input_ids of shape [1, seq_len], got {tuple(input_ids.shape)}")
    seq_len = int(input_ids.shape[1])
    kwargs: dict[str, Any] = {"output_router_logits": True, "use_cache": False}
    if _supports_kwarg(model, "logits_to_keep"):
        kwargs["logits_to_keep"] = 1
    with TraceRecorder(layout, spec, seq_len) as rec:
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **kwargs)
    routing = rec.finalize_routing(getattr(out, "router_logits", None))
    acts = rec.activations()
    next_token = int(out.logits[0, -1].argmax().item())
    return TraceResult(routing=routing, activations=acts, next_token_id=next_token)
