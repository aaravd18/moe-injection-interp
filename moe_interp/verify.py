"""End-to-end consistency checks of the extraction against the model itself.

These checks back the claims in ``tracing.py`` with measurements on the loaded model:

1. Structure: layer, expert, and top-k counts from the modules match the config.
2. Router hooks vs ``output_router_logits``: identical logits; module top-k ids/weights agree with softmax(logits).
3. Dispatched weights: the MoE output is recomputed by hand from the captured ``router_input``, top-k ids, top-k
   weights, and the expert parameters. It must match the captured ``moe_output``. This shows the stored weights are
   the ones the model multiplies in, independently of which experts kernel (eager / grouped_mm) ran.
4. Residual bookkeeping: ``resid_post = resid_mid + moe_output``; ``resid_pre[l+1] = resid_post[l]``;
   ``router_input = post_attention_layernorm(resid_mid)``; ``attn_input = input_layernorm(resid_pre)``;
   ``resid_mid = resid_pre + attn_output``; ``attn_output = o_proj(attn_z)``; and ``resid_pre[l]`` equals
   ``output_hidden_states[l]``.
5. Attention patterns: rows sum to 1; future keys (and, on sliding-window layers, keys ``>= sliding_window``
   positions back) are exactly 0; and the per-head outputs recomputed by hand as ``A_h @ v_proj(attn_input)``
   (with grouped-query KV sharing) match the captured ``attn_z``. This shows the stored patterns are the ones the
   model applied to the values.
6. Generation: greedy generation runs, and its first token matches the traced prefill argmax.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .model import LoadedModel
from .prompts import render_prompt
from .tracing import TraceSpec, eager_attention, run_traced_forward

DEFAULT_MESSAGES = [{"role": "user", "content": "Summarize in one sentence: The museum reopened after renovations."}]
_REL_TOL = {torch.float32: 1e-4, torch.float16: 2e-2, torch.bfloat16: 5e-2}


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


@torch.inference_mode()
def manual_moe_output(moe_block: torch.nn.Module, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Recompute the sparse-MoE output in fp32 for a few tokens.

    Args:
        x: [N, H] router_input. ids, weights: [N, k] captured top-k routing.
    Returns:
        [N, H] = sum_j weights[:, j] * expert_{ids[:, j]}(x)
    """
    experts = moe_block.experts
    x = x.float()
    out = torch.zeros_like(x)
    if hasattr(experts, "gate_up_proj"):  # transformers >= 5: stacked expert weights
        if getattr(experts, "is_transposed", False):
            raise NotImplementedError("Transposed expert weight layout is not supported by the manual check")
        gu, down, act = experts.gate_up_proj, experts.down_proj, experts.act_fn
        for n in range(x.shape[0]):
            for j in range(ids.shape[1]):
                e = int(ids[n, j])
                gate, up = F.linear(x[n], gu[e].float().to(x.device)).chunk(2, dim=-1)
                out[n] += float(weights[n, j]) * F.linear(act(gate) * up, down[e].float().to(x.device))
    else:  # transformers 4.x: ModuleList of OlmoeMLP
        for n in range(x.shape[0]):
            for j in range(ids.shape[1]):
                mlp = experts[int(ids[n, j])]
                p = next(mlp.parameters())
                out[n] += float(weights[n, j]) * mlp(x[n:n + 1].to(p.dtype)).float()[0]
    return out


@torch.inference_mode()
def _check_attention(loaded: LoadedModel, tr: Any, acts: dict) -> dict[str, float]:
    """Attention-side identities over all captured positions (see module docstring, items 4-5)."""
    layout, device, dt = loaded.layout, loaded.device, loaded.dtype
    H, KV, d = layout.num_heads, layout.num_kv_heads, layout.head_dim
    att = tr.attention
    qpos = torch.tensor(att.query_positions)
    w = {k: 0.0 for k in ("attn_norm", "attn_resid_add", "o_proj", "attn_z_manual", "pattern_row_sum", "pattern_masked_max")}
    for layer in range(layout.num_layers):
        attn = layout.attentions[layer]
        a_in = acts[("attn_input", layer)]
        normed = layout.input_norms[layer](acts[("resid_pre", layer)].to(device=device, dtype=dt))
        w["attn_norm"] = max(w["attn_norm"], _rel_err(normed.float().cpu(), a_in))
        w["attn_resid_add"] = max(w["attn_resid_add"], _rel_err(
            acts[("resid_pre", layer)] + acts[("attn_output", layer)], acts[("resid_mid", layer)]))
        o = attn.o_proj(acts[("attn_z", layer)].to(device=device, dtype=dt))
        w["o_proj"] = max(w["o_proj"], _rel_err(o.float().cpu(), acts[("attn_output", layer)]))
        A = att.patterns[layer].float()  # [H, Q, S]
        S = A.shape[-1]
        w["pattern_row_sum"] = max(w["pattern_row_sum"], float((A.sum(-1) - 1).abs().max()))
        kpos = torch.arange(S)
        masked = kpos[None, :] > qpos[:, None]
        window = layout.sliding_window if layout.layer_types and layout.layer_types[layer] == "sliding_attention" else None
        if window:
            masked |= (qpos[:, None] - kpos[None, :]) >= window
        if bool(masked.any()):
            w["pattern_masked_max"] = max(w["pattern_masked_max"], float(A[:, masked].abs().max()))
        # z_h = A_h @ v_{h // group}, v = v_proj(attn_input) over all key positions (positions must be "all").
        v = attn.v_proj(a_in.to(device=device, dtype=dt)).float().cpu().view(S, KV, d).transpose(0, 1)  # [KV, S, d]
        v = v.repeat_interleave(H // KV, dim=0)  # [H, S, d]
        z = (A @ v).transpose(0, 1).reshape(A.shape[1], H * d)  # [Q, H*d]
        w["attn_z_manual"] = max(w["attn_z_manual"], _rel_err(z, acts[("attn_z", layer)][qpos]))
    return w


@torch.inference_mode()
def verify_model(loaded: LoadedModel, messages: list[dict[str, str]] | None = None, n_tokens: int = 4,
                 max_new_tokens: int = 8) -> dict[str, Any]:
    """Run all checks; returns a report dict with a top-level ``passed`` flag. Never raises on a failed check."""
    model, tok, layout, device = loaded.model, loaded.tokenizer, loaded.layout, loaded.device
    tol = _REL_TOL.get(loaded.dtype, 5e-2)
    report: dict[str, Any] = {"architecture": layout.describe(), "dtype": str(loaded.dtype), "checks": {}}
    checks = report["checks"]

    def check(name: str, value: float | bool, ok: bool) -> None:
        checks[name] = {"value": value, "passed": bool(ok)}

    cfg = model.config
    check("num_layers_matches_config", layout.num_layers, layout.num_layers == cfg.num_hidden_layers)
    check("num_experts_matches_config", layout.num_experts, layout.num_experts == cfg.num_experts)
    check("top_k_matches_config", layout.top_k, layout.top_k == cfg.num_experts_per_tok)

    rp = render_prompt(tok, messages or DEFAULT_MESSAGES)
    S = len(rp.input_ids)
    ids = torch.tensor([rp.input_ids], device=device)
    spec = TraceSpec(activation_types=["resid_pre", "attn_input", "attn_z", "attn_output", "resid_mid",
                                       "router_input", "moe_output", "resid_post"],
                     activation_layers=list(range(layout.num_layers)), positions="all",
                     save_router_logits=True, save_router_probs=True,
                     save_attention=True, attention_layers=list(range(layout.num_layers)),
                     attention_query_positions="all", attention_dtype=torch.float32)
    try:
        tr = run_traced_forward(model, layout, ids, spec)
    except Exception as e:  # RoutingVerificationError etc.
        check("traced_forward_and_routing_verification", repr(e), False)
        report["passed"] = False
        return report
    r = tr.routing
    check("traced_forward_and_routing_verification", r.checks, True)
    check("routing_source", r.source, r.source in ("router_module_output", "recomputed_from_logits"))
    check("topk_ids_shape", list(r.topk_ids.shape), list(r.topk_ids.shape) == [layout.num_layers, S, layout.top_k])
    check("router_logits_shape", list(r.router_logits.shape),
          list(r.router_logits.shape) == [layout.num_layers, S, layout.num_experts])
    wsum = r.topk_weights.sum(-1)
    check("topk_weight_sum_range", [float(wsum.min()), float(wsum.max())],
          bool((wsum <= 1 + 1e-2).all() and (wsum > 0).all()))
    report["dispatched_weights_normalized"] = layout.norm_topk_prob
    report["mean_topk_weight_sum"] = float(wsum.mean())

    acts, pos = tr.activations.tensors, tr.activations.positions
    pos = pos[-n_tokens:]  # MoE hand-recompute is slow; check the final n_tokens positions
    rows = [tr.activations.positions.index(p) for p in pos]
    worst = {"moe_manual": 0.0, "resid_add": 0.0, "resid_chain": 0.0, "norm": 0.0, "hidden_states": 0.0,
             "attn_norm": 0.0, "attn_resid_add": 0.0, "o_proj": 0.0, "attn_z_manual": 0.0,
             "pattern_row_sum": 0.0, "pattern_masked_max": 0.0}
    report["attn_implementation_traced"] = tr.attention.attn_implementation if tr.attention else None
    worst.update(_check_attention(loaded, tr, acts))
    # Same attention implementation as the traced pass (eager), so this compares extraction, not kernels.
    with eager_attention(model):
        out_hs = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = out_hs.hidden_states
    for layer in range(layout.num_layers):
        x = acts[("router_input", layer)][rows].to(device)
        sel = r.topk_ids[layer, pos]
        w = r.topk_weights[layer, pos]
        manual = manual_moe_output(layout.moe_blocks[layer], x, sel, w)
        worst["moe_manual"] = max(worst["moe_manual"], _rel_err(manual.cpu(), acts[("moe_output", layer)][rows]))
        worst["resid_add"] = max(worst["resid_add"], _rel_err(
            acts[("resid_mid", layer)] + acts[("moe_output", layer)], acts[("resid_post", layer)]))
        normed = layout.post_attention_norms[layer](acts[("resid_mid", layer)].to(device=device, dtype=loaded.dtype))
        worst["norm"] = max(worst["norm"], _rel_err(normed.float().cpu(), acts[("router_input", layer)]))
        if layer + 1 < layout.num_layers:
            worst["resid_chain"] = max(worst["resid_chain"], _rel_err(
                acts[("resid_pre", layer + 1)], acts[("resid_post", layer)]))
        if hs is not None and len(hs) == layout.num_layers + 1:
            worst["hidden_states"] = max(worst["hidden_states"], _rel_err(
                acts[("resid_pre", layer)], hs[layer][0, tr.activations.positions].float().cpu()))
    check("manual_moe_recompute_rel_err", worst["moe_manual"], worst["moe_manual"] < tol)
    check("resid_post_eq_resid_mid_plus_moe_rel_err", worst["resid_add"], worst["resid_add"] < tol)
    check("router_input_eq_norm_resid_mid_rel_err", worst["norm"], worst["norm"] < tol)
    check("resid_pre_next_eq_resid_post_rel_err", worst["resid_chain"], worst["resid_chain"] == 0.0)
    check("resid_pre_eq_output_hidden_states_rel_err", worst["hidden_states"], worst["hidden_states"] < tol)  # separate forward pass: allow kernel nondeterminism
    check("attn_input_eq_input_layernorm_resid_pre_rel_err", worst["attn_norm"], worst["attn_norm"] < tol)
    check("resid_mid_eq_resid_pre_plus_attn_output_rel_err", worst["attn_resid_add"], worst["attn_resid_add"] < tol)
    check("attn_output_eq_o_proj_attn_z_rel_err", worst["o_proj"], worst["o_proj"] < tol)
    check("attention_pattern_row_sum_max_abs_dev", worst["pattern_row_sum"], worst["pattern_row_sum"] < 2 * tol)
    check("attention_pattern_masked_entries_max", worst["pattern_masked_max"], worst["pattern_masked_max"] == 0.0)
    check("manual_attn_z_from_patterns_rel_err", worst["attn_z_manual"], worst["attn_z_manual"] < tol)

    gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=tok.pad_token_id)
    resp = gen[0, S:].tolist()
    report["generated_text"] = tok.decode(resp, skip_special_tokens=True)
    check("generated_tokens", len(resp), len(resp) > 0)
    check("first_token_matches_prefill_argmax", resp[0] == tr.next_token_id if resp else None,
          bool(resp) and resp[0] == tr.next_token_id)
    report["passed"] = all(c["passed"] for c in checks.values())
    return report
