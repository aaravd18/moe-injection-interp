"""Conditionally routed security expert for OLMoE-style sparse MoE blocks (E08).

Every MoE layer gets a 65th expert that competes for the usual top-k slots:

    logits(x) = [ W_gate x  ;  w_new . x + b ]          (64 frozen rows, 1 new row, fixed calibration offset b)
    p         = softmax(logits)                          (over 65; top-k = 8 unchanged, norm_topk_prob as configured)
    h(x)      = sum over the 8 selected e of p_e * E_e(x),  E_64 = the new expert

- Init: the new expert is a copy of pretrained expert j (`copy_from[l]`, by default the expert this layer routes most
  tokens to), and w_new is a copy of router row j. Alternatively (`init_expert`, E08 ablation) the new expert's weights
  come from elsewhere, e.g. E05's always-active security expert (`security_expert_init`); the row is still row j.
  b is a fixed per-layer scalar chosen so that the new expert is selected for a target share of tokens on
  calibration data (`calibrate_bias`). Without it the copied row ties with row j and would be selected wherever j is.
- Trainable: w_new and the new expert's weights (`gate_up_proj`, `down_proj`). Everything else is frozen, including
  the 64 original router rows and experts and b. (`train_expert=False` freezes the new expert too; loading for
  inference uses it.)
- Mechanism: the block's class is swapped for a subclass. The 64 original experts run through the model's own
  experts module (grouped_mm) with the new expert's slots given weight 0. The new expert runs only on the tokens that
  selected it and is added with its routing weight. That is exactly a 65-expert MoE (tests/test_routed_expert.py).
  Softmax over 65 also rescales the original experts' weights, so the model is not identical to the base at init.
- With `routed_experts_disabled(model)` every block runs its original forward: the exact base model.

Saved format: routed_expert.safetensors (keys `layers.{i}.{row,gate_up_proj,down_proj,bias}`) + routed_expert_config.json.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

WEIGHTS = "routed_expert.safetensors"
CONFIG = "routed_expert_config.json"
PARAMS = ("row", "gate_up_proj", "down_proj")


class _Stop(Exception):
    pass


_ROUTED: dict[type, type] = {}


def _routed_class(cls: type) -> type:
    if cls in _ROUTED:
        return _ROUTED[cls]

    def forward(self, hidden_states, *args, **kwargs):
        if not self.re_enabled:
            return cls.forward(self, hidden_states, *args, **kwargs)
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        gate, n = self.gate, self.gate.num_experts
        orig = F.linear(x, gate.weight)
        new_raw = F.linear(x, self.re_row)                       # [N, 1], before the calibration offset
        logits = torch.cat([orig, new_raw + self.re_bias], dim=-1)
        if self.re_capture is not None:                          # calibration: what b would select this token?
            kth = orig.topk(gate.top_k, dim=-1).values[:, -1]
            self.re_capture((kth - new_raw[:, 0]).detach().float())
            raise _Stop
        probs = F.softmax(logits, dim=-1, dtype=torch.float)
        top_v, top_i = torch.topk(probs, gate.top_k, dim=-1)
        if gate.norm_topk_prob:
            top_v = top_v / top_v.sum(dim=-1, keepdim=True)
        top_v = top_v.to(logits.dtype)
        is_new = top_i == n
        out = self.experts(x, torch.where(is_new, torch.zeros_like(top_i), top_i),
                           torch.where(is_new, torch.zeros_like(top_v), top_v))
        w_new = (top_v * is_new).sum(dim=-1)
        rows = is_new.any(dim=-1).nonzero().squeeze(-1)
        if rows.numel():
            g, u = F.linear(x[rows], self.re_gate_up).chunk(2, dim=-1)
            y = F.linear(self.experts.act_fn(g) * u, self.re_down) * w_new[rows, None]
            out = out.index_add(0, rows, y.to(out.dtype))
        if self.re_recorder is not None:
            self.re_recorder.add(self.re_layer, is_new.any(dim=-1), w_new, probs[:, n])
        return out.reshape(shape)

    _ROUTED[cls] = type(f"Routed{cls.__name__}", (cls,), {"forward": forward, "__module__": __name__})
    return _ROUTED[cls]


def blocks(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if hasattr(m, "re_enabled")]


def _flat_mask(att):
    return att.reshape(-1).bool()


@torch.no_grad()
def routing_frequency(model: nn.Module, layout, batches) -> list[torch.Tensor]:
    """Share of (non-padding) tokens routed to each original expert, per layer, with the model as it is now.
    `batches` = [(input_ids, attention_mask), ...]."""
    counts = [torch.zeros(layout.num_experts, dtype=torch.float64) for _ in layout.moe_blocks]
    state = {"tok": 0, "mask": None}

    def hook(i):
        def f(_m, _inp, out):
            idx = out[2][state["mask"]]
            counts[i] += torch.bincount(idx.reshape(-1).cpu(), minlength=layout.num_experts).double()
            if i == 0:
                state["tok"] += idx.shape[0]
        return f

    hs = [r.register_forward_hook(hook(i)) for i, r in enumerate(layout.routers)]
    try:
        for ids, att in batches:
            state["mask"] = _flat_mask(att)
            model(input_ids=ids, attention_mask=att, use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return [c / max(state["tok"], 1) for c in counts]


def add_routed_experts(model: nn.Module, layout, copy_from: list[int], train_expert: bool = True,
                       row_noise_std: float = 0.0, seed: int = 0,
                       init_expert: list[tuple[torch.Tensor, torch.Tensor]] | None = None) -> None:
    """Attach the 65th expert and router row to every MoE block (bias 0 until `calibrate_bias`). Freezes everything
    else. `copy_from[l]` = the pretrained router row copied in layer l, and the pretrained expert copied too unless
    `init_expert[l]` = (gate_up_proj [2I, H], down_proj [H, I]) gives the new expert's initial weights instead."""
    if blocks(model):
        raise ValueError("model already has routed experts")
    if len(copy_from) != len(layout.moe_blocks):
        raise ValueError("copy_from needs one expert id per MoE layer")
    if init_expert is not None and len(init_expert) != len(layout.moe_blocks):
        raise ValueError("init_expert needs one (gate_up_proj, down_proj) per MoE layer")
    for p in model.parameters():
        p.requires_grad_(False)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for i, (block, j) in enumerate(zip(layout.moe_blocks, copy_from)):
        w, ex = block.gate.weight, block.experts
        row = w[j: j + 1].detach().clone()
        if row_noise_std:
            row += (torch.randn(row.shape, generator=g) * row_noise_std * row.float().pow(2).mean().sqrt().item()
                    ).to(row.device, row.dtype)
        block.re_row = nn.Parameter(row)
        gu, dn = (ex.gate_up_proj[j], ex.down_proj[j]) if init_expert is None else init_expert[i]
        if gu.shape != ex.gate_up_proj[j].shape or dn.shape != ex.down_proj[j].shape:
            raise ValueError(f"layer {i}: init_expert shapes {tuple(gu.shape)}, {tuple(dn.shape)} do not match an expert")
        kw = {"device": ex.gate_up_proj.device, "dtype": ex.gate_up_proj.dtype}
        block.re_gate_up = nn.Parameter(gu.detach().to(**kw).clone(), requires_grad=train_expert)
        block.re_down = nn.Parameter(dn.detach().to(**kw).clone(), requires_grad=train_expert)
        block.register_buffer("re_bias", torch.zeros((), dtype=w.dtype, device=w.device))
        block.re_enabled, block.re_layer, block.re_copy = True, i, int(j)
        block.re_recorder = block.re_capture = None
        block.__class__ = _routed_class(type(block))


def security_expert_init(path, fold_alpha: bool) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], dict]:
    """Per-layer (gate_up_proj, down_proj) of a saved E05 security expert (moe_interp.security_expert), as initial
    weights for the routed expert. E05 adds alpha_l * S_l(x); with fold_alpha the routed expert starts as alpha_l * S_l
    (down_proj scaled by alpha_l), otherwise as S_l itself and the routing weight takes alpha's place."""
    from safetensors.torch import load_file

    from moe_interp import security_expert as SE

    path = Path(path)
    state = load_file(str(path / SE.WEIGHTS))
    n = json.loads((path / SE.CONFIG).read_text())["n_layers"]
    alpha = [state[f"layers.{i}.alpha"].item() for i in range(n)]
    out = [(state[f"layers.{i}.gate_up_proj"],
            state[f"layers.{i}.down_proj"] * (state[f"layers.{i}.alpha"] if fold_alpha else 1)) for i in range(n)]
    info = {"source": str(path), "sha256": hashlib.sha256((path / SE.WEIGHTS).read_bytes()).hexdigest(),
            "fold_alpha": fold_alpha, "alpha": [round(a, 6) for a in alpha]}
    return out, info


@torch.no_grad()
def calibrate_bias(model: nn.Module, batches, target_rate: float) -> list[float]:
    """Set each layer's offset b so that the new expert is selected for `target_rate` of the calibration tokens.

    Layer by layer, because a layer's input depends on the routing of the layers below it: layer l is calibrated with
    layers < l already routed and calibrated, and each forward stops right after layer l's router. The new expert is
    selected iff  w_new.x + b > (k-th largest original logit), so b sits between the k-th and (k+1)-th smallest gap.
    A copied router row ties exactly with its source row; use row_noise_std > 0 so that ties cannot pin the rate."""
    out = []
    for b in blocks(model):
        gaps, state = [], {}
        b.re_capture = lambda g, _s=state, _g=gaps: _g.append(g[_s["mask"]].cpu())
        try:
            for ids, att in batches:
                state["mask"] = _flat_mask(att)
                try:
                    model(input_ids=ids, attention_mask=att, use_cache=False)
                except _Stop:
                    pass
        finally:
            b.re_capture = None
        gap = torch.cat(gaps).double().sort().values
        k = min(max(round(target_rate * gap.numel()), 1), gap.numel() - 1)
        val = ((gap[k - 1] + gap[k]) / 2).item()  # exactly k tokens have gap < b, barring ties
        b.re_bias.fill_(val)
        out.append(val)
    return out


class RoutingRecorder:
    """Per layer, over non-padding tokens: how often the new expert is selected, its mean weight when selected, and
    its mean softmax probability."""

    def __init__(self, n_layers: int):
        self.n = n_layers
        self.sel = [0.0] * n_layers
        self.wsum = [0.0] * n_layers
        self.psum = [0.0] * n_layers
        self.tok = [0] * n_layers
        self.mask = None

    @torch.no_grad()
    def add(self, layer: int, selected, w_new, p_new):
        if self.mask is not None and self.mask.numel() == selected.shape[0]:
            selected, w_new, p_new = selected[self.mask], w_new[self.mask], p_new[self.mask]
        self.sel[layer] += selected.sum().item()
        self.wsum[layer] += w_new.float().sum().item()
        self.psum[layer] += p_new.float().sum().item()
        self.tok[layer] += selected.shape[0]

    def summary(self) -> dict:
        rate = [s / max(t, 1) for s, t in zip(self.sel, self.tok)]
        w = [ws / max(s, 1) for ws, s in zip(self.wsum, self.sel)]
        p = [ps / max(t, 1) for ps, t in zip(self.psum, self.tok)]
        return {"tokens": self.tok[0], "rate_mean": sum(rate) / len(rate), "rate_by_layer": rate,
                "weight_when_selected_by_layer": w, "prob_by_layer": p}


@contextmanager
def record_routing(model: nn.Module):
    bs = blocks(model)
    rec = RoutingRecorder(len(bs))

    def pre(_m, _a, kw):
        am = kw.get("attention_mask")
        rec.mask = am.reshape(-1).bool() if isinstance(am, torch.Tensor) and am.dim() == 2 else None

    h = model.register_forward_pre_hook(pre, with_kwargs=True)
    for b in bs:
        b.re_recorder = rec
    try:
        yield rec
    finally:
        h.remove()
        for b in bs:
            b.re_recorder = None


@contextmanager
def routed_experts_disabled(model: nn.Module):
    bs = blocks(model)
    prev = [b.re_enabled for b in bs]
    for b in bs:
        b.re_enabled = False
    try:
        yield
    finally:
        for b, p in zip(bs, prev):
            b.re_enabled = p


def param_report(model: nn.Module) -> dict:
    trainable, bad = 0, []
    by = {k: 0 for k in PARAMS}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        kind = {"re_row": "row", "re_gate_up": "gate_up_proj", "re_down": "down_proj"}.get(n.rsplit(".", 1)[-1])
        if kind is None:
            bad.append(n)
            continue
        by[kind] += p.numel()
        trainable += p.numel()
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "trainable_pct": round(100 * trainable / total, 4), "trainable_by_group": by,
            "non_routed_expert_trainable": bad, "n_blocks": len(blocks(model)),
            "original_router_trainable": [n for n, p in model.named_parameters() if n.endswith("gate.weight") and p.requires_grad]}


def save_routed_experts(model: nn.Module, out_dir, meta: dict | None = None) -> str:
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bs = blocks(model)
    state = {}
    for i, b in enumerate(bs):
        state |= {f"layers.{i}.row": b.re_row.detach().contiguous().cpu(),
                  f"layers.{i}.gate_up_proj": b.re_gate_up.detach().contiguous().cpu(),
                  f"layers.{i}.down_proj": b.re_down.detach().contiguous().cpu(),
                  f"layers.{i}.bias": b.re_bias.detach().reshape(1).cpu()}
    save_file(state, str(out / WEIGHTS))
    cfg = {"n_layers": len(bs), "hidden_size": model.config.hidden_size,
           "intermediate_size": model.config.intermediate_size, "copy_from": [b.re_copy for b in bs],
           "bias": [b.re_bias.item() for b in bs], "dtype": str(bs[0].re_row.dtype).replace("torch.", ""),
           "formula": "65-expert top-k MoE: new router row (+ fixed offset b) and new expert; 64 original rows/experts frozen",
           **(meta or {})}
    (out / CONFIG).write_text(json.dumps(cfg, indent=1) + "\n")
    return hashlib.sha256((out / WEIGHTS).read_bytes()).hexdigest()


def load_routed_experts(model: nn.Module, layout, path) -> dict:
    from safetensors.torch import load_file

    path = Path(path)
    cfg = json.loads((path / CONFIG).read_text())
    if cfg["n_layers"] != len(layout.moe_blocks) or cfg["hidden_size"] != model.config.hidden_size:
        raise ValueError(f"routed experts in {path} do not match this model")
    add_routed_experts(model, layout, cfg["copy_from"], train_expert=False)
    state = load_file(str(path / WEIGHTS))
    want = {f"layers.{i}.{k}" for i in range(cfg["n_layers"]) for k in (*PARAMS, "bias")}
    if set(state) != want:
        raise ValueError(f"unexpected keys in {path / WEIGHTS}")
    with torch.no_grad():
        for i, b in enumerate(blocks(model)):
            b.re_row.copy_(state[f"layers.{i}.row"])
            b.re_gate_up.copy_(state[f"layers.{i}.gate_up_proj"])
            b.re_down.copy_(state[f"layers.{i}.down_proj"])
            b.re_bias.copy_(state[f"layers.{i}.bias"].reshape(()))
    for b in blocks(model):
        b.re_row.requires_grad_(False)
    return cfg
