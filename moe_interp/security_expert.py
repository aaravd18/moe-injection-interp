"""Always-active security expert for OLMoE-style sparse MoE blocks (E05).

For every decoder layer l, the MoE block's output becomes

    h_new(x) = h_MoE(x) + alpha_l * S_l(x),      S_l(x) = W_down,l ( act(W_gate,l x) * (W_up,l x) )

where x is the MoE block's input (the post-attention RMSNorm output, the same input the router and the routed experts
see) and S_l has exactly one OLMoE expert's shape and computation (`OlmoeExperts.forward` for a single expert:
`gate, up = linear(x, gate_up_proj).chunk(2)`, `linear(act(gate) * up, down_proj)`). The decoder layer then adds the
result to the residual stream as usual (`h = residual + mlp(post_attention_layernorm(h))`), so S_l's output goes
through the same residual path as the routed experts' output.

- S_l always runs on every token. It is not routed, takes no top-k slot, and leaves the router and the 64 original
  experts alone; h_MoE is computed by the unmodified original forward.
- Init: down_proj = 0 (so S_l(x) = 0 exactly and the model equals the base bit for bit), gate_up_proj ~ N(0, std)
  (default std = config.initializer_range, OLMoE's own expert init), alpha_l = alpha_init (small, nonzero). At step 0
  only down_proj receives a gradient (d/d gate_up and d/d alpha are multiplied by down_proj = 0); from the first update
  on, all three do.
- Mechanism: the existing block object's class is swapped for a subclass whose forward calls the original forward and
  adds alpha_l * S_l(x). Module names (`mlp.gate`, `mlp.experts`) are unchanged, so routing capture, output recorders
  and MoELayout keep working; the new parameters are `...mlp.security_expert.{gate_up_proj,down_proj,alpha}`.

Saved format (`save_security_experts`): `security_expert.safetensors` (keys `layers.{i}.{gate_up_proj,down_proj,alpha}`)
plus `security_expert_config.json`.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

WEIGHTS = "security_expert.safetensors"
CONFIG = "security_expert_config.json"
PARAM_NAMES = ("gate_up_proj", "down_proj", "alpha")


class SecurityExpert(nn.Module):
    """One OLMoE-shaped FFN expert (gate_up [2I, H], down [H, I]) with a learnable output scale alpha."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, alpha_init: float = 0.1):
        super().__init__()
        from transformers.activations import ACT2FN

        self.hidden_size, self.intermediate_size, self.hidden_act = hidden_size, intermediate_size, hidden_act
        self.gate_up_proj = nn.Parameter(torch.empty(2 * intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.zeros(hidden_size, intermediate_size))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = F.linear(x, self.gate_up_proj).chunk(2, dim=-1)
        return self.alpha * F.linear(self.act_fn(gate) * up, self.down_proj)

    def extra_repr(self) -> str:
        return f"hidden={self.hidden_size}, intermediate={self.intermediate_size}, act={self.hidden_act}"


_SECURED: dict[type, type] = {}


def _secured_class(cls: type) -> type:
    """Subclass of the model's own MoE block class: original forward, plus alpha * S(x) when enabled."""
    if cls in _SECURED:
        return _SECURED[cls]

    def forward(self, hidden_states, *args, **kwargs):
        out = cls.forward(self, hidden_states, *args, **kwargs)
        if not self.security_expert_enabled:
            return out
        moe = out[0] if isinstance(out, tuple) else out  # transformers 4.x blocks return (hidden, router_logits)
        se = self.security_expert(hidden_states).to(moe.dtype)
        rec = self.security_expert_recorder
        if rec is not None:
            rec.add(self.security_expert_layer, moe, se)
        new = moe + se
        return (new, *out[1:]) if isinstance(out, tuple) else new

    _SECURED[cls] = type(f"Secured{cls.__name__}", (cls,), {"forward": forward, "__module__": __name__})
    return _SECURED[cls]


def security_experts(model: nn.Module) -> list[SecurityExpert]:
    return [m for m in model.modules() if isinstance(m, SecurityExpert)]


def _blocks(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if hasattr(m, "security_expert_enabled")]


def add_security_experts(model: nn.Module, layout, alpha_init: float = 0.1, init_std: float | None = None,
                         seed: int = 0, freeze_base: bool = True) -> list[SecurityExpert]:
    """Attach one SecurityExpert to every MoE block of `layout`, in the model's dtype and device.

    With freeze_base (default), every original parameter gets requires_grad=False and only the new experts train."""
    if _blocks(model):
        raise ValueError("model already has security experts")
    cfg = model.config
    std = cfg.initializer_range if init_std is None else init_std
    p0 = next(model.parameters())
    if freeze_base:
        for p in model.parameters():
            p.requires_grad_(False)
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for i, block in enumerate(layout.moe_blocks):
        se = SecurityExpert(cfg.hidden_size, cfg.intermediate_size, cfg.hidden_act, alpha_init)
        with torch.no_grad():  # generated on CPU in fp32 so init does not depend on device/dtype
            se.gate_up_proj.copy_(torch.randn(se.gate_up_proj.shape, generator=g) * std)
        se.to(device=p0.device, dtype=p0.dtype)
        block.security_expert = se
        block.security_expert_enabled = True
        block.security_expert_layer = i
        block.security_expert_recorder = None
        block.__class__ = _secured_class(type(block))
        out.append(se)
    return out


@contextmanager
def security_experts_disabled(model: nn.Module):
    """The original model: every block skips its security expert (the analogue of PEFT's disable_adapter)."""
    blocks = _blocks(model)
    prev = [b.security_expert_enabled for b in blocks]
    for b in blocks:
        b.security_expert_enabled = False
    try:
        yield
    finally:
        for b, p in zip(blocks, prev):
            b.security_expert_enabled = p


class ContributionRecorder:
    """Per-layer sums over non-padding tokens of ||alpha S(x)||^2, ||h_MoE(x)||^2 and their dot product.

    The token mask is taken from the `attention_mask` passed to the top-level model's forward."""

    def __init__(self, n_layers: int):
        self.n = n_layers
        self.reset()
        self.mask = None

    def reset(self):
        z = lambda: [0.0] * self.n  # noqa: E731
        self.se2, self.moe2, self.dot, self.tok = z(), z(), z(), z()

    @torch.no_grad()
    def add(self, layer: int, moe: torch.Tensor, se: torch.Tensor):
        moe, se = moe.reshape(-1, moe.shape[-1]).float(), se.reshape(-1, se.shape[-1]).float()
        if self.mask is not None and self.mask.numel() == moe.shape[0]:
            moe, se = moe[self.mask], se[self.mask]
        self.se2[layer] += se.pow(2).sum().item()
        self.moe2[layer] += moe.pow(2).sum().item()
        self.dot[layer] += (se * moe).sum().item()
        self.tok[layer] += moe.shape[0]

    def summary(self) -> dict:
        """rms over tokens of ||alpha S|| and ||h_MoE|| (per-token L2 norms), their ratio, and their cosine."""
        rows = []
        for i in range(self.n):
            t = max(self.tok[i], 1)
            se, moe = (self.se2[i] / t) ** 0.5, (self.moe2[i] / t) ** 0.5
            den = (self.se2[i] * self.moe2[i]) ** 0.5
            rows.append({"layer": i, "se_norm": se, "moe_norm": moe, "ratio": se / moe if moe else 0.0,
                         "cos": self.dot[i] / den if den else 0.0})
        r = [x["ratio"] for x in rows]
        return {"tokens": self.tok[0], "ratio_mean": sum(r) / len(r), "ratio_max": max(r), "layers": rows}


@contextmanager
def record_contributions(model: nn.Module):
    """Within this context, forwards (use under no_grad; checkpointed backward would double count) accumulate per-layer
    security-expert contribution statistics into the yielded ContributionRecorder."""
    blocks = _blocks(model)
    rec = ContributionRecorder(len(blocks))

    def pre_hook(_mod, args, kwargs):
        am = kwargs.get("attention_mask")
        rec.mask = am.reshape(-1).bool() if isinstance(am, torch.Tensor) and am.dim() == 2 else None

    h = model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    for b in blocks:
        b.security_expert_recorder = rec
    try:
        yield rec
    finally:
        h.remove()
        for b in blocks:
            b.security_expert_recorder = None


def alphas(model: nn.Module) -> list[float]:
    return [round(se.alpha.item(), 6) for se in security_experts(model)]


def param_report(model: nn.Module) -> dict:
    """Trainable/frozen counts; flags any trainable parameter that is not part of a security expert."""
    se_n = frozen = 0
    bad, by_kind = [], {k: 0 for k in PARAM_NAMES}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            frozen += p.numel()
            continue
        if ".security_expert." not in n:
            bad.append(n)
            continue
        se_n += p.numel()
        by_kind[n.rsplit(".", 1)[-1]] += p.numel()
    se_total = sum(p.numel() for se in security_experts(model) for p in se.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": se_n, "frozen": frozen, "trainable_pct": round(100 * se_n / total, 3),
            "trainable_by_group": {"security_expert": se_n, **by_kind}, "security_expert_params_total": se_total,
            "n_security_experts": len(security_experts(model)), "non_security_expert_trainable": bad,
            "router_trainable": [n for n, p in model.named_parameters() if ".mlp.gate." in n and p.requires_grad]}


def save_security_experts(model: nn.Module, out_dir: str | Path, meta: dict | None = None) -> str:
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ses = security_experts(model)
    state = {f"layers.{i}.{k}": getattr(se, k).detach().contiguous().cpu() for i, se in enumerate(ses) for k in PARAM_NAMES}
    save_file(state, str(out / WEIGHTS))
    s0 = ses[0]
    cfg = {"n_layers": len(ses), "hidden_size": s0.hidden_size, "intermediate_size": s0.intermediate_size,
           "hidden_act": s0.hidden_act, "dtype": str(s0.down_proj.dtype).replace("torch.", ""),
           "formula": "h_new(x) = h_MoE(x) + alpha_l * W_down (act(W_gate x) * (W_up x)), x = MoE block input",
           **(meta or {})}
    (out / CONFIG).write_text(json.dumps(cfg, indent=1) + "\n")
    return hashlib.sha256((out / WEIGHTS).read_bytes()).hexdigest()


def load_security_experts(model: nn.Module, layout, path: str | Path) -> dict:
    """Attach security experts to a freshly loaded base model and fill them from `path` (strict). Returns the config."""
    from safetensors.torch import load_file

    path = Path(path)
    cfg = json.loads((path / CONFIG).read_text())
    if cfg["n_layers"] != len(layout.moe_blocks) or cfg["hidden_size"] != model.config.hidden_size:
        raise ValueError(f"security experts in {path} do not match this model")
    ses = add_security_experts(model, layout, alpha_init=0.0, freeze_base=True)
    state = load_file(str(path / WEIGHTS))
    want = {f"layers.{i}.{k}" for i in range(len(ses)) for k in PARAM_NAMES}
    if set(state) != want:
        raise ValueError(f"unexpected keys in {path / WEIGHTS}: {sorted(set(state) ^ want)[:5]}")
    with torch.no_grad():
        for i, se in enumerate(ses):
            for k in PARAM_NAMES:
                getattr(se, k).copy_(state[f"layers.{i}.{k}"])
    for se in ses:
        se.requires_grad_(False)
    return cfg
