"""Model and tokenizer loading, device/dtype resolution, and MoE structure discovery.

Architecture properties (layers, experts, top-k) are read from the loaded model
and cross-checked against its module structure; nothing is hardcoded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .config import ModelConfig

log = logging.getLogger(__name__)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class ModelStructureError(RuntimeError):
    pass


def transformers_major_version() -> int:
    import transformers

    return int(transformers.__version__.split(".")[0])


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requested but torch.cuda.is_available() is False. Run `doctor`.")
    return dev


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    """``auto``: bf16 on CUDA GPUs that support it natively (Ampere+), else fp16 on CUDA, bf16 on MPS, fp32 on CPU."""
    if dtype != "auto":
        return _DTYPES[dtype]
    if device.type == "cuda":
        try:
            ok = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:  # older torch
            ok = torch.cuda.is_bf16_supported()
        if ok:
            return torch.bfloat16
        log.warning("GPU has no native bf16 support; falling back to float16 (the supported models were trained in bf16).")
        return torch.float16
    if device.type == "mps":
        return torch.bfloat16
    log.warning("Running on CPU in float32: a full model needs 4 bytes per parameter (set model.dtype=bfloat16 to halve it).")
    return torch.float32


@dataclass
class MoELayout:
    """Discovered MoE structure. ``layers[i]`` is decoder layer i; ``moe_blocks[i]`` its sparse MoE block."""

    num_layers: int
    num_experts: int
    top_k: int
    norm_topk_prob: bool
    hidden_size: int
    layers: list[nn.Module]
    moe_blocks: list[nn.Module]
    routers: list[nn.Module]
    post_attention_norms: list[nn.Module]
    attentions: list[nn.Module]
    """``layers[i].self_attn``"""
    input_norms: list[nn.Module]
    """``layers[i].input_layernorm`` (its output is the attention input)"""
    num_heads: int
    num_kv_heads: int
    head_dim: int
    layer_types: list[str] | None = None
    """Per-layer attention type, e.g. ``sliding_attention`` / ``full_attention`` (Mellum); None if uniform."""
    sliding_window: int | None = None
    model_type: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "norm_topk_prob": self.norm_topk_prob,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "layer_types": self.layer_types,
            "sliding_window": self.sliding_window,
            "router_module_class": type(self.routers[0]).__name__,
            "moe_block_class": type(self.moe_blocks[0]).__name__,
            "attention_class": type(self.attentions[0]).__name__,
            "decoder_layer_class": type(self.layers[0]).__name__,
        }


def get_moe_layout(model: nn.Module) -> MoELayout:
    """Find the decoder layers, MoE blocks, and routers, and check them against the config.

    Expected (Transformers OLMoE 4.x/5.x and Mellum 5.x share this layout)::

        model.model.layers[i]                          OlmoeDecoderLayer / MellumDecoderLayer
        model.model.layers[i].input_layernorm          RMSNorm (its output is the attention input)
        model.model.layers[i].self_attn                attention; returns (output after o_proj, patterns or None)
        model.model.layers[i].self_attn.o_proj         nn.Linear [num_heads*head_dim -> hidden]
        model.model.layers[i].post_attention_layernorm RMSNorm (its output is the MoE input)
        model.model.layers[i].mlp                      OlmoeSparseMoeBlock / MellumSparseMoeBlock
        model.model.layers[i].mlp.gate                 router; weight [num_experts, hidden]
                                                         4.x: nn.Linear  -> returns logits
                                                         5.x: *TopKRouter -> returns (logits, top-k weights, top-k ids)
    """
    cfg = model.config
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise ModelStructureError(
            f"{type(model).__name__} has no .model.layers; this code supports OLMoE/Mellum-style causal LMs only."
        )
    num_layers = int(cfg.num_hidden_layers)
    num_experts = int(getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", 0)))
    top_k = int(cfg.num_experts_per_tok)
    norm_topk = bool(getattr(cfg, "norm_topk_prob", False))
    hidden = int(cfg.hidden_size)
    num_heads = int(cfg.num_attention_heads)
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", None) or num_heads)
    head_dim = int(getattr(cfg, "head_dim", None) or hidden // num_heads)
    layer_types = list(cfg.layer_types) if getattr(cfg, "layer_types", None) else None
    sliding_window = getattr(cfg, "sliding_window", None) if layer_types and "sliding_attention" in layer_types else None
    if len(layers) != num_layers:
        raise ModelStructureError(f"config.num_hidden_layers={num_layers} but model has {len(layers)} layers")
    blocks, routers, norms, attns, in_norms = [], [], [], [], []
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate", None)
        norm = getattr(layer, "post_attention_layernorm", None)
        attn = getattr(layer, "self_attn", None)
        in_norm = getattr(layer, "input_layernorm", None)
        if mlp is None or gate is None or norm is None or attn is None or in_norm is None:
            raise ModelStructureError(
                f"Layer {i} lacks mlp / mlp.gate / post_attention_layernorm / self_attn / input_layernorm; not an "
                f"OLMoE-style MoE layer (got {type(layer).__name__}, mlp={type(mlp).__name__}). Dense MLP layers "
                "are not supported because routing tensors are stacked over all layers."
            )
        o_proj = getattr(attn, "o_proj", None)
        if o_proj is None or tuple(o_proj.weight.shape) != (hidden, num_heads * head_dim):
            raise ModelStructureError(
                f"Layer {i} self_attn.o_proj weight has shape "
                f"{None if o_proj is None else tuple(o_proj.weight.shape)}, expected ({hidden}, {num_heads * head_dim})"
            )
        w = getattr(gate, "weight", None)
        if w is None or tuple(w.shape) != (num_experts, hidden):
            raise ModelStructureError(
                f"Layer {i} router weight has shape {None if w is None else tuple(w.shape)}, expected "
                f"({num_experts}, {hidden}) from the config"
            )
        for attr, expected in (("top_k", top_k), ("num_experts", num_experts), ("norm_topk_prob", norm_topk)):
            for mod in (gate, mlp):
                if hasattr(mod, attr) and getattr(mod, attr) != expected:
                    raise ModelStructureError(
                        f"Layer {i} {type(mod).__name__}.{attr}={getattr(mod, attr)} disagrees with config ({expected})"
                    )
        blocks.append(mlp)
        routers.append(gate)
        norms.append(norm)
        attns.append(attn)
        in_norms.append(in_norm)
    return MoELayout(num_layers, num_experts, top_k, norm_topk, hidden, list(layers), blocks, routers, norms,
                     attns, in_norms, num_heads, num_kv_heads, head_dim, layer_types, sliding_window,
                     getattr(cfg, "model_type", None))


def load_tokenizer(cfg: ModelConfig) -> Any:
    from transformers import AutoTokenizer

    src = cfg.local_path or cfg.model_id
    kwargs = {} if cfg.local_path else {"revision": cfg.revision}
    return AutoTokenizer.from_pretrained(src, **kwargs)


def build_tiny_random_olmoe(
    vocab_size: int,
    *,
    num_layers: int = 4,
    num_experts: int = 8,
    top_k: int = 2,
    hidden_size: int = 64,
    norm_topk_prob: bool = False,
    pad_token_id: int | None = None,
    eos_token_id: int | None = None,
    bos_token_id: int | None = None,
    seed: int = 0,
) -> nn.Module:
    """A tiny randomly initialised OlmoeForCausalLM for tests and CPU dry runs. Its outputs are meaningless."""
    from transformers import OlmoeConfig, OlmoeForCausalLM

    config = OlmoeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size // 2,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        norm_topk_prob=norm_topk_prob,
        max_position_embeddings=2048,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
    )
    torch.manual_seed(seed)
    model = OlmoeForCausalLM(config)
    return model.eval()


def build_tiny_random_mellum(
    vocab_size: int,
    *,
    num_layers: int = 4,
    num_experts: int = 8,
    top_k: int = 2,
    hidden_size: int = 64,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    norm_topk_prob: bool = True,
    sliding_window: int = 8,
    pad_token_id: int | None = None,
    eos_token_id: int | None = None,
    bos_token_id: int | None = None,
    seed: int = 0,
) -> nn.Module:
    """A tiny randomly initialised MellumForCausalLM (grouped-query attention, alternating sliding-window / full
    attention layers, renormalised top-k). For tests and CPU dry runs; outputs are meaningless."""
    from transformers import MellumConfig, MellumForCausalLM

    layer_types = ["full_attention" if (i + 1) % 2 == 0 else "sliding_attention" for i in range(num_layers)]
    config = MellumConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        moe_intermediate_size=hidden_size // 2,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=hidden_size // num_heads,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        norm_topk_prob=norm_topk_prob,
        layer_types=layer_types,
        mlp_layer_types=["sparse"] * num_layers,
        sliding_window=sliding_window,
        use_sliding_window=True,
        max_position_embeddings=2048,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
    )
    torch.manual_seed(seed)
    return MellumForCausalLM(config).eval()


TINY_BUILDERS = {"olmoe": build_tiny_random_olmoe, "mellum": build_tiny_random_mellum}


def _model_type(cfg: ModelConfig) -> str:
    """model_type from the Hub/local config.json (no weights downloaded)."""
    from transformers import AutoConfig

    src = cfg.local_path or cfg.model_id
    kwargs = {} if cfg.local_path else {"revision": cfg.revision}
    return AutoConfig.from_pretrained(src, **kwargs).model_type


@dataclass
class LoadedModel:
    model: nn.Module
    tokenizer: Any
    layout: MoELayout
    device: torch.device
    dtype: torch.dtype
    info: dict[str, Any]


def load_model(cfg: ModelConfig, seed: int = 0) -> LoadedModel:
    """Load tokenizer and model in eval mode on the configured device and dtype."""
    from transformers import AutoModelForCausalLM

    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    tokenizer = load_tokenizer(cfg)
    if cfg.tiny_random:
        model_type = _model_type(cfg)
        if model_type not in TINY_BUILDERS:
            raise ModelStructureError(f"No tiny_random builder for model_type={model_type!r}; known: {sorted(TINY_BUILDERS)}")
        log.warning("model.tiny_random=True: using a TINY RANDOM %s. Results are meaningless (pipeline test only).", model_type)
        model = TINY_BUILDERS[model_type](
            len(tokenizer), pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id, seed=seed,
        ).to(device=device, dtype=dtype)
    else:
        src = cfg.local_path or cfg.model_id
        kwargs: dict[str, Any] = {}
        if not cfg.local_path:
            kwargs["revision"] = cfg.revision
        kwargs["dtype" if transformers_major_version() >= 5 else "torch_dtype"] = dtype
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation
        if device.type == "cuda":
            kwargs["device_map"] = {"": device.index if device.index is not None else 0}
        log.info("Loading %s (revision=%s) as %s on %s", src, cfg.revision, dtype, device)
        model = AutoModelForCausalLM.from_pretrained(src, **kwargs)
        if device.type != "cuda":
            model = model.to(device)
        model.eval()
    layout = get_moe_layout(model)
    info = {
        "model_id": cfg.model_id,
        "local_path": cfg.local_path,
        "requested_revision": cfg.revision,
        "resolved_commit_hash": getattr(model.config, "_commit_hash", None),
        "tiny_random": cfg.tiny_random,
        "model_class": type(model).__name__,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "experts_implementation": getattr(model.config, "_experts_implementation", None),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "architecture": layout.describe(),
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template": tokenizer.chat_template,
    }
    return LoadedModel(model, tokenizer, layout, device, dtype, info)
