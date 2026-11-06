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
        log.warning("GPU has no native bf16 support; falling back to float16 (OLMoE was trained in bf16).")
        return torch.float16
    if device.type == "mps":
        return torch.bfloat16
    log.warning("Running on CPU in float32. The full OLMoE-1B-7B needs ~28 GB RAM at fp32 (set model.dtype=bfloat16 for ~14 GB).")
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

    def describe(self) -> dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "norm_topk_prob": self.norm_topk_prob,
            "hidden_size": self.hidden_size,
            "router_module_class": type(self.routers[0]).__name__,
            "moe_block_class": type(self.moe_blocks[0]).__name__,
            "decoder_layer_class": type(self.layers[0]).__name__,
        }


def get_moe_layout(model: nn.Module) -> MoELayout:
    """Find the decoder layers, MoE blocks, and routers, and check them against the config.

    Expected (Transformers OLMoE, 4.x and 5.x)::

        model.model.layers[i]                          OlmoeDecoderLayer
        model.model.layers[i].post_attention_layernorm RMSNorm (its output is the MoE input)
        model.model.layers[i].mlp                      OlmoeSparseMoeBlock
        model.model.layers[i].mlp.gate                 router; weight [num_experts, hidden]
                                                         4.x: nn.Linear      -> returns logits
                                                         5.x: OlmoeTopKRouter -> returns (logits, top-k weights, top-k ids)
    """
    cfg = model.config
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise ModelStructureError(
            f"{type(model).__name__} has no .model.layers; this code supports OLMoE-style causal LMs only."
        )
    num_layers = int(cfg.num_hidden_layers)
    num_experts = int(getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", 0)))
    top_k = int(cfg.num_experts_per_tok)
    norm_topk = bool(getattr(cfg, "norm_topk_prob", False))
    hidden = int(cfg.hidden_size)
    if len(layers) != num_layers:
        raise ModelStructureError(f"config.num_hidden_layers={num_layers} but model has {len(layers)} layers")
    blocks, routers, norms = [], [], []
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate", None)
        norm = getattr(layer, "post_attention_layernorm", None)
        if mlp is None or gate is None or norm is None:
            raise ModelStructureError(
                f"Layer {i} lacks mlp / mlp.gate / post_attention_layernorm; not an OLMoE-style MoE layer "
                f"(got {type(layer).__name__})."
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
    return MoELayout(num_layers, num_experts, top_k, norm_topk, hidden, list(layers), blocks, routers, norms)


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
        log.warning("model.tiny_random=True: using a TINY RANDOM OLMoE. Results are meaningless (pipeline test only).")
        model = build_tiny_random_olmoe(
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
