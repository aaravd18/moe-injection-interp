"""Experiment configuration: plain dataclasses loaded from YAML.

Unknown keys raise an error, so typos in a config file are caught instead of
being silently ignored. Command-line overrides use dotted keys, for example
``--set generation.max_new_tokens=32``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml

DEFAULT_MODEL_ID = "allenai/OLMoE-1B-7B-0125-Instruct"

# Activation types that the tracer can capture. See tracing.ACTIVATION_SEMANTICS.
ACTIVATION_TYPES = ("resid_pre", "resid_mid", "router_input", "moe_output", "resid_post")


@dataclass
class ModelConfig:
    model_id: str = DEFAULT_MODEL_ID
    """Hugging Face model ID. Also used to load the tokenizer when ``tiny_random`` is set."""
    revision: str | None = None
    """Git revision (commit SHA, branch, or tag) on the Hub. Pin a SHA for reproducibility."""
    local_path: str | None = None
    """If set, load weights and tokenizer from this directory instead of ``model_id``."""
    device: str = "auto"
    """``auto`` (cuda > mps > cpu), ``cuda``, ``cuda:0``, ``mps``, or ``cpu``."""
    dtype: str = "auto"
    """``auto``, ``bfloat16``, ``float16``, or ``float32``. ``auto`` = bf16 on capable GPUs, else fp16 on CUDA, else fp32."""
    attn_implementation: str | None = None
    """Passed to ``from_pretrained`` (e.g. ``sdpa``, ``eager``). ``None`` uses the Transformers default."""
    tiny_random: bool = False
    """DEBUG ONLY: build a tiny randomly initialised OLMoE (real tokenizer) to test the pipeline without the 7B download.
    Results are meaningless."""


@dataclass
class PromptConfig:
    add_generation_prompt: bool = True
    """Append the assistant turn header (``<|assistant|>\\n`` for OLMoE) after the last message."""


@dataclass
class GenerationConfig:
    enabled: bool = True
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float | None = None
    """Only used when ``do_sample`` is true."""
    top_p: float | None = None
    """Only used when ``do_sample`` is true."""
    repetition_penalty: float | None = None


@dataclass
class RoutingTraceConfig:
    save_router_logits: bool = False
    """Also save raw pre-softmax router logits, shape [num_layers, seq_len, num_experts]."""
    save_router_probs: bool = False
    """Also save full softmax router probabilities, shape [num_layers, seq_len, num_experts]."""


@dataclass
class ActivationTraceConfig:
    types: list[str] = field(default_factory=lambda: ["resid_mid", "router_input", "moe_output", "resid_post"])
    """Subset of: resid_pre, resid_mid, router_input, moe_output, resid_post. Use [] to disable."""
    layers: list[int] | str = field(default_factory=lambda: [0, 4, 8, 12, -1])
    """Layer indices (negative indices count from the end) or ``"all"``."""
    positions: str | list[int] = "last"
    """``last``, ``all``, ``last_k:N`` (final N prompt tokens), or an explicit list of token indices."""
    dtype: str = "float32"
    """Storage dtype for activations: float32, float16, or bfloat16."""


@dataclass
class TraceConfig:
    enabled: bool = True
    routing: RoutingTraceConfig = field(default_factory=RoutingTraceConfig)
    activations: ActivationTraceConfig = field(default_factory=ActivationTraceConfig)


@dataclass
class DataConfig:
    path: str | None = None
    sample_ids: list[str] | None = None
    task_types: list[str] | None = None
    conditions: list[str] | None = None
    limit: int | None = None
    """Keep only the first N samples after filtering."""


@dataclass
class AnalysisConfig:
    reference: str = "clean"
    """Condition every other condition is compared against."""
    control: str | None = "benign"
    """Control condition for the control-adjusted shift. Set to null to skip."""
    alignment: str = "common_suffix"
    """common_suffix | common_prefix | last_n | span"""
    last_n: int = 8
    """Number of final tokens for ``alignment: last_n``."""
    span: str | None = None
    """Span name for ``alignment: span`` (must be annotated in the dataset)."""
    min_aligned_tokens: int = 1
    save_pdf: bool = False


@dataclass
class ExperimentConfig:
    experiment_name: str | None = None
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


class ConfigError(ValueError):
    pass


def _from_dict(cls: type, data: dict[str, Any], path: str = "") -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config section '{path or '<root>'}' must be a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(
            f"Unknown config key(s) {sorted(unknown)} in section '{path or '<root>'}'. Valid keys: {sorted(names)}"
        )
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _from_dict(hint, value, f"{path}.{name}" if path else name)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    cfg = _from_dict(ExperimentConfig, data)
    validate_config(cfg)
    return cfg


def load_config(path: str | Path | None) -> ExperimentConfig:
    """Load a YAML config. ``None`` returns the defaults."""
    if path is None:
        return config_from_dict({})
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return config_from_dict(data)


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    with Path(path).open("w") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False)


def apply_overrides(cfg: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    """Apply ``key.sub=value`` overrides. Values are parsed as YAML (``null``, ``1``, ``[0, 1]``, ...)."""
    data = config_to_dict(cfg)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override '{item}' must look like section.key=value")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        node = data
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise ConfigError(f"Override '{item}': unknown section '{p}'")
            node = node[p]
        if parts[-1] not in node:
            raise ConfigError(f"Override '{item}': unknown key '{parts[-1]}'")
        node[parts[-1]] = yaml.safe_load(raw)
    return config_from_dict(data)


def validate_config(cfg: ExperimentConfig) -> None:
    acts = cfg.trace.activations
    bad = [t for t in acts.types if t not in ACTIVATION_TYPES]
    if bad:
        raise ConfigError(f"Unknown activation type(s) {bad}. Valid: {list(ACTIVATION_TYPES)}")
    if isinstance(acts.layers, str) and acts.layers != "all":
        raise ConfigError(f"trace.activations.layers must be a list of ints or 'all', got {acts.layers!r}")
    if acts.dtype not in ("float32", "float16", "bfloat16"):
        raise ConfigError(f"trace.activations.dtype must be float32/float16/bfloat16, got {acts.dtype!r}")
    if cfg.model.dtype not in ("auto", "bfloat16", "float16", "float32"):
        raise ConfigError(f"model.dtype must be auto/bfloat16/float16/float32, got {cfg.model.dtype!r}")
    if cfg.generation.max_new_tokens < 1:
        raise ConfigError("generation.max_new_tokens must be >= 1")
    if cfg.analysis.alignment not in ("common_suffix", "common_prefix", "last_n", "span"):
        raise ConfigError(
            f"analysis.alignment must be common_suffix/common_prefix/last_n/span, got {cfg.analysis.alignment!r}"
        )
    if cfg.analysis.alignment == "span" and not cfg.analysis.span:
        raise ConfigError("analysis.alignment=span requires analysis.span to name an annotated span")
