"""On-disk layout and (de)serialisation of traces.

::

    <output_dir>/
      config.yaml            resolved experiment config
      metadata.json          environment, model, architecture, dataset hash
      generations.jsonl      one record per (sample, condition)
      traces/<sample>/<condition>/
          trace.json             metadata: token ids, shapes, dtypes, semantics, spans, checks
          routing.safetensors    topk_ids [L,S,k], topk_weights [L,S,k], optional router_logits/probs [L,S,E]
          activations.safetensors  "<type>.layer<NNN>" -> [P, H], plus "positions" [P]
          attention.safetensors    "attn_pattern.layer<NNN>" -> [heads, Q, S], plus "query_positions" [Q] (optional)
      metrics/  plots/

Tensors are stored with safetensors (no pickle). Directory names come from
:func:`safe_name`, which is deterministic and collision-free.
"""

from __future__ import annotations

import hashlib
import json
import re
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors.torch import load_file, save_file

from .tracing import (ACTIVATION_SEMANTICS, ATTENTION_SEMANTICS, ROUTING_SEMANTICS, ActivationTrace,
                      AttentionTrace, RoutingTrace)

TRACE_FORMAT_VERSION = 1
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_name(name: str) -> str:
    """Filesystem-safe, deterministic directory name.

    Names already made of ``[A-Za-z0-9._-]`` are kept unchanged. Any other name has those characters replaced
    and an 8-character hash of the original appended, so two distinct names never map to the same directory.
    """
    if not name:
        raise ValueError("Empty name")
    cleaned = _SAFE.sub("_", name)[:80]
    if cleaned == name and name not in (".", ".."):
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned}-{digest}"


def trace_dir(output_dir: str | Path, sample_id: str, condition: str) -> Path:
    return Path(output_dir) / "traces" / safe_name(sample_id) / safe_name(condition)


def act_key(kind: str, layer: int) -> str:
    return f"{kind}.layer{layer:03d}"


def attn_key(layer: int) -> str:
    return f"attn_pattern.layer{layer:03d}"


def _tensor_meta(t: torch.Tensor, meaning: str, dims: list[str]) -> dict[str, Any]:
    return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "dims": dims, "meaning": meaning}


def save_trace(
    directory: str | Path,
    *,
    sample_id: str,
    condition: str,
    input_ids: list[int],
    routing: RoutingTrace,
    activations: ActivationTrace | None,
    attention: AttentionTrace | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write one trace. Returns the directory."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    L, S, k = routing.topk_ids.shape
    if S != len(input_ids):
        raise ValueError(f"Routing seq_len {S} != len(input_ids) {len(input_ids)}")
    rt = {"topk_ids": routing.topk_ids.to(torch.int64).contiguous(),
          "topk_weights": routing.topk_weights.to(torch.float32).contiguous()}
    if routing.router_logits is not None:
        rt["router_logits"] = routing.router_logits.to(torch.float32).contiguous()
    if routing.router_probs is not None:
        rt["router_probs"] = routing.router_probs.to(torch.float32).contiguous()
    save_file(rt, str(d / "routing.safetensors"))
    dims = {"topk_ids": ["layer", "token", "k"], "topk_weights": ["layer", "token", "k"],
            "router_logits": ["layer", "token", "expert"], "router_probs": ["layer", "token", "expert"]}
    meta: dict[str, Any] = {
        "format_version": TRACE_FORMAT_VERSION,
        "sample_id": sample_id,
        "condition": condition,
        "num_tokens": S,
        "input_ids": list(map(int, input_ids)),
        "routing": {
            "num_layers": L,
            "top_k": k,
            "num_experts": routing.num_experts,
            "norm_topk_prob": routing.norm_topk_prob,
            "source": routing.source,
            "dispatched_weights_dtype": routing.weights_dtype,
            "checks": routing.checks,
            "tensors": {name: _tensor_meta(t, ROUTING_SEMANTICS[name], dims[name]) for name, t in rt.items()},
        },
    }
    if activations is not None and activations.tensors:
        at = {act_key(kind, layer): t.contiguous() for (kind, layer), t in sorted(activations.tensors.items())}
        at["positions"] = torch.tensor(activations.positions, dtype=torch.int64)
        save_file(at, str(d / "activations.safetensors"))
        meta["activations"] = {
            "positions": list(activations.positions),
            "source_dtype": activations.source_dtype,
            "tensors": {
                act_key(kind, layer): {
                    "type": kind, "layer": layer,
                    **_tensor_meta(t, ACTIVATION_SEMANTICS[kind], ["position", "hidden"]),
                }
                for (kind, layer), t in sorted(activations.tensors.items())
            },
        }
    if attention is not None and attention.patterns:
        att = {attn_key(layer): t.contiguous() for layer, t in sorted(attention.patterns.items())}
        att["query_positions"] = torch.tensor(attention.query_positions, dtype=torch.int64)
        save_file(att, str(d / "attention.safetensors"))
        meta["attention"] = {
            "query_positions": list(attention.query_positions),
            "source_dtype": attention.source_dtype,
            "attn_implementation": attention.attn_implementation,
            "num_key_value_heads": attention.num_kv_heads,
            "sliding_window": attention.sliding_window,
            "tensors": {
                attn_key(layer): {
                    "layer": layer,
                    "layer_type": attention.layer_types[layer] if attention.layer_types else None,
                    **_tensor_meta(t, ATTENTION_SEMANTICS["attn_pattern"], ["head", "query_position", "key_position"]),
                }
                for layer, t in sorted(attention.patterns.items())
            },
        }
    if extra:
        meta.update(extra)
    with (d / "trace.json").open("w") as f:
        json.dump(meta, f, indent=1)
    return d


@dataclass
class LoadedTrace:
    meta: dict[str, Any]
    input_ids: torch.Tensor
    """[S] int64"""
    topk_ids: torch.Tensor
    """[L, S, k] int64"""
    topk_weights: torch.Tensor
    """[L, S, k] float32"""
    router_logits: torch.Tensor | None
    router_probs: torch.Tensor | None
    positions: list[int]
    """Token indices at which activations were captured."""
    activations: dict[tuple[str, int], torch.Tensor]
    """(type, layer) -> [len(positions), H]"""
    query_positions: list[int] = dataclasses.field(default_factory=list)
    """Token indices of the attention-pattern query rows."""
    attention: dict[int, torch.Tensor] = dataclasses.field(default_factory=dict)
    """layer -> [num_heads, len(query_positions), S] attention patterns (only if loaded)"""

    @property
    def spans(self) -> dict[str, dict[str, Any]]:
        return self.meta.get("spans", {})

    @property
    def num_layers(self) -> int:
        return int(self.topk_ids.shape[0])

    @property
    def top_k(self) -> int:
        return int(self.topk_ids.shape[-1])

    def activation_layers(self, kind: str) -> list[int]:
        return sorted(layer for (k, layer) in self.activations if k == kind)

    def activation_types(self) -> list[str]:
        return sorted({k for (k, _) in self.activations})


def load_trace(directory: str | Path, load_activations: bool = True, load_attention: bool = True) -> LoadedTrace:
    d = Path(directory)
    meta_path = d / "trace.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No trace.json in {d}")
    with meta_path.open() as f:
        meta = json.load(f)
    if meta.get("format_version") != TRACE_FORMAT_VERSION:
        raise ValueError(f"{d}: unsupported trace format_version {meta.get('format_version')}")
    rt = load_file(str(d / "routing.safetensors"))
    acts: dict[tuple[str, int], torch.Tensor] = {}
    positions: list[int] = []
    if load_activations and (d / "activations.safetensors").exists():
        at = load_file(str(d / "activations.safetensors"))
        positions = at.pop("positions").tolist()
        for key, info in meta["activations"]["tensors"].items():
            acts[(info["type"], int(info["layer"]))] = at[key]
        if positions != meta["activations"]["positions"]:
            raise ValueError(f"{d}: positions in activations.safetensors disagree with trace.json")
    attn: dict[int, torch.Tensor] = {}
    qpos: list[int] = []
    if load_attention and (d / "attention.safetensors").exists():
        att = load_file(str(d / "attention.safetensors"))
        qpos = att.pop("query_positions").tolist()
        for key, info in meta["attention"]["tensors"].items():
            attn[int(info["layer"])] = att[key]
        if qpos != meta["attention"]["query_positions"]:
            raise ValueError(f"{d}: query_positions in attention.safetensors disagree with trace.json")
    input_ids = torch.tensor(meta["input_ids"], dtype=torch.int64)
    if rt["topk_ids"].shape[1] != input_ids.shape[0]:
        raise ValueError(f"{d}: routing has {rt['topk_ids'].shape[1]} tokens but input_ids has {input_ids.shape[0]}")
    return LoadedTrace(
        meta=meta, input_ids=input_ids, topk_ids=rt["topk_ids"], topk_weights=rt["topk_weights"],
        router_logits=rt.get("router_logits"), router_probs=rt.get("router_probs"),
        positions=positions, activations=acts, query_positions=qpos, attention=attn,
    )


# ---- JSON helpers --------------------------------------------------------------------------------

def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path: str | Path, obj: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
