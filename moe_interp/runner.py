"""Experiment runner: JSONL dataset -> generations.jsonl + routing/activation traces.

Each (sample, condition) gets two independent passes over the same token IDs:

1. Instrumented prefill (``tracing.run_traced_forward``): a single forward pass over the prompt with hooks,
   ``use_cache=False``. Every routing/activation trace comes from this pass.
2. Generation (``model.generate``): runs with no hooks attached. Nothing is captured during decoding.

The two passes see identical ``input_ids``. As a sanity check, each record stores whether the first generated
token equals the argmax of the instrumented prefill logits (``first_token_matches_prefill``). In bf16, kernel
differences between the cached and uncached paths can very occasionally flip a near-tie.
"""

from __future__ import annotations

import copy
import datetime as dt
import logging
import time
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig, config_to_dict, save_config
from .dataset import Sample, filter_samples, load_dataset
from .env import environment_info
from .model import LoadedModel, load_model
from .prompts import RenderedPrompt, render_prompt
from .storage import append_jsonl, read_jsonl, sha256_file, save_trace, trace_dir, write_json
from .tracing import ACTIVATION_SEMANTICS, ATTENTION_SEMANTICS, ROUTING_SEMANTICS, TraceSpec, run_traced_forward

log = logging.getLogger(__name__)


class ExperimentExistsError(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def prepare_prompts(samples: list[Sample], tokenizer: Any, add_generation_prompt: bool,
                    chat_template_kwargs: dict[str, Any] | None = None) -> dict[tuple[str, str], RenderedPrompt]:
    """Render every selected variant before any model work, so template problems fail fast."""
    out = {}
    for s in samples:
        for cond, v in s.variants.items():
            try:
                out[(s.id, cond)] = render_prompt(tokenizer, v.messages, add_generation_prompt, v.spans,
                                                  chat_template_kwargs)
            except Exception as e:
                raise type(e)(f"Sample {s.id!r} condition {cond!r} (line {s.line_number}): {e}") from e
    return out


def _generation_config(model: Any, cfg: ExperimentConfig, tokenizer: Any) -> Any:
    g = cfg.generation
    gc = copy.deepcopy(model.generation_config)
    gc.max_new_tokens = g.max_new_tokens
    gc.do_sample = g.do_sample
    if g.do_sample:
        if g.temperature is not None:
            gc.temperature = g.temperature
        if g.top_p is not None:
            gc.top_p = g.top_p
    else:  # greedy: clear sampling knobs inherited from the checkpoint to avoid warnings / ambiguity
        gc.temperature, gc.top_p, gc.top_k = None, None, None
    if g.repetition_penalty is not None:
        gc.repetition_penalty = g.repetition_penalty
    if gc.pad_token_id is None:
        gc.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return gc


def split_thinking(text: str) -> tuple[str | None, str]:
    """Split a reasoning model's response into (thinking, answer) at the last ``</think>``.

    Returns (None, text) if there is no ``</think>``. If generation stopped inside an unclosed ``<think>``, the
    whole text is thinking and the answer is empty.
    """
    if "</think>" in text:
        thinking, answer = text.rsplit("</think>", 1)
        return thinking.replace("<think>", "", 1).strip(), answer.strip()
    if text.lstrip().startswith("<think>"):
        return text.replace("<think>", "", 1).strip(), ""
    return None, text


def split_response(tokenizer: Any, response_ids: list[int], text: str) -> tuple[str | None, str]:
    """(thinking, answer). Splits on the ``</think>`` token id when it is a single token (tags may be special
    tokens that ``skip_special_tokens`` removes), otherwise on the decoded text."""
    end_id = tokenizer.convert_tokens_to_ids("</think>") if "</think>" in tokenizer.get_vocab() else None
    if end_id is not None and end_id in response_ids:
        cut = len(response_ids) - 1 - response_ids[::-1].index(end_id)
        thinking = tokenizer.decode(response_ids[:cut], skip_special_tokens=True).replace("<think>", "", 1).strip()
        return thinking, tokenizer.decode(response_ids[cut + 1:], skip_special_tokens=True).strip()
    return split_thinking(text)


def _eos_ids(gc: Any) -> set[int]:
    e = gc.eos_token_id
    if e is None:
        return set()
    return {int(x) for x in (e if isinstance(e, (list, tuple)) else [e])}


def _completed(gen_path: Path) -> set[tuple[str, str]]:
    if not gen_path.exists():
        return set()
    return {(r["sample_id"], r["condition"]) for r in read_jsonl(gen_path)}


def _comparable(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    return {k: cfg_dict[k] for k in ("model", "prompt", "generation", "trace", "seed")}


def run_experiment(
    cfg: ExperimentConfig,
    output_dir: str | Path,
    *,
    resume: bool = False,
    loaded: LoadedModel | None = None,
) -> Path:
    """Run generation and/or tracing for every selected (sample, condition). Returns the output directory.

    ``loaded`` lets callers (tests, notebooks) reuse an already loaded model.
    """
    import yaml

    out = Path(output_dir)
    gen_path = out / "generations.jsonl"
    if cfg.data.path is None:
        raise ValueError("No dataset given: set data.path in the config or pass --data")
    if out.exists() and gen_path.exists() and not resume:
        raise ExperimentExistsError(
            f"{out} already contains generations.jsonl. Use --resume to continue it or choose a new --output."
        )
    if resume and (out / "config.yaml").exists():
        with (out / "config.yaml").open() as f:
            old = yaml.safe_load(f)
        if _comparable(old) != _comparable(config_to_dict(cfg)):
            raise ExperimentExistsError(
                f"--resume: model/prompt/generation/trace/seed settings differ from {out / 'config.yaml'}; "
                "refusing to mix incompatible results in one directory."
            )
    out.mkdir(parents=True, exist_ok=True)
    (out / "traces").mkdir(exist_ok=True)

    samples = load_dataset(cfg.data.path)
    samples = filter_samples(samples, cfg.data.sample_ids, cfg.data.task_types, cfg.data.conditions, cfg.data.limit)
    if not samples:
        raise ValueError("No samples left after filtering")
    if not (cfg.generation.enabled or cfg.trace.enabled):
        raise ValueError("Both generation and tracing are disabled; nothing to do")

    torch.manual_seed(cfg.seed)
    if loaded is None:
        loaded = load_model(cfg.model, seed=cfg.seed)
    model, tok, layout, device = loaded.model, loaded.tokenizer, loaded.layout, loaded.device

    prompts = prepare_prompts(samples, tok, cfg.prompt.add_generation_prompt, cfg.prompt.chat_template_kwargs)
    max_pos = getattr(model.config, "max_position_embeddings", None)
    if max_pos:
        need = max(len(p.input_ids) for p in prompts.values()) + (cfg.generation.max_new_tokens if cfg.generation.enabled else 0)
        if need > max_pos:
            raise ValueError(f"Longest prompt + max_new_tokens = {need} exceeds max_position_embeddings = {max_pos}")

    spec = TraceSpec.from_config(cfg.trace, layout.num_layers)
    gc = _generation_config(model, cfg, tok)
    eos = _eos_ids(gc)

    save_config(cfg, out / "config.yaml")
    metadata = {
        "experiment_name": cfg.experiment_name or out.name,
        "started_at": _now(),
        "status": "running",
        "dataset": {"path": str(Path(cfg.data.path).resolve()), "sha256": sha256_file(cfg.data.path),
                    "num_samples": len(samples), "num_variants": sum(len(s.variants) for s in samples)},
        "model": loaded.info,
        "environment": environment_info(),
        "generation_config_effective": gc.to_dict(),
        "trace_spec": spec.describe(),
        "semantics": {"routing": ROUTING_SEMANTICS, "activations": ACTIVATION_SEMANTICS, "attention": ATTENTION_SEMANTICS,
                      "note": "Traces come from a separate instrumented prefill pass, not from generation."},
    }
    if resume and (out / "metadata.json").exists():
        import json

        with (out / "metadata.json").open() as f:
            old_meta = json.load(f)
        metadata["started_at"] = old_meta.get("started_at", metadata["started_at"])
        metadata["resumed_at"] = old_meta.get("resumed_at", []) + [_now()]
    write_json(out / "metadata.json", metadata)

    done =_completed(gen_path) if resume else set()
    todo = [(s, c) for s in samples for c in s.variants if (s.id, c) not in done]
    log.info("%d variants to run (%d already done)", len(todo), len(done))

    for n, (sample, cond) in enumerate(todo, start=1):
        rp = prompts[(sample.id, cond)]
        ids = torch.tensor([rp.input_ids], dtype=torch.long, device=device)
        record: dict[str, Any] = {
            "sample_id": sample.id,
            "condition": cond,
            "task_type": sample.task_type,
            "metadata": sample.metadata,
            "evaluation": sample.evaluation,
            "messages": sample.variants[cond].messages,
            "prompt_text": rp.text,
            "input_ids": rp.input_ids,
            "attention_mask": rp.attention_mask,
            "num_prompt_tokens": len(rp.input_ids),
            "message_token_spans": [None if s is None else s.to_dict() for s in rp.message_token_spans],
            "spans": {k: v.to_dict() for k, v in rp.spans.items()},
        }
        timing: dict[str, float] = {}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        if cfg.trace.enabled:
            torch.manual_seed(cfg.seed)
            _sync(device)
            t0 = time.perf_counter()
            tr = run_traced_forward(model, layout, ids, spec)
            _sync(device)
            timing["trace_forward_s"] = round(time.perf_counter() - t0, 4)
            td = trace_dir(out, sample.id, cond)
            save_trace(
                td, sample_id=sample.id, condition=cond, input_ids=rp.input_ids, routing=tr.routing,
                activations=tr.activations, attention=tr.attention,
                extra={"prompt_text": rp.text, "spans": record["spans"],
                       "message_token_spans": record["message_token_spans"],
                       "prefill_argmax_token_id": tr.next_token_id},
            )
            record["trace_dir"] = str(td.relative_to(out))
            record["routing_source"] = tr.routing.source
            record["prefill_argmax_token_id"] = tr.next_token_id

        if cfg.generation.enabled:
            torch.manual_seed(cfg.seed)
            _sync(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=gc)
            _sync(device)
            dt_gen = time.perf_counter() - t0
            resp = gen[0, ids.shape[1]:].tolist()
            raw = tok.decode(resp, skip_special_tokens=False)
            text = tok.decode(resp, skip_special_tokens=True)
            thinking, answer = split_response(tok, resp, text)
            record.update({
                "response_text": text,
                "response_text_raw": raw,
                "response_thinking": thinking,
                "response_answer": answer,
                "response_ids": resp,
                "num_generated_tokens": len(resp),
                "finish_reason": "eos" if resp and resp[-1] in eos else "length",
            })
            timing["generate_s"] = round(dt_gen, 4)
            timing["generated_tokens_per_s"] = round(len(resp) / dt_gen, 2) if dt_gen > 0 else None
            if "prefill_argmax_token_id" in record and resp:
                record["first_token_matches_prefill"] = resp[0] == record["prefill_argmax_token_id"]
            target = sample.evaluation.get("attack_target")
            if isinstance(target, str) and target:
                # Naive case-insensitive substring check; a convenience, not an attack-success judge.
                record["attack_target_in_response"] = target.lower() in record["response_text"].lower()
                # For reasoning models the thinking may merely *mention* the target; judge the answer separately.
                record["attack_target_in_answer"] = target.lower() in answer.lower()
        record["timing"] = timing
        if device.type == "cuda":
            record["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 3)
        append_jsonl(gen_path, record)
        log.info(
            "[%d/%d] %s/%s: %d prompt tokens%s", n, len(todo), sample.id, cond, len(rp.input_ids),
            f", {record['num_generated_tokens']} generated ({record['finish_reason']})" if "response_ids" in record else "",
        )

    metadata["status"] = "complete"
    metadata["finished_at"] = _now()
    write_json(out / "metadata.json", metadata)
    return out
