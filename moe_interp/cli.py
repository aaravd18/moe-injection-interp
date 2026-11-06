"""Command-line interface. Run ``python -m moe_interp.cli <command> --help`` for every option.

Commands:
  doctor         check Python/torch/CUDA/disk/HF setup (no model download)
  download       download model weights + tokenizer to the HF cache or a local directory
  verify         load the model and run the routing/activation consistency checks (verify.py)
  validate-data  validate a JSONL dataset; optionally tokenize it and preview token alignment
  run            run an experiment: generations + routing/activation traces
  analyze        compute metrics from saved traces (no model), then plots
  plot           (re)draw plots from saved metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import DEFAULT_MODEL_ID, ExperimentConfig, apply_overrides, load_config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("urllib3", "filelock", "huggingface_hub", "matplotlib", "PIL", "httpx", "httpx2", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML config file (default: built-in defaults)")
    p.add_argument("--model-id", help=f"Hugging Face model id (default {DEFAULT_MODEL_ID})")
    p.add_argument("--revision", help="Hub revision (commit SHA / branch / tag)")
    p.add_argument("--local-path", help="Load model + tokenizer from this directory instead of the Hub")
    p.add_argument("--device", help="auto | cuda | cuda:N | mps | cpu")
    p.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--tiny-random", action="store_true",
                   help="DEBUG: tiny random model of the same architecture, real tokenizer (pipeline test; meaningless results)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Override any config key, e.g. --set trace.activations.positions=all (repeatable)")


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = load_config(args.config)
    o: list[str] = []
    m = {"model_id": "model.model_id", "revision": "model.revision", "local_path": "model.local_path",
         "device": "model.device", "dtype": "model.dtype", "data": "data.path",
         "max_new_tokens": "generation.max_new_tokens", "limit": "data.limit"}
    for attr, key in m.items():
        v = getattr(args, attr, None)
        if v is not None:
            o.append(f"{key}={json.dumps(v)}")
    for attr, key in (("sample_ids", "data.sample_ids"), ("task_types", "data.task_types"),
                      ("conditions", "data.conditions")):
        v = getattr(args, attr, None)
        if v:
            o.append(f"{key}={json.dumps(v.split(','))}")
    if getattr(args, "tiny_random", False):
        o.append("model.tiny_random=true")
    if getattr(args, "no_generate", False):
        o.append("generation.enabled=false")
    if getattr(args, "no_trace", False):
        o.append("trace.enabled=false")
    return apply_overrides(cfg, o + list(args.set))


# ---- commands ------------------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> int:
    from .env import doctor_report

    info, warnings = doctor_report(args.output)
    print(json.dumps(info, indent=2, default=str))
    if args.model_id:
        try:
            from transformers import AutoConfig
            c = AutoConfig.from_pretrained(args.model_id, revision=args.revision)
            print(f"\n{args.model_id} config: layers={c.num_hidden_layers} experts={c.num_experts} "
                  f"top_k={c.num_experts_per_tok} norm_topk_prob={getattr(c, 'norm_topk_prob', None)} "
                  f"hidden={c.hidden_size} dtype={getattr(c, 'dtype', getattr(c, 'torch_dtype', None))}")
        except Exception as e:
            warnings.append(f"Could not fetch config for {args.model_id}: {e}")
    print()
    for w in warnings:
        print(f"WARNING: {w}")
    print("doctor: OK" if not warnings else f"doctor: {len(warnings)} warning(s)")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    path = snapshot_download(args.model_id, revision=args.revision, local_dir=args.local_dir,
                             allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.md"])
    print(f"Model files at: {path}")
    if args.local_dir:
        print(f"Use with: --local-path {path}  (or model.local_path in the config)")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from .model import load_model
    from .verify import verify_model

    cfg = _build_config(args)
    loaded = load_model(cfg.model, seed=cfg.seed)
    report = verify_model(loaded, n_tokens=args.n_tokens)
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text)
    print("\nverify: PASSED" if report["passed"] else "\nverify: FAILED (see checks above)")
    return 0 if report["passed"] else 1


def cmd_validate_data(args: argparse.Namespace) -> int:
    from .alignment import align_tokens
    from .dataset import DatasetValidationError, dataset_warnings, load_dataset, summarize_dataset

    try:
        samples = load_dataset(args.data)
    except DatasetValidationError as e:
        print(f"INVALID: {e}")
        return 1
    print(json.dumps(summarize_dataset(samples), indent=2))
    for w in dataset_warnings(samples):
        print(f"WARNING: {w}")
    if not args.tokenizer:
        print("\nvalid (pass --tokenizer MODEL_ID to also check chat-template rendering and token alignment)")
        return 0
    from transformers import AutoTokenizer

    from .prompts import render_prompt

    tok = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    bad = 0
    print(f"\n{'sample':<20} {'condition':<16} {'tokens':>6}  alignment vs '{args.reference}' ({args.alignment})")
    for s in samples:
        rendered = {}
        for c, v in s.variants.items():
            try:
                rendered[c] = render_prompt(tok, v.messages, True, v.spans)
            except Exception as e:
                print(f"{s.id:<20} {c:<16} ERROR: {e}")
                bad += 1
        ref = rendered.get(args.reference)
        for c, rp in rendered.items():
            note = "(reference)" if c == args.reference else "no reference condition"
            if ref is not None and c != args.reference:
                al = align_tokens(ref.input_ids, rp.input_ids, args.alignment, n=args.last_n, span_name=args.span,
                                  spans_a={k: v.to_dict() for k, v in ref.spans.items()},
                                  spans_b={k: v.to_dict() for k, v in rp.spans.items()})
                note = f"{al.n} tokens aligned" if al.ok else f"NOT ALIGNED: {al.reason}"
            print(f"{s.id:<20} {c:<16} {len(rp.input_ids):>6}  {note}")
            if args.show_prompts:
                print("    " + repr(rp.text))
    print("\nvalid" if not bad else f"\n{bad} rendering error(s)")
    return 0 if not bad else 1


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import run_experiment

    cfg = _build_config(args)
    if cfg.experiment_name is None:
        cfg.experiment_name = Path(args.output).name
    out = run_experiment(cfg, args.output, resume=args.resume)
    print(f"Results written to {out}")
    if args.analyze:
        return _analyze(out, cfg, args, plots=True, pdf=cfg.analysis.save_pdf)
    return 0


def _analysis_settings(cfg: ExperimentConfig, args: argparse.Namespace):
    from .analysis import AnalysisSettings

    a = cfg.analysis
    control = getattr(args, "control", None)
    return AnalysisSettings(
        reference=getattr(args, "reference", None) or a.reference,
        control=None if control in ("none", "null", "") else (control or a.control),
        alignment=getattr(args, "alignment", None) or a.alignment,
        last_n=getattr(args, "last_n", None) or a.last_n,
        span=getattr(args, "span", None) or a.span,
        min_aligned_tokens=getattr(args, "min_aligned_tokens", None) or a.min_aligned_tokens,
    )


def _analyze(results: Path, cfg: ExperimentConfig, args: argparse.Namespace, plots: bool, pdf: bool) -> int:
    from .analysis import run_analysis
    from .plotting import make_all_plots

    settings = _analysis_settings(cfg, args)
    tables = run_analysis(results, settings)
    al = tables["alignment"]
    print(f"Metrics written to {results / 'metrics'}  "
          f"({len(al)} pairs, {int(al['ok'].sum()) if not al.empty else 0} aligned)")
    if not tables["overlap_summary"].empty:
        piv = tables["overlap_summary"].pivot(index="layer", columns="condition", values="overlap_mean")
        print(f"\nMean top-k overlap with '{settings.reference}' by layer:")
        print(piv.round(3).to_string())
    if plots:
        written = make_all_plots(results, save_pdf=pdf)
        print(f"\n{len(written)} plot file(s) written to {results / 'plots'}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    results = Path(args.results)
    cfg_path = Path(args.config) if args.config else results / "config.yaml"
    cfg = load_config(cfg_path if cfg_path.exists() else None)
    return _analyze(results, cfg, args, plots=not args.no_plots, pdf=args.pdf or cfg.analysis.save_pdf)


def cmd_plot(args: argparse.Namespace) -> int:
    from .plotting import make_all_plots

    written = make_all_plots(args.results, save_pdf=args.pdf)
    for p in written:
        print(p)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="moe_interp", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="Check environment (CUDA, versions, disk, HF cache)")
    d.add_argument("--output", help="Output directory to check free disk space for")
    d.add_argument("--model-id", help="Also fetch this model's config (small download) and print its architecture")
    d.add_argument("--revision")
    d.set_defaults(func=cmd_doctor)

    dl = sub.add_parser("download", help="Download model weights and tokenizer")
    dl.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    dl.add_argument("--revision")
    dl.add_argument("--local-dir", help="Copy files here (e.g. a Google Drive folder) instead of only the HF cache")
    dl.set_defaults(func=cmd_download)

    v = sub.add_parser("verify", help="Load the model and verify routing/activation extraction")
    _add_model_args(v)
    v.add_argument("--n-tokens", type=int, default=4, help="Number of final prompt tokens to check by hand")
    v.add_argument("--json-out", help="Also write the report to this JSON file")
    v.set_defaults(func=cmd_verify)

    vd = sub.add_parser("validate-data", help="Validate a JSONL dataset (and optionally its tokenization)")
    vd.add_argument("--data", required=True)
    vd.add_argument("--tokenizer", help="Model id/path whose tokenizer + chat template to check (small download)")
    vd.add_argument("--revision")
    vd.add_argument("--reference", default="clean")
    vd.add_argument("--alignment", default="common_suffix", choices=["common_suffix", "common_prefix", "last_n", "span"])
    vd.add_argument("--last-n", type=int, default=8)
    vd.add_argument("--span")
    vd.add_argument("--show-prompts", action="store_true", help="Print each rendered prompt")
    vd.set_defaults(func=cmd_validate_data)

    r = sub.add_parser("run", help="Run generation + tracing over a dataset")
    _add_model_args(r)
    r.add_argument("--data", help="JSONL dataset (overrides data.path)")
    r.add_argument("--output", required=True, help="Experiment output directory (e.g. results/smoke_test or a Drive path)")
    r.add_argument("--resume", action="store_true", help="Continue an interrupted run in the same output directory")
    r.add_argument("--max-new-tokens", type=int)
    r.add_argument("--limit", type=int, help="Only the first N samples")
    r.add_argument("--sample-ids", help="Comma-separated sample ids")
    r.add_argument("--task-types", help="Comma-separated metadata.task_type values")
    r.add_argument("--conditions", help="Comma-separated condition names")
    r.add_argument("--no-generate", action="store_true", help="Only extract routing/activations (prefill pass)")
    r.add_argument("--no-trace", action="store_true", help="Only generate responses")
    r.add_argument("--analyze", action="store_true", help="Run `analyze` (metrics + plots) afterwards")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("analyze", help="Compute comparison metrics from saved traces, then plot")
    a.add_argument("--results", required=True)
    a.add_argument("--config", help="Config for analysis defaults (default: <results>/config.yaml)")
    a.add_argument("--reference", help="Reference condition (default: analysis.reference, 'clean')")
    a.add_argument("--control", help="Control condition for deltas, or 'none' (default: analysis.control, 'benign')")
    a.add_argument("--alignment", choices=["common_suffix", "common_prefix", "last_n", "span"])
    a.add_argument("--last-n", type=int)
    a.add_argument("--span", help="Span name for --alignment span")
    a.add_argument("--min-aligned-tokens", type=int)
    a.add_argument("--no-plots", action="store_true")
    a.add_argument("--pdf", action="store_true", help="Also save PDF versions of plots")
    a.set_defaults(func=cmd_analyze)

    pl = sub.add_parser("plot", help="Draw plots from <results>/metrics")
    pl.add_argument("--results", required=True)
    pl.add_argument("--pdf", action="store_true")
    pl.set_defaults(func=cmd_plot)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        if args.verbose:
            raise
        print(f"error: {type(e).__name__}: {e}\n(re-run with -v for a traceback)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
