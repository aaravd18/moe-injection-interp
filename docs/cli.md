# CLI reference

Generated from `python -m moe_interp.cli <command> --help`. Regenerate it after changing `moe_interp/cli.py`.

```text
usage: moe_interp [-h] [-v]
                  {doctor,download,verify,validate-data,run,analyze,plot} ...

Command-line interface. Run ``python -m moe_interp.cli <command> --help`` for every option.

Commands:
  doctor         check Python/torch/CUDA/disk/HF setup (no model download)
  download       download model weights + tokenizer to the HF cache or a local directory
  verify         load the model and run the routing/activation consistency checks (verify.py)
  validate-data  validate a JSONL dataset; optionally tokenize it and preview token alignment
  run            run an experiment: generations + routing/activation traces
  analyze        compute metrics from saved traces (no model), then plots
  plot           (re)draw plots from saved metrics

positional arguments:
  {doctor,download,verify,validate-data,run,analyze,plot}
    doctor              Check environment (CUDA, versions, disk, HF cache)
    download            Download model weights and tokenizer
    verify              Load the model and verify routing/activation
                        extraction
    validate-data       Validate a JSONL dataset (and optionally its
                        tokenization)
    run                 Run generation + tracing over a dataset
    analyze             Compute comparison metrics from saved traces, then
                        plot
    plot                Draw plots from <results>/metrics

options:
  -h, --help            show this help message and exit
  -v, --verbose
```

## `doctor`

```text
usage: moe_interp doctor [-h] [--output OUTPUT] [--model-id MODEL_ID] [--revision REVISION]

options:
  -h, --help           show this help message and exit
  --output OUTPUT      Output directory to check free disk space for
  --model-id MODEL_ID  Also fetch this model's config (small download) and print its architecture
  --revision REVISION
```

## `download`

```text
usage: moe_interp download [-h] [--model-id MODEL_ID] [--revision REVISION] [--local-dir LOCAL_DIR]

options:
  -h, --help            show this help message and exit
  --model-id MODEL_ID
  --revision REVISION
  --local-dir LOCAL_DIR
                        Copy files here (e.g. a Google Drive folder) instead of only the HF cache
```

## `verify`

```text
usage: moe_interp verify [-h] [--config CONFIG] [--model-id MODEL_ID] [--revision REVISION]
                         [--local-path LOCAL_PATH] [--device DEVICE]
                         [--dtype {auto,bfloat16,float16,float32}] [--tiny-random] [--set KEY=VALUE]
                         [--n-tokens N_TOKENS] [--json-out JSON_OUT]

options:
  -h, --help            show this help message and exit
  --config CONFIG       YAML config file (default: built-in defaults)
  --model-id MODEL_ID   Hugging Face model id (default allenai/OLMoE-1B-7B-0125-Instruct)
  --revision REVISION   Hub revision (commit SHA / branch / tag)
  --local-path LOCAL_PATH
                        Load model + tokenizer from this directory instead of the Hub
  --device DEVICE       auto | cuda | cuda:N | mps | cpu
  --dtype {auto,bfloat16,float16,float32}
  --tiny-random         DEBUG: tiny random OLMoE with the real tokenizer (pipeline test; meaningless
                        results)
  --set KEY=VALUE       Override any config key, e.g. --set trace.activations.positions=all (repeatable)
  --n-tokens N_TOKENS   Number of final prompt tokens to check by hand
  --json-out JSON_OUT   Also write the report to this JSON file
```

## `validate-data`

```text
usage: moe_interp validate-data [-h] --data DATA [--tokenizer TOKENIZER] [--revision REVISION]
                                [--reference REFERENCE]
                                [--alignment {common_suffix,common_prefix,last_n,span}] [--last-n LAST_N]
                                [--span SPAN] [--show-prompts]

options:
  -h, --help            show this help message and exit
  --data DATA
  --tokenizer TOKENIZER
                        Model id/path whose tokenizer + chat template to check (small download)
  --revision REVISION
  --reference REFERENCE
  --alignment {common_suffix,common_prefix,last_n,span}
  --last-n LAST_N
  --span SPAN
  --show-prompts        Print each rendered prompt
```

## `run`

```text
usage: moe_interp run [-h] [--config CONFIG] [--model-id MODEL_ID] [--revision REVISION]
                      [--local-path LOCAL_PATH] [--device DEVICE] [--dtype {auto,bfloat16,float16,float32}]
                      [--tiny-random] [--set KEY=VALUE] [--data DATA] --output OUTPUT [--resume]
                      [--max-new-tokens MAX_NEW_TOKENS] [--limit LIMIT] [--sample-ids SAMPLE_IDS]
                      [--task-types TASK_TYPES] [--conditions CONDITIONS] [--no-generate] [--no-trace]
                      [--analyze]

options:
  -h, --help            show this help message and exit
  --config CONFIG       YAML config file (default: built-in defaults)
  --model-id MODEL_ID   Hugging Face model id (default allenai/OLMoE-1B-7B-0125-Instruct)
  --revision REVISION   Hub revision (commit SHA / branch / tag)
  --local-path LOCAL_PATH
                        Load model + tokenizer from this directory instead of the Hub
  --device DEVICE       auto | cuda | cuda:N | mps | cpu
  --dtype {auto,bfloat16,float16,float32}
  --tiny-random         DEBUG: tiny random OLMoE with the real tokenizer (pipeline test; meaningless
                        results)
  --set KEY=VALUE       Override any config key, e.g. --set trace.activations.positions=all (repeatable)
  --data DATA           JSONL dataset (overrides data.path)
  --output OUTPUT       Experiment output directory (e.g. results/smoke_test or a Drive path)
  --resume              Continue an interrupted run in the same output directory
  --max-new-tokens MAX_NEW_TOKENS
  --limit LIMIT         Only the first N samples
  --sample-ids SAMPLE_IDS
                        Comma-separated sample ids
  --task-types TASK_TYPES
                        Comma-separated metadata.task_type values
  --conditions CONDITIONS
                        Comma-separated condition names
  --no-generate         Only extract routing/activations (prefill pass)
  --no-trace            Only generate responses
  --analyze             Run `analyze` (metrics + plots) afterwards
```

## `analyze`

```text
usage: moe_interp analyze [-h] --results RESULTS [--config CONFIG] [--reference REFERENCE]
                          [--control CONTROL] [--alignment {common_suffix,common_prefix,last_n,span}]
                          [--last-n LAST_N] [--span SPAN] [--min-aligned-tokens MIN_ALIGNED_TOKENS]
                          [--no-plots] [--pdf]

options:
  -h, --help            show this help message and exit
  --results RESULTS
  --config CONFIG       Config for analysis defaults (default: <results>/config.yaml)
  --reference REFERENCE
                        Reference condition (default: analysis.reference, 'clean')
  --control CONTROL     Control condition for deltas, or 'none' (default: analysis.control, 'benign')
  --alignment {common_suffix,common_prefix,last_n,span}
  --last-n LAST_N
  --span SPAN           Span name for --alignment span
  --min-aligned-tokens MIN_ALIGNED_TOKENS
  --no-plots
  --pdf                 Also save PDF versions of plots
```

## `plot`

```text
usage: moe_interp plot [-h] --results RESULTS [--pdf]

options:
  -h, --help         show this help message and exit
  --results RESULTS
  --pdf
```
