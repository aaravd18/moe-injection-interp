# Modal experiments

GPU experiments run on [Modal](https://modal.com) from this laptop. Mellum doesn't fit on the 16 GB laptop, so the
model runs in a Modal container and results come back to this folder.

```
experiments/modal/
  experiments/          one script per experiment (eNN_<name>.py) + modal_common.py (shared image, cache, generation)
  logbooks/             one markdown logbook per experiment: setup, runs, observations
  outputs/<exp>/<tag>/  what each run wrote (CSV / JSONL / run_info.json); small runs are committed
```

## Running

From the repo root, with the venv active (`source .venv/bin/activate`; `modal` is installed there):

```bash
modal run experiments/modal/experiments/e01_sep_baseline_outputs.py            # defaults: 20 items, thinking off
modal run experiments/modal/experiments/e01_sep_baseline_outputs.py --help     # all flags
```

- **GPU**: set per experiment (e01: `GPU_E01 = "L40S"`). `modal_common.GPU` is just a default. Mellum2.1 12B in bf16
  needs ~25 GB, so an L40S (48 GB) is enough: 80 generations × 160 tokens took ~2 min including model load.
- **Weights** are cached in the Modal volume `moe-interp-hf-cache`, so only the first run downloads them.
- **Software** matches the verified H100 / Spark runs: torch 2.14.1, transformers 5.22.1, Mellum revision `92ddae9`.
  The repo's `moe_interp` package is shipped into the container, so model loading, chat templating and batched
  generation are the same code the other experiments use.
- **Pipeline check without a GPU**: `generate_variants` in `modal_common.py` is plain Python and runs locally with
  `{**MELLUM, "tiny_random": True, "device": "cpu", "dtype": "float32", "experts_implementation": None}`.
  This uses the real tokenizer and chat template with a tiny random model, so the outputs are noise.

## Experiments

| id | what | logbook |
|----|------|---------|
| e01 | Baseline Mellum outputs on 20 SEP items, clean vs benign vs injection, read by hand | [logbooks/e01_sep_baseline_outputs.md](logbooks/e01_sep_baseline_outputs.md) |
