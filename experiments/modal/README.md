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
| e02 | Activation steering against injection: injection−benign direction at the last prompt token, val sweep, test vs random control | [logbooks/e02_sep_steering.md](logbooks/e02_sep_steering.md) |
| e03 | OLMoE security expert. Steps 0–1: SecPO data, OLMoE self-generated pairs. Step 2: SecPO LoRA pipeline. Step 3: `secpo_v1` (L40S): ASR 6.7% → 0% but answers collapse. Step 5: `secpo_v2` on paper-style pairs (`y_l` = OLMoE on the seed task alone, 1,500 unfiltered pairs, fp32): ASR 0/208 with base-like answer length (judge pending) | [logbooks/e03_olmoe_security_expert.md](logbooks/e03_olmoe_security_expert.md) |

## Reusable artifacts (`artifacts/`)
- `splits/sep600_witness_grouped_v1.json`: train/val/test item ids for `data/sep_600.jsonl`, grouped by probe
  question (no leakage). Load with `modal_common.load_sep_split()`. Use it for every SEP experiment so held-out
  results are comparable.
- `splits/secpo_olmoe_v1_grouped_v1.json`: train/val ids for `data/secpo_olmoe_v1` (all 1,500 records + the 654
  v1 pairs; v2 trains on `ids`, 1,275 / 225), grouped by shared instruction / data / injection wrapper incl. near-duplicate wrappers. Used by e03 training.
- `directions/mellum_sep_injminusbenign_last_resid_post_v1.npz` (+ `.json`): per-layer injection−benign direction
  (unnormalised) and a random control direction, fitted on the train split.
- Modal volume `moe-interp-artifacts`: full last-token activations for all 600 × 4 prompts × 28 layers (e02).
