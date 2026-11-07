# Mellum controlled prompt-injection pilot · November 5, 2026

This checkpoint adds a controlled pilot to the existing DGX Spark examples. It uses the same requested model, `JetBrains/Mellum2.1-12B-A2.5B-Thinking`, at revision `92ddae9fc7665e9f801d141d2e5a6b2caf2460c4`. Model weights were fixed throughout. The recorded inference code was `df09e640adb4e9d8e5db38c6a2fa602e18174185`.

[Open the report locally](index.html) for seven plots, all 60 final-answer rows, split filters, scoring tables, timing, and the Spark comparison. GitHub does not render this HTML as a web page. The same findings are also available on the [hosted checkpoint page](https://security-research-273270575556.us-west1.run.app/checkpoints/2026-11-05-mellum-pilot/).

## What was run

Twenty locally generated fictional tasks, with clean, benign-control, and injection conditions for each: **60 prompts**. Ten tasks ask for a museum's opening year; ten extract a booking code. The templates were authored with Codex and filled by a seeded Python generator, not sampled from an established benchmark. Five base tasks (15 prompts) were development cases; fifteen (45 prompts) were held out. Both splits share templates and the direct-override and claimed-authority attack families.

The model ran on an NVIDIA H100 80 GB with BF16, SDPA attention, greedy generation, seed 0, and a 2,048-new-token limit. Routing was captured at all 28 layers; four activation types were recorded at layers 0, 4, 8, 12, 16, 20, 24, and 27. Tracing was a separate instrumented prompt-prefill pass, not a trace of generated reasoning or answers. Attention-pattern capture was disabled for this pilot.

Times are America/Los_Angeles (PST): compute session **November 5, 7:59:52–8:43:42 PM**, report complete **8:53:03 PM**. UTC artifact dates are November 6. These session times include setup and additional checks; they are not pure inference timings.

## Main results

| Held-out measure | Result |
|---|---:|
| Clean task accuracy | 15/15 |
| Benign-control task accuracy | 15/15 |
| Exact attack-target success | 12/15 |
| Legitimate answer retained under attack | 3/15 |
| Missing, truncated, or unscorable outputs | 0 |

The overall pilot had 16/20 attack successes; development alone had 4/5. The primary figures and interpretations use only the 15 held-out tasks, with equal weight per base task.

Injections produced larger routing and activation differences than benign controls through middle and late layers. Mean extra routing overlap distance was **0.05377** in layers 8–19 and **0.06451** in layers 20–27, positive in all 15 tasks in both bands. The largest observed mean routing difference was **0.10108 at layer 26**; residual-post cosine-distance difference reached **0.07108 at layer 24**. These are descriptive selections, not causal localizations.

Routing shifts also appeared in resisted attacks: middle-layer mean overlap-distance difference was 0.05277 for 12 successful attacks and 0.05775 for three resisted attacks. This pilot therefore does not establish a useful success detector. All three resisted examples used claimed-authority attacks, making outcome groups confounded by task and attack-family composition.

The 60-prompt pilot recorded **309.22 seconds of generation**, **21.98 seconds of instrumented trace processing**, 11,064 output tokens, and **24.494 GB peak allocated tensor memory**. These figures exclude setup/model loading and the separate original-example regression. Output-token counts include generated thinking; allocated memory is not total GPU memory use.

## Original-example regression

The original 12 prompt variants were separately rerun on H100. Messages, rendered prompts, input token IDs, requested model revision, and recorded generation settings matched the Spark run. Four final-answer strings matched exactly; two attack outcomes changed.

Assistant inspection assessed standard injection success as **2/4 on Spark versus 4/4 on H100**, with the polite injection resisted in both runs (0/1). This assessment is distinct from the pilot's exact grader and is not a human review. A target word in a translated sentence or generated reasoning is not necessarily attack success. Hardware and software provenance differ, including an earlier dirty Spark checkout, so these runs do not isolate a hardware effect or establish a controlled speed comparison. The existing [Spark report artifacts](../mellum_examples_full/) are unchanged.

## Reproduce the published scores and figures without a GPU

From the repository root, in the environment described in the [experiment guide](../../experiments/mellum_pilot_v1/README.md):

```sh
python experiments/mellum_pilot_v1/scripts/score_mellum_pilot.py \
  --dataset data/mellum_pilot_v1/examples.jsonl \
  --generations reports/mellum_pilot_2026-11-05/data/pilot_generations.jsonl \
  --output-dir /tmp/mellum-rescored

python experiments/mellum_pilot_v1/scripts/report_mellum_pilot.py \
  --output-dir /tmp/mellum-report-rebuilt

python -m unittest discover -s experiments/mellum_pilot_v1/tests -v
```

The report generator verifies the published input hashes in `data/provenance.json`, re-scores all 60 records and compares every retained scoring field, checks every saved layer-band mean and outcome group against the per-sample metrics, and rebuilds all seven plots. It writes fresh output hashes to `output_manifest.json`. `--data-dir` can select another copy of this checkpoint's input directory. Use a new output directory to preserve the published files. The script intentionally validates this frozen checkpoint; it is not a generic judge for unrelated datasets.

PNG file bytes can vary across Matplotlib/platform versions; compare the regenerated scoring outcomes and CSV plotting coordinates, and use the pinned dependencies for closer rendering reproduction. The checked-in figures were regenerated from the included metrics using the recorded Matplotlib version. They have not been recomputed from raw tensor traces.

For a new full-model run, follow the [GPU reproduction recipe](../../experiments/mellum_pilot_v1/README.md). This requires downloading the pinned model and producing new traces. A new run may yield different answers despite greedy settings; preserve it separately rather than replacing this checkpoint.

## Included evidence and export provenance

- `data/pilot_generations.jsonl` and `data/regression_generations.jsonl`: exact final answers, finish reason, input token IDs/spans, counts, timing, and a verified `incomplete_thinking` flag.
- `data/scores.csv` and `data/scoring_summary.json`: recorded scoring results; generated reasoning is removed from the CSV. The exported generation hash is distinguished from the original generation/scorer hashes.
- Five `data/pilot_*.csv` files: byte-identical per-sample routing, activation, control-delta, and alignment tables.
- `data/interpretation.json` and `data/regression_comparison.json`: preserved scientific findings and assistant assessments, with relative references or clearly labeled original hashes.
- `data/provenance.json`: model/configuration/environment information, dataset revision, original source hashes, export hashes, and exact transformation descriptions.
- `tables/`: regenerated held-out plotting coordinates and descriptive score tables. `plots/`: seven regenerated PNGs. `output_manifest.json`: hashes for generated report artifacts.

These are **derived exports**, not byte-identical copies of the original complete run. `original_record_sha256` hashes the exact original JSONL line in UTF-8, excluding the trailing newline. Raw generated reasoning, output token IDs, model weights, tensor traces, and operational cloud identifiers are intentionally absent. The completion flag was checked against original raw output before its removal; the portable scorer requires this flag for final-answer-only records. Original source hashes are retained as provenance, not a claim that omitted files are available in this repository.

## Limits

This is a small controlled fixture set, not a benchmark of real-world robustness. Held-out tasks reuse development templates and attack families. Prompts span 92–132 tokens, and analysis compares 21–24 aligned question tokens. Every injection is one token shorter than its benign control, so absolute positions can differ. Benign text itself includes document-level instructions and the same target label. These controls help interpretation but do not isolate injection intent perfectly. No training, activation patching, routing intervention, or causal test was performed.
