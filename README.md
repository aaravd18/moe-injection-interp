# moe-injection-interp

Inference-only infrastructure for studying how prompt-injection attacks change **expert routing**,
**attention patterns**, and **hidden-state activations** in Mixture-of-Experts language models, relative to clean
prompts and benign controls. Supported models:

| Model | Config | Layers | Experts / top-k | Attention | `norm_topk_prob` |
|---|---|---|---|---|---|
| [`allenai/OLMoE-1B-7B-0125-Instruct`](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) | [`configs/olmoe.yaml`](configs/olmoe.yaml) | 16 | 64 / 8 | 16 heads, full | false |
| [`JetBrains/Mellum2.1-12B-A2.5B-Thinking`](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking) | [`configs/mellum.yaml`](configs/mellum.yaml) | 28 | 64 / 8 | 32 query / 4 KV heads (GQA); 3 sliding-window (1024) : 1 full | true |

Mellum needs `transformers >= 5.8.1`. `configs/mellum.yaml` captures **everything at every layer and every prompt
token**: routing (ids, weights, logits, probs), all 8 activation types, and per-head attention patterns.

```
JSONL of prompt variants ──► run ──► generations.jsonl + per-token routing traces + activations (safetensors)
                                     │
                                     └──► analyze (CPU, no model) ──► metrics/*.csv ──► plots/*.png
```

One command, from a JSONL file to responses, traces, metrics, and figures:

```bash
python -m moe_interp.cli run --config configs/olmoe.yaml --data data/examples.jsonl \
    --output results/smoke_test --analyze
```

---

## Contents

1. [Setup](#1-setup): laptop, SSH GPU machine, Google Colab
2. [Workflow](#2-workflow)
3. [Dataset format](#3-dataset-format)
4. [What is extracted, and what it means](#4-what-is-extracted-and-what-it-means) (**read this before interpreting traces**)
5. [Token alignment](#5-token-alignment)
6. [Metrics](#6-metrics)
7. [Output layout](#7-output-layout)
8. [Configuration and CLI](#8-configuration-and-cli)
9. [Tests and verification status](#9-tests-and-verification-status)
10. [Limitations and extension points](#10-limitations-and-extension-points)

---

## 1. Setup

Requirements: Python ≥ 3.10. For real experiments you also need one CUDA GPU with ≥ 16 GB (24 GB is comfortable).
The bf16 weights are ≈ 14 GB. Analysis, plotting, and the unit tests run on CPU.

### Laptop (development + CPU analysis)

```bash
git clone <your-repo-url> moe-injection-interp && cd moe-injection-interp
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m moe_interp.cli doctor          # versions, CUDA, disk, HF cache
python -m pytest                         # 77 CPU tests, no model download

# Full pipeline on CPU with a *tiny random* OLMoE + the real tokenizer (≈2 MB download; results meaningless):
python -m moe_interp.cli run --config configs/debug_tiny.yaml --output results/debug --analyze
```

### Remote GPU machine over SSH

```bash
ssh user@gpu-box
git clone <your-repo-url> moe-injection-interp && cd moe-injection-interp
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# If the machine's CUDA needs a specific torch build, install it first, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m moe_interp.cli doctor --model-id allenai/OLMoE-1B-7B-0125-Instruct

export HF_HOME=/scratch/$USER/hf          # optional: put the 14 GB cache on a large disk
export HF_TOKEN=...                        # optional: faster downloads
tmux new -s moe                            # keep the run alive if SSH drops
python -m moe_interp.cli verify --config configs/olmoe.yaml
python -m moe_interp.cli run --config configs/olmoe.yaml --data data/examples.jsonl \
    --output /scratch/$USER/results/smoke_test --analyze
```

Copy results back for CPU analysis with `rsync -av gpu-box:/scratch/$USER/results/smoke_test results/`.

### NVIDIA DGX Spark (Mellum2.1, full capture)

The Spark is aarch64 with CUDA 13 and 128 GB unified memory; Mellum2.1 in bf16 (~24 GB) fits easily. Use the
PyTorch cu130 wheels (no container needed):

```bash
python3 -m venv ~/moe-venv && . ~/moe-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[dev]" "transformers>=5.8.1"
hf download JetBrains/Mellum2.1-12B-A2.5B-Thinking --revision 92ddae9fc7665e9f801d141d2e5a6b2caf2460c4
python -m moe_interp.cli verify --config configs/mellum.yaml          # 21 checks, incl. attention recompute
nohup python -m moe_interp.cli run --config configs/mellum.yaml --data data/examples.jsonl \
    --output ~/moe-results/mellum_examples_full --analyze > run.log 2>&1 &
```

Mellum is a thinking model: responses start with `<think>…</think>`, so `max_new_tokens` defaults to 2048 in its
config. Each record stores `response_thinking` and `response_answer` separately, and `attack_target_in_answer`
checks only the final answer (the reasoning may merely *mention* the target). Set
`prompt.chat_template_kwargs: {enable_thinking: false}` to disable thinking.

### Google Colab Pro

Open [`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb) in Colab, select a GPU runtime
(A100/L4 preferred; T4 works in fp16 but is tight), set `REPO_URL`, and run all cells. It mounts Drive, clones
the repo, runs `pip install -e .`, and then calls the same CLI. The equivalent commands are:

```python
from google.colab import drive; drive.mount("/content/drive")
%cd /content
!git clone <your-repo-url> moe-injection-interp
%cd /content/moe-injection-interp
!pip install -q -e .
!python -m moe_interp.cli doctor
!python -m moe_interp.cli verify --config configs/olmoe.yaml
!python -m moe_interp.cli run --config configs/olmoe.yaml --data data/examples.jsonl \
    --output /content/drive/MyDrive/moe_interp_results/smoke_test --resume --analyze
```

`--resume` makes a disconnected run continue where it stopped. Each record is flushed to `generations.jsonl` as
soon as it finishes. Supported transformers versions: tested with **4.57.1 and 5.22.1** (see §9). Colab's
preinstalled version is used as long as it is ≥ 4.46.

---

## 2. Workflow

| Step | Command | Needs model? |
|---|---|---|
| Check environment | `python -m moe_interp.cli doctor [--model-id ID]` | no (config only) |
| Download weights | `python -m moe_interp.cli download [--local-dir DIR]` | downloads |
| Validate dataset + preview alignment | `python -m moe_interp.cli validate-data --data D.jsonl --tokenizer allenai/OLMoE-1B-7B-0125-Instruct` | tokenizer only |
| Verify extraction on the real model | `python -m moe_interp.cli verify --config configs/olmoe.yaml` | yes |
| Generate + extract routing/activations | `python -m moe_interp.cli run --config C --data D --output OUT` | yes |
| Extract only (no generation) | `... run ... --no-generate` | yes |
| Compute metrics (+ plots) | `python -m moe_interp.cli analyze --results OUT` | **no** |
| Re-draw plots | `python -m moe_interp.cli plot --results OUT [--pdf]` | **no** |

Run `validate-data --tokenizer ...` on a laptop before using GPU time. It renders every prompt with the real
chat template, reports token counts, and shows whether each condition can be token-aligned with the reference.

---

## 3. Dataset format

JSONL, one base task per line, with any number of named conditions (pairs, trios, or more):

```json
{
  "id": "task_002",
  "metadata": {"task_type": "retrieval_qa", "attack_type": "instruction_override"},
  "variants": {
    "clean":     {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "<document>...</document>\n\nUsing only the document above, answer ..."}],
                  "spans": {"question": {"message": 1, "text": "Using only the document above, answer ..."}}},
    "benign":    {"messages": [...], "spans": {...}},
    "injection": {"messages": [...], "spans": {...}}
  },
  "evaluation": {"attack_target": "ACCESS GRANTED"}
}
```

* `id` (required, unique) is used verbatim in outputs. Directory names are derived deterministically (§7).
* `variants` (required): condition name → `{"messages": [...], "spans": {...}?}`. Condition names are arbitrary
  strings. Messages are `{"role", "content"}` only and are passed to the chat template unchanged.
* `metadata` (optional): free-form. `task_type` is used by `--task-types` filtering.
* `evaluation` (optional): free-form. If `attack_target` is a string, each record gets a naive
  case-insensitive substring check, `attack_target_in_response`. That is a convenience, not an attack-success judge.
* `spans` (optional): named character ranges inside one message, either `{"message": i, "text": "..."}` (must
  occur exactly once, or add `"occurrence": n`) or `{"message": i, "start": s, "end": e}`. They are mapped to
  token ranges at tokenization time and saved, and are used by `--alignment span`.

Validation reports **every** problem with its line number (missing fields, wrong types, unknown keys, duplicate
ids, spans whose text isn't found, ambiguous spans). Lines that are blank or start with `//` are ignored.

> **Important (OLMoE chat template):** the template only renders `system`, `user`, and `assistant` roles. A
> `tool` message is **silently dropped** by the template. The renderer detects this and raises an error. Put tool
> outputs and retrieved documents *inside* a supported role's content, as in `data/examples.jsonl`.

[`data/examples.jsonl`](data/examples.jsonl) holds four smoke-test tasks: the trio from the project brief, a
retrieval-QA trio with an annotated question span, a 4-condition tool-output case (`injection_polite` shows
arbitrary condition names), and a pair. They exist to exercise the pipeline, not as a dataset.

---

## 4. What is extracted, and what it means

### Two passes per variant

For each (sample, condition), the **same token IDs** go through:

1. **An instrumented prefill pass** (`tracing.run_traced_forward`): one forward pass over the prompt with
   `use_cache=False` and `output_router_logits=True`, with forward hooks attached. **All traces come from this
   pass.**
2. **Generation** (`model.generate`, greedy by default), with **no hooks attached**. Nothing is captured
   during decoding.

As a cross-check, each record stores `first_token_matches_prefill`: whether the first generated token equals the
argmax of the instrumented pass's last-position logits.

The rendered prompt text, the token IDs, and the attention mask are saved exactly as fed to the model. Nothing is
reconstructed from decoded text. Token IDs come from `tokenizer(text, add_special_tokens=False)` on the
chat-template output and are checked against `apply_chat_template(tokenize=True)`. A mismatch raises an error.

### Routing: verified against the Transformers OLMoE source

The relevant code is `OlmoeTopKRouter.forward` in transformers 5.x, or `OlmoeSparseMoeBlock.forward` in 4.x.
For every token, at every MoE layer:

```
router_logits = x @ W_router.T                  # [S, E]  pre-softmax ("router logits")
probs         = softmax(router_logits.float())  # [S, E]  fp32
w, ids        = topk(probs, k)                  # [S, k]  sorted by descending prob
if norm_topk_prob: w = w / w.sum(-1)            # OLMoE-1B-7B-0125: norm_topk_prob = False
w             = w.to(model_dtype)               # ← these weights multiply the expert outputs
```

* `output_router_logits=True` returns **pre-softmax logits** only (`outputs.router_logits`, one `[B·S, E]`
  tensor per layer). It does not return the selected experts or their weights.
* For OLMoE-1B-7B-0125, `norm_topk_prob=False`. **The dispatched weights are the raw softmax probabilities of
  the selected experts. They are not renormalized and sum to less than 1** (≈0.3 for the tiny test model). The
  stored `topk_weights` are exactly those dispatched values, losslessly upcast to fp32. `trace.json` records
  `norm_topk_prob` and the dtype of the dispatched weights.
* **transformers ≥ 5**: the router module returns `(logits, weights, ids)`. A forward hook captures the module's
  own output (`routing.source = "router_module_output"`). We also recompute softmax → top-k from the logits and
  raise `RoutingVerificationError` if they disagree.
* **transformers 4.x**: the router is an `nn.Linear` that returns logits only. ids and weights are recomputed with
  the identical algorithm (`routing.source = "recomputed_from_logits"`).
* Every traced forward pass checks that the hooked logits equal `outputs.router_logits`, that every shape is
  `[S, E]` / `[S, k]`, that the selected ids are a valid top-k, and that no id repeats within a token.
* `verify` (and the tests) recompute the MoE block output **by hand** from the captured `router_input`, ids,
  weights, and the expert parameters, and require it to match the captured `moe_output`. This shows the stored
  weights are the ones actually used, whichever experts kernel ran (v5 defaults to `grouped_mm`). A
  negative-control test shows that renormalized weights do *not* match.

| Tensor (`routing.safetensors`) | Shape | Default | Meaning |
|---|---|---|---|
| `topk_ids` | `[L, S, k]` int64 | saved | selected experts, descending probability |
| `topk_weights` | `[L, S, k]` fp32 | saved | dispatched weights (see above) |
| `router_logits` | `[L, S, E]` fp32 | `trace.routing.save_router_logits` | raw pre-softmax logits |
| `router_probs` | `[L, S, E]` fp32 | `trace.routing.save_router_probs` | softmax over all experts |

`L`, `E`, and `k` are read from the loaded model and checked against its module structure. They are never
hardcoded. For OLMoE-1B-7B-0125 they are 16, 64, and 8 (confirmed from the Hub config at revision `b89a7c4`).

### Hidden-state activations

Per decoder layer ℓ, captured with forward hooks:

| Type | Definition | Hook |
|---|---|---|
| `resid_pre` | residual stream entering layer ℓ (= `output_hidden_states[ℓ]`) | pre-hook on `layers[ℓ]` |
| `attn_input` | `input_layernorm(resid_pre)`, the exact input to q/k/v | pre-hook on `self_attn` |
| `attn_z` | per-head attention outputs Σ_j A_h[i,j]·v_h[j], concatenated (width `heads × head_dim`; head h = columns `[h·d, (h+1)·d)`) | pre-hook on `self_attn.o_proj` |
| `attn_output` | `o_proj(attn_z)`, attention block output before the residual add | hook on `self_attn` |
| `resid_mid` | `resid_pre + attn_output`: after attention, **before** the MoE | pre-hook on `post_attention_layernorm` |
| `router_input` | `post_attention_layernorm(resid_mid)`, the exact input to the router and the experts | pre-hook on `mlp` |
| `moe_output` | MoE block output (Σ_j w_j · expert_j), before the residual add | hook on `mlp` |
| `resid_post` | `resid_mid + moe_output` (= `resid_pre[ℓ+1]`), **after** the MoE | hook on `layers[ℓ]` |

These identities are tested exactly on the tiny models and checked by `verify` on the real models.

### Attention patterns (`trace.attention`)

With `trace.attention.save_patterns: true`, every selected layer stores
`A = softmax(q kᵀ · scale + mask)` with dims `[head, query_position, key_position]` in
`attention.safetensors` (`attn_pattern.layer<NNN>`, plus `query_positions`). Query rows follow
`trace.attention.query_positions` (default `all`); all key columns are always kept.

* SDPA / flash kernels never materialise the pattern, so the **traced pass temporarily switches the model to eager
  attention** and switches back afterwards (generation keeps the fast kernel). `trace.json` records
  `attn_implementation: eager`.
* Eager attention computes the softmax in fp32 and casts it to the model dtype before multiplying by V. Storing as
  `bfloat16` (the default) therefore keeps exactly the values the bf16 model used.
* Masked entries are exactly 0: future keys and, on Mellum's `sliding_attention` layers, keys ≥ 1024 positions
  back. Each layer's `layer_type` is in `trace.json`.
* GQA: query head h reads KV head `h // (num_heads / num_kv_heads)` (Mellum: 8 query heads per KV head).
* `verify` recomputes `attn_z` by hand as `A_h @ v_proj(attn_input)` from the stored patterns and requires it to
  match the captured `attn_z`. This shows the stored patterns are the ones applied to the values.
* Load with `load_trace(dir).attention[layer]` (or `Experiment.load(record, attention=True)`).
* Size: `layers × heads × Q × S × 2` bytes. For Mellum with all rows that is ≈ 1.8 KB × S² per variant (40 MB at
  150 tokens). For long prompts, restrict `query_positions` (e.g. `last_k:32`) or `layers`.

Defaults: `resid_mid, router_input, moe_output, resid_post` at layers `[0, 4, 8, 12, -1]`, **final prompt token
only**. Set `trace.activations.positions` to `all`, `last_k:N`, or an explicit index list, and `layers: all`, for
small experiments ([`configs/olmoe_full_capture.yaml`](configs/olmoe_full_capture.yaml)). Hooks slice the
requested positions and move them to CPU inside the hook, so no GPU tensors are retained. Every hook is removed in
a `finally`/context-manager exit, even when the forward pass raises. A test checks for leftover hooks.

Disk per variant: routing ≈ 1.5 KB/token (+ 4 KB/token per full logits/probs tensor). Default activations ≈ 160 KB
total. Full capture ≈ 650 KB **per token**.

---

## 5. Token alignment

Clean and injected prompts have different lengths, so token position *i* in one prompt is generally not
equivalent to position *i* in another. `moe_interp.alignment` pairs positions **only when the token IDs are
identical**. It always returns the original indices of both sequences, and it returns `ok=False` with a reason
otherwise.

| `--alignment` | Pairs | Typical use |
|---|---|---|
| `common_suffix` (default) | the longest identical token suffix | identical instruction (+ assistant header) after different document contents |
| `span` (`--span NAME`) | a dataset-annotated span; fails if it tokenizes differently in context | an instruction in the middle of the prompt |
| `last_n` (`--last-n N`) | exactly the final N tokens; fails unless all N are identical | fixed-length comparisons |
| `common_prefix` | the longest identical prefix | **sanity check**: under causal attention these positions see identical inputs, so overlap ≈ 1 |

Unaligned pairs are listed in `metrics/alignment.csv` with a reason, excluded from every metric, and counted in
`metrics/summary.json`. Note that the common suffix always includes the chat template's `\n<|assistant|>\n`
tokens. Use a span to restrict to the instruction text.

---

## 6. Metrics

All metrics are computed from saved traces (`analysis.py`, using the pure functions in `metrics.py`). Every
non-reference condition is compared against `--reference` (default `clean`).

| | Definition | Table |
|---|---|---|
| **A. Expert overlap** | `|top-k(ref) ∩ top-k(cond)| / k` per aligned token, averaged per layer | `overlap.csv`, `overlap_summary.csv` |
| **B. Selection frequency** | per (layer, expert): count / (tokens · k), i.e. the share of top-k slots (rows sum to 1; × k = share of tokens) | `expert_freq.csv` (all prompt tokens), `expert_freq_aligned.csv` |
| **C. Routing divergence** | Jensen–Shannon divergence (base 2, ∈ [0, 1]) between the selection distributions of ref vs cond on aligned tokens. Zeros are handled exactly (0·log 0 = 0), no ε needed. | `jsd.csv` (per sample), `jsd_pooled.csv` (counts pooled across samples) |
| **D. Activation similarity** | cosine and `‖a−b‖/‖a‖` (ref = a) at aligned positions captured in both traces | `activation_similarity*.csv` |
| **E. Control-adjusted shift** | `Δ = d(ref, treatment) − d(ref, control)` with d ∈ {1 − overlap, JSD, 1 − cosine, rel-L2}, computed on the **same reference tokens** (3-way intersection of alignments) | `control_delta*.csv` |

Per-sample JSD on a handful of tokens is biased upward, because empirical distributions from few samples differ
even under identical routing. Compare against the control condition (E) or use `jsd_pooled`. Summaries report
mean, std, s.e.m., and n across samples. `generation_summary.csv` reports per-condition attack-target hit rate, EOS
rate, and prefill/generation first-token agreement.

Plots (`plots/`): `overlap_by_layer`, `jsd_by_layer`, `activation_cosine_by_layer`, `activation_rel_l2_by_layer`,
`control_delta`, `control_comparison_routing`, `expert_freq_heatmaps`, `expert_freq_diff_<condition>`. They are
PNG files, plus PDF with `--pdf`. Missing or empty tables are skipped, never filled in. Figures from the tiny debug
model are labelled as such.

[`notebooks/analysis_demo.ipynb`](notebooks/analysis_demo.ipynb) loads a results directory on CPU, shows the
tables and figures, and demonstrates working with raw traces. It calls the same package functions as the CLI.

---

## 7. Output layout

```
<output>/                       # --output; any path (local disk, Drive, scratch)
  config.yaml                   # fully resolved config
  metadata.json                 # versions, GPU, model + resolved commit, architecture, dataset sha256, code git commit
  generations.jsonl             # one record per (sample, condition), appended + flushed as each finishes
  traces/<sample>/<condition>/
    trace.json                  # input_ids, prompt_text, spans, shapes/dtypes/dims/meaning of every tensor, checks
    routing.safetensors
    activations.safetensors     # "<type>.layer<NNN>" -> [P, H]; "positions" -> [P]
  metrics/  plots/
```

* A `generations.jsonl` record contains `sample_id`, `condition`, `metadata`, `evaluation`, `messages`,
  `prompt_text`, `input_ids`, `attention_mask`, message and span token ranges, `response_text`
  (and `response_text_raw` with special tokens), `response_ids`, `finish_reason`, `trace_dir`, `routing_source`,
  `first_token_matches_prefill`, `timing`, and `peak_gpu_memory_gb`. The effective generation config is in
  `metadata.json`.
* Directory names: names made only of `[A-Za-z0-9._-]` are used as-is. Anything else is sanitized and gets an
  8-character hash suffix (`t2/with slash` → `t2_with_slash-<8 hex chars>`), so the mapping is deterministic and
  collision-free.
* Tensors use **safetensors** (no pickle). No model objects are stored. `results/` is gitignored.
* Load traces with `moe_interp.storage.load_trace(dir)`, or the whole experiment with
  `moe_interp.analysis.Experiment.open(dir)`.
* An existing output directory is never overwritten: `run` refuses unless `--resume` is given. `--resume` also
  refuses if the model, prompt, generation, trace, or seed settings differ from the saved `config.yaml`.

---

## 8. Configuration and CLI

Configs are YAML ([`configs/olmoe.yaml`](configs/olmoe.yaml) documents every key). Unknown keys are rejected,
so typos don't silently fall back to defaults. Any key can be overridden from the command line:

```bash
python -m moe_interp.cli run --config configs/olmoe.yaml --output results/exp1 \
    --data my_data.jsonl --conditions clean,injection --max-new-tokens 64 \
    --set trace.activations.positions=all --set trace.activations.layers='[0, 7, 15]' \
    --set trace.routing.save_router_logits=true
```

Key settings: `model.{model_id, revision, local_path, device, dtype, attn_implementation}`,
`generation.{max_new_tokens=128, do_sample=false, ...}`, `trace.routing.save_router_{logits,probs}`,
`trace.activations.{types, layers, positions, dtype}`, `data.{path, sample_ids, task_types, conditions, limit}`,
`analysis.{reference, control, alignment, last_n, span}`.

* **dtype** `auto`: bf16 on CUDA GPUs with native bf16 (Ampere+), fp16 on older CUDA (with a warning), bf16 on
  MPS, fp32 on CPU.
* **device** `auto`: cuda → mps → cpu. On CUDA, weights are loaded directly onto the GPU (`device_map`), which
  avoids a 14 GB CPU-RAM spike on Colab.
* **revision** is pinned to a verified Hub commit in `configs/olmoe.yaml`. The resolved commit hash is recorded in
  `metadata.json`.
* **Determinism**: greedy decoding, `torch.manual_seed(seed)` before each pass, batch size 1, no padding.

Every command and argument is listed in [`docs/cli.md`](docs/cli.md), generated from `--help`.

---

## 9. Tests and verification status

```bash
python -m pytest                                            # CPU, ~5 s, no downloads
MOE_INTERP_RUN_INTEGRATION=1 python -m pytest tests/integration -v -s   # GPU, real OLMoE
```

The CPU suite (77 tests) covers dataset parsing and validation, prompt rendering (including dropped-role
detection and span → token mapping), alignment, every metric against hand-computed values, safetensors
round-trips, config validation, and analysis on synthetic traces with known answers. The tracing tests run against
a **tiny randomly initialized `OlmoeForCausalLM` and `MellumForCausalLM` built from the real Transformers classes**
(the Mellum one has GQA, sliding-window + full layers, and renormalised top-k), so the hooks are exercised on the
real module structure. `tests/test_mellum_attention.py` covers attention-pattern capture, masks, the eager
switch-back, attention identities, and storage. They check routing semantics, hand recomputation of the MoE output,
residual identities, hook removal (including on exceptions), determinism, and the end-to-end run →
analyze → plot path, including `--resume`.

**What has been verified (on a 16 GB Apple-silicon laptop, CPU only):**

* The full CPU suite passes with **transformers 5.22.1** (router returns `(logits, weights, ids)`) and
  **transformers 4.57.1** (router is `nn.Linear`), on torch 2.15.0.
* The real OLMoE-1B-7B-0125-Instruct **config and tokenizer** (Hub revision `b89a7c4`): 16 layers, 64
  experts, top-8, `norm_topk_prob=False`, hidden size 2048. The chat template renders `system`/`user`/`assistant`
  only, and token IDs are identical under both transformers versions.
* The complete CLI pipeline (`run --analyze`, `verify`, `plot`) and the analysis notebook, using the real tokenizer
  with a tiny random model.

**Mellum2.1-12B-A2.5B-Thinking, verified on an NVIDIA DGX Spark (GB10, torch 2.14.1+cu130, transformers 5.22.1,
bf16):** `verify --config configs/mellum.yaml` passes all 21 checks. The MoE hand recompute has a relative error of
4.9e-3, the attention-output recompute from stored patterns 1.7e-3, and `resid_pre` vs `output_hidden_states` is
exactly 0. Masked attention entries are exactly 0, weight sums are 1 ± 2e-3 (bf16), and the first generated token
matches the prefill.

**OLMoE: not yet verified on real weights, because the 7B model was never loaded** (≈14 GB in bf16 does not fit next to the OS in 16 GB
of RAM): every behavior on the real weights, including bf16 tolerances in `verify`, the actual generations, GPU
memory and timing, and first-token agreement under bf16. Run `verify` and the integration suite on the GPU first.
Every check should pass.

---

## 10. Limitations and extension points

* **Prefill only.** Routing and activations are captured for prompt tokens, not for generated tokens. To trace
  responses, run `run_traced_forward` on `prompt_ids + response_ids`, which teacher-forces the response through
  the same hooks. That is a small addition and was not built yet.
* **Batch size 1** by design: no padding, so token positions are unambiguous.
* **OLMoE / Mellum layouts only**: discovery expects `model.model.layers[i].{self_attn.o_proj, input_layernorm,
  mlp.gate, mlp.experts, post_attention_layernorm}` with a sparse MoE in every layer, and `get_moe_layout` fails
  loudly otherwise. Other MoE families (Mixtral, Qwen-MoE) have similar layouts but different routers (e.g. shared
  experts), so they need their own verification.
* **Top-k ties**: in the 4.x path, ids are recomputed on CPU, so an exact tie in router probability could break
  differently than on the GPU. With continuous logits this is practically impossible, and `verify` checks the
  result end to end.
* **Attack success** is only a substring check. Plug in a real judge by post-processing `generations.jsonl`.
* **Interventions**: activation patching and routing interventions are not implemented. `TraceRecorder` in
  `tracing.py` is the place to add them: its hooks already sit at every point where an intervention would go, and
  returning a value from a forward hook replaces that module's output.

### Package map

```
moe_interp/
  config.py     dataclass configs, YAML load, --set overrides
  dataset.py    JSONL loading/validation/filtering (no torch)
  prompts.py    chat-template rendering, token ids, message/span token ranges
  model.py      device/dtype resolution, model loading, MoE structure discovery, tiny debug model
  tracing.py    hook-based routing + activation capture with built-in verification
  verify.py     model-level consistency checks (CLI `verify`, integration test)
  runner.py     dataset → generations.jsonl + traces
  storage.py    on-disk layout, safetensors I/O, safe names
  alignment.py  token alignment
  metrics.py    pure metric functions
  analysis.py   traces → metric tables
  plotting.py   metric tables → figures
  env.py, cli.py
```
