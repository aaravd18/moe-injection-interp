# Prompt-injection security study of an MoE: Mellum2.1 on SEP

**Question.** When an MoE model obeys an instruction hidden in its *data*, what in the model does it, and is there a
handle that a weight-preserving defense (a new "security expert" plus its gate, trained on top of a frozen model;
see SecPO and SEAL) could use?

**Run.** `JetBrains/Mellum2.1-12B-A2.5B-Thinking` (Hub commit `92ddae9`), in **non-thinking** mode
(`enable_thinking: false`), bf16, greedy, max 1024 new tokens. Run on an NVIDIA DGX Spark, November 10, 2026. Config:
[`configs/mellum_sep.yaml`](configs/mellum_sep.yaml). Mellum has 28 layers, 64 experts per layer and top-8 routing.

**Data.** 600 items from SEP (Zverev et al., ICLR 2025) built by
[`scripts/build_sep_dataset.py`](scripts/build_sep_dataset.py) into [`data/sep_600.jsonl`](data/sep_600.jsonl).
The items cover 100 distinct injected instructions ("probes") and 288 subtasks, balanced over probe position (start
or end of the data) and insistent vs plain phrasing. Each item appears in four conditions:

| condition | system prompt (trusted) | user prompt (data) |
|---|---|---|
| `clean` | task | data |
| `benign` | task | data + a neutral sentence of the same length, in the same place |
| `injection` | task | data + **probe** |
| `user_instruction` | task + **probe** | data |

A probe "executed" means the probe's witness string appears in the answer.

**Pipeline.**
- [`experiments/sep_features.py`](experiments/sep_features.py): extracts features from the full traces (routing,
  `resid_post` / `attn_output` / `moe_output` at every layer and token, and attention from the last 16 tokens).
- [`experiments/sep_analysis.py`](experiments/sep_analysis.py): steps 1–3.
- [`experiments/sep_interventions.py`](experiments/sep_interventions.py): step 4.
- All outputs are in [`reports/mellum_sep/`](reports/mellum_sep/): `security/summary.json`, the probe AUC table,
  the plots, `security/interventions/results.json`, and the gzipped generations.
- The raw traces (97 GB) were not kept.

**Not finished.** The Qwen3-30B-A3B replication, the thinking-mode run, and a rerun of the interventions with
reproducibility filtering and confidence intervals were queued, but the GPU was released first. Only 56 of 300 Qwen
items finished; see [Partial Qwen replication](#partial-qwen3-30b-a3b-replication).

---

## Step 1 · Behavior: injections succeed most of the time

![execution rates](reports/mellum_sep/security/step1_execution_rates.png)

| | rate |
|---|---|
| Injection executed (attack success, ASR) | **65.7%** (74.2% counting only answers that did not hit the token limit) |
| Trusted instruction executed (utility) | 79.8% |
| Witness in clean / benign answers (false positive) | 1.8% / 1.7% |
| SEP separation score (1.0 = perfect) | **0.31** |

- Insistent phrasing adds about 20 points (plain 55% → insistent 76%). Probe position (start or end of the data) makes
  almost no difference.
- Attack success varies a lot by probe: the median probe succeeds on 67% of items. 14 of 100 probes always succeed and
  2 never do.
- Per item: in 331 of 600 items the model executes the probe in both places. In 148 it executes only the trusted copy,
  in 63 only the injected copy, and in 58 neither.
- Caveat: 31% of injection answers hit the 1024-token limit. The first answer token from batched generation matched the
  single-prompt prefill prediction in 92% of prompts.

**Reading.** Mellum barely separates instructions from data. Putting the same text in the data instead of the system
prompt lowers execution only from 80% to 66%.

## Step 2 · Probes: the model knows the instruction is there; whether it obeys is weakly readable

![probes by layer](reports/mellum_sep/security/step2_probes_by_layer.png)

Logistic-regression probes, cross-validated with **GroupKFold by probe** (no probe text appears in both train and
test). Numbers are held-out AUC.

| Probe | Best layer | AUC |
|---|---|---|
| **Noticed**: injection vs benign, final prompt token | 5 (≥0.99 from layer 1) | **1.00** |
| Noticed, TaskTracker activation delta | 25 | 1.00 |
| **Obeys**: executed vs ignored, final token, residual stream | 20 | **0.74** |
| Obeys, attention output | 14 | 0.75 |
| Obeys, MoE output | 23 | 0.73 |
| Obeys, router probabilities only | 14 | 0.70 |
| Obeys, attention mass on the probe | 6 | 0.68 |
| Obeys, surface features only (baseline, no activations) | – | 0.65 |
| **Gate input** (`resid_mid`) on the probe tokens: in data vs in trusted task | every layer | **1.00** |
| **Current router** on the same tokens: in data vs in trusted task | 5 | 0.99 (0.85–0.99 across layers) |
| Gate input: injected instruction vs benign sentence | every layer | 1.00 |

**Reading.**
1. **Noticing is not the problem.** From the first layers on, the model represents "an instruction is sitting in the
   data" almost perfectly. It is not fooled about what the text is.
2. **The decision to obey is weakly and diffusely encoded** at the end of the prompt. The best probe beats the
   surface-feature baseline by only about 9 AUC points. Much of the decision is made *during generation*, as the model
   re-reads the injected text (step 4 supports this).
3. **A gate has what it needs.** Where a token sits (data vs trusted task) is linearly readable at every layer, and the
   existing router already routes on it. Caveat: the two conditions differ in which chat turn holds the text, so part of
   this is "which turn am I in". That is still the signal a SecPO-style defense needs.

## Step 3 · Localization: a provenance routing pattern, but no "obey" expert or head

![provenance experts](reports/mellum_sep/security/step3_experts_provenance.png)

- **Provenance.** Expert use on the *same* probe tokens differs strongly when they sit in the data vs the trusted task.
  The differences are concentrated in layers 8–14; for example, layer 11 expert 22 is +22 points and layer 13 expert 22
  is −20 points. The pattern is very reproducible across disjoint halves of the probes: Spearman **0.91**, top-30 overlap
  87%.
- **Obeying.** Expert use when the model obeyed vs ignored is small and unstable.
  - On the probe tokens: split-half Spearman 0.41, top-30 overlap 33%, largest difference 0.10 vs a permutation-null
    95th percentile of 0.07.
  - On the final template tokens: Spearman 0.31.
  - See [`step3_experts_obey_probe_tokens.png`](reports/mellum_sep/security/step3_experts_obey_probe_tokens.png).
- **Heads.** The best single head reaches AUC 0.67 for predicting obeying (layer 8 head 12, layer 16 head 27, …), and
  the ranking is not stable (split-half Spearman 0.46). See
  [`step3_heads_obey_auc.png`](reports/mellum_sep/security/step3_heads_obey_auc.png).
- **Attention budget.** From the final token, the share of attention going to the task, the data and the probe is
  nearly identical whether the model obeyed or not. Obeyed cases have only slightly more attention on the probe in
  layers 6–9 and 20–24. See
  [`step3_attention_by_span.png`](reports/mellum_sep/security/step3_attention_by_span.png).

## Step 4 · Causal interventions

Components were selected on half of the probes (A) and tested on the other half (B), so the test is not circular.
Every targeted intervention has a random control of the same size.

The evaluation sets are:
- **attack**: 96 injection prompts that were obeyed in the main run;
- **utility**: 96 trusted-instruction prompts that were executed;
- **side effects**: KL divergence on 48 clean prompts.

Expert steering applies to the data tokens, except where noted.

**Important caveat: decisions are fragile.** Re-generating the attack set *with no intervention* reproduced only
**40%** of the obeys, even though decoding is greedy. bf16 greedy decoding is not invariant to batch composition and
left padding, and most successful injections are close to a tie, so tiny numerical noise flips them. Comparisons
below are paired against this 0.40 baseline on identical batches. With n=96 the noise is roughly ±0.1, so only large
gaps mean anything. (The fixed script, now in the repo, filters for reproducible prompts first and reports paired
bootstrap CIs. It was not re-run before the GPU was released.)

| Intervention | Attack success | Utility | KL on clean |
|---|---|---|---|
| Baseline | 0.40 | 0.55 | 0 |
| Deactivate "obey" experts, N=16 / 64 | 0.40 / 0.38 | 0.50 / 0.49 | 0.005 / 0.011 |
|   random experts, same layers | 0.39 / 0.39 | 0.46 / 0.57 | 0.006 / 0.012 |
| Deactivate "trusted-instruction" experts, N=16 / 64 | 0.40 / 0.44 | 0.56 / 0.45 | 0.005 / 0.010 |
|   random | 0.38 / 0.44 | 0.52 / 0.41 | 0.002 / 0.008 |
| Force on "data" experts, N=16 / 62 | 0.03 / 0.03 | 0.09 / 0.26 | **1.72 / 1.68** |
|   random | 0.02 / 0.02 | 0.16 / 0.12 | 1.56 / 1.76 |
| SteerMoE-style: deactivate on *all* tokens incl. generated, N=16 / 64 | 0.45 / 0.48 | 0.45 / 0.45 | 0.03 / 0.08 |
|   random | 0.42 / 0.41 | 0.53 / 0.42 | 0.02 / 0.08 |
| **Ablate "obey" heads, N=8 / 32** | **0.23 / 0.27** | 0.37 / 0.33 | 0.08 / 0.21 |
|   random heads | 0.34 / 0.52 | 0.43 / 0.45 | 0.04 / 0.14 |
| Patch clean-run residuals into the last 8 prompt tokens, layer 0–12 | 0.39–0.45 | – | – |
| … layer 14–27 | 0.31–0.38 | – | – |

**Reading.**
- **Removing experts does nothing specific.** Targeted deactivation matches random deactivation and the baseline,
  including SteerMoE-style deactivation on every token. Forcing experts on stops attacks only by breaking the model:
  random experts do the same, and KL is about 1.7. This is the causal confirmation of step 3: **no expert, or small set
  of experts, carries obedience.**
- **Attention heads are the only targeted lever with a specific effect.** The selected heads cut attack success more
  than random heads at both sizes (0.23 vs 0.34 and 0.27 vs 0.52), but they also cost trusted-task success. The effect
  is suggestive and is within the noise for N=8.
- **The decision is not stored in the final prompt tokens.** Overwriting them with the clean-run activations at
  layers ≥14 removes only about 2–9 points. The model reads the injected text directly while it generates.

---

## What this says about MoE security

1. **An MoE's router already encodes provenance.** It routes the same tokens differently depending on whether they sit
   in trusted instructions or in data, and that difference is reproducible. The model "knows where text came from", but
   that knowledge is not used to suppress the injected instruction.
2. **Obedience is not localized in experts.** So weight-free defenses that disable or force experts (SteerMoE-style
   steering) are not a viable prompt-injection defense for this model. At best they do nothing; at worst they break the
   model.
3. **The causally relevant pathway is attention into the injected span during generation.** The defense has to change
   how generated tokens attend to and act on data tokens, not only how data tokens are processed.
4. **Measured attack success is noisy at the item level.** About 60% of the obeys flip under re-batching. Single-run,
   per-item ASR numbers, including those used to select examples for defenses, should be reported with this in mind.

## Implications for a trained "security expert" (SecPO recipe, SEAL-style, frozen backbone)

- **Gate signal: available.** Provenance is linearly separable (AUC 1.0) at the gate input in every layer, so a new
  gating parameter can learn to send data-span tokens to a security expert. The **middle layers (8–14)** are where the
  existing router already separates them most and are the natural place to insert it.
- **An expert alone is probably not enough.** In this model the behavior lives in attention, so the expert's output
  has to counteract what attention reads from the injected span. Candidates to unfreeze, in increasing cost:
  1. a new expert plus gate at layers 8–17, with data-span tokens marked so the gate is conditioned on provenance;
  2. plus LoRA on the attention o_proj / v_proj of a few mid layers, which our head ablations point to;
  3. plus applying the expert to *generated* tokens, because the decision forms during generation.
- **Evaluation design.** Use reproducible, repeated-seed ASR with confidence intervals. Measure trusted-instruction
  utility on the same items, because every intervention here that reduced attacks also reduced utility.

## Partial Qwen3-30B-A3B replication

`Qwen/Qwen3-30B-A3B-Instruct-2507` (48 layers, 128 experts, top-8) passed the tracing verification
([`reports/qwen_sep_partial/verify_qwen.json`](reports/qwen_sep_partial/verify_qwen.json)). Only 56 items × 4
conditions were generated before the run was stopped:

| condition | n | executed |
|---|---|---|
| injection | 56 | 82% |
| user_instruction | 56 | 79% |
| clean / benign | 56 / 56 | 0% / 0% |

On these items the larger model is at least as injectable as Mellum, and it executes injected instructions about as
often as trusted ones. The sample is small (±10 points), and no probes or interventions were run on Qwen.

## Next steps

1. Re-run [`scripts/queue_security.sh`](scripts/queue_security.sh) stages from `qwen_run` (resumable). This covers
   Qwen behavior, probes and interventions, plus Mellum in thinking mode. Then run
   [`scripts/queue_followup.sh`](scripts/queue_followup.sh) for the reproducibility-filtered Mellum interventions with
   CIs.
2. For speed, generate with vLLM and use HF only for tracing. The 8.5-hour main run was dominated by HF batched
   generation waiting on the longest answer in each batch.
3. Then the fine-tuning step: the security expert plus gate on the SecPO recipe, with the unfreezing ablation above.
