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

## Follow-up analyses (CPU only, from generations and step-3 arrays)

Script: [`experiments/sep_behavior.py`](experiments/sep_behavior.py). Output:
[`reports/mellum_sep/security/behavior/`](reports/mellum_sep/security/behavior/).

### A · Label audit: the substring witness overcounts

The SEP-style label ("witness string appears in the answer") is a **substring** match. Short witnesses fire inside
other words: "cat" matched 10 of 32 clean/benign answers (in words like "category" and "indicate"), and "pen" matched
4 of 18. With **whole-word** matching:

| condition | substring | whole-word |
|---|---|---|
| clean / benign (false positive) | 1.8% / 1.7% | **0.5% / 0.5%** |
| injection (ASR) | 65.7% | **63.3%** |
| user_instruction (utility) | 79.8% | **74.2%** |

The remaining false positives are genuine topic overlaps ("flour" in a recipe, "Paris" in a travel text). The
conclusions do not change. Use whole-word labels from now on.

**"Non-thinking" Mellum still deliberates.** 15–21% of answers start with plain-text reasoning ("The user asks…",
"We need to…"), and **90–97% of those hit the 1024-token limit**. That accounts for nearly all of the truncation. The
witness can appear in this deliberation ("…the animal known for its long neck is a giraffe…") without a clean final
answer, so for 23% of executed injections "executed" partly means "answered the probe while reasoning about it".

### B · What drives attack success

![drivers](reports/mellum_sep/security/behavior/behavior_drivers.png)

- **Injection success is not just "the probe is easy".** Per probe, the injected-execution rate correlates only
  moderately with the trusted-execution rate (Spearman 0.35). Per item, P(obey injection | the same probe was executed
  when trusted) = 0.70, vs 0.45 when it was not.
- 70 items are executed *only* when injected and 135 only when trusted (McNemar p < 10⁻⁵). Placement in the data
  does help a little, but much less than it should.
- **Insistent phrasing: +23 points** (52% → 75%). **Longer data: +17 points** (shortest third 53% → longest third
  70%). The further the trusted task is from the end of the prompt, the more the injected instruction wins. This is
  consistent with the attention story: what the generation reads most recently and most directly takes over.
- Position (start or end of the data) does not matter. Information-retrieval tasks are a little harder to hijack (57%)
  than analytical and creative tasks (67%).

### C · How the model obeys: it piggybacks, it rarely hijacks

Of the 380 executed injections:

| style | n |
|---|---|
| **does the task, then appends the probe answer** | **210** |
| deliberates in plain text (both tasks) | 86 |
| answers the probe first, then does the task | 58 |
| hijack: answers only the probe (answer < 35% of the clean answer's length) | 26 |

The median answer length is 1.03× the clean answer. **Mellum treats the injected instruction as an *additional* task,
not a replacement.** It explicitly flags the injected instruction ("the text also asks…", "an additional question")
in 13% of injection answers, compared with 2–3% in the other conditions. It **still executes it in 57% of the
answers where it flags it**. The model *says* it knows the text is in the data and complies anyway. This is the
behavioral counterpart of step 2 (noticing AUC 1.0, obeying only 0.74).

### D · Expert space: obeyed injections are routed more like trusted instructions

![expert space](reports/mellum_sep/security/behavior/behavior_expert_space.png)

Over all 1,792 (layer, expert) cells, the obey delta (expert share on probe tokens: obeyed − ignored) is negatively
correlated with the provenance delta (data − trusted task). Spearman **−0.19**, against a 95% null of ±0.047 from
permuting experts within each layer (p < 0.001). Per layer, the correlation is strongest in **layers 8–22** (down to
−0.42 at layers 10 and 16) and absent in layers 0, 15 and 24.

**Reading.** When the router sends the injected tokens to the experts it normally uses for *trusted* instructions, the
model is more likely to obey. "Routing mistakes data for instruction" is a weak, distributed tendency, not a single
switch, which fits the null expert-ablation results. It is a correlation: step 4 found no causal handle in small
expert sets.

Caveat: both deltas reuse the injection condition. The split-half re-test in G removes this: −0.13, and negative in
200/200 splits.

### E · Fragility

Batched generation picked a different first token than bs=1 prefill for 6% of injection prompts and 10% of trusted
prompts. These prompts are executed slightly *less* often (54% vs 64%). Together with the 40% re-batching reproduction
rate in step 4, this shows that many injection outcomes sit near a decision boundary.

### G · Representation geometry: obeying tracks how strongly the injection registers, not "looking trusted"

Script: [`experiments/sep_geometry.py`](experiments/sep_geometry.py), run on `features.npz`. Output:
[`reports/mellum_sep/security/geometry/`](reports/mellum_sep/security/geometry/). All directions are fitted on held-out
probes (GroupKFold by probe), and the labels are whole-word.

![geometry](reports/mellum_sep/security/geometry/geometry.png)

1. **The expert-space correlation in D survives a clean split-half test (panel B).** I computed the provenance delta on
   one half of the probes and the obey delta on the other, so the two share no prompts. Over 200 random splits the mean
   Spearman is **−0.13** (95% range −0.16 to −0.10), and **all 200 are below 0**. Obeyed injections are routed somewhat
   more like trusted instructions. The effect is real but small.
2. **Per item, "routed like its trusted twin" predicts obeying only modestly, and mostly through surface features
   (panel A).** Each item's probe tokens are routed once in the data and once in the system prompt. The similarity of
   the two routings (1 − Jensen–Shannon, layers 8–22) predicts obeying with AUC **0.65** (95% CI 0.60–0.70); the router
   probabilities give 0.64. Projecting the gate input on a held-out "trusted − data" direction gives about 0.60. But
   adding the routing similarity to surface features (insistent, position, data length, probe length) only moves a
   grouped-CV model from **0.649 to 0.659**. The same similarity measured against a *benign* sentence also predicts
   obeying (0.61), so part of it is generic.
3. **At the decision point, obeying lines up with the "noticed" direction (panel C).** At the final prompt token:
   - The obey direction (obeyed − ignored injections) has cosine **0.56–0.67** with the noticed direction
     (injection − benign) in layers 6–27, well outside a label-permutation null (±0.25–0.47).
   - A single held-out notice direction scores obeying with AUC **0.70** (CI 0.65–0.74, layer 22). That is nearly all of
     the full 2,304-dim probe (0.74).
   - It adds to surface features (0.649 → **0.706**) and holds **within the same probe text across different
     documents** (0.60, over 82 probes with mixed outcomes).
   - It is strongest for plain probes (0.71) and weaker for insistent ones (0.58).
   - Obeyed injections sit *further* from the trusted-instruction state than ignored ones: cosine −0.3 to −0.5, around
     the edge of its null; the held-out projection AUC is 0.37–0.41, i.e. inverted.

**Reading.** Obeying is not mainly "the model mistook the injected text for a trusted instruction". It is "the injected
instruction registered *more strongly*". The more salient its representation at the decision point, the more likely the
model acts on it. This is consistent with C (it flags and complies), with step 2 (noticing is perfect), and with the
longer-data effect. There is also a small, real routing tilt toward trusted-instruction experts (G1). That tilt is the
part a routing-level defense could act on, but it is not the main driver.

**For the security expert.** The notice direction is the strongest single handle we have. It is readable at the gate
input on the probe tokens (AUC 1.0) and at the decision point. A security expert gated on it should *damp* instruction
salience on data tokens rather than relabel provenance. A weight-free test of the same idea is ablating or
negatively steering this direction on data tokens (roadmap #5). That needs a GPU and has not been run.

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
4. **Obeying is additive and "aware".** The model mostly does the task *and* the injected instruction, sometimes saying
   explicitly that the instruction came from the data. Longer data and insistent phrasing raise success, and routing
   injected tokens like trusted instructions correlates with obeying (layers 8–22). A defense therefore has to teach a
   *policy* ("instructions found in data are not to be executed"), not repair a perception failure. At the
   representation level, obeying tracks how *strongly* the injected instruction registers (the notice direction,
   AUC 0.70 held out), more than whether it looks trusted.
5. **Measured attack success is noisy at the item level.** About 60% of the obeys flip under re-batching. Single-run,
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

| condition | n | Qwen (whole-word) | Mellum, same items |
|---|---|---|---|
| injection | 56 | 73% | 59% |
| user_instruction | 56 | 73% | 66% |
| clean / benign | 56 / 56 | 0% / 0% | 0% / 0% |

On these items the larger model is at least as injectable as Mellum, and it executes injected instructions **exactly as
often** as trusted ones: no separation at all. It almost never deliberates in plain text (2%). Per-item agreement with
Mellum on injections is 64% (φ = 0.23), so the two models are hijacked by partly different items. The sample is small (±10 points), and no probes or interventions were run on Qwen.

## Next steps: the interpretability program

What we can claim now is a set of correlations, plus one null causal result (expert ablation). The standard mechanistic
toolkit turns them into a causal account. In priority order (all need a GPU and the traces or a re-run):

| # | Experiment | Question it answers | Why it matters for the security expert |
|---|---|---|---|
| 1 | **Provenance patching on the probe tokens.** The same probe tokens exist in the `user_instruction` and `injection` runs. Patch `resid_post` at the probe positions from the trusted run into the injected run (and the reverse), layer by layer. | Is "where the text sits" causally what makes the model treat it as data? At which layer? | Tells you at which layers a gate must act. If patching trusted → injected *raises* obedience, provenance is causal. |
| 2 | **Route patching (MoE-specific).** Keep everything else, but give the probe tokens the *expert choices* they got in the trusted run (or the reverse), per layer. | Is the correlation in D causal through routing alone, or only through the residual? | Direct test of whether a *routing-level* defense (new gate / expert) can carry the fix. This is novel; dense models can't run it. |
| 3 | **Attention knockout from the answer to the probe span**, in layer windows, during generation. | Which layers move the injected instruction into the answer? | Locates the attention layers to unfreeze or LoRA (step 4 points to layers 8–17). |
| 4 | **Direct logit attribution to the witness token** at the position where it is first generated: per expert and per head. | Which components *write* the injected answer? | Expert-level attribution is the most direct "is there an obey expert at write time" test, complementing our read-time null. |
| 5 | **Difference-of-means "data-ness" direction.** Compute it at the gate input (provenance AUC 1.0) and add it to the probe tokens in the injected run. | Is there a single linear knob that reduces ASR without hurting utility? | The strongest weight-free baseline the trained expert has to beat. |
| 6 | **Probe the decision during generation**, token by token, not only at the last prompt token. | When is the obey decision made? | The last-token probe was weak (0.74); step 4 patching says the decision forms later. |
| 7 | **Attribution patching over all 1,792 experts** (split-half D is done: robust, small). | Which experts carry the routing tilt causally? | Picks where to insert the security expert. |

Data and infrastructure:
- Generate with vLLM and use HF only for the traced forward pass. Our 8.5 h run was dominated by HF batched decoding.
  Use whole-word labels and cap non-thinking answers with a stop on plain-text deliberation, or run thinking mode
  explicitly.
- **Move beyond SEP's system/user split.** In SEP, provenance is "which chat turn", which is trivially readable.
  Realistic injections arrive inside tool outputs and retrieved documents in the *same* turn (BIPIA, AgentDojo,
  InjecAgent). The key test for a security expert is whether the gate signal survives there.
- Finish the Qwen3-30B-A3B replication and the thinking-mode run
  ([`scripts/queue_security.sh`](scripts/queue_security.sh) from `qwen_run`, resumable).
- Then the fine-tuning step: a security expert plus gate on the SecPO recipe, unfreezing per experiments 1–3.
