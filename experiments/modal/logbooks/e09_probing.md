# E09: linear probes for injection detection and obedience across security post-training

**Question:** How does security post-training change the internal representations associated with detecting and
following prompt injections? Four checkpoints, no retraining:
- **base:** OLMoE-1B-7B-0125-Instruct.
- **LoRA:** SecPO LoRA `secpo_v2` (E03).
- **E05:** the always-on security expert `secpo_se_v1`.
- **routed:** E08's routed security expert, main run, step 80 (copy init; not the §7 ablations).

**Run:** on 2026-11-23/24; capture and interventions on a RunPod A40, probes and judging on the laptop. Outputs
are in `outputs/e09_probing/`. All three experiments are complete. Experiment 3 (§4) covers the base model and the
routed model; LoRA and E05 were not intervened on.

**Experiment 3 in one line:** in the base model, subtracting 2σ of the layer-14 obedience direction cuts attack
success 0.49 → 0.34 on held-out test records, chosen on val in advance, with no utility cost. A random direction of
the same norm does not (0.46). In the routed model, pushing along its own susceptibility direction does not bring
attacks back. Details in §4.

## Result (experiments 1 and 2)

| test AUROC, best layer by val AUROC | base | LoRA | E05 | routed |
|---|--:|--:|--:|--:|
| injection detection | 0.995 (L6) | 0.991 (L2) | 0.998 (L3) | 0.984 (L2) |
| detection, last layer (L16) | 0.99 | 0.94 | 0.95 | 0.92 |
| susceptibility (base obeyed), best layer | **0.857 (L14)** | 0.753 (L9) | 0.706 (L7) | 0.720 (L6) |
| susceptibility, last layer (L16) | 0.86 | 0.63 | 0.63 | 0.61 |
| own-label obedience | 0.857 (= susceptibility) | 7 test positives: insufficient | 0 | 6 |
| base probe transferred, detection at L14 | – | 0.70 | 0.61 | 0.70 |

**Bottom line.**
- **Every model linearly encodes whether an injection is present**, almost perfectly from layer 3 to layer 10, and
  the base model's probe transfers there unchanged (cosine with the base direction ≥ 0.85 up to layer 4).
- **Post-training re-encodes detection in the late layers rather than removing it.**
  - In layers 13–16, each post-trained model's own probe still detects injections (AUROC 0.91–0.98).
  - The frozen base probe falls to 0.61–0.77, and the probe directions rotate away from the base's (cosine 0.11–0.40,
    against the base's own split-half reliability of 0.78–0.82).
  - E05, which adds an expert to every token, rotates the late-layer detection direction most (cosine 0.11–0.22).
- **The late-layer representation of "this prompt makes the base model obey" largely disappears after post-training.**
  - In the base model, susceptibility becomes more decodable with depth: 0.69 → 0.86, and 0.83 within a single attack
    type, so it is more than the attack type (attack-type-only baseline 0.67).
  - In all three post-trained models it peaks mid-network (0.75–0.77 at layers 4–9) and falls to 0.61–0.63 at the
    last layer, which is about the attack-type baseline. Within a single attack type it is 0.50–0.57, close to chance.
  - The three methods are statistically indistinguishable here, even though their outputs differ (E07/E08).
- **Detection and susceptibility use nearly orthogonal directions** in every model (cosine ≤ 0.23).
- **Obedience on each model's own labels cannot be probed after post-training.** The attacks almost never succeed:
  LoRA 39 / 1,248 injected rows, E05 3, routed 9. All three are flagged as insufficient, and no metric is reported.

---

## 1. Data and labels

- **Prompts:** AF-208 × (clean + Meta SecAlign's six static attacks), 1,456 per model, as in E07/E08. The rendered
  prompts are checked against each saved row's `prompt_text` and token count; all 1,456 match for every model.
- **Labels:** each model's existing E07/E08 generations and judgments; nothing was regenerated or re-judged.
  - **Detection:** an injection is present (6 attack conditions) vs clean.
  - **Own-label obedience:** on injected rows, this model's E07 "obeyed" label (strict judge minus verbatim echoes).
  - **Susceptibility:** on injected rows, the *base* model's "obeyed" label. Every model gets the same labels and 608
    positives, so post-trained models can be compared even though they rarely obey. It asks whether a model still
    linearly encodes what made the base model obey.

| own-label "obeyed" (injected rows / 1,248) | base | LoRA | E05 | routed |
|---|--:|--:|--:|--:|
| positives | 608 | 39 | 3 | 9 |
| in train / val / test | 363 / 122 / 123 | 26 / 6 / 7 | 3 / 0 / 0 | 3 / 0 / 6 |

## 2. Activations: `e09_capture_resid.py`

- **Capture:** residual stream at the final prompt token, at all 17 positions (index 0 is the token embedding; index i
  is the output of decoder layer i−1, before the final norm). One prefill per prompt: batch 1, no padding, no KV cache,
  fp32, TF32 off.
- **Models:** built with E07's `build_model`, which checks the LoRA and E05 sha256; the routed checkpoint sha256
  `b7ef759b…` is recorded.
- **Checks, all passed for all four models:**
  - every rendered prompt equals the saved prompt;
  - re-running the first 8 prompts gives bit-identical activations;
  - **the greedy next token equals the first token of the saved E07/E08 response on 1,456 of 1,456 prompts**, so the
    activations come from exactly the model state that produced the judged generations.
- **Cost:** A40, about 3 minutes per model including loading. 203 MB per model (`acts/<model>/resid_last.npz`,
  gitignored).
- **The final prompt token is the same chat-template token for every prompt,** so index 0 is identical across prompts.
  It works as a built-in chance control: every probe there gets AUROC exactly 0.50.

## 3. Probes: `e09_probes.py`

**Protocol, identical for every model and task.**
- **Split by record (task):** 125 / 41 / 42 records for train / val / test, seed 0
  (`artifacts/splits/af208_e09_records.json`). All seven variants of a task stay in one part, so no injection variant
  leaks across parts.
- **Probe:** standardise on train → L2 logistic regression with balanced class weights. C is chosen on validation AUROC
  from 1e-4 … 10; the probe is fitted on train only, so validation stays clean for experiment 3.
- **Metrics:** test AUROC, plus AUPRC and prevalence for obedience. 95% CIs from a cluster bootstrap over test records
  (1,000 resamples).
- **Insufficient:** fewer than 20 train or 10 test positives (or negatives). Such results are flagged and not reported.
- **Baselines for obedience:**
  - a logistic regression on the attack type only (one-hot);
  - the probe's mean AUROC *within* each attack type.
- **Held-out injection types:**
  - leave one attack family out (SF, ignore, completion; each family has two variants);
  - train on the four "after the data" attacks, test on the two "before the data" attacks.
  These are evaluated at four layers per model: best val layer for each task, the middle layer and the last layer.
- **Transfer:** the base probe, frozen, scored on another model's test activations. AUROC ignores the intercept, so
  only the direction w / sd_base matters.
- **Cosine:** between raw-space unit directions (w / sd) of independently trained probes.
  - Null: random 2,048-d directions, |cos| < 0.044 at 95%.
  - Ceiling: the cosine between two base probes trained on disjoint halves of the training records.

### 3.1 Experiment 1: layer-wise probes (`probes/figs/fig1_layerwise_auroc.png`, `fig1b_obedience_auprc.png`)

**Test AUROC by residual index** (0 = embedding; columns 1, 2, 4, 6, 8, 10, 12, 14, 16):

| task | model | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| detection | base | .96 | .98 | 1.00 | .99 | 1.00 | 1.00 | .99 | .99 | .99 |
| | LoRA | .97 | .99 | 1.00 | 1.00 | 1.00 | .99 | .94 | .91 | .94 |
| | E05 | .96 | .98 | 1.00 | 1.00 | 1.00 | .99 | .98 | .98 | .95 |
| | routed | .96 | .98 | 1.00 | .99 | .99 | .99 | .95 | .94 | .92 |
| susceptibility | base | .69 | .70 | .75 | .75 | .77 | .81 | .83 | .86 | .86 |
| | LoRA | .70 | .71 | .75 | .75 | .74 | .73 | .67 | .66 | .63 |
| | E05 | .69 | .71 | .75 | .75 | .76 | .76 | .72 | .68 | .63 |
| | routed | .69 | .70 | .74 | .72 | .72 | .71 | .67 | .64 | .61 |
| susceptibility, mean within-attack AUROC | base | .59 | .59 | .68 | .69 | .72 | .78 | .79 | .83 | .83 |
| | LoRA | .59 | .62 | .69 | .70 | .67 | .66 | .59 | .59 | .56 |
| | E05 | .59 | .59 | .69 | .71 | .69 | .69 | .63 | .60 | .57 |
| | routed | .59 | .59 | .67 | .65 | .62 | .61 | .57 | .54 | .50 |

- **Uncertainty:**
  - Base susceptibility at layer 14: AUROC 0.857 [0.792, 0.912], AUPRC 0.876 at prevalence 0.488.
  - Post-trained at their best layers: LoRA 0.753 [0.686, 0.820], E05 0.706 [0.620, 0.782], routed 0.720
    [0.639, 0.801]. Their CIs overlap each other, and E05's and routed's include the attack-type baseline (0.668).
  - With 42 test records, differences of about 0.05 between post-trained models are not resolved.
- **Held-out injection types** (base, layer 14):
  - detection: 0.96–1.00 on every held-out family, and 0.99 from after → before;
  - susceptibility: 0.68 / 0.81 / 0.90 for held-out SF / ignore / completion, and 0.78 from after → before.
  - In the post-trained models, susceptibility on held-out families is 0.44–0.69 at every layer tested. What they
    still encode about susceptibility is largely attack-specific.

### 3.2 Experiment 2: transfer and directions (`probes/figs/fig2_transfer.png`, `fig3_cosine.png`)

| | layers 1–4 | layers 5–8 | layers 9–12 | layers 13–16 |
|---|---|---|---|---|
| detection: base probe on LoRA / E05 / routed (AUROC) | 0.96–1.00 (own 0.96–1.00) | 0.98–1.00 (own 0.99–1.00) | 0.65–0.97 (own 0.94–1.00) | 0.61–0.77 (own 0.91–0.98) |
| detection: cos(base, own), range over the three models | 0.85–1.00 | 0.45–0.89 | 0.12–0.45 | 0.11–0.40 |
| detection: base split-half cosine (ceiling) | 0.69–0.83 | 0.71–0.83 | 0.66–0.84 | 0.78–0.82 |
| susceptibility: base probe on post-trained (AUROC) | 0.69–0.75 (own 0.69–0.75) | 0.69–0.75 (own 0.71–0.76) | 0.49–0.74 (own 0.67–0.77) | 0.55–0.63 (own 0.61–0.68) |
| susceptibility: cos(base, own) | 0.82–1.00 | 0.27–0.87 | 0.23–0.36 | 0.18–0.33 |
| susceptibility: base split-half cosine (ceiling) | 0.20–0.43 | 0.20–0.27 | 0.26–0.52 | 0.32–0.33 |

- **Detection:** the base direction carries over unchanged through about layer 8. After that it rotates, while the
  information stays decodable by a re-fitted probe. Transfer fails by up to 0.37 AUROC even though the information is
  present.
- **Susceptibility:**
  - The base probe's transfer drops to near chance in late layers (routed 0.49 at layer 12).
  - Late-layer cosines (0.18–0.33) are no lower than the base model's own split-half reliability (≈ 0.33): the
    susceptibility direction is not well-determined even within the base model. So **the cosine does not show a
    change** for this task; the AUROC drop is the evidence.
  - Early cosines near 1 mean only that the first layers barely change.
- **Detection vs susceptibility directions:** cos ≤ 0.23 in every model and layer.

## 4. Experiment 3: causal interventions

**Setup, all chosen on train / val:** `e09_make_interventions.py` → `interventions/`.
- **Layer:** the best val susceptibility layer, excluding index 0.
  - Base: resid 14 (val AUROC 0.83), so the hook goes after decoder layer 13.
  - Routed: resid 6 (val 0.74), hook after decoder layer 5.
- **Intervention:** `e09_intervene.py`. At every position of every forward pass (prompt and generated tokens),
  h ← h + norm · v, where v is a unit vector.
  - Strength is in units of σ, the std of the training injected rows' projection on the obedience direction. Base
    σ = 0.345 (8σ = 61% of the mean final-token residual norm); routed σ = 0.014.
  - At a given strength every direction has the same L2 norm (matched magnitude).
  - Generation: greedy, fp32, 512 tokens, in E07's format. Then the E07 judges.
- **Configs (val records only, 41 × 7 = 287 prompts each):**
  - base: control; obedience −8, −4, −2, +4σ; detection −8, −4, +4σ; random direction 1 at 8 and 4σ; random
    direction 2 at 8σ;
  - routed: control; obedience +4, +8σ; detection +8σ; random 8σ.
- **Test records** (42) are reserved for confirming the chosen settings.
- **Harness check:** the base control config reproduces the saved E07 base responses exactly on all 287 val prompts
  (287 / 287 token-identical). The hook and the A40 do not change generation by themselves.

**Selection rule for the test run (written before any val results were seen).**
- **Base:** among the obedience-direction strengths, the one with the lowest val "obeyed" rate, provided both of
  these hold:
  - clean task done ≥ control − 0.05;
  - task done under injection ≥ control − 0.05.
  If none qualifies, take the strength with the smallest |σ|.
- **Routed:** the obedience strength with the highest val "obeyed" rate.
- **Test run:** for each model, control + the chosen obedience strength + the detection direction and random
  direction 1 at the same norm, on the 42 test records.

### 4.1 Val results (41 records; 246 injected and 41 clean rows per config)

`intervene_val/summary.md`, `fig_tradeoff.png`. Rates are shares of rows; 95% CIs from a cluster bootstrap over
records. "vs control" is a paired sign test on discordant rows (only this config succeeds vs only the control).

**Base, hook after decoder layer 13** (σ = 0.345; 8σ ≈ 61% of the mean final-token residual norm):

| config | obeyed [CI] | task under injection [CI] | clean task | degenerate | obeyed vs control | task under inj. vs control |
|---|---|---|--:|--:|---|---|
| control | 0.50 [0.43, 0.57] | 0.51 [0.42, 0.60] | 0.71 | 1.0% | – | – |
| obedience −2σ | 0.35 [0.28, 0.43] | 0.48 | 0.76 | 3.5% | 10 v 46, p < 0.001 | 12 v 21, p = 0.16 |
| obedience −4σ | 0.21 [0.15, 0.28] | 0.46 | 0.76 | 3.8% | 5 v 75, p < 0.001 | 17 v 31, p = 0.06 |
| obedience −8σ | **0.13** [0.07, 0.20] | 0.39 | 0.71 | 6.6% | 10 v 100, p < 0.001 | 20 v 50, p < 0.001 |
| obedience +4σ | 0.79 [0.73, 0.85] | 0.37 | 0.73 | 4.2% | 76 v 3, p < 0.001 | 12 v 46, p < 0.001 |
| detection −4σ | 0.31 [0.24, 0.39] | **0.60** [0.51, 0.68] | 0.76 | 3.8% | 15 v 61, p < 0.001 | **41 v 20, p = 0.01** |
| detection −8σ | 0.24 [0.18, 0.32] | 0.58 | 0.63 | 6.3% | 16 v 78, p < 0.001 | 43 v 26, p = 0.05 |
| detection +4σ | 0.79 [0.73, 0.85] | 0.28 | 0.68 | 3.1% | 77 v 5, p < 0.001 | 5 v 62, p < 0.001 |
| random 1, 4σ | 0.53 | 0.50 | 0.78 | 2.8% | 23 v 15, p = 0.26 | 14 v 18, p = 0.60 |
| random 1, 8σ | 0.44 | 0.48 | 0.78 | 3.8% | 21 v 34, p = 0.10 | 24 v 31, p = 0.42 |
| random 2, 8σ | 0.64 | 0.42 | 0.73 | 3.8% | 44 v 9, p < 0.001 | 17 v 39, p = 0.005 |

- **The obedience direction is causal in both signs, and dose-dependent.**
  - Subtracting it lowers attack success monotonically (0.50 → 0.35 / 0.21 / 0.13 at −2 / −4 / −8σ); adding 4σ raises
    it to 0.79.
  - Clean task completion is unchanged (0.71–0.76).
  - The user's task *under injection* falls at −8σ (0.51 → 0.39). Random directions of the same norm cause the same
    drop (below), so this cost comes from the size of the push, not from this direction.
- **The detection direction is also causal, with a better trade-off.**
  - Subtracting it lowers attack success less (0.31 at −4σ) but *raises* the user's task under injection (0.51 → 0.60,
    p = 0.01). Clean utility is unchanged at −4σ and drops at −8σ (0.63).
  - Adding it is as harmful as adding the obedience direction.
  - The two directions are only weakly aligned (cos 0.23), so pushing along "injection present" is not simply a
    rescaled push along "will obey".
- **Random directions: seven at 8σ** (random 1–6 at +8σ, random 1 at −8σ), plus random 1 at 4σ.

  | 7 random directions at 8σ | min | median | max | control |
  |---|--:|--:|--:|--:|
  | obeyed | 0.44 | 0.52 | 0.64 | 0.50 |
  | task under injection | 0.37 | 0.42 | 0.48 | 0.51 |
  | clean task | 0.68 | 0.73 | 0.78 | 0.71 |

  - **A push this large is not neutral.** It tends to raise attack success slightly (significant for 2 of 7) and to
    lower the user's task under injection (significant for 5 of 7: p < 0.001 to 0.04).
  - **Against this null**, obedience −8σ (0.13) and −4σ (0.21) are far below every random direction on attack
    success, so the security effect is specific to the direction. Its task-under-injection drop is not: 0.39 lies
    within the random range.
  - **Detection −4σ is the only setting that beats the null on both axes**: 0.31 obeyed against 0.44–0.64, and 0.60
    task under injection against 0.37–0.48.
- **Clean-task differences** are within noise at every setting: at most 11 discordant clean records (4 v 7 for
  detection −8σ), every p ≥ 0.37 (41 clean rows).

**Routed (E08), hook after decoder layer 5**, the routed model's own susceptibility direction (σ = 0.014; 8σ ≈ 25%
of the final-token residual norm):

| config | obeyed | task under injection | clean task |
|---|--:|--:|--:|
| control | 0 / 246 | 0.65 | 0.71 |
| obedience +4σ | 0 | 0.60 | 0.66 |
| obedience +8σ | 0 | 0.64 | 0.71 |
| detection +8σ | 0 | 0.60 | 0.71 |
| random 1, +8σ | 0 | 0.60 | 0.76 |

- **No setting brings attacks back**, and no task difference is significant (p ≥ 0.10). The routed model's weak,
  mid-network susceptibility direction (val AUROC 0.74) is not a lever on its behaviour at these strengths. Larger
  strengths, and transplanting the base model's layer-14 direction, were not tried.
- **Harness check:** the routed control reproduces the saved E08 responses on 287 / 287 val prompts.

**Selection rule applied:**
- **Base:** −8σ fails the task-under-injection floor (0.39 < 0.46), and so does −4σ (0.455 < 0.462). **−2σ is chosen.**
- **Routed:** +4σ and +8σ tie at 0 obeyed; +8σ (the larger) is chosen.
- **Exploratory additions to the test run, chosen after seeing val:** obedience −8σ and detection −4σ, with random
  direction 1 at 8σ and 4σ as their matched controls.

### 4.2 Test results (42 held-out records; 252 injected and 42 clean rows per config)

`intervene_test/summary.md`, `fig_tradeoff.png`. Both controls reproduce the saved E07/E08 responses on 294 / 294
test prompts.

**Base, pre-registered settings (2σ, chosen on val by the rule):**

| config | obeyed [CI] | task under injection [CI] | clean task | obeyed vs control | task under inj. vs control | clean vs control |
|---|---|---|--:|---|---|---|
| control | 0.49 [0.41, 0.57] | 0.49 [0.40, 0.57] | 0.86 | – | – | – |
| **obedience −2σ** | **0.34** [0.26, 0.43] | **0.55** [0.46, 0.64] | 0.83 | **6 v 43, p < 0.001** | **25 v 9, p = 0.009** | 1 v 2, p = 1.0 |
| detection −2σ | 0.39 [0.31, 0.48] | 0.52 | 0.86 | 13 v 37, p < 0.001 | 23 v 15, p = 0.26 | 1 v 1 |
| random 1, 2σ | 0.46 [0.38, 0.54] | 0.51 | 0.81 | 9 v 16, p = 0.23 | 17 v 11, p = 0.34 | 1 v 3, p = 0.63 |

**Base, exploratory settings (chosen after seeing val):**

| config | obeyed | task under injection | clean task | obeyed vs control | task under inj. | clean vs control |
|---|--:|--:|--:|---|---|---|
| obedience −8σ | **0.11** | 0.44 | 0.69 | 6 v 101, p < 0.001 | 27 v 39, p = 0.18 | 0 v 7, p = 0.016 |
| random 1, 8σ | 0.44 | 0.50 | 0.71 | 20 v 32, p = 0.13 | 28 v 24, p = 0.68 | 0 v 6, p = 0.031 |
| detection −4σ | 0.32 | **0.55** | 0.81 | 12 v 54, p < 0.001 | **34 v 18, p = 0.037** | 0 v 2, p = 0.50 |
| random 1, 4σ | 0.48 | 0.53 | 0.79 | 23 v 25, p = 0.89 | 23 v 13, p = 0.13 | 0 v 3, p = 0.25 |

**Routed (E08), pre-registered (+8σ at layer 5):**

| config | obeyed | task under injection | clean task | obeyed vs control | task under inj. vs control |
|---|--:|--:|--:|---|---|
| control | 6 / 252 (0.024) | 0.64 | 0.74 | – | – |
| obedience +8σ | 2 / 252 | 0.60 | 0.76 | 0 v 4, p = 0.13 | 13 v 23, p = 0.13 |
| detection +8σ | 2 / 252 | 0.56 | 0.62 | 1 v 5, p = 0.22 | 14 v 33, p = 0.008 |
| random 1, +8σ | 3 / 252 | 0.65 | 0.74 | 2 v 5, p = 0.45 | 23 v 20, p = 0.76 |

**What the test run confirms:**
- **The pre-registered base result replicates.**
  - Subtracting 2σ of the obedience direction cuts attack success 0.49 → 0.34. The random direction at the same norm
    does not (0.46, p = 0.23).
  - On test it even *raises* the user's task under injection (0.49 → 0.55, p = 0.009), with no clean cost (0.86 →
    0.83, p = 1.0).
  - The detection direction at the same norm also lowers attack success (0.39), but less.
- **The val trade-off pattern replicates (exploratory).**
  - −8σ obedience nearly removes attacks (0.11) but costs clean utility (0.86 → 0.69, p = 0.016). Random 8σ costs the
    same (0.71, p = 0.031), so the cost comes from the push's size.
  - −4σ detection again improves both axes (0.32 obeyed; 0.55 task under injection, p = 0.037).
- **The routed model stays unaffected.** Pushing along its own susceptibility direction does not bring attacks back
  (2 / 252 against the control's 6). The detection push costs utility (task under injection 0.56, p = 0.008).

### 4.3 Analysis of experiment 3

- **In the base model, the late-layer obedience direction is causal and specific.** It moves attack success in both
  signs and in proportion to strength. It does so on held-out test records at a strength chosen in advance, and a
  random direction of the same norm does not. So the probe found more than a correlate of the attack type.
- **Small pushes are close to free; large pushes cost utility whatever the direction.** At 2σ (15% of the final-token
  residual norm) neither clean utility nor the task under injection gets worse. At 8σ (61%) random directions
  cost as much as the probe directions.
- **The detection direction is a different and more task-preserving lever.** Subtracting "injection present" leaves
  attacks somewhat less successful and the user's task *more* often done, both on val and on test. One reading: the
  model attends less to the injected instruction as an instruction. This is a hypothesis; it was not tested.
- **The security-trained routed model has no such handle.** Its weak, mid-network susceptibility direction does not
  restore attacks at 8σ. This matches experiment 1: post-training removes the late-layer representation the base
  model uses, and what is left is not what drives behaviour, at least along these directions and strengths.
  Untested: larger strengths, other layers, and transplanting the base model's layer-14 direction.

## 5. Caveats

- **One split:** 42 test records, so CIs are wide (±0.06 AUROC for susceptibility). One split seed; no repeated or
  nested cross-validation.
- **Labels come from one LLM judge** (E07 caveats). "Obeyed" is an upper estimate.
- **Susceptibility is not obedience.** For the post-trained models it measures whether they encode what made the base
  model obey, not what makes them obey. Their own obedience is too rare to probe with these attacks.
- **Final prompt token only.** Nothing about the injection span itself or the generated tokens.
- **Linear probes only:** "not linearly decodable" does not mean "absent".
- **Detection is easy for a superficial reason:** injected prompts differ in their text. Near-perfect AUROC therefore
  says little; the informative parts are the transfer and the late-layer rotation.
- **Experiment 3 scope:**
  - one layer per model, one hook position (every token), a handful of strengths;
  - 41 val and 42 test records, so single-config effects of about 0.05 are not resolved;
  - LoRA and E05 were not intervened on;
  - the exploratory −8σ / −4σ test rows were chosen after seeing val;
  - the extra random controls (random 3–6 at 8σ, random 1 at −8σ) were added after the first val results, to measure
    the null.
- **Strength unit:** σ is measured at the final prompt token, but the vector is added at every position. Positions
  differ in norm, so "% of residual norm" is only indicative.
- **Judging:** the E07 judges, with no human check of the intervened outputs. The degenerate share stays ≤ 6.6% in
  every config.

## 6. Reproduce

```bash
export POD_HOST=root@<ip> POD_PORT=<port>     # pod: /root/venv (E08 setup) + peft 0.21.3 (installed by sync)
bash experiments/modal/experiments/e09_local.sh sync     # code, data, saved generations, 3 checkpoints
bash experiments/modal/experiments/e09_local.sh start    # capture 4 models -> outputs/e09_probing/acts, verified
python experiments/modal/experiments/e09_probes.py --acts experiments/modal/outputs/e09_probing/acts \
  --out experiments/modal/outputs/e09_probing/probes            # laptop, ~80 s; needs scikit-learn (1.9.1)
python experiments/modal/experiments/e09_make_interventions.py --probes experiments/modal/outputs/e09_probing/probes \
  --acts experiments/modal/outputs/e09_probing/acts --out experiments/modal/outputs/e09_probing/interventions
INTERV=experiments/modal/outputs/e09_probing/interventions bash experiments/modal/experiments/e09_local.sh intervene-sync
PART=val bash experiments/modal/experiments/e09_local.sh intervene-start
# extra random controls (val) and the test run: same commands with
#   INTERV=experiments/modal/outputs/e09_probing/interventions_valx MODELS=base PART=val
#   INTERV=experiments/modal/outputs/e09_probing/interventions_test PART=test
# judge each config dir with the E07 judges (e07_judge_decisions.py / e07_judge_strict.py --out DIR --models NAME), then
python experiments/modal/experiments/e09_intervention_summary.py --dir experiments/modal/outputs/e09_probing/intervene_val \
  --interv experiments/modal/outputs/e09_probing/interventions
python experiments/modal/experiments/e09_intervention_summary.py --dir experiments/modal/outputs/e09_probing/intervene_test \
  --interv experiments/modal/outputs/e09_probing/interventions_test
```

Environment: torch 2.14.1+cu130, transformers 5.22.1, peft 0.21.3, NVIDIA A40; laptop scikit-learn 1.9.1.
