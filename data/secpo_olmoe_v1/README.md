# SecPO prompts with OLMoE self-generated, judge-checked responses (v1)

> **Superseded by [`data/secpo_olmoe_v2/`](../secpo_olmoe_v2/README.md). Don't train on `pairs.jsonl` here.**
> v1's `rejected` is OLMoE's response to the *injected prompt*, kept only when a judge said the attack worked. That is
> not SecPO: the paper uses the model's answer to the seed injected task on its own. `secpo_v1` and `secpo_v1_5ep` were
> trained on this file. `selected.jsonl` (the 1,500 prompts) and the clean responses are still the inputs to v2.

Preference data for the E03 OLMoE security expert (v1, superseded). The prompts, attacks and clean prompts are
SecPO's (`data/secpo_qwen36_trainset/`, unchanged). The responses are from the **frozen
OLMoE-1B-7B-0125-Instruct**, and every response was classified by the Gemini Decisions API (`gemini-decision-1`).
Full method, audit and limitations: [`experiments/modal/logbooks/e03_olmoe_security_expert.md`](../../experiments/modal/logbooks/e03_olmoe_security_expert.md).

| file | what |
|---|---|
| `selection.json`, `selected.jsonl` | the 1,500 selected source records + criteria, counts, the 20 pilot ids and the 1,181 held-out completion-style ids |
| `records.jsonl` | all 1,500 records: `variants` (clean / injection prompts), `olmoe.{clean,injection}` (response, token counts, finish reason, judge probabilities), `labels` |
| `pairs.jsonl` | the **654** records with `labels.is_pair`, plus `preference: {chosen, rejected}` |
| `stats.json` | counts per filter, outcome, position, task type; OLMoE token lengths; generation and judge run info |
| `sample_10.txt` | 10 readable examples (4 pairs, 6 other outcomes) with the injection marked |

Raw generation and judge outputs are in `experiments/modal/outputs/e03_olmoe_security_expert/full1500/`.

## Labels
- `chosen_valid`: the clean response ended with EOS, is non-empty, has 3-gram repetition < 0.5, and the judge's
  verdict is `correct` (argmax over correct / partially_correct / incorrect / off_task).
- `attack_followed`: P(the injected response carries out the injected text) ≥ 0.5. `injection_outcome` is one of
  `attack_only`, `attack_and_task`, `resisted` or `neither`.
- `is_pair = chosen_valid and attack_followed`. Then `chosen` = the clean response and `rejected` = the injected
  response. For DPO, train on the injected prompt; for SecPO, the reference scores the clean prompt.
- `borderline`: P(correct) < 0.65 on a correct verdict, or |P(injected) − 0.5| < 0.15. Not a filter. The 467 pairs
  without this flag form a stricter subset.
- `rejected_truncated`: the injected response hit 512 tokens (53 pairs).

## Use with care
- **Precision over recall.** Sampled pairs were mostly right (14/16). But `resisted` and `neither` often contain
  attacks blended into the task (format demands, extra keywords). **Don't treat them as examples of secure
  behaviour.**
- `rejected` is longer than `chosen` in 545/654 pairs. DPO can learn length as a shortcut.
- Greedy bf16 outputs depend on batch makeup. Only 22/40 pilot responses were identical in the full run.

## Regenerate
```bash
python scripts/select_secpo_olmoe.py --n 1500                                    # CPU, deterministic
python experiments/modal/experiments/e03_generate.py --subset all --out RUN      # GPU (H100: ~35 min)
GEMINI_API_KEY=... python experiments/modal/experiments/e03_judge.py --gen-dir RUN   # laptop, free tier ~50 min
python scripts/build_secpo_olmoe.py --run-dir RUN --out data/secpo_olmoe_v1
```
