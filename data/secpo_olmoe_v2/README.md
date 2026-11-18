# SecPO pairs with OLMoE responses, built as in the paper (v2)

Training data for the E03 OLMoE security expert. **1,500 pairs, no filtering.** It replaces v1's `rejected` side,
which did not follow SecPO. Method and decisions:
[`experiments/modal/logbooks/e03_olmoe_security_expert.md`](../../experiments/modal/logbooks/e03_olmoe_security_expert.md),
Step 5.

SecPO ([arXiv:2610.41998](https://arxiv.org/abs/2610.41998), Algorithm 1 line 13) sets `y_w = f_undef(z)` and
`y_l = f_undef(z′)`: the undefended model on the clean sample, and on the seed injected task by itself. Here `f_undef`
is the frozen OLMoE-1B-7B-0125-Instruct @ `b89a7c4` (greedy, bf16, 512 new tokens, transformers 5.22.1, H100).

| field | what |
|---|---|
| `variants.clean`, `variants.injection` | the prompts, copied unchanged from `data/secpo_olmoe_v1/selected.jsonl`. The policy is trained on `injection`; the SecPO reference scores `clean` |
| `preference.chosen` = `olmoe.clean.response` | OLMoE on the clean prompt (system = instruction, user = data). Run `outputs/e03_olmoe_security_expert/full1500/` |
| `preference.rejected` = `olmoe.seed_task.response` | OLMoE on `metadata.seed_task.strip()` as the only (user) message: no trusted instruction, no clean data, no attack wrapper. Run `outputs/e03_olmoe_security_expert/seed1500/` |
| `olmoe.*` | response, token counts, `finish_reason` (`length` = hit 512; the trainer then adds no EOS), 3-gram repetition, the exact prompt text |

Differences from v1:
- `rejected` now answers the seed task alone. In v1 it was OLMoE's response to the injected prompt, kept only when a
  judge said the attack worked.
- There is no judge and no filter on either side, so every selected record is a pair (v1 had 654).

## Know before training (`stats.json`)
- **Length gap:** `rejected` is longer than `chosen` in 1,153/1,500 pairs. Median tokens are 237 vs 33.
  - 376 `rejected` hit the 512 cap, against 34 `chosen`.
  - The clean Alpaca-with-input tasks have short answers, while seed tasks are often open-ended.
  - SecPO's summed log-probs are exposed to this, so watch answer length after training.
- **Seed task vs final injection:** `rejected` answers the seed task, not the PAIR-optimized injection text. This is
  the paper's design. Only 535/1,500 injections contain the seed task verbatim; most others reword it, and some add
  demands such as exact strings or "output only".
- **Chat layout:** the seed task is one string (the source joins instruction + input), so it is a single user turn.
  OLMoE has no `input` role.

## Regenerate
```bash
python experiments/modal/experiments/e03_generate.py --subset all --out RUN_CLEAN                             # clean (+ injection)
python experiments/modal/experiments/e03_generate.py --subset all --conditions seed_task --out RUN_SEED      # y_l
python scripts/build_secpo_olmoe_v2.py --clean-run RUN_CLEAN --seed-run RUN_SEED                             # CPU
```
The split is `experiments/modal/artifacts/splits/secpo_olmoe_v1_grouped_v1.json` (`ids`: 1,275 train / 225 val). It is
made over the same 1,500 records, so it applies unchanged.
