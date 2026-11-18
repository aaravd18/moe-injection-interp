# SecPO training set (Qwen3.6-27B responses), converted

The released SecPO training set ([arXiv:2610.41998](https://arxiv.org/abs/2610.41998);
[`Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-trainset`](https://huggingface.co/datasets/Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-trainset)
@ `0cbfd6e`, SHA-256 checked), converted to our JSONL format. It has **19,157** preference records built from
Alpaca-cleaned tasks that have an input. Analysis and caveats are in
[`experiments/modal/logbooks/e03_olmoe_security_expert.md`](../../experiments/modal/logbooks/e03_olmoe_security_expert.md).
This is **training data only**. The evaluation sets (`data/alpacafarm_secalign_208.jsonl`, `data/sep_600.jsonl`) are
separate files and are never mixed in.

| file | what | in git |
|---|---|---|
| `secpo_trainset.jsonl` | all 19,157 records (117 MB) | no, regenerate |
| `sample.jsonl` | 25 random records (seed 0), same format | yes |
| `manifest.json` | source revision + hashes, counts, output hashes | yes |
| `stats_olmoe.json` | OLMoE token lengths, attack / response stats, duplicates, eval overlap (with every match listed) | yes |
| `lengths_olmoe.csv` | per-record OLMoE token counts (prompt, responses, full sequences) | yes |

## Record format

The layout is the same as `data/sep_600.jsonl`: trusted instruction → `system`, untrusted data → `user`. The Qwen
source has a separate `input` role, which OLMoE's template does not have.

```json
{"id": "secpo_08484",
 "metadata": {"instruction": "...", "clean_data": "...", "injection": "<all attacker text>",
              "injected_instruction": "<source rejected_input_whole>", "seed_task": "<source rejected_input>",
              "position": "start|end", "completion_style": false, "seed_verbatim_in_injection": true,
              "clean_reconstruction": "chosen_data", "source_index": 8484, ...},
 "variants": {"clean":     {"messages": [system, user = clean_data],        "spans": {"task", "data"}},
              "injection": {"messages": [system, user = injected data],     "spans": {"task", "data", "probe",
                                                                                        "injected_instruction"}}},
 "preference": {"chosen": "<Qwen3.6-27B answer to the trusted task>", "rejected": "<Qwen answer to seed_task>"}}
```

- **`clean`**: the original data (`chosen_data`). The dataset card's recipe, which deletes `rejected_input_whole` from
  the prompt, would leave the fake answer and fake role header of the 1,181 completion-style attacks inside the
  "clean" prompt. We don't use that recipe. 8 rows with small source inconsistencies are flagged in
  `clean_reconstruction`.
- **`injection`**: the source's injected data, byte for byte. Re-rendering it with the Qwen template reproduces the
  source `prompt` exactly for all 19,157 rows.
- **`probe` span**: all attacker text. `injected_instruction` is the instruction inside it. The two differ only for
  completion-style attacks, and by surrounding whitespace.
- **`preference`**: the responses with Qwen's `<|im_end|>` removed. `rejected` answers the *seed* task, not the
  final optimized injection. The card says so too.

## Regenerate / inspect

```bash
python scripts/build_secpo_trainset.py                    # download (~100 MB), hash-check, convert, sample, manifest
python scripts/analyze_secpo_trainset.py                  # OLMoE tokenizer only; writes stats_olmoe.json + lengths_olmoe.csv
python scripts/inspect_secpo_trainset.py --n 8 --seed 0   # readable examples, injection marked [[INJECTION>> … <<INJECTION]]
python -m pytest tests/test_secpo_trainset.py
```

Licensing: the card asserts no license. Upstream terms apply: Stanford Alpaca (CC BY-NC 4.0), alpaca-cleaned
(CC BY 4.0), and Qwen3.6-27B output terms. Research use only.
