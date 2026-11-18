"""Convert the released SecPO training set into our JSONL format (clean / injection variants + preference pair).

Source: the Hugging Face dataset `Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-trainset` (SecPO, Chen et al. 2026,
arXiv:2610.41998), one JSON array of 19,157 preference records. We download it at a pinned revision and check its
SHA-256. Each source record has:

  prompt                 attacked prompt, already rendered with the Qwen chat template:
                         <|im_start|>user\\n{instruction}<|im_end|>\\n<|im_start|>input\\n{data + injection}<|im_end|>
                         \\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
  chosen / rejected      Qwen3.6-27B responses (to the trusted task / to the injected task), ending in <|im_end|>
  chosen_data            the benign data before injection
  rejected_input_whole   the (optimized) injected instruction, exactly once in `prompt`
  rejected_input         the earlier seed task the `rejected` response answers
  chosen_input           instruction + "\\n\\n" + chosen_data
  inject_at_the_end      True = injection after the data, False = before it

Conversion. We parse `prompt` back into (instruction, injected data) and lay it out as in data/sep_600.jsonl:
instruction (trusted) -> system message, data (untrusted) -> user message. The injected user message is the source's
data text byte for byte. The clean user message is `chosen_data`, i.e. the data before any attack. This is NOT the
dataset card's recipe (`prompt.replace(rejected_input_whole, "")`): in ~1.2k completion-style rows the attacker text
also includes a fake response and a fake role header *before* `rejected_input_whole`, and the card's recipe would
leave those inside the "clean" prompt. Everything in the injected data that is not `chosen_data` is the attacker
payload (span "probe"). `rejected_input_whole` is a sub-span ("injected_instruction"). A handful of rows whose
`chosen_data` is not a literal prefix/suffix of the injected data fall back to the card's recipe and are flagged in
`metadata.clean_reconstruction`. Responses have the trailing Qwen end token removed. Nothing else is changed.

Output (data/secpo_qwen36_trainset/):
  secpo_trainset.jsonl   all records (~120 MB, gitignored; regenerate with this script)
  sample.jsonl           25 random records (seed 0), committed, for reading without downloading
  manifest.json          source revision + hashes, counts, output hashes

Usage (standard library only, ~100 MB download):
    python scripts/build_secpo_trainset.py
    python scripts/build_secpo_trainset.py --source-json PATH/preference_...optimized.json     # offline, still hash-checked
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "data" / "secpo_qwen36_trainset"

SOURCE_REPO = "Sizhe-Chen/Qwen3.6-27B-Instruct-SecPO-trainset"
SOURCE_REVISION = "0cbfd6e6a2c2966dbccf4fa89b6ea7fe20c8703e"
SOURCE_FILE = "preference_Qwen3.6-27B_dpo_NaiveCompletion_randpos_synthetic_alpaca_optimized.json"
SOURCE_SHA256 = "cd349baa406eae0bf7e4ee2d938d087d6656469d88076d68ba76baeb7e5ec669"
SOURCE_URL = f"https://huggingface.co/datasets/{SOURCE_REPO}/resolve/{SOURCE_REVISION}/{SOURCE_FILE}"
N_SOURCE = 19157

QWEN_PROMPT = re.compile(r"<\|im_start\|>user\n(.*)<\|im_end\|>\n<\|im_start\|>input\n(.*)<\|im_end\|>\n"
                         r"<\|im_start\|>assistant\n<think>\n\n</think>\n\n", re.S)
QWEN_EOS = "<|im_end|>"
SAMPLE_N, SAMPLE_SEED = 25, 0


def fetch(url: str, sha256: str, path: str | None) -> list[dict]:
    raw = Path(path).read_bytes() if path else urllib.request.urlopen(url, timeout=300).read()
    got = hashlib.sha256(raw).hexdigest()
    if got != sha256:
        raise SystemExit(f"SHA-256 mismatch for {path or url}: expected {sha256}, got {got}")
    return json.loads(raw)


def span(text: str, sub: str, start: int | None = None) -> dict:
    s = text.index(sub) if start is None else start
    assert text[s:s + len(sub)] == sub
    return {"message": 1, "start": s, "end": s + len(sub)}


def convert(e: dict, idx: int) -> dict:
    m = QWEN_PROMPT.fullmatch(e["prompt"])
    if not m:
        raise ValueError(f"row {idx}: prompt does not match the Qwen user/input template")
    instruction, data = m.groups()
    inj, at_end = e["rejected_input_whole"], e["inject_at_the_end"]
    if data.count(inj) != 1:
        raise ValueError(f"row {idx}: injected instruction occurs {data.count(inj)} times in the data")

    # Split the injected data into (original data, attacker payload) at the side given by inject_at_the_end.
    clean, method, data_start = None, "card_recipe_fallback", None
    for cand, how in ((e["chosen_data"], "chosen_data"), (e["chosen_data"].strip(), "chosen_data_stripped")):
        rest = data[len(cand):] if at_end else data[:len(data) - len(cand)]
        if cand and (data.startswith(cand) if at_end else data.endswith(cand)) and inj in rest:
            clean, method, data_start = cand, how, (0 if at_end else len(data) - len(cand))
            break
    if clean is None:                                     # dataset card's recipe; payload = injected instruction only
        clean = data.replace(inj, "", 1).strip()
        payload, payload_start = inj, data.index(inj)
    else:
        rest = data[len(clean):] if at_end else data[:len(data) - len(clean)]
        payload = rest.strip()
        payload_start = (len(clean) + rest.index(payload)) if at_end else rest.index(payload)

    inj_spans = {"task": {"message": 0, "start": 0, "end": len(instruction)},
                 "probe": span(data, payload, payload_start),
                 "injected_instruction": span(data, inj)}
    if data_start is not None:
        inj_spans["data"] = span(data, clean, data_start)

    def msgs(user: str) -> list[dict]:
        return [{"role": "system", "content": instruction}, {"role": "user", "content": user}]

    def strip_eos(s: str) -> str:
        if not s.endswith(QWEN_EOS):
            raise ValueError(f"row {idx}: response does not end with {QWEN_EOS}")
        return s[: -len(QWEN_EOS)]

    return {
        "id": f"secpo_{idx:05d}",
        "metadata": {
            "source": "SecPO-Qwen3.6-27B-trainset", "source_index": idx, "task_type": "secpo_alpaca_cleaned",
            "instruction": instruction, "clean_data": clean,
            "injection": payload,                                   # all attacker-controlled text in the data
            "injected_instruction": inj,                            # source `rejected_input_whole`
            "seed_task": e["rejected_input"],                       # source `rejected_input`; what `rejected` answers
            "position": "end" if at_end else "start",
            "completion_style": payload.strip() != inj.strip(),     # payload has a fake response / role header too
            "seed_verbatim_in_injection": e["rejected_input"].strip() in inj,
            "clean_reconstruction": method,
            "response_model": "Qwen3.6-27B (non-thinking)",
        },
        "variants": {
            "clean": {"messages": msgs(clean),
                      "spans": {"task": {"message": 0, "start": 0, "end": len(instruction)},
                                "data": {"message": 1, "start": 0, "end": len(clean)}}},
            "injection": {"messages": msgs(data), "spans": inj_spans},
        },
        "preference": {"chosen": strip_eos(e["chosen"]), "rejected": strip_eos(e["rejected"])},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source-json", help="local copy of the source JSON instead of downloading it")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    src = fetch(SOURCE_URL, SOURCE_SHA256, args.source_json)
    if len(src) != N_SOURCE:
        raise SystemExit(f"expected {N_SOURCE} records, got {len(src)}")
    rows = [convert(e, i) for i, e in enumerate(src)]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    full, sample = out / "secpo_trainset.jsonl", out / "sample.jsonl"
    full.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    picked = sorted(random.Random(SAMPLE_SEED).sample(range(len(rows)), SAMPLE_N))
    sample.write_text("".join(json.dumps(rows[i], ensure_ascii=False) + "\n" for i in picked))

    meta = [r["metadata"] for r in rows]
    manifest = {
        "dataset": "secpo_qwen36_trainset", "generator": "scripts/build_secpo_trainset.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": {"repo": SOURCE_REPO, "revision": SOURCE_REVISION, "file": SOURCE_FILE, "sha256": SOURCE_SHA256,
                   "n_records": len(src), "paper": "arXiv:2610.41998 (SecPO)"},
        "layout": "system = trusted instruction, user = untrusted data (as data/sep_600.jsonl)",
        "conditions": ["clean", "injection"],
        "n_records": len(rows),
        "position": dict(collections.Counter(m["position"] for m in meta)),
        "completion_style": sum(m["completion_style"] for m in meta),
        "seed_verbatim_in_injection": sum(m["seed_verbatim_in_injection"] for m in meta),
        "clean_reconstruction": dict(collections.Counter(m["clean_reconstruction"] for m in meta)),
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (full, sample)},
        "sample": {"n": SAMPLE_N, "seed": SAMPLE_SEED, "source_indices": picked},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(rows)} records to {full} ({full.stat().st_size / 1e6:.0f} MB), {SAMPLE_N} to {sample}; "
          f"position {manifest['position']}, completion-style {manifest['completion_style']}, "
          f"reconstruction {manifest['clean_reconstruction']}")


if __name__ == "__main__":
    main()
