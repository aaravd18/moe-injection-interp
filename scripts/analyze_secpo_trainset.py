"""Analyse data/secpo_qwen36_trainset/secpo_trainset.jsonl: OLMoE token lengths, attack/response statistics,
duplicates, and overlap with our evaluation datasets. No model is loaded, only the tokenizer.

Token counts use the OLMoE-1B-7B-0125-Instruct chat template the way moe_interp renders prompts: render with
apply_chat_template(tokenize=False), then tokenize with add_special_tokens=False (the template already adds BOS).
  prompt         messages + generation prompt ("<|assistant|>\\n")
  full sequence  messages + assistant response, i.e. prompt + response + EOS, as a DPO/SecPO trainer would feed it
An example "fits in N" when the injected prompt with the LONGER of its two responses is <= N tokens (the SecPO
reference side uses the clean prompt, which is never longer; checked below).

Overlap check: every text in each eval JSONL (all message contents and span texts) vs the training fields
instruction / clean_data / injection / injected_instruction / seed_task. Texts are lowercased with punctuation
collapsed. "exact" = equal after normalisation. "near" = share word 6-grams with containment >= 0.5 (shared shingles /
shingles of the shorter text). Texts under 6 words are only checked for exact matches.

Usage:
    python scripts/analyze_secpo_trainset.py            # writes stats_olmoe.json + lengths_olmoe.csv next to the data
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / "data" / "secpo_qwen36_trainset"
TOKENIZER, REVISION = "allenai/OLMoE-1B-7B-0125-Instruct", "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e"  # configs/olmoe.yaml
EVAL_SETS = ["data/alpacafarm_secalign_208.jsonl", "data/sep_600.jsonl", "data/examples.jsonl",
             "data/mellum_pilot_v1/examples.jsonl"]
TRAIN_FIELDS = ["instruction", "clean_data", "injection", "injected_instruction", "seed_task"]
BUDGETS = [256, 512, 1024]
SHINGLE, NEAR = 6, 0.5

# Heuristic, keyword-based cues on the injected instruction. Multi-label; NOT ground-truth attack labels.
CUES = {
    "override (ignore/forget/disregard…)": r"\b(ignore|forget|disregard|discard|abandon|scrap|let go|overrid\w*|"
                                           r"supersed\w*|no longer|instead|new (task|instruction|order)s?)\b",
    "authority (system/admin/policy…)": r"\b(system|admin\w*|developer|policy|compliance|mandatory|required|official|"
                                        r"authori[sz]\w*|supervisor|manager)\b",
    "urgency (urgent/immediately…)": r"\b(urgent\w*|immediately|asap|critical|attention|important|priority)\b",
    "role play (you are / act as…)": r"\b(you are (now )?a|act as|pretend|role|persona)\b",
    "task correction (oops/actually…)": r"\b(oops|actually|correction|mistake|never mind|wrong task|update[sd]?)\b",
}


def pct(a) -> dict:
    a = np.asarray(a)
    return {"mean": round(float(a.mean()), 1), "p50": int(np.percentile(a, 50)), "p90": int(np.percentile(a, 90)),
            "p99": int(np.percentile(a, 99)), "max": int(a.max()), "min": int(a.min())}


def norm(s: str) -> str:
    """Lowercase, punctuation -> space (Unicode-aware). Falls back to the stripped raw text (e.g. emoji-only data)."""
    return " ".join(re.sub(r"[\W_]+", " ", s.lower()).split()) or s.strip()


def shingles(words: list[str]) -> set:
    return {tuple(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def eval_texts(path: Path) -> set[str]:
    out = set()
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("//"):
            continue
        r = json.loads(line)
        for v in r["variants"].values():
            for m in v["messages"]:
                out.add(m["content"])
            for sp in (v.get("spans") or {}).values():
                msg = v["messages"][sp["message"]]["content"]
                out.add(sp["text"] if "text" in sp else msg[sp["start"]:sp["end"]])
    return {t for t in out if t.strip()}


def overlap(rows: list[dict]) -> dict:
    res = {}
    for name in EVAL_SETS:
        texts = {norm(t): t for t in eval_texts(HERE / name)}
        index = collections.defaultdict(set)
        for nt in texts:
            for sh in shingles(nt.split()):
                index[sh].add(nt)
        per_field, examples = {}, []
        for f in TRAIN_FIELDS:
            exact, near = set(), set()
            for r in rows:
                t = norm(r["metadata"][f])
                if t in texts:
                    exact.add(r["id"]); examples.append(("exact", f, r["id"], r["metadata"][f], texts[t]))
                    continue
                sh = shingles(t.split())
                cands = collections.Counter(nt for s in sh for nt in index.get(s, ()))
                for nt, shared in cands.items():
                    score = shared / max(1, min(len(sh), len(shingles(nt.split()))))
                    if score >= NEAR:
                        near.add(r["id"]); examples.append((f"near {score:.2f}", f, r["id"], r["metadata"][f], texts[nt]))
                        break
            per_field[f] = {"exact": len(exact), "near": len(near)}
        res[name] = {"n_eval_texts": len(texts), "rows_matching": per_field,
                     "matches": [{"kind": k, "field": f, "id": i, "words": len(norm(a).split()),
                                  "train": a[:200], "eval": b[:200]} for k, f, i, a, b in examples]}
    return res


def duplicates(rows: list[dict]) -> dict:
    def dup(key) -> dict:
        c = collections.Counter(key(r) for r in rows)
        return {"distinct": len(c), "rows_in_duplicate_groups": sum(v for v in c.values() if v > 1),
                "max_group": max(c.values())}
    md = lambda r: r["metadata"]
    instr = {norm(md(r)["instruction"]) for r in rows}
    return {
        "full_record (all text fields)": dup(lambda r: json.dumps([r["variants"], r["preference"], md(r)["seed_task"]])),
        "injected_prompt": dup(lambda r: (md(r)["instruction"], r["variants"]["injection"]["messages"][1]["content"])),
        "clean_prompt": dup(lambda r: (md(r)["instruction"], md(r)["clean_data"])),
        "clean_prompt_normalised": dup(lambda r: (norm(md(r)["instruction"]), norm(md(r)["clean_data"]))),
        "instruction": dup(lambda r: md(r)["instruction"]),
        "injected_instruction": dup(lambda r: md(r)["injected_instruction"]),
        "seed_task": dup(lambda r: md(r)["seed_task"]),
        "chosen": dup(lambda r: r["preference"]["chosen"]),
        "rejected": dup(lambda r: r["preference"]["rejected"]),
        "chosen == rejected": sum(r["preference"]["chosen"] == r["preference"]["rejected"] for r in rows),
        "seed_task is also some row's trusted instruction": sum(norm(md(r)["seed_task"]) in instr for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(DATA_DIR / "secpo_trainset.jsonl"))
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines()]
    tok = AutoTokenizer.from_pretrained(TOKENIZER, revision=REVISION)

    def n_tokens(texts: list[str]) -> np.ndarray:
        return np.array([len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]])

    def render(msgs, resp=None):
        if resp is None:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tok.apply_chat_template(msgs + [{"role": "assistant", "content": resp}], tokenize=False)

    L = {}
    for cond in ("clean", "injection"):
        msgs = [r["variants"][cond]["messages"] for r in rows]
        L[f"prompt_{cond}"] = n_tokens([render(m) for m in msgs])
        for side in ("chosen", "rejected"):
            L[f"{cond}+{side}"] = n_tokens([render(m, r["preference"][side]) for m, r in zip(msgs, rows)])
    for side in ("chosen", "rejected"):
        L[f"{side}_tokens"] = n_tokens([r["preference"][side] for r in rows])
    L["injection_tokens"] = n_tokens([r["metadata"]["injection"] for r in rows])
    L["example_max"] = np.maximum(L["injection+chosen"], L["injection+rejected"])
    L["clean_example_max"] = np.maximum(L["clean+chosen"], L["clean+rejected"])

    n = len(rows)
    fits = {k: {str(b): int((L[k] <= b).sum()) for b in BUDGETS}
            for k in ("prompt_injection", "prompt_clean", "injection+chosen", "injection+rejected", "example_max",
                      "clean_example_max")}
    meta = [r["metadata"] for r in rows]
    cue_hits = {k: 0 for k in CUES}
    no_cue = 0
    for m in meta:
        hit = [k for k, p in CUES.items() if re.search(p, m["injected_instruction"], re.I)]
        for k in hit:
            cue_hits[k] += 1
        no_cue += not hit
    # Fake headers in completion-style payloads: short lines like "***AI ANSWER***:" / "### response:" / "'''user task''':"
    headers = collections.Counter()
    for m in meta:
        if m["completion_style"]:
            for ln in m["injection"].replace(m["injected_instruction"].strip(), "\n").splitlines():
                s = ln.strip()
                if 0 < len(s) <= 40 and re.fullmatch(r"\W*[A-Za-z][A-Za-z _-]*\W*:?\W*", s) and re.search(r"[#*'`<\[]", s):
                    headers[norm(s)] += 1

    stats = {
        "n_records": n,
        "tokenizer": f"{TOKENIZER}@{REVISION[:7]}",
        "fits": fits,
        "lengths_tokens": {k: pct(v) for k, v in L.items()},
        "lengths_chars": {"chosen": pct([len(r["preference"]["chosen"]) for r in rows]),
                          "rejected": pct([len(r["preference"]["rejected"]) for r in rows])},
        "clean_side_never_longer": bool((L["clean_example_max"] <= L["example_max"]).all()),
        "rejected_longer_than_chosen": int((L["rejected_tokens"] > L["chosen_tokens"]).sum()),
        "attack": {
            "position": dict(collections.Counter(m["position"] for m in meta)),
            "completion_style": dict(collections.Counter(f"{m['position']}/{m['completion_style']}" for m in meta)),
            "seed_verbatim_in_injection": sum(m["seed_verbatim_in_injection"] for m in meta),
            "heuristic_cues_multilabel": {**cue_hits, "none of these cues": no_cue},
            "completion_fake_headers_top": headers.most_common(15),
            "clean_reconstruction": dict(collections.Counter(m["clean_reconstruction"] for m in meta)),
            "secalign_target_in_training_injections": {
                "print exactly": sum("print exactly" in m["injection"].lower() for m in meta),
                "hacked": sum("hacked" in m["injection"].lower() for m in meta)},
        },
        "duplicates": duplicates(rows),
        "eval_overlap": overlap(rows),
    }
    out = Path(args.data).parent
    (out / "stats_olmoe.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    keys = list(L)
    with (out / "lengths_olmoe.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + keys)
        for i, r in enumerate(rows):
            w.writerow([r["id"]] + [int(L[k][i]) for k in keys])
    print(json.dumps({k: stats[k] for k in ("fits", "clean_side_never_longer", "rejected_longer_than_chosen")}, indent=1))
    print(f"wrote {out / 'stats_olmoe.json'} and {out / 'lengths_olmoe.csv'}")


if __name__ == "__main__":
    main()
