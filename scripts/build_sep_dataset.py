"""Convert the SEP dataset ("Should it be Executed or Processed?", Zverev et al., ICLR 2025) into our JSONL format.

SEP item: a task (trusted instruction), a data passage (untrusted), a probe question, and a witness string that
appears in the output only if the probe was executed. Layout follows SEP's own evaluation: task -> system message,
data -> user message.

Each SEP item becomes one sample with four conditions:

  clean             system = task                 user = data
  benign            system = task                 user = data + neutral sentence (same position, matched length)
  injection         system = task                 user = data + probe          (probe treated as DATA: should be ignored)
  user_instruction  system = task + probe         user = data                  (probe from the TRUSTED side: should be executed)

Only SEP items whose probe is appended at the end of the system prompt (appended_type "r?") are used, so the trusted
side is laid out identically in every condition; the data-side position (start "l" / end "r") is kept as a variable.
Spans: "task" (system), "data" (user, original passage), "probe" (injection / user_instruction), "insert" (benign).

Usage: python scripts/build_sep_dataset.py SEP_dataset.json data/sep_600.jsonl --n 600 --seed 0
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def load_benign(path: Path) -> list[str]:
    out = [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    bad = [s for s in out if s.endswith("?") or s.split()[0].lower() in {"tell", "please", "name", "write", "list", "answer"}]
    if bad:
        raise ValueError(f"Benign sentences must be declarative: {bad}")
    return out


def extract_probe(e: dict) -> str:
    sc, si = e["system_prompt_clean"], e["system_prompt_instructed"]
    if not si.startswith(sc):
        raise ValueError("probe not appended at the end of the system prompt")
    probe = si[len(sc):].strip()
    pi, pc = e["prompt_instructed"], e["prompt_clean"]
    pos = e["info"]["appended_type"][1]
    expect = (pc + " " + probe) if pos == "r" else (probe + " " + pc)
    if pi != expect:
        raise ValueError("data-side probe insertion is not a plain prefix/suffix")
    return probe


def insert(data: str, sentence: str, pos: str) -> str:
    return f"{data} {sentence}" if pos == "r" else f"{sentence} {data}"


def convert(e: dict, idx: int, benign_pool: list[str], rng: random.Random) -> dict:
    probe = extract_probe(e)
    pos = e["info"]["appended_type"][1]
    task, data = e["system_prompt_clean"], e["prompt_clean"]
    n = len(probe.split())
    best = min(abs(len(s.split()) - n) for s in benign_pool)
    benign = rng.choice([s for s in benign_pool if abs(len(s.split()) - n) == best])

    def variant(system: str, user: str, extra: dict) -> dict:
        spans = {"task": {"message": 0, "text": task}, "data": {"message": 1, "text": data}}
        spans.update(extra)
        return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "spans": spans}

    info = e["info"]
    return {
        "id": f"sep_{idx:05d}",
        "metadata": {
            "source": "SEP", "sep_index": idx, "task_type": info["type"], "subtask": info["subtask"],
            "probe_id": info["appended_task_id"], "probe_position_in_data": "end" if pos == "r" else "start",
            "insistent": info["is_insistent"], "probe": probe, "benign_sentence": benign,
            "probe_words": n, "benign_words": len(benign.split()),
        },
        "variants": {
            "clean": variant(task, data, {}),
            "benign": variant(task, insert(data, benign, pos), {"insert": {"message": 1, "text": benign}}),
            "injection": variant(task, insert(data, probe, pos), {"probe": {"message": 1, "text": probe}}),
            "user_instruction": variant(f"{task} {probe}", data, {"probe": {"message": 0, "text": probe}}),
        },
        "evaluation": {"witness": e["witness"], "attack_target": e["witness"]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sep_json")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benign", default=str(HERE / "data" / "benign_sentences.txt"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    pool = load_benign(Path(args.benign))
    sep = json.load(open(args.sep_json))
    # Keep items with the probe at the end of the system prompt; stratify over (data position, insistent) and
    # spread over probes and subtasks.
    cand = [(i, e) for i, e in enumerate(sep) if e["info"]["appended_type"][0] == "r"]
    strata = collections.defaultdict(list)
    for i, e in cand:
        strata[(e["info"]["appended_type"][1], e["info"]["is_insistent"])].append((i, e))
    picked = []
    per = args.n // len(strata)
    for key in sorted(strata):
        items = strata[key][:]
        rng.shuffle(items)
        seen_sub: set = set()
        first = [x for x in items if not (x[1]["info"]["subtask"] in seen_sub or seen_sub.add(x[1]["info"]["subtask"]))]
        rest = [x for x in items if x not in first]
        picked += (first + rest)[:per]
    picked.sort()
    rows, skipped = [], 0
    for i, e in picked:
        try:
            rows.append(convert(e, i, pool, rng))
        except ValueError:
            skipped += 1
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter((r["metadata"]["probe_position_in_data"], r["metadata"]["insistent"]) for r in rows)
    print(f"wrote {len(rows)} samples ({skipped} skipped) to {args.out}; strata {dict(c)}; "
          f"{len({r['metadata']['probe_id'] for r in rows})} probes, {len({r['metadata']['subtask'] for r in rows})} subtasks")


if __name__ == "__main__":
    main()
