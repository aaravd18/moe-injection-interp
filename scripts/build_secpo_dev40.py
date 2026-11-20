"""E06 dev set: 40 records of the SecPO grouped *val* split with the injection placed after the data.

Why this set (E06 logbook, section 2): it is already held out from training (grouped split: no shared instruction, seed
task, injection, clean data or wrapper with train) and was checked against AF-208 for overlap at selection time; it has
the clean/injection variants and metadata (instruction, clean_data, injection) that e03_judge_flash.py reads; and its
attacks (the paper's PAIR-optimised injections) are strong enough that attack success varies across checkpoints,
unlike AF-208's naive "Print exactly 'hacked!'" (~0 for every trained model). Only position == "end" records are used,
matching AF-208's placement and the judge prompt ("appended the <injected_text> to the end of the data").

Selection (no model outputs are looked at): records sorted by id, shuffled with random.Random(seed), then taken
round-robin over task_type_e03 so every type is represented before any type gets a second record, until 40.

Records are copied verbatim from data/secpo_olmoe_v1/selected.jsonl (plus metadata.dev_set).

Usage: python scripts/build_secpo_dev40.py   -> data/secpo_dev40.jsonl + data/secpo_dev40.manifest.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "secpo_olmoe_v1" / "selected.jsonl"
SPLIT = ROOT / "experiments" / "modal" / "artifacts" / "splits" / "secpo_olmoe_v1_grouped_v1.json"
AF = ROOT / "data" / "alpacafarm_secalign_208.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "secpo_dev40.jsonl"))
    args = ap.parse_args()

    split = json.loads(SPLIT.read_text())
    val = set(split["ids"]["val"])
    recs = sorted((json.loads(l) for l in SRC.open()), key=lambda r: r["id"])
    pool = [r for r in recs if r["id"] in val and r["metadata"]["position"] == "end"]
    random.Random(args.seed).shuffle(pool)
    by_type = collections.OrderedDict()
    for r in pool:
        by_type.setdefault(r["metadata"]["task_type_e03"], []).append(r)
    chosen = []
    while len(chosen) < args.n:
        for t in sorted(by_type):
            if by_type[t] and len(chosen) < args.n:
                chosen.append(by_type[t].pop(0))
    chosen.sort(key=lambda r: r["id"])

    # sanity: disjoint from AF-208 (instruction or data text), from train, and end-placed
    af = [json.loads(l) for l in AF.open()]
    af_text = {r["metadata"]["instruction"].strip().lower() for r in af} | {r["metadata"]["input"].strip().lower() for r in af}
    train = set(split["ids"]["train"])
    for r in chosen:
        m = r["metadata"]
        assert r["id"] not in train
        assert m["instruction"].strip().lower() not in af_text and m["clean_data"].strip().lower() not in af_text
        assert r["variants"]["injection"]["messages"][1]["content"].startswith(m["clean_data"].rstrip()[:20])
        m["dev_set"] = "secpo_dev40"

    out = Path(args.out)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen))
    manifest = {"name": "secpo_dev40", "source": str(SRC.relative_to(ROOT)),
                "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(), "split": split["name"],
                "pool": f"val ids with position == 'end' ({len(pool)} records)", "n": len(chosen),
                "seed": args.seed, "ids": [r["id"] for r in chosen],
                "task_types": dict(collections.Counter(r["metadata"]["task_type_e03"] for r in chosen)),
                "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "ids"}, indent=1))


if __name__ == "__main__":
    main()
