"""Grouped, reproducible train/val split of data/secpo_olmoe_v1 for E03 SecPO training (CPU, deterministic).

The split is made over all 1,500 records (not just the 654 pairs), so later experiments that use non-pair records
(e.g. `resisted`) inherit the same partition. Records are first joined into groups with union-find. Two records share
a group if they share any of:
  * the injection wrapper: the injected instruction with the seed task cut out (seed-verbatim rows; the selection
    allowed each wrapper twice),
  * the trusted instruction, the seed task or the injected instruction (normalised; unique by selection, kept as a
    guard),
  * the clean data string (14 short strings like "apple", "N/A" repeat),
  * a near-duplicate wrapper: word 4-gram containment >= 0.5 between wrappers (or full injected instructions for
    rewritten injections). This catches "forget / delete / ignore all previous instructions and instead ..." families.
Whole groups then go to val in a seeded order until val holds VAL_FRAC of the pairs within each attack position and
VAL_FRAC of all records; every other group is train. There is no test split: evaluation stays on
data/alpacafarm_secalign_208.jsonl (and the 1,181 held-out completion-style ids in selection.json).

Usage:
    python scripts/split_secpo_olmoe.py      # writes experiments/modal/artifacts/splits/secpo_olmoe_v1_grouped_v1.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "secpo_olmoe_v1" / "records.jsonl"
OUT = ROOT / "experiments" / "modal" / "artifacts" / "splits" / "secpo_olmoe_v1_grouped_v1.json"
NAME = "secpo_olmoe_v1_grouped_v1"
VAL_FRAC = 0.15
NEAR_DUP = 0.5
SEED = 0


def norm(s: str) -> str:
    return " ".join(re.sub(r"[\W_]+", " ", s.lower()).split())


def wrapper(m: dict) -> str | None:
    """Injected instruction minus the verbatim seed task, i.e. the attack template. None for rewritten injections."""
    inj, seed = norm(m["injected_instruction"]), norm(m["seed_task"])
    return " ".join(inj.replace(seed, " ").split()) if m["seed_verbatim_in_injection"] and seed in inj else None


def grams(s: str, n: int = 4) -> set:
    w = s.split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def group_records(recs: list[dict]) -> list[list[int]]:
    par = list(range(len(recs)))

    def find(x: int) -> int:
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    keys = collections.defaultdict(list)
    for i, r in enumerate(recs):
        m = r["metadata"]
        for kind in ("instruction", "seed_task", "injected_instruction", "clean_data"):
            keys[(kind, norm(m[kind]))].append(i)
        if (w := wrapper(m)) is not None:
            keys[("wrapper", w)].append(i)
    for members in keys.values():
        for j in members[1:]:
            par[find(j)] = find(members[0])
    g = [grams(wrapper(r["metadata"]) or norm(r["metadata"]["injected_instruction"])) for r in recs]
    for i, j in itertools.combinations(range(len(recs)), 2):
        if g[i] and g[j] and len(g[i] & g[j]) / min(len(g[i]), len(g[j])) >= NEAR_DUP:
            par[find(j)] = find(i)
    comps = collections.defaultdict(list)
    for i in range(len(recs)):
        comps[find(i)].append(i)
    return sorted(comps.values(), key=lambda v: recs[v[0]]["id"])  # canonical order before the seeded shuffle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    raw = Path(args.data).read_bytes()
    recs = [json.loads(l) for l in raw.decode().splitlines()]
    pair = [r["labels"]["is_pair"] for r in recs]
    pos = [r["metadata"]["position"] for r in recs]
    groups = group_records(recs)

    pair_target = {p: round(VAL_FRAC * sum(pair[i] for i in range(len(recs)) if pos[i] == p)) for p in ("start", "end")}
    rec_target = round(VAL_FRAC * len(recs))
    order = list(range(len(groups)))
    random.Random(SEED).shuffle(order)
    val_pairs, val_recs, val_groups = collections.Counter(), 0, set()
    for gi in order:
        g = groups[gi]
        add = collections.Counter(pos[i] for i in g if pair[i])
        if val_recs + len(g) > rec_target or any(val_pairs[p] + add[p] > pair_target[p] for p in add):
            continue
        val_groups.add(gi)
        val_pairs.update(add)
        val_recs += len(g)

    split_of = {}
    for gi, g in enumerate(groups):
        for i in g:
            split_of[i] = "val" if gi in val_groups else "train"
    ids = {s: [recs[i]["id"] for i in range(len(recs)) if split_of[i] == s] for s in ("train", "val")}
    pair_ids = {s: [recs[i]["id"] for i in range(len(recs)) if split_of[i] == s and pair[i]] for s in ("train", "val")}

    # Leakage check: no grouping key may appear on both sides.
    for kind in ("instruction", "seed_task", "injected_instruction", "clean_data"):
        a = {norm(recs[i]["metadata"][kind]) for i in range(len(recs)) if split_of[i] == "train"}
        b = {norm(recs[i]["metadata"][kind]) for i in range(len(recs)) if split_of[i] == "val"}
        assert not a & b, f"{kind} leaks across splits"
    wa = {wrapper(recs[i]["metadata"]) for i in range(len(recs)) if split_of[i] == "train"} - {None}
    wb = {wrapper(recs[i]["metadata"]) for i in range(len(recs)) if split_of[i] == "val"} - {None}
    assert not wa & wb, "injection wrapper leaks across splits"

    def breakdown(s: str) -> dict:
        rows = [recs[i] for i in range(len(recs)) if split_of[i] == s]
        prs = [r for r in rows if r["labels"]["is_pair"]]
        return {"records": len(rows), "pairs": len(prs),
                "pairs_strict": sum(not r["labels"]["borderline"] for r in prs),
                "pairs_rejected_truncated": sum(r["labels"]["rejected_truncated"] for r in prs),
                "pairs_by_position": dict(sorted(collections.Counter(r["metadata"]["position"] for r in prs).items())),
                "pairs_by_outcome": dict(sorted(collections.Counter(r["labels"]["injection_outcome"] for r in prs).items())),
                "pairs_by_task_type": dict(sorted(collections.Counter(r["metadata"]["task_type_e03"] for r in prs).items()))}

    sizes = sorted((len(g) for g in groups), reverse=True)
    out = {
        "name": NAME, "data": "data/secpo_olmoe_v1/records.jsonl", "data_sha256": hashlib.sha256(raw).hexdigest(),
        "generator": "scripts/split_secpo_olmoe.py",
        "method": "union-find groups over shared instruction / seed task / injected instruction / clean data / "
                  f"injection wrapper, plus wrapper word-4-gram containment >= {NEAR_DUP}; seeded shuffle of groups, "
                  f"greedy fill of val to {VAL_FRAC:.0%} of pairs per position and {VAL_FRAC:.0%} of records",
        "seed": SEED, "val_frac": VAL_FRAC,
        "groups": {"n": len(groups), "multi_record": sum(s > 1 for s in sizes), "largest": sizes[:5]},
        "summary": {s: breakdown(s) for s in ("train", "val")},
        "ids": ids, "pair_ids": pair_ids,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("groups", "summary")}, indent=1))


if __name__ == "__main__":
    main()
