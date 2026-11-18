"""Select N (default 1,500; v1 pilot drawn from the first 500) diverse, short SecPO training examples for the OLMoE self-generated preference set (E03), plus a
20-example pilot subset. CPU only, deterministic (seed 0). Standard library only.

Input: data/secpo_qwen36_trainset/secpo_trainset.jsonl (prompts, attacks and clean reconstructions are kept as they
are) and lengths_olmoe.csv / stats_olmoe.json from scripts/analyze_secpo_trainset.py.

Eligibility:
  * not completion-style (all 1,181 are HELD OUT for a later generalization eval; ids saved)
  * clean prompt reconstructed from the original data (clean_reconstruction == "chosen_data")
  * injected prompt <= 128 OLMoE tokens (chat template incl. BOS)
  * no exact text match with an eval set (stats_olmoe.json), and not one of the 3 duplicated clean prompts
Selection: N/2 "start" + N/2 "end" injections, spread evenly over a keyword-based task type of the TRUSTED
instruction (10 types, first rule that matches). Within each (position, task type) cell, a seeded random draw that
skips any example whose instruction, seed task, injected instruction or injection template (seed-verbatim
injections with the seed removed) was already used. Cells that run short are topped up round-robin from the others.
Pilot: 20 records, 10 per position, round-robin over task types.

Output: data/secpo_olmoe_v1/selected.jsonl (the N source records, unchanged, plus metadata.task_type_e03 and
metadata.pilot) and data/secpo_olmoe_v1/selection.json (criteria, counts, ids, held-out completion-style ids).

Usage: python scripts/select_secpo_olmoe.py [--n 1500]
The draw is a seeded round-robin, so a larger --n extends the same sequence: the 1,500 contain the earlier 500
and the same pilot (checked when the 1,500 were drawn).
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "data" / "secpo_qwen36_trainset"
OUT = HERE / "data" / "secpo_olmoe_v1"
N_DEFAULT, N_PILOT, MAX_PROMPT_TOKENS, SEED = 1500, 20, 128, 0
V1_PER_POSITION = 250   # the first (pilot) draw was 500 = 250 + 250; kept as stage 1 of every larger draw

# First matching rule wins. Keyword heuristics on the trusted instruction, not ground-truth labels.
TASK_TYPES = [
    ("code", r"\b(code|program|function|python|java(script)?|sql|html|css|c\+\+|algorithm|script|regex|json|api|"
             r"variable|loop|array)\b"),
    ("translate", r"\btranslat"),
    ("math", r"\b(calculate|compute|solve|equation|sum of|average|percent|multipl|divid|area|perimeter|"
             r"probability|convert .* (to|into) (celsius|fahrenheit|meters?|miles|kilograms?|pounds|feet|inches|"
             r"cups|grams|dollars|euros|binary|decimal))\b"),
    ("summarize", r"\b(summari[sz]e|summary|condense|main (idea|point)s?|key points?|tl;?dr)\b"),
    ("classify", r"\b(classify|categori[sz]e|label|sentiment|determine (whether|if)|decide (whether|if)|is (it|this|the "
                 r"following) (a |an )?(true|false|fact|opinion)|true or false|detect|assign|which (category|type|"
                 r"class))\b"),
    ("rewrite", r"\b(rewrite|re-write|edit|rephrase|paraphrase|reword|correct|fix|proofread|rearrange|reorder|shorten|"
                r"simplify|transform|replace|change|convert|make (it|this|the)|passive|active voice|tense|punctuat|"
                r"grammar|capitali[sz]e|insert|delete|remove|reverse)\b"),
    ("extract", r"\b(identify|find|extract|list|name|pick|select|choose|locate|highlight|count|which|match)\b"),
    ("generate", r"\b(generate|create|write|compose|construct|design|come up|develop|formulate|invent|draft|"
                 r"brainstorm|story|poem|slogan|headline|title|tweet|email|letter)\b"),
    ("explain", r"\b(explain|describe|analy[sz]e|compare|contrast|evaluate|assess|why|how|what|suggest|provide|give|"
                r"predict|discuss|answer|interpret|tell)\b"),
]


def task_type(instr: str) -> str:
    s = instr.lower()
    return next((name for name, pat in TASK_TYPES if re.search(pat, s)), "other")


def norm(s: str) -> str:
    return " ".join(re.sub(r"[\W_]+", " ", s.lower()).split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=N_DEFAULT, help="even; half start, half end")
    N = ap.parse_args().n
    rows = [json.loads(l) for l in (SRC / "secpo_trainset.jsonl").open()]
    plen = {r["id"]: int(r["prompt_injection"]) for r in csv.DictReader((SRC / "lengths_olmoe.csv").open())}
    stats = json.loads((SRC / "stats_olmoe.json").read_text())
    overlap = {m["id"] for o in stats["eval_overlap"].values() for m in o["matches"] if m["kind"] == "exact"}
    clean_ct = collections.Counter((r["metadata"]["instruction"], r["metadata"]["clean_data"]) for r in rows)

    heldout = [r["id"] for r in rows if r["metadata"]["completion_style"]]
    reasons = collections.Counter()
    cand = []
    for r in rows:
        m = r["metadata"]
        why = ("completion_style_heldout" if m["completion_style"] else
               "clean_reconstruction_fallback" if m["clean_reconstruction"] != "chosen_data" else
               f"injected_prompt_over_{MAX_PROMPT_TOKENS}_tokens" if plen[r["id"]] > MAX_PROMPT_TOKENS else
               "eval_overlap_exact" if r["id"] in overlap else
               "duplicate_clean_prompt" if clean_ct[(m["instruction"], m["clean_data"])] > 1 else None)
        if why:
            reasons[why] += 1
        else:
            cand.append(r)

    rng = random.Random(SEED)
    cells = collections.defaultdict(list)
    for r in cand:
        cells[(r["metadata"]["position"], task_type(r["metadata"]["instruction"]))].append(r)
    for k in sorted(cells):
        rng.shuffle(cells[k])
    pool_types = dict(sorted(collections.Counter(t for (_, t), v in cells.items() for _ in v).items()))

    used = {"instr": set(), "seed": set(), "inj": set(), "tmpl": collections.Counter()}

    def ok(r) -> bool:
        m = r["metadata"]
        tmpl = norm(m["injected_instruction"].replace(m["seed_task"].strip(), " ")) if m["seed_verbatim_in_injection"] else None
        if norm(m["instruction"]) in used["instr"] or norm(m["seed_task"]) in used["seed"] \
                or norm(m["injected_instruction"]) in used["inj"] or (tmpl is not None and used["tmpl"][tmpl] >= 2):
            return False
        used["instr"].add(norm(m["instruction"])); used["seed"].add(norm(m["seed_task"]))
        used["inj"].add(norm(m["injected_instruction"]))
        if tmpl is not None:
            used["tmpl"][tmpl] += 1
        return True

    types = sorted({t for _, t in cells})
    # Staged round-robin over task types: stage 1 fills each position to V1_PER_POSITION (start first, then end) and
    # reproduces the original 500-record draw exactly; later stages continue the same queues, so a larger N is a
    # strict superset of a smaller one. Empty cells drop out automatically.
    queues = {pos: {t: cells[(pos, t)] for t in types} for pos in ("start", "end")}
    got: dict[str, list] = {"start": [], "end": []}
    for quota in sorted({min(V1_PER_POSITION, N // 2), N // 2}):
        for pos in ("start", "end"):
            while len(got[pos]) < quota and any(queues[pos].values()):
                for t in types:
                    q = queues[pos][t]
                    while q:
                        r = q.pop()
                        if ok(r):
                            got[pos].append((t, r))
                            break
                    if len(got[pos]) == quota:
                        break
    picked: list = got["start"] + got["end"]

    by_pos_type = collections.defaultdict(list)
    for t, r in picked:
        r["metadata"]["task_type_e03"] = t
        by_pos_type[(r["metadata"]["position"], t)].append(r)
    pilot = set()
    for pos in ("start", "end"):
        k = 0
        while sum(1 for i in pilot if i[1] == pos) < N_PILOT // 2:
            t = types[k % len(types)]
            pool = [r for r in by_pos_type[(pos, t)] if (r["id"], pos) not in pilot]
            if pool:
                pilot.add((pool[0]["id"], pos))
            k += 1
    pilot_ids = {i for i, _ in pilot}
    picked_rows = sorted((r for _, r in picked), key=lambda r: r["metadata"]["source_index"])
    for r in picked_rows:
        r["metadata"]["pilot"] = r["id"] in pilot_ids

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selected.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked_rows))
    meta = [r["metadata"] for r in picked_rows]
    sel = {
        "source": "data/secpo_qwen36_trainset/secpo_trainset.jsonl", "generator": "scripts/select_secpo_olmoe.py",
        "seed": SEED, "n": len(picked_rows), "n_pilot": len(pilot_ids),
        "criteria": {"max_injected_prompt_tokens_olmoe": MAX_PROMPT_TOKENS, "exclude_completion_style": True,
                     "require_clean_reconstruction": "chosen_data", "exclude_eval_overlap": "exact",
                     "unique": ["instruction", "seed_task", "injected_instruction"], "max_uses_per_template": 2,
                     "balance": f"position {N // 2}/{N // 2} x task_type_e03 round-robin"},
        "pool": {"source_records": len(rows), "eligible": len(cand), "excluded_by_reason": dict(reasons)},
        "counts": {
            "position": dict(collections.Counter(m["position"] for m in meta)),
            "task_type_e03": dict(sorted(collections.Counter(m["task_type_e03"] for m in meta).items())),
            "position_x_task_type": {f"{p}/{t}": len(v) for (p, t), v in sorted(by_pos_type.items())},
            "seed_verbatim_in_injection": sum(m["seed_verbatim_in_injection"] for m in meta),
            "pilot_task_type": dict(collections.Counter(m["task_type_e03"] for m in meta if m["pilot"])),
            "eligible_pool_task_type": pool_types,
        },
        "ids": [r["id"] for r in picked_rows],
        "pilot_ids": sorted(pilot_ids),
        "heldout_completion_style_ids": heldout,
    }
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1) + "\n")
    print(json.dumps({k: sel[k] for k in ("n", "n_pilot", "pool")}, indent=1))
    print(json.dumps(sel["counts"], indent=1))


if __name__ == "__main__":
    main()
