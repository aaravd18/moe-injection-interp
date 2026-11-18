"""Print randomly chosen SecPO training records in a readable form, marking exactly where the injection sits.

In the injected user message:
  [[INJECTION>> ... <<INJECTION]]   all attacker-controlled text (span "probe": fake answer/header + instruction)
  {{ ... }}                         the injected instruction itself (source field `rejected_input_whole`)
Everything outside [[...]] is the original data, which is exactly the clean user message. On a terminal the
injection is also shown in red.

Usage (standard library only):
    python scripts/inspect_secpo_trainset.py                  # 8 random records, seed 0
    python scripts/inspect_secpo_trainset.py --n 5 --seed 3 --max-chars 600
    python scripts/inspect_secpo_trainset.py --ids secpo_00000 secpo_01830 --completion-only
Reads data/secpo_qwen36_trainset/secpo_trainset.jsonl, or the committed sample.jsonl if the full file is absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import textwrap
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "secpo_qwen36_trainset"


def mark(text: str, probe: dict, inj: dict, color: bool) -> str:
    red, bold, off = ("\033[31m", "\033[1m", "\033[0m") if color else ("", "", "")
    marks = [(probe["start"], 0, f"{red}[[INJECTION>>"), (max(inj["start"], probe["start"]), 1, f"{{{{{bold}"),
             (min(inj["end"], probe["end"]), 2, f"{off}{red}}}}}"), (probe["end"], 3, f"<<INJECTION]]{off}")]
    out, last = [], 0
    for pos, _, m in sorted(marks):
        out += [text[last:pos], m]
        last = pos
    return "".join(out + [text[last:]])


def clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f" … [+{len(s) - n} chars]"


def block(title: str, body: str) -> str:
    return f"  ── {title}\n" + textwrap.indent(body, "     │ ")


def show(r: dict, lengths: dict, max_chars: int, color: bool) -> str:
    m, inj_v = r["metadata"], r["variants"]["injection"]
    system, user = (x["content"] for x in inj_v["messages"])
    sp = inj_v["spans"]
    head = (f"{r['id']}  position={m['position']}  completion_style={m['completion_style']}  "
            f"seed_verbatim={m['seed_verbatim_in_injection']}  clean_from={m['clean_reconstruction']}")
    if r["id"] in lengths:
        L = lengths[r["id"]]
        head += (f"\n  OLMoE tokens: injected prompt {L['prompt_injection']}, chosen {L['chosen_tokens']}, "
                 f"rejected {L['rejected_tokens']}, longest full sequence {L['example_max']}")
    parts = [
        "=" * 100, head,
        block("SYSTEM (trusted instruction)", m["instruction"]),
        block(f"USER, injected (data {len(user)} chars; injection at chars {sp['probe']['start']}–{sp['probe']['end']})",
              mark(user, sp["probe"], sp["injected_instruction"], color)),
        block("USER, clean (original data)", r["variants"]["clean"]["messages"][1]["content"]),
        block("seed task (what `rejected` answers)", m["seed_task"].strip()),
        block("CHOSEN (preferred: does the trusted task)", clip(r["preference"]["chosen"], max_chars)),
        block("REJECTED (dispreferred: does the injected task)", clip(r["preference"]["rejected"], max_chars)),
    ]
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", nargs="*", help="show these record ids instead of a random draw")
    ap.add_argument("--completion-only", action="store_true", help="draw only completion-style injections")
    ap.add_argument("--max-chars", type=int, default=400, help="truncate responses (0 = no limit)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    path = Path(args.data) if args.data else DATA_DIR / "secpo_trainset.jsonl"
    if not path.exists():
        path = DATA_DIR / "sample.jsonl"
        print(f"(full dataset not built; reading {path}. Build it with scripts/build_secpo_trainset.py)\n")
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    lengths = {}
    if (DATA_DIR / "lengths_olmoe.csv").exists():
        lengths = {d["id"]: d for d in csv.DictReader((DATA_DIR / "lengths_olmoe.csv").open())}
    if args.ids:
        by_id = {r["id"]: r for r in rows}
        picked = [by_id[i] for i in args.ids]
    else:
        pool = [r for r in rows if r["metadata"]["completion_style"]] if args.completion_only else rows
        picked = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    color = sys.stdout.isatty() and not args.no_color
    for r in picked:
        print(show(r, lengths, args.max_chars or 10**9, color))
    print("=" * 100)


if __name__ == "__main__":
    main()
