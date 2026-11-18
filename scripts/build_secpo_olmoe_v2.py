"""Build data/secpo_olmoe_v2: SecPO preference pairs with OLMoE responses, constructed as in the paper (CPU only).

SecPO (arXiv:2610.41998, Algorithm 1 line 13): y_w = f_undef(z) and y_l = f_undef(z'), where z is the clean sample and
z' the seed injected task on its own. Here f_undef is the frozen OLMoE-1B-7B-0125-Instruct (greedy, bf16, 512 new
tokens):
  chosen   = its response to the clean prompt          (full1500 run, condition "clean")
  rejected = its response to metadata.seed_task alone  (seed1500 run, condition "seed_task": one user message)
No judge and no filter: every one of the 1,500 selected records is a pair (the paper uses its pairs unfiltered).
Prompts (variants.clean / variants.injection) are copied unchanged from data/secpo_olmoe_v1/selected.jsonl. The OLMoE
responses to the injected prompt (v1's `rejected`) are not used.

Usage:
    python scripts/build_secpo_olmoe_v2.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "experiments" / "modal" / "outputs" / "e03_olmoe_security_expert"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_gens(path: Path, cond: str) -> dict:
    out = {}
    for r in map(json.loads, path.open()):
        if r["condition"] == cond:
            assert r["id"] not in out, r["id"]
            out[r["id"]] = r
    return out


def stats(xs: list[int]) -> dict:
    s = sorted(xs)
    return {"mean": round(statistics.mean(s), 1), "p50": s[len(s) // 2], "p90": s[int(0.9 * len(s))], "max": s[-1]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selected", default=str(ROOT / "data" / "secpo_olmoe_v1" / "selected.jsonl"))
    ap.add_argument("--clean-run", default=str(RUNS / "full1500"))
    ap.add_argument("--seed-run", default=str(RUNS / "seed1500"))
    ap.add_argument("--out", default=str(ROOT / "data" / "secpo_olmoe_v2"))
    args = ap.parse_args()

    sel, clean_run, seed_run = Path(args.selected), Path(args.clean_run), Path(args.seed_run)
    rows = [json.loads(l) for l in sel.open()]
    clean = load_gens(clean_run / "generations.jsonl", "clean")
    seed = load_gens(seed_run / "generations.jsonl", "seed_task")
    ids = [r["id"] for r in rows]
    assert set(clean) == set(ids) == set(seed) and len(ids) == len(set(ids)), "runs do not cover the selection"
    for run in (clean_run, seed_run):  # both runs generated from this exact selection file
        assert json.loads((run / "run_info.json").read_text())["data_sha256"] == sha(sel), run

    keep = ("response", "n_response_tokens", "finish_reason", "repetition_3gram", "prompt_text", "n_prompt_tokens")
    out_rows = []
    for r in rows:
        c, s = clean[r["id"]], seed[r["id"]]
        assert s["prompt_text"].endswith(f"<|user|>\n{r['metadata']['seed_task'].strip()}\n<|assistant|>\n"), r["id"]
        out_rows.append({
            "id": r["id"], "metadata": r["metadata"], "variants": r["variants"],
            "olmoe": {"clean": {k: c[k] for k in keep}, "seed_task": {k: s[k] for k in keep}},
            "preference": {"chosen": c["response"], "rejected": s["response"]},
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = out / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))
    ch = [r["olmoe"]["clean"] for r in out_rows]
    rj = [r["olmoe"]["seed_task"] for r in out_rows]
    st = {
        "n_pairs": len(out_rows),
        "construction": "chosen = OLMoE(clean prompt); rejected = OLMoE(seed_task alone, user turn); no filtering",
        "chosen_tokens": stats([x["n_response_tokens"] for x in ch]),
        "rejected_tokens": stats([x["n_response_tokens"] for x in rj]),
        "chosen_truncated_512": sum(x["finish_reason"] != "stop" for x in ch),
        "rejected_truncated_512": sum(x["finish_reason"] != "stop" for x in rj),
        "rejected_longer_than_chosen": sum(b["n_response_tokens"] > a["n_response_tokens"] for a, b in zip(ch, rj)),
        "chosen_repetition_ge_0.5": sum(x["repetition_3gram"] >= 0.5 for x in ch),
        "rejected_repetition_ge_0.5": sum(x["repetition_3gram"] >= 0.5 for x in rj),
        "by_position": {p: sum(r["metadata"]["position"] == p for r in out_rows) for p in ("start", "end")},
        "inputs": {"selected": str(sel.relative_to(ROOT)), "selected_sha256": sha(sel),
                   "clean_generations_sha256": sha(clean_run / "generations.jsonl"),
                   "seed_generations_sha256": sha(seed_run / "generations.jsonl")},
        "pairs_sha256": sha(pairs),
    }
    (out / "stats.json").write_text(json.dumps(st, indent=2) + "\n")
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
