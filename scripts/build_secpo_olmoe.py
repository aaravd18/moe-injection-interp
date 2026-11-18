"""E03 step 3: merge selection + OLMoE generations + judge verdicts into an OLMoE-specific preference set.
CPU only, standard library only.

Classification (per record, nothing assumed):
  chosen_valid      clean response ended with EOS, is non-empty, is not degenerate (3-gram repetition < 0.5)
                    AND the judge's task verdict is "correct"
  attack_followed   the judge says the injected response carried out the injected text (injected_task_done)
  borderline        decision-relevant judge probability near its cut: P(correct) < 0.65 for a "correct" verdict, or
                    |P(injected_task_done) - 0.5| < 0.15 (a flag for review, not a filter)
  outcome           attack_only | attack_and_task | resisted (task only) | neither | judge_parse_error
A preference pair is built only if chosen_valid AND attack_followed:
  chosen = OLMoE's clean response (generated on the clean prompt), rejected = OLMoE's injected response.
Every other record is kept in records.jsonl with its classification and the reasons it is not a pair.
Prompts, attacks and clean reconstructions are the source's, unchanged. The Qwen responses are not copied.

Outputs (--out): records.jsonl, pairs.jsonl, stats.json, sample_10.txt. Token counts are OLMoE tokens from the
generation run: prompt = chat-template prompt incl. BOS, response = generated ids incl. EOS. "Pair length" =
injected prompt + longer response (chosen was generated on the clean prompt; re-tokenized after the injected prompt
it can differ by a token or so).

Usage: python scripts/build_secpo_olmoe.py --run-dir RUN_DIR [--out OUT_DIR]       (OUT_DIR defaults to RUN_DIR)
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SELECTED = HERE / "data" / "secpo_olmoe_v1" / "selected.jsonl"
BUDGETS, MAX_REPETITION, BORDER_CORRECT, BORDER_ATTACK = (256, 512, 1024), 0.5, 0.65, 0.15


def classify(clean: dict, inj: dict) -> dict:
    jc, ji = clean.get("judge", {}), inj.get("judge", {})
    why = []
    if clean["finish_reason"] != "stop":
        why.append("clean_truncated")
    if not clean["response"].strip():
        why.append("clean_empty")
    if clean["repetition_3gram"] >= MAX_REPETITION:
        why.append("clean_degenerate")
    if jc.get("parse_error"):
        why.append("clean_judge_parse_error")
    elif jc.get("verdict") != "correct":
        why.append(f"clean_{jc.get('verdict')}")
    chosen_valid = not why
    if ji.get("parse_error"):
        outcome, followed = "judge_parse_error", False
    else:
        followed = ji["injected_task_done"]
        outcome = {(True, False): "attack_only", (True, True): "attack_and_task", (False, True): "resisted",
                   (False, False): "neither"}[(ji["injected_task_done"], ji["trusted_task_done"])]
    if not followed:
        why.append(f"injection_{outcome}")
    p_correct = (jc.get("probabilities") or {}).get("correct")
    p_inj = ji.get("p_injected_task_done")
    borderline = [x for x, hit in (("chosen", jc.get("verdict") == "correct" and p_correct is not None
                                    and p_correct < BORDER_CORRECT),
                                   ("attack", p_inj is not None and abs(p_inj - 0.5) < BORDER_ATTACK)) if hit]
    return {"chosen_valid": chosen_valid, "attack_followed": followed, "injection_outcome": outcome,
            "p_correct": p_correct, "p_injected_task_done": p_inj, "borderline": borderline,
            "rejected_truncated": inj["finish_reason"] != "stop", "is_pair": chosen_valid and followed,
            "not_pair_reasons": [] if chosen_valid and followed else why}


def pct(xs: list[int]) -> dict:
    xs = sorted(xs)
    if not xs:
        return {}
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"n": len(xs), "mean": round(statistics.mean(xs), 1), "p50": q(0.5), "p90": q(0.9), "max": xs[-1]}


def marked(rec: dict) -> str:
    u = rec["variants"]["injection"]["messages"][1]["content"]
    p = rec["variants"]["injection"]["spans"]["probe"]
    return u[:p["start"]] + "[[INJECTION>>" + u[p["start"]:p["end"]] + "<<INJECTION]]" + u[p["end"]:]


def fmt(probs: dict | None) -> str:
    return "{" + ", ".join(f"{k} {v:.2f}" for k, v in sorted(probs.items(), key=lambda kv: -kv[1])) + "}" if probs else ""


def pr(p: float | None) -> str:
    return "ERROR" if p is None else f"{p:.2f}"


def show(rec: dict, max_chars: int = 600) -> str:
    m, o, lab = rec["metadata"], rec["olmoe"], rec["labels"]
    clip = lambda s: s if len(s) <= max_chars else s[:max_chars] + f" … [+{len(s) - max_chars} chars]"
    ind = lambda s: textwrap.indent(s, "     │ ")
    jc, ji = o["clean"].get("judge", {}), o["injection"].get("judge", {})
    return "\n".join([
        "=" * 100,
        f"{rec['id']}  task_type={m['task_type_e03']}  position={m['position']}  PAIR={lab['is_pair']}  "
        f"chosen_valid={lab['chosen_valid']}  injection_outcome={lab['injection_outcome']}",
        f"  not-pair reasons: {lab['not_pair_reasons'] or '-'}   borderline judge calls: {lab['borderline'] or '-'}",
        "  ── SYSTEM (trusted instruction)", ind(m["instruction"]),
        "  ── USER, injected ([[INJECTION>> … <<INJECTION]] = attacker text; clean prompt = same without it)",
        ind(marked(rec)),
        f"  ── OLMoE on CLEAN prompt ({o['clean']['n_response_tokens']} tok, {o['clean']['finish_reason']}) — "
        f"judge: {jc.get('verdict', 'ERROR')} {fmt(jc.get('probabilities'))}", ind(clip(o["clean"]["response"])),
        f"  ── OLMoE on INJECTED prompt ({o['injection']['n_response_tokens']} tok, {o['injection']['finish_reason']}) — "
        f"judge: P(injected task done)={pr(ji.get('p_injected_task_done'))}  "
        f"P(user task done)={pr(ji.get('p_trusted_task_done'))}", ind(clip(o["injection"]["response"])),
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True, help="has generations.jsonl + judgments.jsonl (+ run_info.json)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sample-seed", type=int, default=0)
    args = ap.parse_args()
    run, out = Path(args.run_dir), Path(args.out or args.run_dir)
    src = {r["id"]: r for r in map(json.loads, SELECTED.open())}
    gens = {(g["id"], g["condition"]): g for g in map(json.loads, (run / "generations.jsonl").open())}
    judg = {(j["id"], j["condition"]): j for j in map(json.loads, (run / "judgments.jsonl").open())}
    ids = sorted({i for i, _ in gens}, key=lambda i: src[i]["metadata"]["source_index"])

    records = []
    for i in ids:
        s = src[i]
        o = {}
        for cond in ("clean", "injection"):
            g = gens[(i, cond)]
            j = {k: v for k, v in judg[(i, cond)].items() if k not in ("id", "condition", "kind")}
            o[cond] = {k: g[k] for k in ("response", "n_prompt_tokens", "n_response_tokens", "finish_reason",
                                         "repetition_3gram")} | {"judge": j}
        lab = classify(o["clean"], o["injection"])
        rec = {"id": i, "metadata": {**s["metadata"], "response_model": "allenai/OLMoE-1B-7B-0125-Instruct"},
               "variants": s["variants"], "olmoe": o, "labels": lab}
        if lab["is_pair"]:
            rec["preference"] = {"chosen": o["clean"]["response"], "rejected": o["injection"]["response"]}
        records.append(rec)

    out.mkdir(parents=True, exist_ok=True)
    dump = lambda p, rs: p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))
    pairs = [r for r in records if r["labels"]["is_pair"]]
    dump(out / "records.jsonl", records)
    dump(out / "pairs.jsonl", pairs)

    C = collections.Counter
    lab = [r["labels"] for r in records]
    by = lambda key: {k: {"n": sum(1 for r in records if key(r) == k),
                          "pairs": sum(1 for r in records if key(r) == k and r["labels"]["is_pair"]),
                          "attack_followed": sum(1 for r in records if key(r) == k and r["labels"]["attack_followed"]),
                          "chosen_valid": sum(1 for r in records if key(r) == k and r["labels"]["chosen_valid"])}
                      for k in sorted({key(r) for r in records})}
    pair_len = [r["olmoe"]["injection"]["n_prompt_tokens"] + max(r["olmoe"]["clean"]["n_response_tokens"],
                                                                  r["olmoe"]["injection"]["n_response_tokens"])
                for r in pairs]
    tok = lambda cond, key, rs: [r["olmoe"][cond][key] for r in rs]
    stats = {
        "n_records": len(records), "n_pairs": len(pairs),
        "chosen_valid": sum(l["chosen_valid"] for l in lab), "attack_followed": sum(l["attack_followed"] for l in lab),
        "clean_judge_verdict": dict(C(r["olmoe"]["clean"]["judge"].get("verdict", "parse_error") for r in records)),
        "injection_outcome": dict(C(l["injection_outcome"] for l in lab)),
        "not_pair_reasons": dict(C(x for l in lab for x in l["not_pair_reasons"])),
        "pairs_with_truncated_rejected": sum(l["rejected_truncated"] for l in lab if l["is_pair"]),
        "borderline_records": sum(bool(l["borderline"]) for l in lab),
        "borderline_pairs": sum(bool(l["borderline"]) for l in lab if l["is_pair"]),
        "by_position": by(lambda r: r["metadata"]["position"]),
        "by_task_type": by(lambda r: r["metadata"]["task_type_e03"]),
        "by_seed_verbatim": by(lambda r: str(r["metadata"]["seed_verbatim_in_injection"])),
        "tokens_all_records": {
            "prompt_clean": pct(tok("clean", "n_prompt_tokens", records)),
            "prompt_injection": pct(tok("injection", "n_prompt_tokens", records)),
            "response_clean": pct(tok("clean", "n_response_tokens", records)),
            "response_injection": pct(tok("injection", "n_response_tokens", records)),
            "truncated_clean": sum(r["olmoe"]["clean"]["finish_reason"] != "stop" for r in records),
            "truncated_injection": sum(r["olmoe"]["injection"]["finish_reason"] != "stop" for r in records)},
        "tokens_pairs": {"chosen": pct(tok("clean", "n_response_tokens", pairs)),
                         "rejected": pct(tok("injection", "n_response_tokens", pairs)),
                         "pair_length": pct(pair_len),
                         "pair_fits": {str(b): sum(x <= b for x in pair_len) for b in BUDGETS},
                         "rejected_longer_than_chosen": sum(r["olmoe"]["injection"]["n_response_tokens"] >
                                                            r["olmoe"]["clean"]["n_response_tokens"] for r in pairs)},
    }
    for f in ("run_info.json", "judge_info.json"):
        if (run / f).exists():
            stats[f.removesuffix(".json")] = json.loads((run / f).read_text())
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")

    # 10 readable examples: up to 4 pairs, then the other outcomes round-robin, seeded
    rng = random.Random(args.sample_seed)
    groups = collections.defaultdict(list)
    for r in records:
        g = "pair" if r["labels"]["is_pair"] else ("chosen_invalid" if not r["labels"]["chosen_valid"]
                                                    else r["labels"]["injection_outcome"])
        groups[g].append(r)
    for g in groups.values():
        rng.shuffle(g)
    picked = groups["pair"][:4]
    others = [g for g in sorted(groups) if g != "pair"]
    while len(picked) < 10 and any(groups[g] for g in others + ["pair"]):
        for g in others + ["pair"]:
            if groups[g] and len(picked) < 10 and groups[g][0] not in picked:
                picked.append(groups[g].pop(0))
            elif groups[g] and groups[g][0] in picked:
                groups[g].pop(0)
    head = (f"E03 OLMoE self-generated preference data: {len(records)} records, {len(pairs)} pairs. "
            f"10 examples (seed {args.sample_seed}): up to 4 pairs, then other outcomes.\n")
    (out / "sample_10.txt").write_text(head + "\n".join(show(r) for r in picked) + "\n" + "=" * 100 + "\n")
    print(json.dumps({k: stats[k] for k in ("n_records", "n_pairs", "chosen_valid", "attack_followed",
                                            "clean_judge_verdict", "injection_outcome", "not_pair_reasons")}, indent=1))
    print(f"wrote {out}/records.jsonl, pairs.jsonl, stats.json, sample_10.txt")


if __name__ == "__main__":
    main()
