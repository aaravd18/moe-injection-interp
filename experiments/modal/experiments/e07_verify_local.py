"""E07: check, on the laptop, that the pod can be shut down (every generation is here and intact).

For each model: file checksums equal the pod's, exactly one row per record x condition, the saved prompt equals this
repo's dataset rendered with OLMoE's template, the token ids decode to the saved response, and run_info names the
expected data, dtype, settings and checkpoint. Exits non-zero, and says so, if anything is off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e07_generate import ATTACKS, MODEL_ID, REPO, REVISION, WEIGHTS, read_jsonl, sha256  # noqa: E402

from moe_interp.prompts import render_prompt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="base lora e05 a40 b80")
    ap.add_argument("--dry-run", action="store_true", help="tiny-model rehearsal outputs")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    data = REPO / "data" / "alpacafarm_secalign_208x6.jsonl"
    recs, out, bad, sums = read_jsonl(data), Path(args.out), [], {}
    if (out / "SHA256SUMS").exists():
        for line in (out / "SHA256SUMS").read_text().splitlines():
            h, p = line.split(maxsplit=1)
            sums[p.removeprefix("./")] = h
    for p, h in sums.items():
        if not (out / p).exists() or sha256(out / p) != h:
            bad.append(f"{p}: missing or differs from the pod's copy")

    rendered: dict = {}
    for m in args.models.split():
        g, i = out / m / "generations.jsonl", out / m / "run_info.json"
        if not (g.exists() and i.exists()):
            bad.append(f"{m}: no finished output")
            continue
        info, rows = json.loads(i.read_text()), read_jsonl(g)
        if f"{m}/generations.jsonl" not in sums or info["generations_sha256"] != sha256(g):
            bad.append(f"{m}: checksum missing or differs from run_info")
        conds = tuple(info["conditions"])
        if not set(ATTACKS) <= set(conds):
            bad.append(f"{m}: attacks missing from {conds}")
        if conds not in rendered:
            rendered[conds] = {(r["id"], c): render_prompt(tok, r["variants"][c]["messages"], True, None, {})
                               for c in conds for r in recs}
        want = rendered[conds]
        ids_sha = hashlib.sha256(json.dumps([x.input_ids for x in want.values()]).encode()).hexdigest()
        if info["prompt_ids_sha256"] != ids_sha:
            bad.append(f"{m}: prompt token ids differ")
        if len(rows) != len(want) or {(r["id"], r["condition"]) for r in rows} != set(want):
            bad.append(f"{m}: {len(rows)} rows, expected {len(want)} distinct")
            continue
        if any(r["prompt_text"] != want[r["id"], r["condition"]].text for r in rows):
            bad.append(f"{m}: prompts differ from the dataset")
        if any(tok.decode(r["response_ids"], skip_special_tokens=True) != r["response"] for r in rows):
            bad.append(f"{m}: responses do not match their token ids")
        if info["data_sha256"] != sha256(data) or info["dtype"] != "float32":
            bad.append(f"{m}: wrong data or dtype")
        if not args.dry_run:
            expect = (WEIGHTS[m][1] if WEIGHTS[m] else None) if m in WEIGHTS else info["checkpoint_sha256"]
            if info["tiny_random"] or info["checkpoint_sha256"] != expect or info["max_new_tokens"] != 512 \
                    or (m not in WEIGHTS and not expect):
                bad.append(f"{m}: wrong checkpoint or settings")
        print(f"{m}: {len(rows)} rows, {sum(r['finish_reason'] == 'length' for r in rows)} hit the token limit, "
              f"{info['generate_seconds']} s, peak {info['peak_gpu_gb']} GB")

    if bad:
        print("NOT SAFE TO SHUT DOWN:\n  " + "\n  ".join(bad))
        sys.exit(1)
    if args.dry_run:
        print(f"REHEARSAL VERIFIED: {out}")
    else:
        print(f"VERIFIED: all generations are in {out}. The pod can be shut down.")


if __name__ == "__main__":
    main()
