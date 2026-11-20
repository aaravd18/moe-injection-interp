"""Judge generations with Gemini 3.5 Flash Lite *while* the GPU pod is still producing them (E05/E06 helper).

Same judge as e03_judge_flash.py (its build_jobs / generate / parse / RequestRate are imported, so prompts, schema and
settings are identical). What this adds:
  * polls the pod (rsync over SSH) for each run dir's generations.partial.jsonl (written batch by batch by
    e03_generate.py) or the final generations.jsonl, and judges rows as they appear;
  * spreads requests over several API keys (GEMINI_API_KEYS="k1,k2", optional weights "k1*3,k2"), with ONE sliding-window rate limiter per key
    shared across all run dirs, so a key never exceeds --rpm;
  * parse_error rows are dropped and re-sent on the next poll.
A run dir is complete when its final generations.jsonl has been copied and every row has a label. Afterwards, run
e03_judge_flash.py once per dir (it sends nothing when all rows are judged) to sort the file and write
judge_flash_info.json.

Usage (laptop):
    GEMINI_API_KEYS=k1,k2 python experiments/modal/experiments/e06_stream_judge.py --ssh "root@IP -p PORT" \
        --remote-root /workspace/out/dev40 --local-root OUT/dev40 --names base,lora_v2,e05_final \
        --data data/secpo_dev40.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e03_judge_flash as J  # noqa: E402

OUT_NAME = "judgments_flash.jsonl"


def pull(ssh: str, remote: str, local: Path) -> None:
    local.mkdir(parents=True, exist_ok=True)
    user_host, *port = ssh.split()
    e = f"ssh -o ConnectTimeout=20 -i {Path.home() / '.ssh' / 'id_ed25519'} {' '.join(port)}"
    for f in ("generations.partial.jsonl", "generations.jsonl", "run_info.json"):
        subprocess.run(["rsync", "-az", "-e", e, f"{user_host}:{remote}/{f}", str(local / f)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # a partial last line while the file is being copied
            pass
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ssh", required=True, help='"user@host -p PORT"')
    ap.add_argument("--remote-root", required=True)
    ap.add_argument("--local-root", required=True)
    ap.add_argument("--names", required=True, help="comma-separated run dir names under both roots")
    ap.add_argument("--data", required=True)
    ap.add_argument("--rpm", type=int, default=13, help="per key")
    ap.add_argument("--workers-per-key", type=int, default=2)
    ap.add_argument("--poll", type=int, default=45, help="seconds between polls when idle")
    args = ap.parse_args()
    # "k1,k2*3": optional integer weight per key = its share of the jobs (e.g. to fit per-key daily request budgets)
    spec = [k.rsplit("*", 1) if "*" in k else (k, "1") for k in os.environ.get("GEMINI_API_KEYS", "").split(",") if k]
    if not spec:
        raise SystemExit("set GEMINI_API_KEYS=k1,k2")
    keys = [k for k, _ in spec]
    sched = [i for i, (_, w) in enumerate(spec) for _ in range(int(w))]  # job j goes to key sched[j % len(sched)]
    records = {r["id"]: r for r in map(json.loads, Path(args.data).open())}
    names = [n for n in args.names.split(",") if n]
    rates = [J.RequestRate(args.rpm) for _ in keys]
    lock = threading.Lock()
    done_dirs: set[str] = set()
    t0 = time.time()

    def judge(name: str, job: dict, k: int) -> None:
        rates[k].wait()
        try:
            resp = J.generate(job, keys[k])
        except Exception as e:  # noqa: BLE001  e.g. http.client.RemoteDisconnected, not caught by generate()
            resp = {"error": f"exception: {type(e).__name__}: {e}"}
        rec = {"id": job["id"], "condition": job["condition"], "model": J.MODEL} | J.parse(job, resp)
        with lock, (Path(args.local_root) / name / OUT_NAME).open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with cf.ThreadPoolExecutor(len(keys) * args.workers_per_key) as pool:
        while len(done_dirs) < len(names):
            futures, status = [], []
            for name in names:
                if name in done_dirs:
                    continue
                local = Path(args.local_root) / name
                pull(args.ssh, f"{args.remote_root}/{name}", local)
                final = local / "generations.jsonl"
                gens = read_jsonl(final) if final.exists() else read_jsonl(local / "generations.partial.jsonl")
                out = local / OUT_NAME
                judged = [r for r in read_jsonl(out) if not r.get("parse_error")]
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in judged))  # drop errors
                have = {(r["id"], r["condition"]) for r in judged}
                todo = J.build_jobs(records, [g for g in gens if (g["id"], g["condition"]) not in have])
                for i, job in enumerate(todo):
                    futures.append(pool.submit(judge, name, job, sched[i % len(sched)]))
                status.append(f"{name}:{len(have)}/{len(gens)}{'(final)' if final.exists() else ''}")
                if final.exists() and not todo and len(have) == len(gens) > 0:
                    done_dirs.add(name)
            print(f"[{time.time() - t0:5.0f}s] {' '.join(status)}  submitting {len(futures)}", flush=True)
            for f in futures:
                f.result()
            if not futures and len(done_dirs) < len(names):
                time.sleep(args.poll)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
