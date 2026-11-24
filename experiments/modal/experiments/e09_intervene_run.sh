#!/bin/bash
# E09 experiment 3 on the pod: e09_intervene.py for each model in MODELS with its configs_<model>.json, on the records
# in RECORDS (default records_val.json). Inputs in $R/interv (uploaded by e09_local.sh intervene-sync).
set -uo pipefail
R=/workspace/e09
C=$R/code
I=$R/interv
OUT=${OUT:-$R/interv_out_val}
MODELS=${MODELS:-base routed}
RECORDS=${RECORDS:-records_val.json}
EXTRA=${EXTRA:-}
export HF_HUB_CACHE=/workspace/security-project/models/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/venv/bin/python

mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock -n 9 || { echo "already running"; exit 1; }
rm -f "$OUT/ALL_DONE" "$OUT/SHA256SUMS"
status=""
for m in $MODELS; do
  case $m in
    routed) args="--checkpoint routed_expert:$R/ckpt/routed_step80" ;;
    *)      args="--ckpt-root $R/ckpt" ;;
  esac
  for gb in 10 8 6; do
    echo "=== $(date -u +%T) $m, cache $gb GB"
    $PY $C/experiments/modal/experiments/e09_intervene.py --model "$m" --model-dir /dev/shm/olmoe_b89a7c4 \
      --configs "$I/configs_$m.json" --vectors "$I/vectors.npz" --records "$I/$RECORDS" --out "$OUT" \
      --cache-gb "$gb" $args $EXTRA
    rc=$?
    [ $rc -eq 0 ] && break   # a crash (e.g. OOM) is retried with a smaller cache; finished configs are skipped
  done
  status="$status $m=$rc"
done
(cd "$OUT" && find . -type f \( -name 'generations.jsonl' -o -name 'run_info.json' \) -print0 | sort -z \
  | xargs -0 -r sha256sum > SHA256SUMS)
echo "$status" > "$OUT/ALL_DONE"
echo "=== $(date -u +%T) finished:$status"
