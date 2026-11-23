#!/bin/bash
# E08 on the pod: routed security expert (new expert + its router row). Real-model smoke checks -> SecPO training
# (checkpoints at 20/40/60/80) -> E07 generations (clean + 6 attacks) for GEN_STEPS (set GEN_STEPS= to skip
# generation). ALL_DONE records every exit code.
# Started by e08_local.sh. Rehearsal on CPU: EXTRA_TRAIN='--tiny --limit 24 ...' EXTRA_GEN='--tiny ...'.
set -uo pipefail
R=/workspace/e08
C=$R/code
OUT=${OUT:-$R/out}
EXTRA_TRAIN=${EXTRA_TRAIN:-}
EXTRA_GEN=${EXTRA_GEN:-}
GEN_STEPS=${GEN_STEPS-40 80}
export HF_HUB_CACHE=/workspace/security-project/models/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/venv/bin/python
T=$C/experiments/modal/experiments/e03_secpo_train.py
G=$C/experiments/modal/experiments/e07_generate.py
CONDS=clean,straightforward,straightforward_before,ignore,ignore_before,completion,completion_ignore

mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock -n 9 || { echo "already running"; exit 1; }
rm -f "$OUT/ALL_DONE" "$OUT/SHA256SUMS"

COMMON="--method routed_expert --data $C/data/secpo_olmoe_v2/pairs.jsonl --dtype float32 --micro-batch 8 --grad-accum 2 \
  --epochs 1 --re-copy most_used --re-target-rate 0.125 --re-row-noise 1e-3 --re-calib-pairs 128 \
  --ref-cache $R/ref_logps.json"

status=""
for run in main; do
  d=$OUT/train_$run
  echo "=== $(date -u +%T) $run: smoke"
  $PY $T $COMMON --smoke --out "$d/smoke" $EXTRA_TRAIN
  rc=$?
  if [ $rc -ne 0 ]; then status="$status $run-smoke=$rc"; continue; fi
  echo "=== $(date -u +%T) $run: train"
  $PY $T $COMMON --save-steps 20,40,60,80 --out "$d" $EXTRA_TRAIN
  rc=$?
  status="$status $run-train=$rc"
  [ $rc -eq 0 ] || continue
  for s in $GEN_STEPS; do
    echo "=== $(date -u +%T) $run: generate step $s"
    $PY $G --model "re_${run}_s$s" --checkpoint "routed_expert:$d/routed_expert_step$s" --model-dir /dev/shm/olmoe_b89a7c4 \
      --out "$OUT/gen" --conditions $CONDS --controls 0 $EXTRA_GEN
    status="$status re_${run}_s$s=$?"
  done
done

(cd "$OUT" && find . -type f \( -name 'generations.jsonl' -o -name 'run_info.json' -o -name 'metrics.jsonl' \
  -o -name 'config.json' -o -name 'smoke_report.json' -o -name '*.safetensors' -o -name '*_config.json' \) \
  -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS)
echo "$status" > "$OUT/ALL_DONE"
echo "=== $(date -u +%T) finished:$status"
