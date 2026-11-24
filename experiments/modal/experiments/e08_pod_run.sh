#!/bin/bash
# E08 on the pod: routed security expert (new expert + its router row). Real-model smoke checks -> SecPO training
# (checkpoints at 20/40/60/80) -> E07 generations (clean + 6 attacks) for GEN_STEPS (set GEN_STEPS= to skip
# generation). ALL_DONE records every exit code.
# Started by e08_local.sh. Rehearsal on CPU: EXTRA_TRAIN='--tiny --limit 24 ...' EXTRA_GEN='--tiny ...'.
#   RUN       run name (default main = E08); models are re_<RUN>_s<step>
#   RUN_ARGS  extra trainer arguments for this run, e.g. the E08 ablation
#             RUN=seinit RUN_ARGS="--re-expert-init security_expert:/workspace/e08/e05_se"
#   SMOKE_ONLY=1  stop after the smoke checks
#   GEN_CACHE KV-cache sizes (GB) to try in turn; only an out-of-memory exit (75) moves to the next, and it resumes
set -uo pipefail
R=/workspace/e08
C=$R/code
OUT=${OUT:-$R/out}
RUN=${RUN:-main}
RUN_ARGS=${RUN_ARGS:-}
SMOKE_ONLY=${SMOKE_ONLY:-0}
SKIP_SMOKE=${SKIP_SMOKE:-0}   # 1: go straight to training (e.g. a variant whose code path an earlier smoke run covered)
EXTRA_TRAIN=${EXTRA_TRAIN:-}
EXTRA_GEN=${EXTRA_GEN:-}
GEN_STEPS=${GEN_STEPS-40 80}
GEN_CACHE=${GEN_CACHE:-26}
E05_SHA=ed9b3c196083e69a23282aa5b8b0e59fb7d5152ab2f978ab050407d694d77367   # E05 secpo_se_v1, as in e07_generate.py
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

if [[ "$RUN_ARGS" == *security_expert:* && "$EXTRA_TRAIN" != *--tiny* ]]; then
  se=${RUN_ARGS#*security_expert:}; se=${se%% *}
  got=$(sha256sum "$se/security_expert.safetensors" | cut -d' ' -f1)
  if [ "$got" != "$E05_SHA" ]; then
    echo "E05 checkpoint sha256 $got, expected $E05_SHA" | tee "$OUT/ALL_DONE"
    exit 1
  fi
  echo "E05 checkpoint sha256 ok"
fi

COMMON="--method routed_expert --data $C/data/secpo_olmoe_v2/pairs.jsonl --dtype float32 --micro-batch 8 --grad-accum 2 \
  --epochs 1 --re-copy most_used --re-target-rate 0.125 --re-row-noise 1e-3 --re-calib-pairs 128 \
  --ref-cache $R/ref_logps.json $RUN_ARGS"

status=""
for run in $RUN; do
  d=$OUT/train_$run
  if [ "$SKIP_SMOKE" != 1 ]; then
    echo "=== $(date -u +%T) $run: smoke"
    $PY $T $COMMON --smoke --out "$d/smoke" $EXTRA_TRAIN
    rc=$?
    status="$status $run-smoke=$rc"
    if [ $rc -ne 0 ] || [ "$SMOKE_ONLY" = 1 ]; then continue; fi
  fi
  echo "=== $(date -u +%T) $run: train"
  $PY $T $COMMON --save-steps 20,40,60,80 --out "$d" $EXTRA_TRAIN
  rc=$?
  status="$status $run-train=$rc"
  [ $rc -eq 0 ] || continue
  for s in $GEN_STEPS; do
    for gb in $GEN_CACHE; do
      echo "=== $(date -u +%T) $run: generate step $s, cache $gb GB"
      $PY $G --model "re_${run}_s$s" --checkpoint "routed_expert:$d/routed_expert_step$s" --model-dir /dev/shm/olmoe_b89a7c4 \
        --out "$OUT/gen" --conditions $CONDS --controls 0 --cache-gb "$gb" $EXTRA_GEN
      rc=$?
      [ $rc -eq 75 ] || break
    done
    status="$status re_${run}_s$s=$rc"
  done
done

# e07_verify_local.py reads gen/SHA256SUMS (paths relative to gen/)
[ -d "$OUT/gen" ] && (cd "$OUT/gen" && find . -type f \( -name 'generations.jsonl' -o -name 'run_info.json' \) \
  -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS)
(cd "$OUT" && find . -type f \( -name 'generations.jsonl' -o -name 'run_info.json' -o -name 'metrics.jsonl' \
  -o -name 'config.json' -o -name 'smoke_report.json' -o -name '*.safetensors' -o -name '*_config.json' \) \
  -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS)
echo "$status" > "$OUT/ALL_DONE"
echo "=== $(date -u +%T) finished:$status"
