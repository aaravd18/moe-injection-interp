#!/bin/bash
# E09 on the pod: final-prompt-token residual capture (e09_capture_resid.py) for base, LoRA (secpo_v2), E05 and the
# E08 routed expert (main, step 80), one after another. Started by e09_local.sh. ALL_DONE records every exit code.
#   MODELS   default "base lora e05 routed"
#   EXTRA    extra capture arguments (rehearsal: EXTRA='--tiny --device cpu --limit 6')
set -uo pipefail
R=/workspace/e09
C=$R/code
OUT=${OUT:-$R/out}
MODELS=${MODELS:-base lora e05 routed}
EXTRA=${EXTRA:-}
MODEL_DIR=/dev/shm/olmoe_b89a7c4
SNAP=/workspace/security-project/models/huggingface/hub/models--allenai--OLMoE-1B-7B-0125-Instruct/snapshots/b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e
export HF_HUB_CACHE=/workspace/security-project/models/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/venv/bin/python
CAP=$C/experiments/modal/experiments/e09_capture_resid.py

mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock -n 9 || { echo "already running"; exit 1; }
rm -f "$OUT/ALL_DONE" "$OUT/SHA256SUMS"

if [[ "$EXTRA" != *--tiny* ]]; then
  [ -f "$MODEL_DIR/model-00003-of-00003.safetensors" ] || { mkdir -p "$MODEL_DIR"; cp -L "$SNAP"/* "$MODEL_DIR"/; }
  (cd "$MODEL_DIR" && sha256sum -c --quiet - <<'S'
61874210ca7c360f43f8c622cecc12441083d40190eae3b56bc9d6e1c0a30c1e  model-00001-of-00003.safetensors
c523a43b8a17269d5fab33395048a83633f4d1d89c1958570cea738e2bbe80c9  model-00002-of-00003.safetensors
97ae01e3519c52e63a018bca96ab17a89c4cd5cab1c6d742efed0fa5c0e2bb17  model-00003-of-00003.safetensors
S
  ) || { echo "base model checksum failed" | tee "$OUT/ALL_DONE"; exit 1; }
  echo "base model verified"
fi

status=""
for m in $MODELS; do
  echo "=== $(date -u +%T) $m"
  case $m in
    routed) args="--checkpoint routed_expert:$R/ckpt/routed_step80 --ref $R/ref/routed.jsonl" ;;
    *)      args="--ckpt-root $R/ckpt --ref $R/ref/$m.jsonl" ;;
  esac
  # EXTRA and args are deliberately word-split
  $PY $CAP --model "$m" --model-dir "$MODEL_DIR" --data $C/data/alpacafarm_secalign_208x6.jsonl --out "$OUT" $args $EXTRA
  status="$status $m=$?"
done

(cd "$OUT" && find . -type f \( -name '*.npz' -o -name '*.json' \) -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS)
echo "$status" > "$OUT/ALL_DONE"
echo "=== $(date -u +%T) finished:$status"
