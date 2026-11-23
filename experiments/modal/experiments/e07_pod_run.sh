#!/bin/bash
# E07 on the pod: all models, one after another (each needs ~60 GB of the 80 GB GPU). Started by e07_local.sh.
# A model that fails does not stop the others; ALL_DONE records every exit code.
set -uo pipefail
R=/workspace/e07
MODEL=/dev/shm/olmoe_b89a7c4
SNAP=/workspace/security-project/models/huggingface/hub/models--allenai--OLMoE-1B-7B-0125-Instruct/snapshots/b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e
OUT=${OUT:-$R/out}
MODELS=${MODELS:-base lora e05 a40}
CONDS=${CONDS:-straightforward,straightforward_before,ignore,ignore_before,completion,completion_ignore}
EXTRA=${EXTRA:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16 PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock -n 9 || { echo "already running"; exit 1; }
rm -f "$OUT/ALL_DONE" "$OUT/SHA256SUMS"

# SHA-256 of the pinned revision's shards, from the Hub. (The cache's blob file names are Xet hashes, not SHA-256.)
verify_base() {
  (cd "$MODEL" 2>/dev/null && sha256sum -c --quiet - <<'S'
61874210ca7c360f43f8c622cecc12441083d40190eae3b56bc9d6e1c0a30c1e  model-00001-of-00003.safetensors
c523a43b8a17269d5fab33395048a83633f4d1d89c1958570cea738e2bbe80c9  model-00002-of-00003.safetensors
97ae01e3519c52e63a018bca96ab17a89c4cd5cab1c6d742efed0fa5c0e2bb17  model-00003-of-00003.safetensors
S
  )
}

# /dev/shm is empty again after a pod restart.
if [ ! -f "$MODEL/model-00003-of-00003.safetensors" ]; then
  mkdir -p "$MODEL"
  cp -L "$SNAP"/* "$MODEL"/
fi
if [[ "$EXTRA" != *--tiny* ]]; then
  echo "verifying base model"
  if ! verify_base; then
    echo "base model checksum failed" | tee "$OUT/ALL_DONE"
    exit 1
  fi
fi

status=""
for m in $MODELS; do
  rc=75
  for gb in 26 16 8; do
    echo "=== $(date -u +%T) $m, cache $gb GB"
    # EXTRA is deliberately word-split
    /root/venv/bin/python "$R/code/experiments/modal/experiments/e07_generate.py" --model "$m" --model-dir "$MODEL" \
      --ckpt-root "$R/ckpt" --ref-root "$R/ref" --out "$OUT" --conditions "$CONDS" --cache-gb "$gb" $EXTRA
    rc=$?
    [ $rc -eq 75 ] || break   # only an out-of-memory exit is retried; it resumes from the partial file
  done
  status="$status $m=$rc"
done

(cd "$OUT" && find . -type f \( -name 'generations.jsonl' -o -name 'run_info.json' -o -name 'controls.json*' \) \
  -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS)
echo "$status" > "$OUT/ALL_DONE"
echo "=== $(date -u +%T) finished:$status"
