#!/bin/bash
# E09 from the laptop.
#   e09_local.sh sync    code, data, saved generations (prompt check) and the three checkpoints -> pod;
#                        installs peft==0.21.3 into /root/venv (LoRA) if missing. No GPU.
#   e09_local.sh start   start e09_pod_run.sh on the pod (survives a dropped connection), then pull
#   e09_local.sh pull    copy outputs back every 30 s; when the pod finishes, copy once more and verify checksums
# Pod address: POD_HOST, POD_PORT. MODELS (default "base lora e05 routed"). Rehearsal: DRY=1 DRY_OUT=DIR (pod CPU).
set -uo pipefail
HERE=$(dirname "$0")
cd "$HERE/../../.." || exit 1
HOST=${POD_HOST:?set POD_HOST, e.g. root@1.2.3.4}
PORT=${POD_PORT:?set POD_PORT}
SSH="ssh -p $PORT -i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15"
UP="rsync -rc"   # no -t: a RunPod network volume refuses to set file times
R=/workspace/e09
O=experiments/modal/outputs
X=experiments/modal/experiments

if [ "${DRY:-0}" = 1 ]; then
  ROUT=$R/dryrun; LOUT=${DRY_OUT:?set DRY_OUT}
  RENV="OUT=$ROUT CUDA_VISIBLE_DEVICES= EXTRA='--tiny --device cpu --limit 6'"
else
  ROUT=$R/out; LOUT=$O/e09_probing/acts
  RENV="OUT=$ROUT"
fi
[ -n "${MODELS:-}" ] && RENV="$RENV MODELS='$MODELS'"

sync_up() {
  $SSH "$HOST" "mkdir -p $R/code/$X $R/code/data $R/ref $R/ckpt/lora $R/ckpt/e05 $R/ckpt/routed_step80" || return 1
  $UP -e "$SSH" --exclude __pycache__ moe_interp "$HOST:$R/code/" || return 1
  $UP -e "$SSH" $X/e07_generate.py $X/e09_capture_resid.py $X/e09_pod_run.sh "$HOST:$R/code/$X/" || return 1
  $UP -e "$SSH" data/alpacafarm_secalign_208x6.jsonl "$HOST:$R/code/data/" || return 1
  $UP -e "$SSH" $O/e07_af208_six_attacks/base/generations.jsonl "$HOST:$R/ref/base.jsonl" || return 1
  $UP -e "$SSH" $O/e07_af208_six_attacks/lora/generations.jsonl "$HOST:$R/ref/lora.jsonl" || return 1
  $UP -e "$SSH" $O/e07_af208_six_attacks/e05/generations.jsonl "$HOST:$R/ref/e05.jsonl" || return 1
  $UP -e "$SSH" $O/e08_routed_expert/gen/re_main_s80/generations.jsonl "$HOST:$R/ref/routed.jsonl" || return 1
  $UP -e "$SSH" $O/e03_olmoe_security_expert/secpo_v2/adapter/ "$HOST:$R/ckpt/lora/" || return 1
  $UP -e "$SSH" $O/e05_security_expert/secpo_se_v1/security_expert/ "$HOST:$R/ckpt/e05/" || return 1
  $UP -e "$SSH" $O/e08_routed_expert/train_main/routed_expert_step80/ "$HOST:$R/ckpt/routed_step80/" || return 1
  $SSH "$HOST" "/root/venv/bin/python -c 'import peft' 2>/dev/null || /root/venv/bin/pip install -q peft==0.21.3" || return 1
  # e07_generate.py pins the LoRA and E05 sha256; the routed checkpoint is recorded by the capture (b7ef759b...)
  $SSH "$HOST" "cd $R/ckpt && sha256sum lora/adapter_model.safetensors e05/security_expert.safetensors routed_step80/routed_expert.safetensors" || return 1
  echo "synced"
}

start() {
  $SSH "$HOST" "mkdir -p $ROUT && rm -f $ROUT/ALL_DONE && { env $RENV nohup setsid bash $R/code/$X/e09_pod_run.sh > $ROUT/run.log 2>&1 < /dev/null & } && echo started"
}

pull() {
  mkdir -p "$LOUT"
  while true; do
    finished=$($SSH "$HOST" "cat $ROUT/ALL_DONE 2>/dev/null")
    if rsync -rt -e "$SSH" --exclude .lock "$HOST:$ROUT/" "$LOUT/" && [ -n "$finished" ]; then
      echo "pod finished:$finished"
      break
    fi
    if [ -f "$LOUT/run.log" ]; then tail -n 1 "$LOUT/run.log"; fi
    sleep "${POLL:-30}"
  done
  (cd "$LOUT" && shasum -a 256 -c SHA256SUMS --quiet) && echo "checksums: all files match the pod; the pod can be shut down"
}

# Experiment 3: INTERV=local dir from e09_make_interventions.py; PART=val|test (records_<PART>.json)
PART=${PART:-val}
IOUT=$R/interv_out_$PART
ILOUT=$O/e09_probing/intervene_$PART
interv_sync() {
  $SSH "$HOST" "mkdir -p $R/interv $R/code/$X" || return 1
  $UP -e "$SSH" $X/e07_generate.py $X/e09_intervene.py $X/e09_intervene_run.sh "$HOST:$R/code/$X/" || return 1
  $UP -e "$SSH" --exclude __pycache__ moe_interp "$HOST:$R/code/" || return 1
  $UP -e "$SSH" "${INTERV:?set INTERV}/" "$HOST:$R/interv/" || return 1
  echo "synced interventions"
}
interv_start() {
  local env="OUT=$IOUT RECORDS=records_$PART.json"
  [ -n "${MODELS:-}" ] && env="$env MODELS='$MODELS'"
  $SSH "$HOST" "mkdir -p $IOUT && rm -f $IOUT/ALL_DONE && { env $env nohup setsid bash $R/code/$X/e09_intervene_run.sh > $IOUT/run.log 2>&1 < /dev/null & } && echo started"
}
interv_pull() {
  mkdir -p "$ILOUT"
  while true; do
    finished=$($SSH "$HOST" "cat $IOUT/ALL_DONE 2>/dev/null")
    if rsync -rt -e "$SSH" --exclude .lock "$HOST:$IOUT/" "$ILOUT/" && [ -n "$finished" ]; then
      echo "pod finished:$finished"; break
    fi
    sleep "${POLL:-30}"
  done
  (cd "$ILOUT" && shasum -a 256 -c SHA256SUMS --quiet) && echo "checksums: all files match the pod"
}

case "${1:-}" in
  sync) sync_up ;;
  start) start && pull ;;
  pull) pull ;;
  intervene-sync) interv_sync ;;
  intervene-start) interv_start && interv_pull ;;
  intervene-pull) interv_pull ;;
  *) echo "usage: $0 sync|start|pull|intervene-sync|intervene-start|intervene-pull"; exit 2 ;;
esac
