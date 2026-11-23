#!/bin/bash
# E07 from the laptop.
#   e07_local.sh sync    code, data and references -> pod (no GPU)
#   e07_local.sh start   start the run on the pod (survives a dropped connection), then pull
#   e07_local.sh pull    copy outputs back every 20 s; when the pod finishes, copy once more and verify
# Pod address: POD_HOST, POD_PORT. Optional CONDS=clean,straightforward,... Rehearsal: DRY=1 DRY_OUT=DIR.
set -uo pipefail
HERE=$(dirname "$0")
cd "$HERE/../../.." || exit 1
HOST=${POD_HOST:?set POD_HOST, e.g. root@1.2.3.4}
PORT=${POD_PORT:?set POD_PORT}
SSH="ssh -p $PORT -i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15"
R=/workspace/e07
O=experiments/modal/outputs
X=experiments/modal/experiments

if [ "${DRY:-0}" = 1 ]; then
  ROUT=$R/dryrun
  LOUT=${DRY_OUT:?set DRY_OUT}
  VERIFY=--dry-run
  RENV="OUT=$ROUT CUDA_VISIBLE_DEVICES= EXTRA='--tiny --controls 0 --max-new-tokens 3'"
else
  ROUT=$R/out
  LOUT=$O/e07_af208_six_attacks
  VERIFY=
  RENV="OUT=$ROUT"
  if [ -n "${MODELS:-}" ]; then RENV="$RENV MODELS='$MODELS'"; fi
  if [ -n "${CONDS:-}" ]; then RENV="$RENV CONDS=$CONDS"; fi
fi

send_ref() {
  rsync -rt -e "$SSH" "$O/$2/generations.jsonl" "$HOST:$R/ref/$1.jsonl"
}

sync_up() {
  $SSH "$HOST" "mkdir -p $R/code/$X $R/code/data $R/ref" || return 1
  rsync -rt -e "$SSH" --exclude __pycache__ moe_interp "$HOST:$R/code/" || return 1
  rsync -rt -e "$SSH" "$X/e07_generate.py" "$X/e07_pod_run.sh" "$HOST:$R/code/$X/" || return 1
  rsync -rt -e "$SSH" data/alpacafarm_secalign_208x6.jsonl "$HOST:$R/code/data/" || return 1
  send_ref base e03_olmoe_security_expert/eval_v1/base_af208 || return 1
  send_ref lora e03_olmoe_security_expert/eval_v2/secpo_v2_af208_fp32 || return 1
  send_ref e05 e05_security_expert/eval_se_v1/secpo_se_v1_af208_fp32 || return 1
  send_ref a40 e06_se_lr_sweep/af208/A40 || return 1
  send_ref b80 e06_se_lr_sweep/af208/B80 || return 1
  echo "synced"
}

start() {
  # ALL_DONE is cleared before launch, so that pull cannot mistake an earlier run's marker for this one's.
  # The braces matter: only the runner is backgrounded, with every stream detached from ssh.
  $SSH "$HOST" "mkdir -p $ROUT && rm -f $ROUT/ALL_DONE && { env $RENV nohup setsid bash $R/code/$X/e07_pod_run.sh > $ROUT/run.log 2>&1 < /dev/null & } && echo started"
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
    sleep "${POLL:-20}"
  done
  HF_HUB_OFFLINE=1 .venv/bin/python "$X/e07_verify_local.py" --out "$LOUT" $VERIFY
}

case "${1:-}" in
  sync) sync_up ;;
  start) start && pull ;;
  pull) pull ;;
  *) echo "usage: $0 sync|start|pull"; exit 2 ;;
esac
