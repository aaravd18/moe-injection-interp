#!/bin/bash
# E08 from the laptop.
#   e08_local.sh sync    code, data, the SecPO reference cache and the E05 security expert -> pod (no GPU)
#   e08_local.sh start   start e08_pod_run.sh on the pod (survives a dropped connection), then pull
#   e08_local.sh pull    copy outputs back every 30 s; when the pod finishes, copy once more and verify
# Pod address: POD_HOST, POD_PORT. Rehearsal: DRY=1 DRY_OUT=DIR (tiny model on the pod's CPU).
# Run selection (passed to the pod): RUN (default main = E08), RUN_ARGS, SMOKE_ONLY, GEN_CACHE. A run other than
# main writes to /workspace/e08/out_<RUN> and experiments/modal/outputs/e08_routed_expert_<RUN>. E08 ablation:
#   RUN=seinit RUN_ARGS="--re-expert-init security_expert:/workspace/e08/e05_se" GEN_CACHE="10 8 6"
set -uo pipefail
HERE=$(dirname "$0")
cd "$HERE/../../.." || exit 1
HOST=${POD_HOST:?set POD_HOST, e.g. root@1.2.3.4}
PORT=${POD_PORT:?set POD_PORT}
SSH="ssh -p $PORT -i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15"
UP="rsync -rc"   # no -t: a RunPod network volume refuses to set file times
R=/workspace/e08
X=experiments/modal/experiments
RUN=${RUN:-main}
GEN_STEPS=${GEN_STEPS-40 80}
MODELS=""
for s in $GEN_STEPS; do MODELS="$MODELS re_${RUN}_s$s"; done
PASS="RUN=$RUN RUN_ARGS='${RUN_ARGS:-}' SMOKE_ONLY=${SMOKE_ONLY:-0} SKIP_SMOKE=${SKIP_SMOKE:-0} GEN_CACHE='${GEN_CACHE:-26}'"

if [ "${DRY:-0}" = 1 ]; then
  ROUT=$R/dryrun
  LOUT=${DRY_OUT:?set DRY_OUT}
  VERIFY=--dry-run
  TINY="--tiny --limit 24 --micro-batch 4 --grad-accum 1 --eval-batch 8 --eval-every 2 --re-calib-pairs 16 --re-target-rate 0.25"
  RENV="OUT=$ROUT CUDA_VISIBLE_DEVICES= EXTRA_TRAIN='$TINY' EXTRA_GEN='--tiny --limit 3 --max-new-tokens 3' GEN_STEPS=2 $PASS"
  MODELS="re_${RUN}_s2"
else
  if [ "$RUN" = main ]; then ROUT=$R/out; LOUT=experiments/modal/outputs/e08_routed_expert
  else ROUT=$R/out_$RUN; LOUT=experiments/modal/outputs/e08_routed_expert_$RUN; fi
  VERIFY=
  RENV="OUT=$ROUT GEN_STEPS='$GEN_STEPS' $PASS"
fi

sync_up() {
  $SSH "$HOST" "mkdir -p $R/code/$X $R/code/data/secpo_olmoe_v2 $R/code/experiments/modal/artifacts/splits $R/e05_se" || return 1
  $UP -e "$SSH" --exclude __pycache__ moe_interp "$HOST:$R/code/" || return 1
  $UP -e "$SSH" $X/e03_secpo_train.py $X/e03_generate.py $X/e07_generate.py $X/e08_pod_run.sh "$HOST:$R/code/$X/" || return 1
  $UP -e "$SSH" data/secpo_olmoe_v2/pairs.jsonl "$HOST:$R/code/data/secpo_olmoe_v2/" || return 1
  $SSH "$HOST" "mkdir -p $R/code/data/secpo_olmoe_v1" || return 1
  $UP -e "$SSH" data/secpo_olmoe_v1/records.jsonl "$HOST:$R/code/data/secpo_olmoe_v1/" || return 1  # the split checks its sha
  $UP -e "$SSH" data/alpacafarm_secalign_208x6.jsonl "$HOST:$R/code/data/" || return 1
  $UP -e "$SSH" experiments/modal/artifacts/splits/secpo_olmoe_v1_grouped_v1.json "$HOST:$R/code/experiments/modal/artifacts/splits/" || return 1
  $UP -e "$SSH" experiments/modal/outputs/e05_security_expert/secpo_se_v1/ref_logps.json "$HOST:$R/ref_logps.json" || return 1
  $UP -e "$SSH" experiments/modal/outputs/e05_security_expert/secpo_se_v1/security_expert/security_expert.safetensors \
    experiments/modal/outputs/e05_security_expert/secpo_se_v1/security_expert/security_expert_config.json "$HOST:$R/e05_se/" || return 1
  echo "synced"
}

start() {
  $SSH "$HOST" "mkdir -p $ROUT && rm -f $ROUT/ALL_DONE && { env $RENV nohup setsid bash $R/code/$X/e08_pod_run.sh > $ROUT/run.log 2>&1 < /dev/null & } && echo started"
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
  (cd "$LOUT" && shasum -a 256 -c SHA256SUMS --quiet) && echo "checksums: all files match the pod"
  if [ "${SMOKE_ONLY:-0}" != 1 ] && [ -n "$GEN_STEPS" ]; then
    HF_HUB_OFFLINE=1 .venv/bin/python "$X/e07_verify_local.py" --out "$LOUT/gen" --models "$MODELS" $VERIFY
  fi
}

case "${1:-}" in
  sync) sync_up ;;
  start) start && pull ;;
  pull) pull ;;
  *) echo "usage: $0 sync|start|pull"; exit 2 ;;
esac
