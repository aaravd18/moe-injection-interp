#!/usr/bin/env bash
# Runs the SEP security study stages back to back on one GPU (DGX Spark). Each stage logs to ~/moe-results/<stage>.log.
# Started after the main Mellum run (configs/mellum_sep.yaml) is already running; waits for it first.
set -u
cd ~/moe-injection-interp
. ~/moe-venv/bin/activate
export HF_HUB_OFFLINE=1
R=~/moe-results
stage() { echo "$(date +%T) START $1" >> $R/queue.log; shift; "$@"; echo "$(date +%T) END rc=$?" >> $R/queue.log; }

while pgrep -f "configs/mellum_sep.yaml --output" > /dev/null; do sleep 60; done
stage mellum_features python experiments/sep_features.py $R/mellum_sep > $R/mellum_features.log 2>&1
stage mellum_analysis python experiments/sep_analysis.py $R/mellum_sep --model "Mellum2.1-12B-A2.5B (non-thinking)" > $R/mellum_analysis.log 2>&1
stage mellum_interventions python experiments/sep_interventions.py $R/mellum_sep --config configs/mellum_sep.yaml > $R/mellum_interventions.log 2>&1
while ! grep -q "Downloaded\|path:" $R/qwen_download.log; do sleep 60; done
stage qwen_verify python -m moe_interp.cli verify --config configs/qwen3_sep.yaml --json-out $R/verify_qwen.json > $R/qwen_verify.log 2>&1
stage qwen_run python -m moe_interp.cli run --config configs/qwen3_sep.yaml --output $R/qwen_sep --resume > $R/qwen_sep.log 2>&1
stage qwen_features python experiments/sep_features.py $R/qwen_sep > $R/qwen_features.log 2>&1
stage qwen_analysis python experiments/sep_analysis.py $R/qwen_sep --model "Qwen3-30B-A3B-Instruct-2507" > $R/qwen_analysis.log 2>&1
stage mellum_thinking python -m moe_interp.cli run --config configs/mellum_sep_thinking.yaml --output $R/mellum_sep_thinking --resume > $R/mellum_sep_thinking.log 2>&1
stage qwen_interventions python experiments/sep_interventions.py $R/qwen_sep --config configs/qwen3_sep.yaml > $R/qwen_interventions.log 2>&1
echo "$(date +%T) ALL DONE" >> $R/queue.log
