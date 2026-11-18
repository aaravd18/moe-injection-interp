#!/usr/bin/env bash
# E03 step 3 evaluation generations (pod). Same settings as the dataset generation: greedy, max 512 new tokens, batch 32.
set -euo pipefail
cd /workspace/moe-injection-interp
export HF_HOME=/workspace/hf TOKENIZERS_PARALLELISM=false
PY=/root/venv/bin/python; G=experiments/modal/experiments/e03_generate.py; A=experiments/modal/outputs/e03_olmoe_security_expert/secpo_v1/adapter
$PY $G --data data/alpacafarm_secalign_208.jsonl --subset all --out experiments/modal/outputs/e03_olmoe_security_expert/eval_v1/base_af208
$PY $G --data data/alpacafarm_secalign_208.jsonl --subset all --adapter $A --out experiments/modal/outputs/e03_olmoe_security_expert/eval_v1/secpo_af208
$PY $G --subset all --split val --out experiments/modal/outputs/e03_olmoe_security_expert/eval_v1/base_secpoval
$PY $G --subset all --split val --adapter $A --out experiments/modal/outputs/e03_olmoe_security_expert/eval_v1/secpo_secpoval
echo ALL_EVAL_DONE
