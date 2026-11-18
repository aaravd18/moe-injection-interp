#!/usr/bin/env bash
cd /workspace/security-project/repos/moe-injection-interp-e03
export HF_HOME=/workspace/security-project/models/huggingface TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
/workspace/security-project/environments/mellum-pilot-mellum_pilot_20261106/bin/python experiments/modal/experiments/e03_generate.py --subset pilot --out /workspace/security-project/results/e03_olmoe_security_expert/pilot20 --max-new-tokens 512 --batch-size 32
echo $? > /workspace/security-project/results/e03_olmoe_security_expert/pilot20/EXIT_CODE
