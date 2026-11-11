#!/usr/bin/env bash
# After queue_security.sh finishes: re-run the Mellum interventions with the reproducibility filter + paired CIs.
set -u
cd ~/moe-injection-interp
. ~/moe-venv/bin/activate
export HF_HUB_OFFLINE=1
R=~/moe-results
while ! grep -q "ALL DONE" $R/queue.log; do sleep 60; done
echo "$(date +%T) START mellum_interventions_v2" >> $R/queue.log
python experiments/sep_interventions.py $R/mellum_sep --config configs/mellum_sep.yaml \
  --out $R/mellum_sep/security/interventions_v2 > $R/mellum_interventions_v2.log 2>&1
echo "$(date +%T) END rc=$?" >> $R/queue.log
echo "$(date +%T) FOLLOWUP DONE" >> $R/queue.log
