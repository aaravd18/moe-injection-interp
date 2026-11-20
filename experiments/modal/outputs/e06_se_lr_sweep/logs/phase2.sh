# usage: bash phase2.sh "lr16e-5:10 lr4e-5:10 ..."   (run:step pairs, generated in the given order)
set -e
cd /workspace/repo && . /root/venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
for rs in $1; do
  r=${rs%%:*}; s=${rs##*:}
  python experiments/modal/experiments/e03_generate.py --data data/secpo_dev40.jsonl --subset all --dtype float32 \
    --batch-size 32 --max-new-tokens 512 --security-expert /workspace/out/se_$r/security_expert_step$s \
    --out /workspace/out/dev40/${r}_step$s > /workspace/out/dev40.${r}_step$s.log 2>&1
done
echo PHASE2_DONE > /workspace/out/phase2.status
