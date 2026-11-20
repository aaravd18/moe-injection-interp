set -e
cd /workspace/repo && . /root/venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
G="python experiments/modal/experiments/e03_generate.py --data data/secpo_dev40.jsonl --subset all --dtype float32 --batch-size 32 --max-new-tokens 512"
D=/workspace/out/dev40
$G --out $D/base > $D.base.log 2>&1
$G --adapter /workspace/ckpt/lora_v2 --out $D/lora_v2 > $D.lora_v2.log 2>&1
$G --security-expert /workspace/ckpt/e05_final --out $D/e05_final > $D.e05_final.log 2>&1
echo DEV_REF_DONE > /workspace/out/phase1.status
T="python experiments/modal/experiments/e03_secpo_train.py --method security_expert --data data/secpo_olmoe_v2/pairs.jsonl --dtype float32 --micro-batch 8 --grad-accum 2 --epochs 1 --se-alpha-init 0.1 --save-steps 10,20,40,60,80"
$T --lr 1.6e-4 --ref-cache /workspace/out/se_lr16e-5/ref_logps.json --out /workspace/out/se_lr16e-5 > /workspace/out/se_lr16e-5.log 2>&1
echo TRAIN_A_DONE > /workspace/out/phase1.status
$T --lr 4e-5 --ref-cache /workspace/out/se_lr4e-5/ref_logps.json --out /workspace/out/se_lr4e-5 > /workspace/out/se_lr4e-5.log 2>&1
echo PHASE1_DONE > /workspace/out/phase1.status
