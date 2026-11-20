set -e
cd /workspace/repo && . /root/venv/bin/activate && uv pip install -q -e . --no-deps
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
for c in A40 B80; do
  python experiments/modal/experiments/e03_generate.py --data data/alpacafarm_secalign_208.jsonl --subset all --dtype float32 \
    --batch-size 32 --max-new-tokens 512 --security-expert /workspace/ckpt/$c --out /workspace/out/af208/$c > /workspace/out/af208.$c.log 2>&1
done
echo AF208_DONE > /workspace/out/af208.status
