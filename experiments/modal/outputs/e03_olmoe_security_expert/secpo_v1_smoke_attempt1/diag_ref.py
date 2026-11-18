"""Diagnose the smoke-test mismatch between cached reference log-probs and the online reference (GPU, bf16)."""
import json
import sys
from pathlib import Path

import torch

REPO = Path("/workspace/moe-injection-interp")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/modal/experiments"))
import e03_secpo_train as t  # noqa: E402
from e03_generate import OLMOE  # noqa: E402
from moe_interp.config import config_from_dict  # noqa: E402
from moe_interp.model import load_model  # noqa: E402

OUT = REPO / "experiments/modal/outputs/e03_olmoe_security_expert"
torch.manual_seed(42)
lm = load_model(config_from_dict({"model": OLMOE, "trace": {"enabled": False}}).model, seed=42)
print("impl", lm.info["attn_implementation"], lm.info["experts_implementation"], lm.info["dtype"], flush=True)
tok, dev, pad = lm.tokenizer, lm.device, lm.tokenizer.pad_token_id
rows = {r["id"]: r for r in map(json.loads, (REPO / "data/secpo_olmoe_v1/pairs.jsonl").open())}
split = json.loads((REPO / "experiments/modal/artifacts/splits/secpo_olmoe_v1_grouped_v1.json").read_text())
train = t.tokenize_pairs(tok, [rows[i] for i in split["pair_ids"]["train"][:16]], 1024)
cache = json.loads((OUT / "secpo_v1_smoke/ref_logps.json").read_text())["logps"]
ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
m = lm.model


def run(model, exs, key, bs):
    out = []
    with torch.no_grad():
        for b in range(0, len(exs), bs):
            c, r, _ = t.pair_logps(model, exs[b:b + bs], key, 1024, pad, dev, ac)
            out += list(zip(c.tolist(), r.tolist()))
    return torch.tensor(out[:8])


def d(a, b):
    x = (a - b).abs()
    return f"max {x.max().item():.3f}  mean {x.mean().item():.3f}"


cached = torch.tensor([[cache[e["id"]]["ref_chosen"], cache[e["id"]]["ref_rejected"]] for e in train[:8]])
m.eval()
A16 = run(m, train, "ref_prompt", 16)
A16b = run(m, train, "ref_prompt", 16)
A8 = run(m, train[:8], "ref_prompt", 8)
A8b = run(m, train[:8], "ref_prompt", 8)
A1 = run(m, train[:8], "ref_prompt", 1)
print("eval bs16 vs cache      ", d(A16, cached))
print("eval bs16 repeat        ", d(A16, A16b))
print("eval bs8 repeat         ", d(A8, A8b))
print("eval bs8 vs bs16        ", d(A8, A16))
print("eval bs1 vs bs8         ", d(A1, A8))
print("eval bs1 vs bs16        ", d(A1, A16))
m.train()
B8 = run(m, train[:8], "ref_prompt", 8)
print("train bs8 vs eval bs8   ", d(B8, A8))
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
C8 = run(m, train[:8], "ref_prompt", 8)
print("train+ckpt bs8 vs eval8 ", d(C8, A8))
print("per-example |bs1-bs8| (chosen, rejected):", [[round(v, 2) for v in r] for r in (A1 - A8).abs().tolist()])
print("response lengths (chosen, rejected):", [(len(e["chosen"]), len(e["rejected"])) for e in train[:8]])

# Token-level and routing view for the worst example: batch of 8 vs alone.
m.eval()
w = int((A1 - A8).abs().max(1).values.argmax())
side = "chosen" if (A1 - A8).abs()[w, 0] >= (A1 - A8).abs()[w, 1] else "rejected"
ids8, att8, lab8, _ = t.collate(train[:8], "ref_prompt", 1024, pad, dev)
row = w if side == "chosen" else 8 + w
ids1, att1, lab1, _ = t.collate([train[w]], "ref_prompt", 1024, pad, dev)
r1 = 0 if side == "chosen" else 1
with torch.no_grad(), ac():
    o8 = m(input_ids=ids8, attention_mask=att8, use_cache=False, output_router_logits=True)
    o1 = m(input_ids=ids1, attention_mask=att1, use_cache=False, output_router_logits=True)
L = int(att1[r1].sum())
lp = lambda lo, lab: (torch.gather(lo[:-1].float(), -1, lab[1:].clamp_min(0)[:, None]).squeeze(-1)  # noqa: E731
                      - lo[:-1].float().logsumexp(-1))
mask = lab1[r1, 1:L].ne(-100)
t8, t1 = lp(o8.logits[row, :L], lab8[row, :L])[mask], lp(o1.logits[r1, :L], lab1[r1, :L])[mask]
diff = (t8 - t1).abs()
print(f"worst example {train[w]['id']} {side}: {mask.sum().item()} scored tokens; summed diff "
      f"{(t8 - t1).sum().item():.3f}; per-token max {diff.max().item():.3f}, >0.1: {(diff > 0.1).sum().item()}, "
      f">0.01: {(diff > 0.01).sum().item()}")
T8, T1 = ids8.shape[1], ids1.shape[1]
flips, tot = 0, 0
for layer, (g8, g1) in enumerate(zip(o8.router_logits, o1.router_logits)):
    g8 = g8.view(ids8.shape[0], T8, -1)[row, :L]
    g1 = g1.view(ids1.shape[0], T1, -1)[r1, :L]
    k8, k1 = g8.topk(8, -1).indices.sort(-1).values, g1.topk(8, -1).indices.sort(-1).values
    f = (k8 != k1).any(-1)
    flips += f.sum().item()
    tot += f.numel()
print(f"routing: (token, layer) slots whose top-8 set differs, batch 8 vs alone: {flips}/{tot}")

# PEFT wrapper with adapter disabled, train mode, same batch.
policy, _, _ = t.attach_lora(m, lm.layout, 16, 4.0, True)
policy.train()
with policy.disable_adapter():
    D8 = run(policy, train[:8], "ref_prompt", 8)
print("peft disabled train bs8 vs train+ckpt bs8", d(D8, C8))
print("peft disabled train bs8 vs cache        ", d(D8, cached))
print("mem peak GB", round(torch.cuda.max_memory_allocated() / 2**30, 1))
