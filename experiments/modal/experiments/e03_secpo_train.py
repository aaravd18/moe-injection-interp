"""E03 step 2: SecPO LoRA fine-tuning of OLMoE-1B-7B-0125-Instruct on SecPO preference pairs.

Default data: data/secpo_olmoe_v2 (paper construction: rejected = OLMoE on the seed task alone; all 1,500 records,
1,275 train / 225 val by the grouped split). data/secpo_olmoe_v1/pairs.jsonl (rejected = OLMoE on the injected prompt,
judge-filtered, 556 / 98) still works via --data.

Plain Python (no Modal) for the SSH H100, or locally with --tiny (tiny random OLMoE, real tokenizer; output is noise).

Objective: the official SecPO loss (github.com/Sizhe-Chen/SecPO @ da0372a, src/secpo/loss.py), unchanged:
    -log sigmoid(beta * ([log pi(y_w | x_adv) - log pi(y_l | x_adv)] - [log ref(y_w | x_clean) - log ref(y_l | x_clean)]))
with summed (not length-normalised) response log-probabilities. pi = OLMoE + LoRA on the injected prompt; ref = the
same OLMoE with the adapter off, on the clean prompt. Tokenization also follows the official collator: the prompt is the
chat-rendered text (system = instruction, user = data, generation prompt "<|assistant|>\\n") tokenized without extra
special tokens; the response is tokenized separately and EOS is appended; only response tokens are scored; prompts are
left-truncated to fit max_len and responses are never truncated.

Differences from the official trainer (all documented in the E03 logbook, Step 2):
  * The reference log-probs are computed once before training (frozen base, clean prompt) and cached, instead of two
    adapter-disabled forwards per step. Valid because the base weights never change and OLMoE has no train-mode-only
    computation (attention dropout 0, no router jitter); `--smoke` checks cache == online.
  * LoRA also targets OLMoE's fused expert weights (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`, one rank-r
    adapter per expert) via PEFT `target_parameters`. PEFT requires lora_dropout = 0 for that (paper: 0.1).
  * Rank 16 / alpha 4 instead of 64 / 8: same alpha/sqrt(r) = 1 per-step update size, ~122M trainable parameters
    (the paper's Llama-8B LoRA has ~141M). Effective batch 16 pairs (paper: 64-256) because there are only 556 train
    pairs; one epoch is then ~35 optimizer steps.
  * EOS is appended only to responses whose generation ended with EOS (the 53 truncated `rejected` get none).
  * bf16 caveat for expert LoRA: PEFT folds it into the fused weight (W + dW, in bf16), so update elements below half a
    bf16 step of W are rounded away in the forward pass. Attention LoRA runs unmerged and is unaffected. run_info.json
    and the smoke report give `expert_delta_realization` (relative error of the realized vs intended update). If it
    is large, `--dtype float32` trains the whole model in full fp32 (~28 GB weights; fits and is fast on one H100).
  * chosen and rejected run in one concatenated forward pass (same math as two passes).
The router (`mlp.gate`), embeddings, norms and all base weights stay frozen.

Outputs (--out): config.json, ref_logps.json (cache), metrics.jsonl (train/val rows), run_info.json, adapter/.
With --smoke: smoke_report.json only (no adapter).

Usage:
    python experiments/modal/experiments/e03_secpo_train.py --smoke --out OUT_SMOKE          # H100, ~2 min
    python experiments/modal/experiments/e03_secpo_train.py --out OUT                        # H100, full run
    python experiments/modal/experiments/e03_secpo_train.py --tiny --smoke --out /tmp/x      # CPU pipeline check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e03_generate import OLMOE  # noqa: E402  (pinned model id + revision shared with generation)

IGNORE = -100
DATA = REPO / "data" / "secpo_olmoe_v2" / "pairs.jsonl"
SPLIT = REPO / "experiments" / "modal" / "artifacts" / "splits" / "secpo_olmoe_v1_grouped_v1.json"
ATTN_TARGETS = r".*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
EXPERT_PARAMS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]   # transformers 5.x fused experts
EXPERT_MODULES = r".*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)"  # transformers 4.x per-expert nn.Linear


# ----------------------------------------------------------------------------------------------------------- data

def response_ids(tok, text: str, ended_with_eos: bool) -> list[int]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return ids + [tok.eos_token_id] if ended_with_eos and (not ids or ids[-1] != tok.eos_token_id) else ids


def build_input_and_labels(prompt: list[int], response: list[int], max_len: int) -> tuple[list[int], list[int]]:
    """Official SecPO `build_input_and_labels`: left-truncate only the prompt; never truncate the response."""
    if not prompt or not response:
        raise ValueError("empty prompt or response")
    if len(response) >= max_len:
        raise ValueError(f"response has {len(response)} tokens >= max_len={max_len}; responses are not truncated")
    prompt = prompt[-(max_len - len(response)):]
    return prompt + response, [IGNORE] * len(prompt) + response


def tokenize_pairs(tok, rows: list[dict], max_len: int) -> list[dict]:
    from moe_interp.prompts import render_prompt

    out = []
    for r in rows:
        pref, gen = r["preference"], r["olmoe"]
        # v2 (SecPO as in the paper): rejected = OLMoE(seed task alone). v1: rejected = OLMoE(injected prompt).
        rej = "seed_task" if "seed_task" in gen else "injection"
        assert pref["chosen"] == gen["clean"]["response"] and pref["rejected"] == gen[rej]["response"], r["id"]
        ex = {"id": r["id"], "strict": not r.get("labels", {}).get("borderline", False),
              "chosen": response_ids(tok, pref["chosen"], gen["clean"]["finish_reason"] == "stop"),
              "rejected": response_ids(tok, pref["rejected"], gen[rej]["finish_reason"] == "stop")}
        for key, variant in (("policy_prompt", "injection"), ("ref_prompt", "clean")):
            ex[key] = render_prompt(tok, r["variants"][variant]["messages"], True, None, {}).input_ids
        ex["truncated_prompt"] = any(len(ex[p]) + len(ex[y]) > max_len
                                     for p in ("policy_prompt", "ref_prompt") for y in ("chosen", "rejected"))
        ex["sha"] = hashlib.sha256(json.dumps([ex[k] for k in ("policy_prompt", "ref_prompt", "chosen", "rejected")])
                                   .encode()).hexdigest()[:16]
        out.append(ex)
    return out


def collate(exs: list[dict], prompt_key: str, max_len: int, pad_id: int, device):
    """Batch of 2B sequences: [prompt + chosen] * B then [prompt + rejected] * B, right-padded."""
    import torch

    seqs = [build_input_and_labels(e[prompt_key], e[y], max_len) for y in ("chosen", "rejected") for e in exs]
    T = max(len(s) for s, _ in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    lab = torch.full((len(seqs), T), IGNORE, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, (s, l) in enumerate(seqs):
        ids[i, :len(s)], lab[i, :len(l)], att[i, :len(s)] = torch.tensor(s), torch.tensor(l), 1
    return ids.to(device), att.to(device), lab.to(device), int(att.sum())


# ----------------------------------------------------------------------------------------------------------- loss

def sequence_logps(logits, labels):
    """Summed causal log-prob of the labelled tokens (official `sequence_logps`; token t scored by logits[t-1]).
    Computed in fp32 as gather - logsumexp, which avoids materialising a second vocab-sized tensor."""
    import torch

    logits, labels = logits[:, :-1].float(), labels[:, 1:]
    mask = labels.ne(IGNORE)
    if bool((mask.sum(-1) == 0).any()):
        raise ValueError("a sequence has no scored token")
    tok = torch.gather(logits, -1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
    return (tok * mask).sum(-1)


def secpo_loss(pc, pr, rc, rr, beta: float) -> dict:
    """Official `secpo_loss` (sigmoid). Returns per-example tensors."""
    import torch.nn.functional as F

    logits = (pc - pr) - (rc - rr)
    return {"losses": -F.logsigmoid(beta * logits), "logits": logits,
            "chosen_rewards": beta * (pc - rc).detach(), "rejected_rewards": beta * (pr - rr).detach()}


def pair_logps(model, exs, prompt_key, max_len, pad_id, device, autocast):
    ids, att, lab, ntok = collate(exs, prompt_key, max_len, pad_id, device)
    with autocast():
        logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits
    lp = sequence_logps(logits, lab)
    return lp[: len(exs)], lp[len(exs):], ntok


# ---------------------------------------------------------------------------------------------------------- model

def attach_lora(model, layout, r: int, alpha: float, grad_ckpt: bool, revision: str | None = None):
    import torch
    from peft import LoraConfig, get_peft_model

    experts = getattr(layout.moe_blocks[0], "experts", None)
    fused = isinstance(getattr(experts, "gate_up_proj", None), torch.nn.Parameter)
    kw = ({"target_modules": ATTN_TARGETS, "target_parameters": EXPERT_PARAMS} if fused
          else {"target_modules": f"({ATTN_TARGETS})|({EXPERT_MODULES})"})
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", revision=revision, **kw)
    model.config.use_cache = False
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy = get_peft_model(model, cfg)
    return policy, cfg, ("fused 3D experts (target_parameters)" if fused else "per-expert nn.Linear (target_modules)")


def param_report(policy) -> dict:
    groups = {"attention": 0, "experts": 0, "other": 0}
    frozen, bad = 0, []
    for n, p in policy.named_parameters():
        if not p.requires_grad:
            frozen += p.numel()
            continue
        if "lora_" not in n:
            bad.append(n)
        groups["attention" if ".self_attn." in n else "experts" if ".mlp.experts" in n else "other"] += p.numel()
    router_trainable = [n for n, p in policy.named_parameters() if ".mlp.gate." in n and p.requires_grad]
    router_wrapped = [n for n, _ in policy.named_modules() if ".mlp.gate." in n and "lora_" in n]
    total = sum(groups.values())
    return {"trainable": total, "frozen": frozen, "trainable_pct": round(100 * total / (total + frozen), 3),
            "trainable_by_group": groups, "non_lora_trainable": bad, "router_trainable": router_trainable,
            "router_lora_modules": router_wrapped}


def expert_delta_realization(policy) -> dict | None:
    """How much of the intended expert LoRA update survives the forward pass.

    PEFT applies expert LoRA by folding it into the fused weight, W' = W + s * lhs @ rhs, computed in W's dtype. In bf16,
    an element of the update smaller than half a bf16 step of W is rounded away, so small updates are partly invisible to
    the forward pass (the factor gradients are still exact). Returns ||realized - intended|| / ||intended|| over all
    expert tensors (0 = fully realized; 1 = nothing realized) and the cosine between realized and intended updates.
    None for the per-expert nn.Linear layout (transformers 4.x), where LoRA runs unmerged like attention LoRA."""
    import torch
    from peft.tuners.lora.layer import ParamWrapper

    err = ref = dot = rn = 0.0
    n = 0
    with torch.no_grad():
        for m in policy.modules():
            if not isinstance(m, ParamWrapper) or "default" not in m.lora_A:
                continue
            W = m.get_param()
            lhs, rhs, s = m.get_delta_factors("default")
            intended = s * torch.bmm(lhs.float(), rhs.float())
            realized = torch.baddbmm(W, lhs, rhs, alpha=s).float() - W.float()
            err += (realized - intended).pow(2).sum().item()
            ref += intended.pow(2).sum().item()
            dot += (realized * intended).sum().item()
            rn += realized.pow(2).sum().item()
            n += 1
    if not n or ref == 0:
        return None if not n else {"tensors": n, "intended_norm": 0.0}
    return {"tensors": n, "rel_error": round(math.sqrt(err / ref), 4),
            "cosine": round(dot / math.sqrt(ref * rn), 4) if rn else 0.0, "intended_norm": round(math.sqrt(ref), 6)}


# ------------------------------------------------------------------------------------------------------ reference

def compute_cache(base, exs, args, pad_id, device, autocast) -> dict:
    """Frozen base model, no grad: ref = clean prompt (the SecPO reference); base = injected prompt (policy at init)."""
    import torch

    base.eval()
    out = {}
    with torch.no_grad():
        for b in range(0, len(exs), args.eval_batch):
            chunk = exs[b: b + args.eval_batch]
            rc, rr, _ = pair_logps(base, chunk, "ref_prompt", args.max_len, pad_id, device, autocast)
            bc, br, _ = pair_logps(base, chunk, "policy_prompt", args.max_len, pad_id, device, autocast)
            for i, e in enumerate(chunk):
                out[e["id"]] = {"sha": e["sha"], "ref_chosen": rc[i].item(), "ref_rejected": rr[i].item(),
                                "base_inj_chosen": bc[i].item(), "base_inj_rejected": br[i].item()}
    return out


def load_or_build_cache(path: Path, header: dict, base, exs, args, pad_id, device, autocast) -> tuple[dict, bool]:
    if path.exists():
        c = json.loads(path.read_text())
        if c.get("header") == header and all(c["logps"].get(e["id"], {}).get("sha") == e["sha"] for e in exs):
            return c["logps"], True
        print(f"[ref] cache {path} does not match this config/tokenization; recomputing", flush=True)
    logps = compute_cache(base, exs, args, pad_id, device, autocast)
    path.write_text(json.dumps({"header": header, "logps": logps}) + "\n")
    return logps, False


def ref_tensors(cache, exs, device):
    import torch

    t = lambda k: torch.tensor([cache[e["id"]][k] for e in exs], dtype=torch.float32, device=device)  # noqa: E731
    return t("ref_chosen"), t("ref_rejected")


# ------------------------------------------------------------------------------------------------------- evaluate

def evaluate(policy, exs, cache, args, pad_id, device, autocast) -> dict:
    import torch

    policy.eval()
    rows = []
    with torch.no_grad():
        for b in range(0, len(exs), args.eval_batch):
            chunk = exs[b: b + args.eval_batch]
            pc, pr, _ = pair_logps(policy, chunk, "policy_prompt", args.max_len, pad_id, device, autocast)
            o = secpo_loss(pc, pr, *ref_tensors(cache, chunk, device), args.beta)
            for i, e in enumerate(chunk):
                rows.append((e["strict"], o["losses"][i].item(), o["logits"][i].item(), (pc[i] - pr[i]).item(),
                             o["chosen_rewards"][i].item(), o["rejected_rewards"][i].item(), pc[i].item(), pr[i].item()))
    policy.train()

    def agg(rs):
        n = len(rs)
        return {} if not n else {
            "n": n, "loss": sum(r[1] for r in rs) / n,
            "secpo_margin": sum(r[2] for r in rs) / n,             # policy logratio (adv) - ref logratio (clean)
            "secpo_acc": sum(r[2] > 0 for r in rs) / n,
            "policy_logratio_adv": sum(r[3] for r in rs) / n,      # log pi(y_w|x_adv) - log pi(y_l|x_adv)
            "policy_prefers_chosen_adv": sum(r[3] > 0 for r in rs) / n,
            "chosen_reward": sum(r[4] for r in rs) / n, "rejected_reward": sum(r[5] for r in rs) / n,
            "logp_chosen": sum(r[6] for r in rs) / n, "logp_rejected": sum(r[7] for r in rs) / n}

    return {"all": agg(rows), "strict": agg([r for r in rows if r[0]])}


def gpu_mem(device) -> dict:
    import torch

    if device.type != "cuda":
        return {}
    g = 2 ** 30
    return {"mem_alloc_gb": round(torch.cuda.memory_allocated(device) / g, 2),
            "mem_peak_gb": round(torch.cuda.max_memory_allocated(device) / g, 2),
            "mem_reserved_gb": round(torch.cuda.max_memory_reserved(device) / g, 2)}


# ---------------------------------------------------------------------------------------------------------- smoke

def smoke(policy, train, cache, args, pad_id, device, autocast, rep: dict, pre: dict) -> dict:
    """Checks on one small batch before any real training: parameters, cache == online reference, loss at init,
    gradient flow, and one optimizer step that moves only LoRA weights and lowers the loss on that batch.

    `pre` holds the frozen base model's log-probs on this exact batch, computed before LoRA was attached. In bf16 the
    batch shape (padding) changes summed log-probs by up to ~2 nats (E03 logbook, Step 3), so exact checks compare
    against `pre` (same batch) and the cache (computed in batches of --eval-batch) gets a loose sanity bound."""
    import torch

    exs = train[: args.micro_batch]
    tol, loose = 1e-3, 5.0  # exact (same batch) / gross-error bound across batch shapes (a wrong prompt is off by >>5)
    checks = {}

    def check(name, ok, **info):
        checks[name] = {"ok": bool(ok), **info}
        print(f"[smoke] {'PASS' if ok else 'FAIL'} {name} {json.dumps(info)}", flush=True)

    check("only_lora_trainable", not rep["non_lora_trainable"], n=len(rep["non_lora_trainable"]))
    check("router_frozen_and_unwrapped", not rep["router_trainable"] and not rep["router_lora_modules"])
    check("attention_and_expert_lora_present",
          rep["trainable_by_group"]["attention"] > 0 and rep["trainable_by_group"]["experts"] > 0,
          **rep["trainable_by_group"])

    policy.train()
    with torch.no_grad(), policy.disable_adapter():  # official online reference path
        rc, rr, _ = pair_logps(policy, exs, "ref_prompt", args.max_len, pad_id, device, autocast)
    d = max((rc - pre["rc"]).abs().max().item(), (rr - pre["rr"]).abs().max().item())
    check("online_reference_equals_frozen_base", d <= tol, max_abs_diff=round(d, 5), tol=tol)
    crc, crr = ref_tensors(cache, exs, device)
    dc = torch.cat([rc - crc, rr - crr]).abs()
    check("ref_cache_within_batch_shape_noise", dc.max().item() <= loose, max_abs_diff=round(dc.max().item(), 4),
          mean_abs_diff=round(dc.mean().item(), 4), bound=loose)

    pc, pr, _ = pair_logps(policy, exs, "policy_prompt", args.max_len, pad_id, device, autocast)
    d0 = max((pc.detach() - pre["bc"]).abs().max().item(), (pr.detach() - pre["br"]).abs().max().item())
    check("policy_equals_base_at_init", d0 <= tol, max_abs_diff=round(d0, 5), tol=tol)
    o = secpo_loss(pc, pr, crc, crr, args.beta)
    loss0 = o["losses"].mean()
    check("loss_finite", torch.isfinite(loss0).item(), loss=round(loss0.item(), 4),
          secpo_margin=round(o["logits"].mean().item(), 3), log2=round(math.log(2), 4))
    loss0.backward()

    named = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    nz = lambda p: p.grad is not None and p.grad.abs().sum().item() > 0  # noqa: E731
    b_attn = [nz(p) for n, p in named if "lora_B" in n and ".self_attn." in n]
    b_exp = [nz(p) for n, p in named if "lora_B" in n and ".mlp.experts" in n]
    check("grad_on_every_attention_lora_B", all(b_attn), nonzero=sum(b_attn), total=len(b_attn))
    check("grad_on_every_expert_lora_B", all(b_exp), nonzero=sum(b_exp), total=len(b_exp))
    frozen_grads = [n for n, p in policy.named_parameters() if not p.requires_grad and p.grad is not None]
    check("no_grad_on_frozen_params", not frozen_grads, n=len(frozen_grads))
    gn = torch.nn.utils.clip_grad_norm_([p for _, p in named], float("inf")).item()
    check("grad_norm_finite_positive", math.isfinite(gn) and gn > 0, grad_norm=round(gn, 4))

    router = [p for n, p in policy.named_parameters() if ".mlp.gate." in n]
    snap = lambda ps: [p.detach().float().sum().item() for p in ps]  # noqa: E731
    router_before = snap(router)
    lora_before = [p.detach().clone() for _, p in named if "lora_B" in _]
    opt = torch.optim.AdamW([p for _, p in named], lr=args.lr, weight_decay=0.0)
    t0 = time.time()
    opt.step()
    opt.zero_grad(set_to_none=True)
    moved = sum(not torch.equal(a, p.detach()) for a, (n, p) in zip(lora_before, [x for x in named if "lora_B" in x[0]]))
    check("optimizer_step_moves_lora_B", moved == len(lora_before), moved=moved, total=len(lora_before))
    check("router_unchanged", snap(router) == router_before)
    with torch.no_grad():
        pc1, pr1, _ = pair_logps(policy, exs, "policy_prompt", args.max_len, pad_id, device, autocast)
        loss1 = secpo_loss(pc1, pr1, crc, crr, args.beta)["losses"].mean().item()
        with policy.disable_adapter():
            rc1, rr1, _ = pair_logps(policy, exs, "ref_prompt", args.max_len, pad_id, device, autocast)
    check("loss_decreases_after_one_step", loss1 < loss0.item(), before=round(loss0.item(), 5), after=round(loss1, 5))
    d1 = max((rc1 - pre["rc"]).abs().max().item(), (rr1 - pre["rr"]).abs().max().item())
    check("reference_unchanged_after_step", d1 <= tol, max_abs_diff=round(d1, 5), tol=tol)
    real = expert_delta_realization(policy)  # informational: one step is far smaller than a full run's update
    print(f"[smoke] INFO expert_delta_realization_after_one_step {json.dumps(real)}", flush=True)
    return {"checks": checks, "all_ok": all(c["ok"] for c in checks.values()), "n_examples": len(exs),
            "expert_delta_realization_after_one_step": real, "step_seconds": round(time.time() - t0, 2),
            **gpu_mem(device)}


def frozen_fingerprint(model) -> dict:
    """Per-tensor (sum, sum of squares, sha of a strided sample) of every frozen parameter: detects any change."""
    import torch

    fp = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                continue
            f = p.detach().float()
            flat = p.detach().reshape(-1)
            sample = flat[:: max(1, flat.numel() // 4096)].float().cpu().numpy().tobytes()
            fp[n] = (f.sum().item(), f.pow(2).sum().item(), hashlib.sha256(sample).hexdigest()[:16])
    return fp


def router_topk(model, exs, prompt_key, max_len, pad_id, device, autocast, layout) -> list:
    """Top-k expert ids and weights per (layer, token) from every router, on one batch."""
    import torch

    got = []
    hooks = [r.register_forward_hook(lambda _m, _i, o: got.append((o[2].detach().clone(), o[1].detach().clone())))
             for r in layout.routers]
    try:
        with torch.no_grad():
            pair_logps(model, exs, prompt_key, max_len, pad_id, device, autocast)
    finally:
        for h in hooks:
            h.remove()
    return got


def smoke_security_expert(model, layout, train, cache, args, pad_id, device, autocast, rep: dict, pre: dict,
                          out: Path) -> dict:
    """Security-expert analogue of `smoke` (same batch, tolerances and reference checks), plus: exact equality with the
    base at init (log-probs, logits and router top-k), the zero-init gradient pattern (step 0: only down_proj; after one
    update: gate_up_proj and alpha too), frozen weights bit-identical after two optimizer steps, contribution magnitudes,
    and a save -> reload round trip."""
    import torch

    from moe_interp import security_expert as SE

    exs = train[: args.micro_batch]
    tol, loose = 1e-3, 5.0
    checks = {}

    def check(name, ok, **info):
        checks[name] = {"ok": bool(ok), **info}
        print(f"[smoke] {'PASS' if ok else 'FAIL'} {name} {json.dumps(info)}", flush=True)

    ses = SE.security_experts(model)
    n_l = len(layout.moe_blocks)
    H, I = model.config.hidden_size, model.config.intermediate_size
    check("only_security_experts_trainable", not rep["non_security_expert_trainable"],
          n_bad=len(rep["non_security_expert_trainable"]))
    check("router_frozen", not rep["router_trainable"])
    want = n_l * (2 * I * H + H * I + 1)
    check("one_expert_per_moe_layer_with_olmoe_expert_shape",
          len(ses) == n_l and rep["trainable"] == want
          and all(tuple(s.gate_up_proj.shape) == tuple(layout.moe_blocks[0].experts.gate_up_proj.shape[1:])
                  and tuple(s.down_proj.shape) == tuple(layout.moe_blocks[0].experts.down_proj.shape[1:]) for s in ses),
          n_experts=len(ses), trainable=rep["trainable"], expected=want, trainable_pct=rep["trainable_pct"])
    check("down_proj_zero_alpha_nonzero_at_init",
          all(s.down_proj.abs().max().item() == 0 for s in ses) and all(s.alpha.item() != 0 for s in ses),
          alphas=sorted(set(SE.alphas(model))))

    model.train()
    # 1. equivalence with the frozen base at init (pre = base before the experts were attached, same batch)
    with torch.no_grad():
        pc0, pr0, _ = pair_logps(model, exs, "policy_prompt", args.max_len, pad_id, device, autocast)
    d0 = max((pc0 - pre["bc"]).abs().max().item(), (pr0 - pre["br"]).abs().max().item())
    check("policy_equals_base_at_init", d0 <= tol, max_abs_diff=d0, tol=tol)
    ids, att, _, _ = collate(exs, "policy_prompt", args.max_len, pad_id, device)
    with torch.no_grad(), autocast():
        lg_on = model(input_ids=ids, attention_mask=att, use_cache=False).logits
        with SE.security_experts_disabled(model):
            lg_off = model(input_ids=ids, attention_mask=att, use_cache=False).logits
    dl = (lg_on - lg_off).abs().max().item()
    del lg_on, lg_off
    check("logits_identical_to_base_at_init", dl == 0.0, max_abs_diff=dl)
    with SE.security_experts_disabled(model):
        rt_off = router_topk(model, exs, "policy_prompt", args.max_len, pad_id, device, autocast, layout)
    rt_on = router_topk(model, exs, "policy_prompt", args.max_len, pad_id, device, autocast, layout)
    check("router_topk_identical_to_base_at_init",
          all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(rt_on, rt_off)), layers=len(rt_on))

    # 2. reference = base on the clean prompt (experts disabled), and the cached reference
    with torch.no_grad(), SE.security_experts_disabled(model):
        rc, rr, _ = pair_logps(model, exs, "ref_prompt", args.max_len, pad_id, device, autocast)
    d = max((rc - pre["rc"]).abs().max().item(), (rr - pre["rr"]).abs().max().item())
    check("online_reference_equals_frozen_base", d <= tol, max_abs_diff=round(d, 5), tol=tol)
    crc, crr = ref_tensors(cache, exs, device)
    dc = torch.cat([rc - crc, rr - crr]).abs()
    check("ref_cache_within_batch_shape_noise", dc.max().item() <= loose, max_abs_diff=round(dc.max().item(), 4),
          mean_abs_diff=round(dc.mean().item(), 4), bound=loose)

    fp_before = frozen_fingerprint(model)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([p for _, p in named], lr=args.lr, weight_decay=0.0)
    nz = lambda p: p.grad is not None and p.grad.abs().sum().item() > 0  # noqa: E731
    kind = lambda n: n.rsplit(".", 1)[-1]  # noqa: E731
    losses, grads = [], []
    t0 = time.time()
    for step in range(2):
        pc, pr, _ = pair_logps(model, exs, "policy_prompt", args.max_len, pad_id, device, autocast)
        loss = secpo_loss(pc, pr, crc, crr, args.beta)["losses"].mean()
        losses.append(loss.item())
        if step == 0:
            check("loss_finite", torch.isfinite(loss).item(), loss=round(loss.item(), 4), log2=round(math.log(2), 4))
        loss.backward()
        g = {k: [nz(p) for n, p in named if kind(n) == k] for k in SE.PARAM_NAMES}
        gn = torch.nn.utils.clip_grad_norm_([p for _, p in named], float("inf")).item()
        grads.append({k: sum(v) for k, v in g.items()} | {"grad_norm": gn})
        if step == 0:
            check("step0_grad_on_every_down_proj", all(g["down_proj"]), nonzero=sum(g["down_proj"]), total=n_l)
            check("step0_no_grad_on_gate_up_and_alpha_as_expected_from_zero_init",
                  not any(g["gate_up_proj"]) and not any(g["alpha"]),
                  gate_up_nonzero=sum(g["gate_up_proj"]), alpha_nonzero=sum(g["alpha"]))
            frozen_grads = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
            check("no_grad_on_frozen_params", not frozen_grads, n=len(frozen_grads))
            check("grad_norm_finite_positive", math.isfinite(gn) and gn > 0, grad_norm=round(gn, 6))
        else:
            check("step1_grad_flows_to_every_gate_up_and_alpha", all(g["gate_up_proj"]) and all(g["alpha"]),
                  gate_up_nonzero=sum(g["gate_up_proj"]), alpha_nonzero=sum(g["alpha"]), down_nonzero=sum(g["down_proj"]),
                  grad_norm=round(gn, 6))
        before = {n: p.detach().clone() for n, p in named}
        opt.step()
        opt.zero_grad(set_to_none=True)
        moved = {k: sum(not torch.equal(before[n], p.detach()) for n, p in named if kind(n) == k) for k in SE.PARAM_NAMES}
        if step == 0:
            check("step0_update_moves_every_down_proj", moved["down_proj"] == n_l, **moved)
        else:
            check("step1_update_moves_every_parameter", all(v == n_l for v in moved.values()), **moved)
        del before
    step_s = time.time() - t0

    with torch.no_grad():
        with SE.record_contributions(model) as rec:
            pc1, pr1, _ = pair_logps(model, exs, "policy_prompt", args.max_len, pad_id, device, autocast)
        loss2 = secpo_loss(pc1, pr1, crc, crr, args.beta)["losses"].mean().item()
        with SE.security_experts_disabled(model):
            rc1, rr1, _ = pair_logps(model, exs, "ref_prompt", args.max_len, pad_id, device, autocast)
    contrib = rec.summary()
    check("loss_decreases_after_two_steps", loss2 < losses[0], losses=[round(x, 5) for x in losses + [loss2]])
    check("security_expert_contribution_nonzero_after_update", contrib["ratio_mean"] > 0,
          ratio_mean=contrib["ratio_mean"], ratio_max=contrib["ratio_max"], tokens=contrib["tokens"])
    fp_after = frozen_fingerprint(model)
    changed = [n for n in fp_before if fp_before[n] != fp_after[n]]
    check("frozen_weights_bit_identical_after_updates", not changed and len(fp_before) == len(fp_after),
          n_frozen_tensors=len(fp_before), changed=changed[:5])
    d1 = max((rc1 - pre["rc"]).abs().max().item(), (rr1 - pre["rr"]).abs().max().item())
    check("reference_unchanged_after_updates", d1 <= tol, max_abs_diff=round(d1, 5), tol=tol)
    with SE.security_experts_disabled(model):
        rt_off1 = router_topk(model, exs, "policy_prompt", args.max_len, pad_id, device, autocast, layout)
    check("base_routing_unchanged_after_updates",
          all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(rt_off1, rt_off)))

    # save -> wipe -> reload reproduces the trained policy
    sha = SE.save_security_experts(model, out / "smoke_security_expert")
    keep = [(s.gate_up_proj.detach().clone(), s.down_proj.detach().clone(), s.alpha.detach().clone()) for s in ses]
    from safetensors.torch import load_file

    state = load_file(str(out / "smoke_security_expert" / SE.WEIGHTS))
    with torch.no_grad():
        for s in ses:
            s.down_proj.zero_()
            s.gate_up_proj.normal_()
        for i, s in enumerate(ses):
            for k in SE.PARAM_NAMES:
                getattr(s, k).copy_(state[f"layers.{i}.{k}"])
        pc2, pr2, _ = pair_logps(model, exs, "policy_prompt", args.max_len, pad_id, device, autocast)
    same = all(torch.equal(a, s.gate_up_proj) and torch.equal(b, s.down_proj) and torch.equal(c, s.alpha)
               for (a, b, c), s in zip(keep, ses))
    d2 = max((pc2 - pc1).abs().max().item(), (pr2 - pr1).abs().max().item())
    check("save_reload_round_trip", same and d2 <= tol, max_abs_diff=d2, sha256=sha[:12])
    return {"checks": checks, "all_ok": all(c["ok"] for c in checks.values()), "n_examples": len(exs),
            "grads": grads, "losses": losses + [loss2], "contribution_after_two_steps": contrib,
            "alphas_after_two_steps": SE.alphas(model), "step_seconds": round(step_s, 2), **gpu_mem(device)}


def smoke_routed_expert(model, layout, train, cache, args, pad_id, device, autocast, rep: dict, pre: dict,
                        out: Path, calib: dict, init_expert=None) -> dict:
    """E08 checks on one batch: trainable set, copy init, exact base when disabled, reference, calibrated rates,
    gradients only on the new row (+ expert), frozen weights bit-identical after two updates, loss decreases, reload."""
    import torch

    from moe_interp import routed_expert as RE

    exs = train[: args.micro_batch]
    tol, loose = 1e-3, 5.0
    checks = {}

    def check(name, ok, **info):
        checks[name] = {"ok": bool(ok), **info}
        print(f"[smoke] {'PASS' if ok else 'FAIL'} {name} {json.dumps(info)}", flush=True)

    bs = RE.blocks(model)
    n_l = len(layout.moe_blocks)
    H, I = model.config.hidden_size, model.config.intermediate_size
    want = n_l * (H + 3 * I * H)
    check("trainable_set_is_new_rows_and_new_experts",
          rep["trainable"] == want and not rep["non_routed_expert_trainable"] and not rep["original_router_trainable"]
          and len(bs) == n_l, trainable=rep["trainable"], expected=want, trainable_pct=rep["trainable_pct"])
    check("original_64_rows_and_experts_frozen",
          all(not b.gate.weight.requires_grad and not b.experts.gate_up_proj.requires_grad
              and not b.experts.down_proj.requires_grad for b in bs))
    rel = max(((b.re_row[0] - b.gate.weight[b.re_copy]).norm() / b.gate.weight[b.re_copy].norm()).item() for b in bs)
    if init_expert is None:
        check("new_expert_is_exact_copy_and_row_is_copy_plus_small_noise",
              all(torch.equal(b.re_gate_up, b.experts.gate_up_proj[b.re_copy])
                  and torch.equal(b.re_down, b.experts.down_proj[b.re_copy]) for b in bs)
              and rel < 10 * args.re_row_noise + 1e-12, copy_from=[b.re_copy for b in bs], max_row_rel_diff=rel)
    else:
        check("new_expert_equals_given_init_and_row_is_copy_plus_small_noise",
              all(torch.equal(b.re_gate_up, gu.to(b.re_gate_up)) and torch.equal(b.re_down, dn.to(b.re_down))
                  for b, (gu, dn) in zip(bs, init_expert))
              and rel < 10 * args.re_row_noise + 1e-12, expert_init=args.re_expert_init,
              fold_alpha=args.re_fold_alpha, row_copied_from=[b.re_copy for b in bs], max_row_rel_diff=rel)
    check("calibrated_rate_on_calibration_tokens", all(abs(r - args.re_target_rate) < 0.01 for r in calib["rate_by_layer"]),
          target=args.re_target_rate, rates=[round(r, 4) for r in calib["rate_by_layer"]])

    model.train()
    ids, att, _, _ = collate(exs, "policy_prompt", args.max_len, pad_id, device)
    with torch.no_grad(), autocast():
        with RE.routed_experts_disabled(model):
            lg_off = model(input_ids=ids, attention_mask=att, use_cache=False).logits
        pc0, pr0, _ = pair_logps(model, exs, "policy_prompt", args.max_len, pad_id, device, autocast)
        with RE.routed_experts_disabled(model):
            pc_off, pr_off, _ = pair_logps(model, exs, "policy_prompt", args.max_len, pad_id, device, autocast)
    d_off = max((pc_off - pre["bc"]).abs().max().item(), (pr_off - pre["br"]).abs().max().item())
    check("disabled_equals_base", d_off <= tol, max_abs_diff=d_off, tol=tol)
    shift = torch.cat([pc0 - pre["bc"], pr0 - pre["br"]])
    check("init_changes_the_model_only_moderately", shift.abs().mean().item() < 50,
          mean_abs_logp_shift=round(shift.abs().mean().item(), 4), max_abs_logp_shift=round(shift.abs().max().item(), 4),
          note="softmax over 65 rescales the original experts and the copy displaces the 8th expert")
    del lg_off
    with torch.no_grad(), RE.routed_experts_disabled(model):
        rc, rr, _ = pair_logps(model, exs, "ref_prompt", args.max_len, pad_id, device, autocast)
    d = max((rc - pre["rc"]).abs().max().item(), (rr - pre["rr"]).abs().max().item())
    check("online_reference_equals_frozen_base", d <= tol, max_abs_diff=round(d, 5), tol=tol)
    crc, crr = ref_tensors(cache, exs, device)
    dc = torch.cat([rc - crc, rr - crr]).abs()
    check("ref_cache_within_batch_shape_noise", dc.max().item() <= loose, max_abs_diff=round(dc.max().item(), 4))

    fp_before = frozen_fingerprint(model)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([p for _, p in named], lr=args.lr, weight_decay=0.0)
    kinds = ("re_row", "re_gate_up", "re_down")
    kind = lambda n: n.rsplit(".", 1)[-1]  # noqa: E731
    losses, grads = [], []
    t0 = time.time()
    for step in range(2):
        with RE.record_routing(model) as rec:
            pc, pr, _ = pair_logps(model, exs, "policy_prompt", args.max_len, pad_id, device, autocast)
        loss = secpo_loss(pc, pr, crc, crr, args.beta)["losses"].mean()
        losses.append(loss.item())
        loss.backward()
        nz = {k: sum(p.grad is not None and p.grad.abs().sum().item() > 0 for n, p in named if kind(n) == k) for k in kinds}
        gn = torch.nn.utils.clip_grad_norm_([p for _, p in named], float("inf")).item()
        grads.append(nz | {"grad_norm": gn, "rate_mean": rec.summary()["rate_mean"]})
        frozen_grads = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
        check(f"step{step}_grad_on_every_trainable_tensor_and_none_on_frozen",
              all(v == n_l for v in nz.values()) and not frozen_grads and math.isfinite(gn) and gn > 0,
              nonzero=nz, frozen_with_grad=len(frozen_grads), grad_norm=round(gn, 6))
        before = {n: p.detach().clone() for n, p in named}
        opt.step()
        opt.zero_grad(set_to_none=True)
        moved = {k: sum(not torch.equal(before[n], p.detach()) for n, p in named if kind(n) == k) for k in kinds}
        check(f"step{step}_update_moves_every_trainable_tensor", all(v == n_l for v in moved.values()), **moved)
        del before
    step_s = time.time() - t0
    with torch.no_grad():
        pc1, pr1, _ = pair_logps(model, exs, "policy_prompt", args.max_len, pad_id, device, autocast)
        loss2 = secpo_loss(pc1, pr1, crc, crr, args.beta)["losses"].mean().item()
        with RE.routed_experts_disabled(model):
            rc1, rr1, _ = pair_logps(model, exs, "ref_prompt", args.max_len, pad_id, device, autocast)
    check("loss_decreases_after_two_steps", loss2 < losses[0], losses=[round(x, 5) for x in losses + [loss2]])
    fp_after = frozen_fingerprint(model)
    changed = [n for n in fp_before if fp_before[n] != fp_after[n]]
    check("frozen_weights_bit_identical_after_updates", not changed, n_frozen_tensors=len(fp_before), changed=changed[:5])
    d1 = max((rc1 - pre["rc"]).abs().max().item(), (rr1 - pre["rr"]).abs().max().item())
    check("reference_unchanged_after_updates", d1 <= tol, max_abs_diff=round(d1, 5))

    sha = RE.save_routed_experts(model, out / "smoke_routed_expert")
    keep = [(b.re_row.detach().clone(), b.re_gate_up.detach().clone(), b.re_down.detach().clone()) for b in bs]
    from safetensors.torch import load_file

    state = load_file(str(out / "smoke_routed_expert" / RE.WEIGHTS))
    with torch.no_grad():
        for i, b in enumerate(bs):
            b.re_row.normal_()
            b.re_row.copy_(state[f"layers.{i}.row"])
            b.re_gate_up.copy_(state[f"layers.{i}.gate_up_proj"])
            b.re_down.copy_(state[f"layers.{i}.down_proj"])
        pc2, pr2, _ = pair_logps(model, exs, "policy_prompt", args.max_len, pad_id, device, autocast)
    same = all(torch.equal(a, b.re_row) and torch.equal(c, b.re_gate_up) and torch.equal(e, b.re_down)
               for (a, c, e), b in zip(keep, bs))
    d2 = max((pc2 - pc1).abs().max().item(), (pr2 - pr1).abs().max().item())
    check("save_reload_round_trip", same and d2 <= tol, max_abs_diff=d2, sha256=sha[:12])
    return {"checks": checks, "all_ok": all(c["ok"] for c in checks.values()), "n_examples": len(exs),
            "grads": grads, "losses": losses + [loss2], "calibration": calib, "step_seconds": round(step_s, 2),
            **gpu_mem(device)}


# ----------------------------------------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--split", default=str(SPLIT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="run the pre-training checks on one batch, then exit")
    ap.add_argument("--tiny", action="store_true", help="tiny random OLMoE on CPU (pipeline check; output is noise)")
    ap.add_argument("--dtype", default=None, help="override model dtype (default: bfloat16; float32 with --tiny)")
    ap.add_argument("--limit", type=int, default=None, help="first N train / N val pairs only (debugging)")
    ap.add_argument("--strict-only", action="store_true", help="train only on the 402 non-borderline train pairs")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1.6e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--micro-batch", type=int, default=8, help="pairs per forward (2x sequences)")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=4.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=10, help="optimizer steps between val evaluations")
    ap.add_argument("--save-every-epoch", action="store_true",
                    help="at each epoch end: val evaluation + adapter saved to OUT/adapter_epochN")
    ap.add_argument("--save-steps", default="",
                    help="comma-separated optimizer steps at which to save OUT/<adapter|security_expert>_stepN "
                         "(after that step's update; val metrics are logged at the same steps)")
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref-cache", default=None, help="default: OUT/ref_logps.json")
    ap.add_argument("--method", choices=["lora", "security_expert", "routed_expert"], default="lora",
                    help="lora: LoRA on attention + experts (E03). security_expert: freeze all of OLMoE and train one "
                         "new always-active expert per MoE layer, h = h_MoE + alpha_l * S_l(x) (E05, "
                         "moe_interp/security_expert.py)")
    ap.add_argument("--se-alpha-init", type=float, default=0.1, help="security_expert: initial alpha_l")
    ap.add_argument("--se-init-std", type=float, default=None,
                    help="security_expert: std of the gate_up_proj init (default: config.initializer_range)")
    ap.add_argument("--re-copy", default="most_used",
                    help="routed_expert: 'most_used' (per layer, the expert the base routes most calibration tokens "
                         "to) or a comma-separated expert id per layer")
    ap.add_argument("--re-target-rate", type=float, default=0.125,
                    help="routed_expert: share of tokens that select the new expert at init (8/64 = an average expert)")
    ap.add_argument("--re-row-noise", type=float, default=1e-3,
                    help="routed_expert: Gaussian noise on the copied router row, relative to its RMS (breaks ties)")
    ap.add_argument("--re-calib-pairs", type=int, default=128,
                    help="routed_expert: train pairs (policy prompt + chosen and + rejected) used for calibration")
    ap.add_argument("--re-expert-init", default="copy",
                    help="routed_expert: 'copy' (the new expert is a copy of the expert whose router row is copied) or "
                         "'security_expert:DIR' (a saved E05 security expert's gate_up_proj / down_proj; the router "
                         "row is still copied per --re-copy)")
    ap.add_argument("--re-fold-alpha", action="store_true",
                    help="routed_expert with --re-expert-init security_expert:DIR: scale down_proj by E05's alpha_l")
    args = ap.parse_args()
    se_method = args.method == "security_expert"
    re_method = args.method == "routed_expert"
    if args.re_expert_init != "copy" and not args.re_expert_init.startswith("security_expert:"):
        raise SystemExit("--re-expert-init must be 'copy' or 'security_expert:DIR'")
    if args.re_fold_alpha and args.re_expert_init == "copy":
        raise SystemExit("--re-fold-alpha needs --re-expert-init security_expert:DIR")

    import torch
    import transformers

    try:
        import peft  # only the LoRA method needs it
    except ImportError:
        if not (se_method or re_method):
            raise
        peft = None

    from moe_interp.config import config_from_dict
    from moe_interp.model import load_model

    out = Path(args.out)
    save_name = "security_expert" if se_method else "routed_expert" if re_method else "adapter"
    if not args.smoke and (out / save_name).exists():
        raise SystemExit(f"{out}/{save_name} already exists; choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    raw = Path(args.data).read_bytes()
    rows = {r["id"]: r for r in map(json.loads, raw.decode().splitlines())}
    split = json.loads(Path(args.split).read_text())
    assert split["data_sha256"] == hashlib.sha256((REPO / split["data"]).read_bytes()).hexdigest(), "split is stale"
    # v2 has a pair for every record -> the split's record ids; v1 has pairs for a subset -> its pair ids.
    split_key = "ids" if all(i in rows for k in ("train", "val") for i in split["ids"][k]) else "pair_ids"
    train_rows = [rows[i] for i in split[split_key]["train"]]
    val_rows = [rows[i] for i in split[split_key]["val"]]
    print(f"[split] {split['name']} using {split_key}: {len(train_rows)} train / {len(val_rows)} val", flush=True)
    if args.strict_only:
        if not all("labels" in r for r in train_rows):
            raise SystemExit("--strict-only needs judge labels (v1 data); v2 is unfiltered by design")
        train_rows = [r for r in train_rows if not r["labels"]["borderline"]]
    if args.limit:
        train_rows, val_rows = train_rows[: args.limit], val_rows[: args.limit]

    model_cfg = {**OLMOE, **({"tiny_random": True, "device": "cpu", "dtype": "float32"} if args.tiny else {})}
    if args.dtype:
        model_cfg["dtype"] = args.dtype
    t0 = time.time()
    lm = load_model(config_from_dict({"model": model_cfg, "trace": {"enabled": False}}).model, seed=args.seed)
    load_s = time.time() - t0
    tok, device = lm.tokenizer, lm.device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    bf16 = lm.dtype == torch.bfloat16
    autocast = (lambda: torch.autocast(device.type, dtype=torch.bfloat16)) if bf16 else nullcontext

    train, val = tokenize_pairs(tok, train_rows, args.max_len), tokenize_pairs(tok, val_rows, args.max_len)
    lens = sorted(max(len(e["policy_prompt"]), len(e["ref_prompt"])) + max(len(e["chosen"]), len(e["rejected"]))
                  for e in train + val)
    print(f"[data] train {len(train)} / val {len(val)} pairs; longest sequence {lens[-1]} tokens; "
          f"prompt-truncated {sum(e['truncated_prompt'] for e in train + val)}", flush=True)

    # Reference cache from the untouched base model, before any adapter exists.
    header = {"model": lm.info["model_id"], "revision": lm.info["requested_revision"], "dtype": lm.info["dtype"],
              "tiny_random": args.tiny, "max_len": args.max_len, "data_sha256": hashlib.sha256(raw).hexdigest(),
              "autocast_bf16": bf16}
    t1 = time.time()
    cache_path = Path(args.ref_cache) if args.ref_cache else out / "ref_logps.json"
    cache, reused = load_or_build_cache(cache_path, header, lm.model, train + val, args, pad_id, device, autocast)
    ref_s = time.time() - t1
    print(f"[ref] {'reused' if reused else 'computed'} {len(cache)} reference entries in {ref_s:.1f}s -> {cache_path}",
          flush=True)

    pre = None
    if args.smoke:  # frozen base on the smoke batch itself, same batch shape as the online checks
        lm.model.eval()
        with torch.no_grad():
            b = train[: args.micro_batch]
            rc_, rr_, _ = pair_logps(lm.model, b, "ref_prompt", args.max_len, pad_id, device, autocast)
            bc_, br_, _ = pair_logps(lm.model, b, "policy_prompt", args.max_len, pad_id, device, autocast)
        pre = {"rc": rc_, "rr": rr_, "bc": bc_, "br": br_}
    calib = init_expert = init_info = None
    if re_method:
        from moe_interp import routed_expert as RE

        lm.model.eval()
        cal_exs = train[: args.re_calib_pairs]
        cal_batches = [collate(cal_exs[b: b + args.eval_batch], "policy_prompt", args.max_len, pad_id, device)[:2]
                       for b in range(0, len(cal_exs), args.eval_batch)]
        tc = time.time()
        with torch.no_grad(), autocast():
            freq = RE.routing_frequency(lm.model, lm.layout, cal_batches)
        copy_from = ([int(f.argmax()) for f in freq] if args.re_copy == "most_used"
                     else [int(x) for x in args.re_copy.split(",")])
        if args.re_expert_init != "copy":
            init_expert, init_info = RE.security_expert_init(args.re_expert_init.split(":", 1)[1], args.re_fold_alpha)
            print(f"[routed_expert] expert init {json.dumps(init_info)}", flush=True)
        RE.add_routed_experts(lm.model, lm.layout, copy_from, row_noise_std=args.re_row_noise, seed=args.seed,
                              init_expert=init_expert)
        with autocast():
            bias = RE.calibrate_bias(lm.model, cal_batches, args.re_target_rate)
        with torch.no_grad(), autocast(), RE.record_routing(lm.model) as rec:
            for ids_, att_ in cal_batches:
                lm.model(input_ids=ids_, attention_mask=att_, use_cache=False)
        calib = rec.summary() | {"copy_from": copy_from, "source_rate": [round(f[j].item(), 4) for f, j in zip(freq, copy_from)],
                                 "bias": bias, "n_pairs": len(cal_exs), "seconds": round(time.time() - tc, 1)}
        print(f"[routed_expert] calibration {json.dumps(calib)}", flush=True)
        lm.model.config.use_cache = False
        if not args.no_grad_ckpt:
            lm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        policy, lora_cfg, expert_kind = lm.model, None, "conditionally routed 65th expert per MoE layer (top-8 of 65)"
        rep = RE.param_report(policy)
        print(f"[routed_expert] {json.dumps(rep)}", flush=True)
        if rep["non_routed_expert_trainable"] or rep["original_router_trainable"]:
            raise SystemExit("only the new router rows (and new experts) may be trainable")
    elif se_method:
        from moe_interp import security_expert as SE

        SE.add_security_experts(lm.model, lm.layout, alpha_init=args.se_alpha_init, init_std=args.se_init_std,
                                seed=args.seed)
        lm.model.config.use_cache = False
        if not args.no_grad_ckpt:
            lm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        policy, lora_cfg, expert_kind = lm.model, None, "always-active security expert per MoE layer"
        rep = SE.param_report(policy)
        print(f"[security_expert] {json.dumps(rep)}", flush=True)
        if rep["non_security_expert_trainable"] or rep["router_trainable"]:
            raise SystemExit("only security-expert weights may be trainable and the router must stay frozen")
    else:
        policy, lora_cfg, expert_kind = attach_lora(lm.model, lm.layout, args.lora_r, args.lora_alpha,
                                                    not args.no_grad_ckpt, lm.info["requested_revision"])
        rep = param_report(policy)
        print(f"[lora] experts: {expert_kind}; {json.dumps(rep)}", flush=True)
        if rep["non_lora_trainable"] or rep["router_trainable"] or rep["router_lora_modules"]:
            raise SystemExit("only LoRA weights may be trainable and the router must stay frozen")
    se_meta = {"alpha_init": args.se_alpha_init, "init_std": args.se_init_std if args.se_init_std is not None
               else lm.model.config.initializer_range, "init_seed": args.seed,
               "base_model": lm.info["model_id"], "base_revision": lm.info["requested_revision"]} if se_method else None
    re_meta = {"copy": args.re_copy, "expert_init": init_info or "copy", "target_rate": args.re_target_rate,
               "row_noise_std": args.re_row_noise, "calibration": calib, "init_seed": args.seed,
               "base_model": lm.info["model_id"], "base_revision": lm.info["requested_revision"]} if re_method else None

    steps_per_epoch = math.ceil(len(train) / (args.micro_batch * args.grad_accum))
    total_steps = max(1, math.ceil(args.epochs * steps_per_epoch))
    config = {"args": vars(args), "model": lm.info["model_id"], "revision": lm.info["requested_revision"],
              "dtype": lm.info["dtype"], "device": lm.info["device"], "experts": expert_kind,
              "method": args.method,
              "lora": None if lora_cfg is None else {
                  k: (sorted(v) if isinstance(v, set) else v) for k, v in lora_cfg.to_dict().items()
                  if k in ("r", "lora_alpha", "lora_dropout", "target_modules", "target_parameters", "bias")},
              "security_expert": se_meta, "routed_expert": re_meta,
              "params": rep, "objective": "official SecPO sigmoid loss, summed response log-probs, ref = base on clean prompt",
              "secpo_reference": "github.com/Sizhe-Chen/SecPO@da0372a0e598badf4e51c98c175a962147cac2ac",
              "data": args.data, "data_sha256": header["data_sha256"], "split": split["name"],
              "n_train": len(train), "n_val": len(val), "effective_batch": args.micro_batch * args.grad_accum,
              "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
              "warmup_steps": math.ceil(args.warmup_ratio * total_steps), "lr_schedule": "linear warmup + linear decay",
              "optimizer": "AdamW (betas 0.9/0.999, eps 1e-8, weight_decay 0)", "gradient_checkpointing": not args.no_grad_ckpt,
              "torch": torch.__version__, "transformers": transformers.__version__, "peft": getattr(peft, "__version__", None),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    if args.smoke:
        res = (smoke_routed_expert(policy, lm.layout, train, cache, args, pad_id, device, autocast, rep, pre, out, calib,
                                   init_expert)
               if re_method else
               smoke_security_expert(policy, lm.layout, train, cache, args, pad_id, device, autocast, rep, pre, out)
               if se_method else smoke(policy, train, cache, args, pad_id, device, autocast, rep, pre))
        res.update({"config": config, "load_seconds": round(load_s, 1), "ref_seconds": round(ref_s, 1)})
        (out / "smoke_report.json").write_text(json.dumps(res, indent=1) + "\n")
        print(f"[smoke] {'ALL CHECKS PASSED' if res['all_ok'] else 'SOME CHECKS FAILED'} -> {out / 'smoke_report.json'}")
        sys.exit(0 if res["all_ok"] else 1)

    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, **({"fused": True} if device.type == "cuda" else {}))
    sched = transformers.get_linear_schedule_with_warmup(opt, config["warmup_steps"], total_steps)
    save_steps = {int(x) for x in args.save_steps.split(",") if x}
    metrics = (out / "metrics.jsonl").open("w")

    def log(row: dict) -> None:
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()
        print(json.dumps(row), flush=True)

    def run_eval(step: int) -> dict:
        t = time.time()
        if se_method:  # also measure ||alpha S(x)|| vs ||h_MoE(x)|| over the val tokens (policy prompts)
            with SE.record_contributions(policy) as rec:
                v = evaluate(policy, val, cache, args, pad_id, device, autocast)
            c = rec.summary()
            extra = {"se_ratio_mean": round(c["ratio_mean"], 6), "se_ratio_max": round(c["ratio_max"], 6),
                     "se_ratio_by_layer": [round(x["ratio"], 6) for x in c["layers"]],
                     "se_cos_by_layer": [round(x["cos"], 4) for x in c["layers"]],
                     "se_norm_by_layer": [round(x["se_norm"], 5) for x in c["layers"]],
                     "moe_norm_by_layer": [round(x["moe_norm"], 4) for x in c["layers"]],
                     "se_alpha": SE.alphas(policy), "se_tokens": c["tokens"]}
        elif re_method:
            with RE.record_routing(policy) as rec:
                v = evaluate(policy, val, cache, args, pad_id, device, autocast)
            policy.eval()
            with torch.no_grad(), RE.record_routing(policy) as rec_clean:
                for b in range(0, len(val), args.eval_batch):
                    pair_logps(policy, val[b: b + args.eval_batch], "ref_prompt", args.max_len, pad_id, device, autocast)
            policy.train()
            c, cc = rec.summary(), rec_clean.summary()
            extra = {"re_rate_injected": round(c["rate_mean"], 5), "re_rate_clean": round(cc["rate_mean"], 5),
                     "re_rate_injected_by_layer": [round(x, 4) for x in c["rate_by_layer"]],
                     "re_rate_clean_by_layer": [round(x, 4) for x in cc["rate_by_layer"]],
                     "re_weight_when_selected_by_layer": [round(x, 4) for x in c["weight_when_selected_by_layer"]],
                     "re_tokens": c["tokens"]}
        else:
            v, extra = evaluate(policy, val, cache, args, pad_id, device, autocast), {}
        log({"type": "val", "step": step, **{f"{s}/{k}": round(x, 5) if isinstance(x, float) else x
                                            for s in v for k, x in v[s].items()},
             **extra, "eval_seconds": round(time.time() - t, 1), **gpu_mem(device)})
        return v

    def save(path: Path):
        if re_method:
            sha = RE.save_routed_experts(policy, path, re_meta)
            tok.save_pretrained(path)
            return sha
        if se_method:
            sha = SE.save_security_experts(policy, path, se_meta)
            tok.save_pretrained(path)
            return sha
        policy.save_pretrained(path)
        tok.save_pretrained(path)
        return None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    first_val = run_eval(0)
    policy.train()
    g = torch.Generator().manual_seed(args.seed)
    t_train, step, epoch = time.time(), 0, 0
    while step < total_steps:
        order = torch.randperm(len(train), generator=g).tolist()
        micro = [[train[i] for i in order[b: b + args.micro_batch]] for b in range(0, len(order), args.micro_batch)]
        for s in range(0, len(micro), args.grad_accum):
            if step >= total_steps:
                break
            group = micro[s: s + args.grad_accum]
            n_pairs = sum(len(m) for m in group)
            ts, acc, ntok = time.time(), {"loss": 0.0, "margin": 0.0, "acc": 0.0, "plr": 0.0, "cr": 0.0, "rr": 0.0}, 0
            rec_ctx = RE.record_routing(policy) if re_method else nullcontext()
            rec = rec_ctx.__enter__()
            lp = {"c": 0.0, "r": 0.0}
            for exs in group:
                pc, pr, nt = pair_logps(policy, exs, "policy_prompt", args.max_len, pad_id, device, autocast)
                lp["c"] += pc.detach().sum().item() / n_pairs
                lp["r"] += pr.detach().sum().item() / n_pairs
                o = secpo_loss(pc, pr, *ref_tensors(cache, exs, device), args.beta)
                (o["losses"].sum() / n_pairs).backward()  # mean over the pairs of this optimizer step
                ntok += nt
                for k, v in (("loss", o["losses"]), ("margin", o["logits"]), ("acc", (o["logits"] > 0).float()),
                             ("plr", pc - pr), ("cr", o["chosen_rewards"]), ("rr", o["rejected_rewards"])):
                    acc[k] += v.detach().sum().item() / n_pairs
            rec_ctx.__exit__(None, None, None)
            gn = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm if args.max_grad_norm > 0 else float("inf"))
            lr = sched.get_last_lr()[0]
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            dt = time.time() - ts
            log({"type": "train", "step": step, "epoch": round(epoch + (s + len(group)) / len(micro), 3),
                 "loss": round(acc["loss"], 5), "secpo_margin": round(acc["margin"], 4), "secpo_acc": round(acc["acc"], 4),
                 "policy_logratio_adv": round(acc["plr"], 4), "chosen_reward": round(acc["cr"], 4),
                 "rejected_reward": round(acc["rr"], 4), "logp_chosen": round(lp["c"], 3), "logp_rejected": round(lp["r"], 3),
                 "grad_norm": round(gn.item(), 4), "lr": lr, "pairs": n_pairs,
                 **({"re_rate": round(rec.summary()["rate_mean"], 5)} if re_method else {}),
                 **({"se_alpha_mean": round(sum(SE.alphas(policy)) / len(SE.alphas(policy)), 6)} if se_method else {}),
                 "tokens": ntok, "step_seconds": round(dt, 2), "tokens_per_s": round(ntok / dt),
                 "elapsed_s": round(time.time() - t_train, 1), **gpu_mem(device)})
            if step < total_steps and (step % args.eval_every == 0 or step in save_steps):
                run_eval(step)
            if step in save_steps and step < total_steps:
                log({"type": "checkpoint", "step": step, "dir": f"{save_name}_step{step}",
                     "sha256": save(out / f"{save_name}_step{step}")})
                policy.train()
        epoch += 1
        if args.save_every_epoch:
            v = run_eval(step) if step < total_steps else None  # the final epoch is evaluated after the loop
            save(out / f"{save_name}_epoch{epoch}")
            log({"type": "epoch_end", "epoch": epoch, "step": step, "adapter": f"{save_name}_epoch{epoch}",
                 "val_loss": v["all"]["loss"] if v else None,
                 "expert_delta_realization": None if (se_method or re_method) else expert_delta_realization(policy)})
            policy.train()
    train_s = time.time() - t_train
    last_val = run_eval(step)
    metrics.close()

    sha = save(out / save_name)
    if step in save_steps:  # the final checkpoint under its step name too (same weights as OUT/<save_name>)
        save(out / f"{save_name}_step{step}")
    info = {"step": "e03_secpo_train", "method": args.method, "optimizer_steps": step, "load_seconds": round(load_s, 1),
            "ref_cache_seconds": round(ref_s, 1), "ref_cache_reused": reused, "train_seconds": round(train_s, 1),
            "total_seconds": round(time.time() - t0, 1), **gpu_mem(device),
            "expert_delta_realization": None if (se_method or re_method) else expert_delta_realization(policy),
            "val_step0": first_val, "val_final": last_val,
            "trainable_params": rep["trainable"], "trainable_pct": rep["trainable_pct"],
            **({"se_alpha_final": SE.alphas(policy), "security_expert_sha256": sha} if se_method else {}),
            **({"routed_expert_sha256": sha, "calibration": calib} if re_method else {}),
            "adapter_dir": str(out / save_name), "config": "config.json"}
    (out / "run_info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "val_step0"}, indent=1))


if __name__ == "__main__":
    main()
