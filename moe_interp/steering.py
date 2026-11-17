"""Small, notebook-friendly helpers for direction steering on SEP prompts (any supported MoE model).

Unlike ``experiments/sep_steering.py`` (which needs a full traced run + ``features.npz``), everything here works
directly on a loaded model, with no traces on disk:

* :func:`build_items`      SEP samples -> one dict per (sample, condition) with token ids and token spans
* :func:`capture`          one batched forward pass -> mean activation over a span, per layer
* :func:`diff_of_means`    the steering direction per layer (unit vectors)
* :func:`steer_intervention` ``ResidSteer`` on resid_post, gated to the untrusted user message (or to all tokens)
* :func:`generate`         batched greedy generation, optionally inside an intervention, scored by whole-word witness
* :func:`check_steer`      verifies the hook really removes the direction where intended and nowhere else
* :func:`clean_kl`         how much steering moves the next-token distribution on clean (uninjected) prompts

Activation sites (per layer ``l``):
``mid``  = resid_mid, the residual stream after attention, before the MoE (the input of post_attention_layernorm,
           i.e. what the router reads, before its norm) -- the site the Mellum direction was fitted on.
``post`` = resid_post, the decoder layer's output -- the site ``ResidSteer`` edits.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np
import torch

from .interventions import Intervention, ResidSteer, padded_mask
from .runner import generate_batch


# ---- data ---------------------------------------------------------------------------------------------------------

def build_items(samples: list, rendered: dict, conditions: tuple[str, ...] | None = None) -> list[dict]:
    """One dict per (sample, condition). Spans are token ``(start, end)`` in the unpadded prompt."""
    items = []
    for s in samples:
        for cond in s.variants:
            if conditions and cond not in conditions:
                continue
            rp = rendered[(s.id, cond)]
            us = rp.message_token_spans[1]
            items.append({
                "sample_id": s.id, "condition": cond, "probe_id": s.metadata.get("probe_id"),
                "probe_position": s.metadata.get("probe_position_in_data"),
                "probe": s.metadata.get("probe"), "witness": s.evaluation.get("witness"),
                "input_ids": rp.input_ids, "text": rp.text,
                "user_span": (us.start, us.end),
                "spans": {k: (v.start, v.end) for k, v in rp.spans.items()},
            })
    return items


def split_probes(items: list[dict], seed: int = 0) -> set:
    """Half of the probe ids (identical split to experiments/sep_steering.py). Fit on half A, evaluate on half B,
    so the direction is never tested on a probe instruction it was fitted on."""
    probes = sorted({r["probe_id"] for r in items})
    rng = np.random.default_rng(seed)
    rng.shuffle(probes)
    return set(probes[: len(probes) // 2])


def show_spans(tok: Any, item: dict, names: tuple[str, ...] = ("probe", "insert")) -> str:
    """The prompt with ⟦ ⟧ around the tokens of the given spans and ‹ › around the user message (for eyeballing)."""
    ids = item["input_ids"]
    marks: dict[int, list[str]] = {}
    us, ue = item["user_span"]
    marks.setdefault(us, []).append("‹")
    marks.setdefault(ue, []).insert(0, "›")
    for n in names:
        if n in item["spans"]:
            a, b = item["spans"][n]
            marks.setdefault(a, []).append("⟦")
            marks.setdefault(b, []).insert(0, "⟧")
    out = []
    for i in range(len(ids) + 1):
        out += marks.get(i, [])
        if i < len(ids):
            out.append(tok.decode([ids[i]]))
    return "".join(out)


def wb_hit(text: str, witness: str) -> bool:
    """Whole-word, case-insensitive witness match (the substring check over-counts, see results_security.md)."""
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(witness) + r"(?![A-Za-z0-9])", text or "", re.I) is not None


# ---- activations --------------------------------------------------------------------------------------------------

def _pad(rows: list[dict], pad_id: int, device: torch.device):
    T = max(len(r["input_ids"]) for r in rows)
    ids = torch.tensor([[pad_id] * (T - len(r["input_ids"])) + r["input_ids"] for r in rows], device=device)
    mask = torch.tensor([[0] * (T - len(r["input_ids"])) + [1] * len(r["input_ids"]) for r in rows], device=device)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids, mask, pos, T


@torch.inference_mode()
def capture(lm: Any, items: list[dict], spans: tuple[str, ...] = ("probe", "insert"), site: str = "mid",
            batch: int = 8, intervention: Callable[[list[dict], int], Intervention] | None = None,
            pad_id: int | None = None) -> dict[str, np.ndarray]:
    """Per item and layer: the mean activation over each span's tokens, and at the final prompt token ("last").

    Returns ``{span: float32 [N, L, H]}`` (rows of items lacking the span are NaN). ``intervention(rows, T)`` lets
    you capture *under* steering (the capture hooks run after the steering hook, so they see the edited stream).
    """
    model, layout, device = lm.model, lm.layout, lm.device
    L, H = layout.num_layers, layout.hidden_size
    pad_id = pad_id if pad_id is not None else (lm.tokenizer.pad_token_id or lm.tokenizer.eos_token_id)
    out = {s: np.full((len(items), L, H), np.nan, dtype=np.float32) for s in (*spans, "last")}
    for i0 in range(0, len(items), batch):
        rows = items[i0:i0 + batch]
        ids, mask, pos, T = _pad(rows, pad_id, device)
        got: dict[int, torch.Tensor] = {}
        iv = intervention(rows, T) if intervention else None
        if iv:
            iv.__enter__()
        hs = []
        try:
            for l in range(L):
                if site == "mid":
                    hs.append(layout.post_attention_norms[l].register_forward_pre_hook(
                        lambda m, a, l=l: got.__setitem__(l, a[0].detach().float())))
                else:
                    hs.append(layout.layers[l].register_forward_hook(
                        lambda m, a, o, l=l: got.__setitem__(l, (o[0] if isinstance(o, tuple) else o).detach().float())))
            model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
        finally:
            for h in hs:
                h.remove()
            if iv:
                iv.__exit__(None, None, None)
        X = torch.stack([got[l] for l in range(L)], 1)  # [B, L, T, H]
        for j, r in enumerate(rows):
            off = T - len(r["input_ids"])
            out["last"][i0 + j] = X[j, :, T - 1].cpu().numpy()
            for s in spans:
                if s in r["spans"]:
                    a, b = r["spans"][s]
                    out[s][i0 + j] = X[j, :, off + a:off + b].mean(1).cpu().numpy()
    return out


def diff_of_means(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """[N1, L, H], [N2, L, H] -> unit direction per layer [L, H] (mean(pos) - mean(neg), normalised)."""
    d = np.nanmean(pos, 0) - np.nanmean(neg, 0)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def random_directions(pos: np.ndarray, neg: np.ndarray, seed: int = 0) -> np.ndarray:
    """Control: per layer, a random combination of the centred activations (so it lives in the same subspace and
    has typical activation variance, unlike an isotropic Gaussian direction)."""
    rng = np.random.default_rng(seed)
    X = np.concatenate([pos, neg], 0)
    out = []
    for l in range(X.shape[1]):
        Xc = X[:, l] - X[:, l].mean(0)
        v = Xc.T @ rng.standard_normal(len(Xc))
        out.append(v / np.linalg.norm(v))
    return np.stack(out)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative)."""
    s = np.r_[pos, neg]
    r = s.argsort().argsort() + 1.0
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def separation(dirs: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> dict[str, np.ndarray]:
    """Per layer: AUROC and Cohen's d of the projections of pos vs neg onto dirs."""
    pp, pn = np.einsum("nlh,lh->nl", pos, dirs), np.einsum("nlh,lh->nl", neg, dirs)
    au = np.array([auroc(pp[:, l], pn[:, l]) for l in range(dirs.shape[0])])
    sd = np.sqrt((pp.var(0) + pn.var(0)) / 2)
    return {"auroc": au, "cohen_d": (pp.mean(0) - pn.mean(0)) / sd, "proj_pos": pp, "proj_neg": pn}


# ---- steering -----------------------------------------------------------------------------------------------------

def steer_intervention(layout: Any, dirs: np.ndarray, layers: list[int], coef: float = 1.0, gate: str = "user",
                       decode: bool = False, extra: int = 0) -> Callable[[list[dict], int], Intervention]:
    """Factory ``(rows, T) -> Intervention``. ``h <- h - coef (h·d) d`` on resid_post at ``layers``.

    gate="user": only the untrusted user-message tokens (incl. any injected probe), prompt pass only.
    gate="all":  every prompt token (and generated tokens if ``decode``).
    ``extra``: number of teacher-forced answer tokens appended after the prompt (never steered when gated).
    """
    dd = {l: torch.from_numpy(np.ascontiguousarray(dirs[l])) for l in layers}

    def make(rows: list[dict], T: int) -> Intervention:
        if gate == "user":
            n = [len(r["input_ids"]) for r in rows]
            m = padded_mask([r["user_span"] for r in rows], n, T)
            if extra:
                m = torch.cat([m, torch.zeros(m.shape[0], extra, dtype=torch.bool)], 1)
            return Intervention(layout, steer=ResidSteer(dd, token_mask=m, coef=coef, decode=False))
        return Intervention(layout, steer=ResidSteer(dd, token_mask=None, coef=coef, decode=decode))
    return make


@torch.inference_mode()
def check_steer(lm: Any, item: dict, dirs: np.ndarray, layers: list[int], coef: float = 1.0) -> list[dict]:
    """Single prompt: |projection| of resid_post on the direction, inside vs outside the user message, before and
    after steering. With coef=1 'after, inside' must be ~0 at steered layers and 'outside' unchanged at the first
    steered layer (later layers can change outside because attention mixes positions)."""
    mk = steer_intervention(lm.layout, dirs, layers, coef)
    T = len(item["input_ids"])
    rows = []
    us, ue = item["user_span"]
    for tag, iv in (("before", None), ("after", mk)):
        got: dict[int, torch.Tensor] = {}
        ctx = iv([item], T) if iv else None
        if ctx:
            ctx.__enter__()
        hs = [lm.layout.layers[l].register_forward_hook(
            lambda m, a_, o, l=l: got.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))
            for l in range(lm.layout.num_layers)]
        try:
            lm.model(input_ids=torch.tensor([item["input_ids"]], device=lm.device), use_cache=False)
        finally:
            for h in hs:
                h.remove()
            if ctx:
                ctx.__exit__(None, None, None)
        for l in range(lm.layout.num_layers):
            p = (got[l] @ torch.from_numpy(dirs[l]).float()).abs()
            inside = p[us:ue].mean().item()
            outside = torch.cat([p[:us], p[ue:]]).mean().item()
            rows.append({"layer": l, "when": tag, "steered": l in layers, "proj_user": inside, "proj_outside": outside})
    return rows


def generate(lm: Any, gc: Any, rows: list[dict], make_iv: Callable | None = None, batch: int = 16,
             skip_special: bool = True) -> tuple[list[str], list[bool]]:
    """Batched generation in the given order (keep the order fixed across conditions: bf16 greedy decoding depends on
    batch composition). Returns (texts, whole-word witness hits)."""
    texts, hits = [], []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        ids = [r["input_ids"] for r in chunk]
        T = max(map(len, ids))
        if make_iv is None:
            resp = generate_batch(lm.model, ids, gc, lm.device)
        else:
            with make_iv(chunk, T):
                resp = generate_batch(lm.model, ids, gc, lm.device)
        for r, x in zip(chunk, resp):
            t = lm.tokenizer.decode(x, skip_special_tokens=skip_special)
            texts.append(t)
            hits.append(wb_hit(t, r["witness"]) if r.get("witness") else False)
    return texts, hits


@torch.inference_mode()
def clean_kl(lm: Any, rows: list[dict], responses: list[list[int]], dirs: np.ndarray, layers: list[int],
             coef: float = 1.0, gate: str = "user", n_answer: int = 64) -> float:
    """Mean per-token KL(unsteered || steered) over the first ``n_answer`` tokens of each row's own baseline answer
    (teacher-forced). Small = steering leaves the uninjected model's behaviour alone."""
    kls = []
    for r, resp in zip(rows, responses):
        p, a = r["input_ids"], resp[:n_answer]
        if not a:
            continue
        x = torch.tensor([p + a], device=lm.device)
        b = lm.model(input_ids=x, use_cache=False).logits[0, len(p) - 1:-1].float().log_softmax(-1)
        mk = steer_intervention(lm.layout, dirs, layers, coef, gate=gate, extra=len(a))
        with mk([r], len(p)):
            n = lm.model(input_ids=x, use_cache=False).logits[0, len(p) - 1:-1].float().log_softmax(-1)
        kls.append(float((b.exp() * (b - n)).sum(-1).mean()))
    return float(np.mean(kls))


def paired_ci(new: list[bool] | np.ndarray, base: list[bool] | np.ndarray, n_boot: int = 2000,
              seed: int = 7) -> tuple[float, float, float]:
    """Mean paired difference new - base with a bootstrap 95% CI."""
    d = np.asarray(new, float) - np.asarray(base, float)
    b = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    m = d[b].mean(1)
    return float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))
