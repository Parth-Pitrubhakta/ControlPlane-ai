"""Three-valued grounding. SUPPORTED, CONTRADICTED, or UNVERIFIABLE -- never binary.

UNVERIFIABLE means we found no evidence either way, which is not the same as
false. Invariant 2 caps it at annotate; nothing here may push it higher.
"""

from __future__ import annotations

import os
import re
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from det import embed
from det.batcher import Batcher
from det.schema import Finding

MODEL = os.getenv(
    "NLI_MODEL", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
)
DEV = os.getenv("DET_DEVICE_NLI") or os.getenv("DET_DEVICE", "cuda")
VER = "nli-deberta-v3-large-v1"
MAXLEN = int(os.getenv("NLI_MAXLEN", "384"))
# the spec's 16-item cap is for micro-batching across concurrent requests. One
# request's sentence x chunk cross product is already tens of pairs, and
# splitting it into 16s costs several sequential forward passes for no gain, so
# NLI takes a wider window and still does one pass per flush.
BATCH_MAX = int(os.getenv("NLI_BATCH_MAX", "64"))
# Chunks considered per sentence, chosen by embedding similarity. 0 disables it
# and falls back to the full cross product. Measured off by default: at 5 chunks
# the extra embedding round trip costs more than the pairs it saves (199 ms vs
# 133 ms p50). Turn it on when a tenant retrieves more than about ten chunks.
TOPK = int(os.getenv("NLI_TOPK", "0"))

# spec thresholds. contradiction is checked first, so a sentence that is both
# strongly entailed by one chunk and contradicted by another resolves to
# CONTRADICTED -- the conservative reading when the corpus disagrees with itself.
THR_CONTRA = float(os.getenv("NLI_THR_CONTRA", "0.7"))
THR_ENTAIL = float(os.getenv("NLI_THR_ENTAIL", "0.6"))

_tok: Any = None
_mod: Any = None
_ix: dict[str, int] = {}
_bat: Batcher[tuple[str, str], tuple[float, float]] | None = None

# sentence split that keeps char offsets. Deliberately simple: no model, no
# dependency, and it never merges across a sentence boundary.
_SENT = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")


def sents(txt: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for m in _SENT.finditer(txt):
        a, b = m.start(), m.end()
        while a < b and txt[a].isspace():
            a += 1
        while b > a and txt[b - 1].isspace():
            b -= 1
        # a fragment with no letters or digits is punctuation, not a claim:
        # sending it to NLI as a hypothesis invites a spurious finding
        if b - a >= 2 and any(txt[i].isalnum() for i in range(a, b)):
            out.append((a, b))
    return out


def load() -> None:
    global _tok, _mod, _ix, _bat
    cfg = AutoConfig.from_pretrained(MODEL)
    _ix = {v.lower(): k for k, v in cfg.id2label.items()}
    if not {"entailment", "contradiction"} <= set(_ix):
        raise RuntimeError(f"unexpected NLI label map: {cfg.id2label}")
    _tok = AutoTokenizer.from_pretrained(MODEL)
    _mod = AutoModelForSequenceClassification.from_pretrained(
        MODEL, dtype=torch.float16
    ).to(DEV).eval()
    _fwd([("The warranty is three years.", "The warranty is three years.")])  # warm cuda
    _bat = Batcher(_fwd, "nli", max_items=BATCH_MAX)
    _bat.start()


def ready() -> bool:
    return _mod is not None


def info() -> dict[str, Any]:
    return {"model": MODEL, "ver": VER, "ready": ready(), "dev": DEV,
            "batch": dict(_bat.stat) if _bat else {}}


def _fwd(pairs: list[tuple[str, str]]) -> list[tuple[float, float]]:
    """One batched forward pass. Returns (p_entail, p_contra) per pair."""
    prem = [p for p, _ in pairs]
    hyp = [h for _, h in pairs]
    b = _tok(prem, hyp, return_tensors="pt", padding=True, truncation=True,
             max_length=MAXLEN).to(DEV)
    with torch.no_grad():
        p = _mod(**b).logits.softmax(-1).float().cpu()
    e, c = _ix["entailment"], _ix["contradiction"]
    return [(float(r[e]), float(r[c])) for r in p]


async def check(resp: str, ctx: list[dict[str, Any]]) -> list[Finding]:
    """Cross product of response sentences against context chunks, batched.

    With no context every sentence is UNVERIFIABLE: absence of retrieval is
    absence of evidence, not evidence of falsehood.
    """
    sp = sents(resp)
    if not sp:
        return []
    if not ctx:
        return [
            Finding(span=s, dim="perf", label="unverifiable", sev=1, conf=0.5,
                    evid=None, det=VER)
            for s in sp
        ]

    sel = (await embed.select(resp, sp, ctx, TOPK)) if TOPK > 0 else None

    pairs: list[tuple[str, str]] = []
    owner: list[tuple[int, int]] = []   # (sentence index, ctx index)
    for i, (a, b) in enumerate(sp):
        for j in (sel[i] if sel is not None else range(len(ctx))):
            pairs.append((ctx[j].get("text", ""), resp[a:b]))
            owner.append((i, j))

    assert _bat is not None
    res = await _bat.submit_many(pairs)

    best_e = [0.0] * len(sp)
    best_c = [0.0] * len(sp)
    src_e: list[str | None] = [None] * len(sp)
    src_c: list[str | None] = [None] * len(sp)
    for k, (e, c) in enumerate(res):
        i, j = owner[k]
        d = ctx[j].get("id")
        if e > best_e[i]:
            best_e[i], src_e[i] = e, d
        if c > best_c[i]:
            best_c[i], src_c[i] = c, d

    out: list[Finding] = []
    for i, s in enumerate(sp):
        if best_c[i] > THR_CONTRA:
            out.append(Finding(span=s, dim="perf", label="contradicted", sev=2,
                               conf=round(best_c[i], 4), evid=src_c[i], det=VER))
        elif best_e[i] > THR_ENTAIL:
            continue   # SUPPORTED: grounded, so no finding
        else:
            out.append(Finding(span=s, dim="perf", label="unverifiable", sev=1,
                               conf=round(1.0 - max(best_e[i], best_c[i]), 4),
                               evid=None, det=VER))
    return out
