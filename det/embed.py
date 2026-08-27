"""bge-small: chunk retrieval, and a semantic cache in front of /check.

The cache is the reason this model earns its VRAM. Support traffic repeats
itself, and a near-duplicate response against the same context has already been
verified, so we can return the earlier findings instead of paying for NLI again.
Keyed on context identity as well as text: same words against different
evidence is a different question.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from det.batcher import Batcher
from det.schema import Finding

MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
DEV = os.getenv("DET_DEVICE_EMBED") or os.getenv("DET_DEVICE", "cuda")
VER = "bge-small-en-v1.5"
Q_PREFIX = "Represent this sentence for searching relevant passages: "
CACHE_N = int(os.getenv("CACHE_N", "512"))
CACHE_THR = float(os.getenv("CACHE_THR", "0.97"))
CACHE_TTL_S = float(os.getenv("CACHE_TTL_S", "900"))

_tok: Any = None
_mod: Any = None
_bat: Batcher[str, list[float]] | None = None

# semantic cache: parallel arrays so lookup is one matmul
_vec: Any = None                      # (n, d) normalised, on device
_key: list[str] = []                  # context identity per row
_val: list[list[Finding]] = []
_ts: list[float] = []
_hit = _miss = 0


def load() -> None:
    global _tok, _mod, _bat, _vec
    _tok = AutoTokenizer.from_pretrained(MODEL)
    _mod = AutoModel.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
    _fwd(["Refunds take three business days."])  # warm cuda
    _bat = Batcher(_fwd, "embed")
    _bat.start()
    _vec = torch.zeros((0, _mod.config.hidden_size), dtype=torch.float16, device=DEV)


def ready() -> bool:
    return _mod is not None


def info() -> dict[str, Any]:
    return {"model": MODEL, "ver": VER, "ready": ready(), "dev": DEV,
            "cache": {"n": len(_key), "hit": _hit, "miss": _miss,
                      "thr": CACHE_THR},
            "batch": dict(_bat.stat) if _bat else {}}


def _fwd(txt: list[str]) -> list[list[float]]:
    b = _tok(txt, return_tensors="pt", padding=True, truncation=True,
             max_length=512).to(DEV)
    with torch.no_grad():
        h = _mod(**b).last_hidden_state[:, 0]          # CLS pooling, per bge
        h = torch.nn.functional.normalize(h, p=2, dim=-1)
    return h.float().cpu().tolist()


async def vec(txt: str) -> list[float]:
    assert _bat is not None
    return await _bat.submit(txt)


async def rank(q: str, docs: list[dict[str, Any]], k: int = 5) -> list[dict[str, Any]]:
    """Score context chunks against a query and keep the top k."""
    if not docs:
        return []
    assert _bat is not None
    vs = await _bat.submit_many([Q_PREFIX + q] + [d.get("text", "") for d in docs])
    qv = torch.tensor(vs[0])
    dv = torch.tensor(vs[1:])
    s = (dv @ qv).tolist()
    out = [{**d, "score": round(float(x), 4)} for d, x in zip(docs, s)]
    out.sort(key=lambda d: d["score"], reverse=True)
    return out[:k]


def ctx_key(ctx: list[dict[str, Any]]) -> str:
    return "|".join(sorted(str(c.get("id", "")) for c in ctx))


async def cache_get(resp: str, ctx: list[dict[str, Any]]) -> list[Finding] | None:
    global _hit, _miss
    if _vec is None or len(_key) == 0:
        _miss += 1
        return None
    v = torch.tensor(await vec(resp), dtype=torch.float16, device=DEV)
    sim = (_vec @ v).float()
    k = ctx_key(ctx)
    now = time.time()
    best_i, best_s = -1, 0.0
    for i in range(len(_key)):
        if _key[i] != k or now - _ts[i] > CACHE_TTL_S:
            continue
        s = float(sim[i])
        if s > best_s:
            best_i, best_s = i, s
    if best_i >= 0 and best_s >= CACHE_THR:
        _hit += 1
        return _val[best_i]
    _miss += 1
    return None


async def cache_put(resp: str, ctx: list[dict[str, Any]], fnd: list[Finding]) -> None:
    global _vec
    v = torch.tensor([await vec(resp)], dtype=torch.float16, device=DEV)
    _vec = torch.cat([_vec, v], dim=0)
    _key.append(ctx_key(ctx))
    _val.append(fnd)
    _ts.append(time.time())
    if len(_key) > CACHE_N:
        _vec = _vec[1:]
        _key.pop(0)
        _val.pop(0)
        _ts.pop(0)


def cache_clear() -> None:
    global _vec, _hit, _miss
    if _mod is not None:
        _vec = torch.zeros((0, _mod.config.hidden_size), dtype=torch.float16, device=DEV)
    _key.clear()
    _val.clear()
    _ts.clear()
    _hit = _miss = 0


async def select(
    resp: str,
    sp: list[tuple[int, int]],
    ctx: list[dict[str, Any]],
    k: int = 2,
) -> list[list[int]]:
    """Top-k context chunks per response sentence.

    Scoring every sentence against every chunk is quadratic and mostly wasted:
    a sentence about warranty length has nothing to settle against the shipping
    doc. One cheap embedding pass picks the chunks that could plausibly ground
    or contradict each sentence, and NLI only reads those.
    """
    if not sp or not ctx:
        return [[] for _ in sp]
    if len(ctx) <= k:
        return [list(range(len(ctx))) for _ in sp]
    assert _bat is not None
    vs = await _bat.submit_many(
        [Q_PREFIX + resp[a:b] for a, b in sp] + [c.get("text", "") for c in ctx]
    )
    sv = torch.tensor(vs[: len(sp)])
    cv = torch.tensor(vs[len(sp):])
    sim = sv @ cv.T                              # (nsent, nctx)
    return [row.topk(k).indices.tolist() for row in sim]
