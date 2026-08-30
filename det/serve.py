"""Detector service. Runs on the H200, holds the four models, serves /check.

The gateway never imports torch; it talks to this over HTTP. Finding is
mirrored from api/schemas.py and re-exported here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from det import bias, embed, nli, safety, t2
from det.schema import Finding  # noqa: F401  re-exported: this is the wire contract

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
CACHE_ON = os.getenv("DET_CACHE", "1") == "1"

# "nli=8101,safety=8102,bias=8103" fans /check out to worker processes so the
# detectors run genuinely concurrently. Empty means load everything in-process,
# which is simpler and still correct, just serialised by the GIL.
WORKERS: dict[str, str] = {
    k: v for k, v in
    (kv.split("=", 1) for kv in os.getenv("DET_WORKERS", "").split(",") if kv)
}
# embed holds the semantic cache and is small, so it always stays in-process
LOCAL = ["embed"] + [n for n in ("nli", "safety", "bias") if n not in WORKERS]

logging.basicConfig(format="%(message)s", level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("det")

MODS = {"nli": nli, "safety": safety, "bias": bias, "embed": embed}


class CheckReq(BaseModel):
    resp: str
    ctx: list[dict[str, Any]] = Field(default_factory=list)
    need: list[str] = Field(default_factory=lambda: ["nli", "safety", "bias"])
    prompt: str = ""            # guard models score the pair, not the reply alone
    cache: bool = True


class CheckRes(BaseModel):
    findings: list[Finding]
    lat: dict[str, float]
    cached: bool = False


_wcli: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _wcli
    t = time.perf_counter()
    for n in LOCAL:
        t1 = time.perf_counter()
        MODS[n].load()
        log.info('{"event": "loaded", "model": "%s", "ms": %.0f}',
                 n, (time.perf_counter() - t1) * 1000)
    await t2.open_t2()
    if WORKERS:
        _wcli = httpx.AsyncClient(timeout=float(os.getenv("DET_WORKER_TIMEOUT_S", "30")))
        log.info('{"event": "workers", "map": "%s"}', WORKERS)
    log.info('{"event": "ready", "ms": %.0f}', (time.perf_counter() - t) * 1000)
    yield
    await t2.close_t2()
    if _wcli is not None:
        await _wcli.aclose()


app = FastAPI(title="ControlPlane.ai detectors", version="0.1.0", lifespan=lifespan)


def _vram() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2),
        "free_gb": round(free / 2**30, 2),
        "total_gb": round(total / 2**30, 2),
    }


async def _worker_health(name: str, port: str) -> dict[str, Any]:
    if _wcli is None:
        return {"ready": False, "err": "no worker client"}
    try:
        r = await _wcli.get(f"http://127.0.0.1:{port}/health", timeout=3.0)
        return {**r.json(), "port": port}
    except Exception as e:
        return {"ready": False, "port": port, "err": type(e).__name__}


@app.get("/health")
async def health() -> dict[str, Any]:
    mods = {n: MODS[n].info() for n in LOCAL}
    mods["t2"] = t2.info()
    if WORKERS:
        got = await asyncio.gather(*(_worker_health(n, p) for n, p in WORKERS.items()))
        mods.update(dict(zip(WORKERS, got)))
    return {
        "status": "ok" if all(m.get("ready") for m in mods.values()) else "degraded",
        "models": mods,
        "vram": _vram(),
        "device": os.getenv("DET_DEVICE", "cuda"),
        "mode": "workers" if WORKERS else "in-process",
    }


@app.post("/check", response_model=CheckRes)
async def check(r: CheckReq) -> CheckRes:
    t0 = time.perf_counter()
    lat: dict[str, float] = {}

    if CACHE_ON and r.cache:
        tc = time.perf_counter()
        hit = await embed.cache_get(r.resp, r.ctx)
        lat["cache"] = round((time.perf_counter() - tc) * 1000, 2)
        if hit is not None:
            lat["total"] = round((time.perf_counter() - t0) * 1000, 2)
            return CheckRes(findings=hit, lat=lat, cached=True)

    # sentence spans are shared: NLI needs them, and bias scores per sentence
    # so its finding highlights the offending clause rather than the whole reply
    sp = nli.sents(r.resp)

    async def timed(name: str, coro: Any) -> list[Finding]:
        t = time.perf_counter()
        try:
            out = await coro
        except Exception as e:
            log.warning('{"event": "det_failed", "det": "%s", "err": "%s"}',
                        name, type(e).__name__)
            out = []
        lat[name] = round((time.perf_counter() - t) * 1000, 2)
        return out

    async def remote(name: str) -> list[Finding]:
        assert _wcli is not None
        body = {"resp": r.resp, "ctx": r.ctx, "prompt": r.prompt, "spans": sp}
        res = await _wcli.post(f"http://127.0.0.1:{WORKERS[name]}/run", json=body)
        res.raise_for_status()
        return [Finding(**f) for f in res.json()["findings"]]

    local = {"nli": lambda: nli.check(r.resp, r.ctx),
             "safety": lambda: safety.check(r.resp, r.prompt),
             "bias": lambda: bias.check(r.resp, sp)}

    jobs = []
    # Tier 2 replaces tier 1 grounding rather than adding to it. Both answer the
    # same question; t2 answers it with evidence chosen per claim, so running
    # both would double-count every grounding finding.
    deep = "t2" in r.need
    if deep:
        jobs.append(timed("t2", t2.check(r.resp, r.ctx, r.prompt)))
    for name in ("nli", "safety", "bias"):
        if name not in r.need or (name == "nli" and deep):
            continue
        jobs.append(timed(name, remote(name) if name in WORKERS else local[name]()))

    res = await asyncio.gather(*jobs)
    fnd = [f for group in res for f in group]
    lat["total"] = round((time.perf_counter() - t0) * 1000, 2)

    if CACHE_ON and r.cache:
        await embed.cache_put(r.resp, r.ctx, fnd)

    return CheckRes(findings=fnd, lat=lat, cached=False)


@app.post("/cache/clear")
async def cache_clear() -> dict[str, str]:
    embed.cache_clear()
    return {"status": "cleared"}
