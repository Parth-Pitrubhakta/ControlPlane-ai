"""One detector, one process, one port.

The detectors are kernel-launch bound rather than compute bound: a DeBERTa
forward on a (35, 27) batch spends 32 ms in Python launching kernels, holding
the GIL the whole time. In a single process that makes asyncio.gather a lie --
three detectors "in parallel" measured 144 ms against 41+34+15 of real work.
Separate processes give the concurrency the tier budget assumes.

DET_ROLE picks which module this process loads. serve.py fans out to these.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from pydantic import BaseModel, Field

from det import bias, embed, nli, safety
from det.schema import Finding

ROLE = os.getenv("DET_ROLE", "nli")
MODS = {"nli": nli, "safety": safety, "bias": bias, "embed": embed}
if ROLE not in MODS:
    raise SystemExit(f"DET_ROLE must be one of {sorted(MODS)}, got {ROLE!r}")
MOD = MODS[ROLE]

logging.basicConfig(format="%(message)s",
                    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO"), logging.INFO))
log = logging.getLogger(f"det.{ROLE}")


class RunReq(BaseModel):
    resp: str = ""
    ctx: list[dict[str, Any]] = Field(default_factory=list)
    prompt: str = ""
    spans: list[tuple[int, int]] | None = None


class RunRes(BaseModel):
    findings: list[Finding]
    ms: float


class PairReq(BaseModel):
    pairs: list[tuple[str, str]]      # (premise, hypothesis)


class PairRes(BaseModel):
    scores: list[tuple[float, float]]  # (p_entail, p_contra)
    ms: float


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    t = time.perf_counter()
    MOD.load()
    log.info('{"event": "loaded", "role": "%s", "ms": %.0f}',
             ROLE, (time.perf_counter() - t) * 1000)
    yield


app = FastAPI(title=f"ControlPlane.ai detector [{ROLE}]", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"role": ROLE, **MOD.info()}


@app.post("/pairs", response_model=PairRes)
async def pairs(r: PairReq) -> PairRes:
    """Raw entailment scoring. Tier 2 drives its own claim-level comparisons and
    needs the scores, not nli.check's sentence-level verdicts. Keeping this here
    means one copy of the model rather than a second one in the coordinator."""
    if ROLE != "nli":
        return PairRes(scores=[], ms=0.0)
    t = time.perf_counter()
    out = await nli._bat.submit_many([(a, b) for a, b in r.pairs])
    return PairRes(scores=out, ms=round((time.perf_counter() - t) * 1000, 2))


@app.post("/run", response_model=RunRes)
async def run(r: RunReq) -> RunRes:
    t = time.perf_counter()
    if ROLE == "nli":
        f = await nli.check(r.resp, r.ctx)
    elif ROLE == "safety":
        f = await safety.check(r.resp, r.prompt)
    elif ROLE == "bias":
        f = await bias.check(r.resp, r.spans)
    else:
        f = []
    return RunRes(findings=f, ms=round((time.perf_counter() - t) * 1000, 2))
