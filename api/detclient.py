"""HTTP client to the H200 detector service.

Invariant 7: nothing in this module imports torch or transformers. All inference
is an HTTP call. MOCK_H200=1 replaces the call with canned findings so the
gateway keeps working with the H200 powered off.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any

import httpx

from api.schemas import Finding

DET_URL = os.getenv("DET_URL", "http://localhost:8100")
MOCK = os.getenv("MOCK_H200", "0") == "1"
MOCK_LAT_MS = int(os.getenv("MOCK_LAT_MS", "50"))
DET_TIMEOUT_S = float(os.getenv("DET_TIMEOUT_S", "5.0"))

_cli: httpx.AsyncClient | None = None

_SENT = re.compile(r"[^.!?]+[.!?]|[^.!?]+$")


async def open_det() -> None:
    global _cli
    if not MOCK:
        _cli = httpx.AsyncClient(base_url=DET_URL, timeout=DET_TIMEOUT_S)


async def close_det() -> None:
    global _cli
    if _cli is not None:
        await _cli.aclose()
        _cli = None


def _sents(txt: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _SENT.finditer(txt) if m.group().strip()]


def _canned(resp: str, ctx: list[dict]) -> list[Finding]:
    """Mock findings. Deterministic, offline, never claims support it did not check.

    With no context there is no evidence either way, so every sentence is
    UNVERIFIABLE -- which per invariant 2 can never escalate to a block.
    """
    if ctx:
        return []
    return [
        Finding(
            span=sp,
            dim="perf",
            label="unverifiable",
            sev=1,
            conf=0.5,
            evid=None,
            det="mock-h200-v1",
        )
        for sp in _sents(resp)[:3]
    ]


async def check(
    resp: str,
    ctx: list[dict],
    need: list[str],
) -> tuple[list[Finding], dict[str, Any]]:
    """Run tier 1 detectors on the H200. Returns (findings, latency dict).

    Never raises: a detector outage degrades to zero findings plus an err marker,
    it does not take the gateway down with it.
    """
    t = time.perf_counter()
    if MOCK:
        await asyncio.sleep(MOCK_LAT_MS / 1000.0)
        return _canned(resp, ctx), {"t1": (time.perf_counter() - t) * 1000, "mock": True}

    if _cli is None:
        return [], {"t1": 0.0, "err": "det client not opened"}
    try:
        r = await _cli.post("/check", json={"resp": resp, "ctx": ctx, "need": need})
        r.raise_for_status()
        body = r.json()
        fnd = [Finding(**f) for f in body.get("findings", [])]
        lat = {"t1": (time.perf_counter() - t) * 1000, "det": body.get("lat", {})}
        return fnd, lat
    except Exception as e:
        return [], {"t1": (time.perf_counter() - t) * 1000, "err": f"{type(e).__name__}"}


async def health() -> dict[str, Any]:
    if MOCK:
        return {"det": "mock"}
    if _cli is None:
        return {"det": "down: client not opened"}
    try:
        r = await _cli.get("/health", timeout=2.0)
        return {"det": "up", **r.json()}
    except Exception as e:
        return {"det": f"down: {type(e).__name__}"}
