"""Per-session risk that accumulates across turns.

A single mildly suspicious turn is noise. Five in a row from the same session is
someone probing. The ledger keeps a decaying score in Redis so later turns in a
session that has already misbehaved start at a higher verification tier.

The score never decides an action -- that is decide.py's job and it reads
findings, not numbers (invariant 3). All this does is raise the tier floor,
which buys more evidence, not a harsher verdict.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import structlog

from api import store
from api.schemas import Finding

log = structlog.get_logger("ledger")

HALFLIFE_S = float(os.getenv("LEDGER_HALFLIFE_S", "900"))
TTL_S = int(os.getenv("LEDGER_TTL_S", "7200"))
# score at or above which the session floor rises to that tier
T1_AT = float(os.getenv("LEDGER_T1_AT", "3.0"))
T2_AT = float(os.getenv("LEDGER_T2_AT", "8.0"))

_K = "cp:led:"


def weight(fnd: list[Finding]) -> float:
    """Severity-weighted contribution of one turn. Cost never counts."""
    return float(sum(f.sev for f in fnd if f.dim != "cost"))


def _decay(v: float, age_s: float) -> float:
    return v * math.pow(0.5, age_s / HALFLIFE_S) if age_s > 0 else v


async def bump(sess: str, fnd: list[Finding]) -> float:
    """Add this turn's weight to the session and return the decayed total."""
    w = weight(fnd)
    now = time.time()
    k = _K + sess
    try:
        r = store.rd()
        cur = await r.hgetall(k)
        prev = float(cur.get("v", 0.0)) if cur else 0.0
        ts = float(cur.get("t", now)) if cur else now
        v = _decay(prev, now - ts) + w
        await r.hset(k, mapping={"v": v, "t": now})
        await r.expire(k, TTL_S)
        return v
    except Exception as e:
        log.warning("ledger_bump_failed", sess=sess, err=str(e))
        return w


async def score(sess: str) -> float:
    try:
        cur = await store.rd().hgetall(_K + sess)
        if not cur:
            return 0.0
        return _decay(float(cur.get("v", 0.0)), time.time() - float(cur.get("t", 0)))
    except Exception:
        return 0.0


def floor(v: float) -> int:
    """Minimum verification tier this session has earned."""
    if v >= T2_AT:
        return 2
    if v >= T1_AT:
        return 1
    return 0


async def clear(sess: str) -> None:
    try:
        await store.rd().delete(_K + sess)
    except Exception:
        pass


async def info(sess: str) -> dict[str, Any]:
    v = await score(sess)
    return {"sess": sess, "score": round(v, 3), "floor": floor(v),
            "halflife_s": HALFLIFE_S}
