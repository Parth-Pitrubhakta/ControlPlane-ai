"""Policy load, versioning and hot reload.

One document per (tenant, geo) in the `policies` collection. Documents are
versioned and never edited in place: a change writes a new version with a new
`effective_from`, so any decision can be replayed against the exact policy that
produced it (invariant 8). Every trace carries the resolved `pol_ver`.

A background poll refreshes the in-memory cache, so a policy edit takes effect
across the fleet without a restart.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import structlog

from api import store
from api.schemas import Policy

log = structlog.get_logger("policy")

POLL_S = float(os.getenv("POLICY_POLL_S", "30"))
TENANTS = ("CS-BOT", "KB-COPILOT", "DECIDE")
GEOS = ("IN", "EU", "US")

_cache: dict[tuple[str, str], Policy] = {}
_task: asyncio.Task[None] | None = None

# Label -> action floor. This is the rulebook. There is deliberately no weighted
# sum anywhere in this file: an action comes from what was found, not from a
# score (invariant 3).
#
# Two entries are load-bearing and must not be "tightened" later:
#   unverifiable never exceeds annotate  (invariant 2)
#   cost_anom    never exceeds annotate  (invariant 4)
_BASE_FLOORS: dict[str, str] = {
    "contradicted": "edit",
    "unverifiable": "annotate",
    "pii": "edit",
    "bias": "annotate",
    "unsafe": "block",
    "inject": "block",
    "cost_anom": "annotate",
    "format": "annotate",
}


def _p(
    tenant: str,
    geo: str,
    ver: str,
    lat_budget_ms: int,
    thr: dict[str, float],
    floors: dict[str, str],
    sample: dict[str, float],
    retention_days: int,
    escalate_if: dict[str, Any] | None = None,
) -> Policy:
    return Policy(
        tenant=tenant,
        geo=geo,
        ver=ver,
        effective_from=0.0,
        lat_budget_ms=lat_budget_ms,
        thr=thr,
        floors={**_BASE_FLOORS, **floors},
        escalate_if=escalate_if or {"sev": 3, "irrev_tool": False},
        sample=sample,
        retention_days=retention_days,
    )


def defaults() -> list[Policy]:
    """Seed policies. Per-tenant posture comes from section 11 of the spec.

    Thresholds are placeholders until the router is calibrated; recalibration
    rewrites them as a new version rather than editing these.
    """
    out: list[Policy] = []
    for geo in GEOS:
        eu = geo == "EU"

        # external customer support: balanced, alerts <= 5/100
        out.append(_p(
            "CS-BOT", geo, "v1", 150,
            {"med": 0.30, "high": 0.70},
            # EU blocks personal data outright; elsewhere we redact and continue
            {"pii": "block"} if eu else {"pii": "edit"},
            {"t2": 0.05, "t3": 0.01},
            30 if eu else 90,
        ))

        # internal knowledge assistant: permissive, alerts <= 3/100
        out.append(_p(
            "KB-COPILOT", geo, "v1", 200,
            {"med": 0.45, "high": 0.80},
            {"pii": "block" if eu else "annotate", "contradicted": "annotate",
             "unverifiable": "allow", "cost_anom": "allow", "format": "allow"},
            {"t2": 0.03, "t3": 0.005},
            30 if eu else 60,
        ))

        # regulated decision support: recall >= 0.98, human on irreversible
        out.append(_p(
            "DECIDE", geo, "v1", 500,
            {"med": 0.10, "high": 0.35},
            {"pii": "block", "contradicted": "escalate", "inject": "escalate",
             "bias": "escalate"},
            {"t2": 0.25, "t3": 0.05},
            365,
            {"sev": 3, "irrev_tool": True},
        ))
    return out


async def seed() -> int:
    """Insert default policies for any (tenant, geo) that has none. Idempotent."""
    col = store.db()["policies"]
    await col.create_index([("tenant", 1), ("geo", 1), ("effective_from", -1)])
    n = 0
    for p in defaults():
        if await col.count_documents({"tenant": p.tenant, "geo": p.geo}, limit=1):
            continue
        await col.insert_one(p.model_dump())
        n += 1
    return n


async def refresh() -> int:
    """Pull the active policy for every (tenant, geo) into the cache."""
    col = store.db()["policies"]
    now = time.time()
    n = 0
    for t in TENANTS:
        for g in GEOS:
            # _id breaks the tie: two versions can legitimately share an
            # effective_from, and without a deterministic second key Mongo may
            # hand back the older one, so a published policy silently does not
            # take effect.
            d = await col.find_one(
                {"tenant": t, "geo": g, "effective_from": {"$lte": now}},
                sort=[("effective_from", -1), ("_id", -1)],
                projection={"_id": 0},
            )
            if d:
                _cache[(t, g)] = Policy(**d)
                n += 1
    return n


def get(tenant: str, geo: str) -> Policy:
    """Resolve a policy. Falls back tenant-wide, then to a safe default.

    Never raises: a missing policy must not take the gateway down, and the
    fallback is the strictest thing we can serve without knowing the tenant.
    """
    p = _cache.get((tenant, geo)) or _cache.get((tenant, "IN"))
    if p is not None:
        return p
    return _p("CS-BOT", geo, "fallback", 150, {"med": 0.3, "high": 0.7},
              {"pii": "block"}, {"t2": 0.05, "t3": 0.01}, 30)


def ver(tenant: str, geo: str) -> str:
    """The stamp that goes on the trace. Identifies the exact policy used."""
    p = get(tenant, geo)
    return f"{p.tenant}:{p.geo}:{p.ver}"


async def put(p: Policy) -> str:
    """Write a new policy version. Never mutates an existing one.

    A document posted back from the editor still carries the timestamp of the
    version it was copied from. Re-using it would date the new version to the
    old one, so anything not explicitly scheduled for the future is stamped now.
    """
    now = time.time()
    if not p.effective_from or p.effective_from <= now:
        p.effective_from = now
    await store.db()["policies"].insert_one(p.model_dump())
    await refresh()
    log.info("policy_version", tenant=p.tenant, geo=p.geo, ver=p.ver)
    return f"{p.tenant}:{p.geo}:{p.ver}"


async def _poll() -> None:
    while True:
        try:
            await asyncio.sleep(POLL_S)
            await refresh()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("policy_poll_failed", err=str(e))


async def open_policy() -> None:
    global _task
    try:
        await seed()
        await refresh()
    except Exception as e:
        log.warning("policy_seed_failed", err=str(e))
    _task = asyncio.create_task(_poll())


async def close_policy() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        _task = None
