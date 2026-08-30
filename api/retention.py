"""Enforce the retention window each policy promises.

Every policy document carries retention_days, and the EU tenants set it to 30.
Until now nothing read that field, so the promise was written down and never
kept -- which is worse than not promising, because the policy editor displays it
as though it were in force.

A Mongo TTL index cannot do this: TTL is one expiry for a whole collection, and
retention differs per tenant and per geography. So this sweeps instead, deleting
by (tenant, geo) against each policy's own window.

Traces hold the prompt, the response and every finding, so they are the records
that actually carry personal data. Reviewer feedback references a trace and
carries the finding text with it, so it expires on the same clock.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import structlog

from api import policy, store

log = structlog.get_logger("retention")

SWEEP_S = float(os.getenv("RETENTION_SWEEP_S", "3600"))
BATCH = int(os.getenv("RETENTION_BATCH", "5000"))
ON = os.getenv("RETENTION", "1") == "1"

_task: asyncio.Task[None] | None = None
_last: dict[str, Any] = {"ran": None, "deleted": 0, "by": {}}


async def sweep(dry: bool = False) -> dict[str, Any]:
    """Delete traces past their tenant's retention window.

    Returns what it removed, per (tenant, geo), so the sweep is auditable rather
    than a silent background deletion.
    """
    now = time.time()
    out: dict[str, Any] = {}
    total = 0
    tr = store.db()["traces"]
    fb = store.db()["feedback"]

    for (tenant, geo), pol in sorted(policy._cache.items()):
        days = int(pol.retention_days or 0)
        if days <= 0:
            continue
        cutoff = now - days * 86400
        q = {"tenant": tenant, "geo": geo, "ts": {"$lt": cutoff}}
        n = await tr.count_documents(q, limit=BATCH)
        if not n:
            continue
        key = f"{tenant}/{geo}"
        out[key] = {"days": days, "cutoff": round(cutoff, 0), "traces": n}
        if not dry:
            ids = [d["id"] async for d in tr.find(q, {"_id": 0, "id": 1}).limit(BATCH)]
            r = await tr.delete_many({"id": {"$in": ids}})
            f = await fb.delete_many({"trace": {"$in": ids}})
            out[key]["deleted"] = r.deleted_count
            out[key]["feedback_deleted"] = f.deleted_count
            total += r.deleted_count
    res = {"dry": dry, "ts": now, "total_deleted": total, "by": out}
    if not dry:
        _last.update({"ran": now, "deleted": total, "by": out})
        if total:
            log.info("retention_sweep", deleted=total, by=list(out))
    return res


def last() -> dict[str, Any]:
    return dict(_last)


async def _loop() -> None:
    while True:
        try:
            await asyncio.sleep(SWEEP_S)
            await sweep()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("retention_sweep_failed", err=str(e))


async def open_retention() -> None:
    global _task
    if not ON:
        log.warning("retention_disabled", note="RETENTION=0; policy windows not enforced")
        return
    _task = asyncio.create_task(_loop())


async def close_retention() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        _task = None
