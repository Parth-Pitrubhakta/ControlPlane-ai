"""Dashboard data. Read-only aggregates over the trace store.

The header numbers are the point of the demo, so they are computed here rather
than in the browser: the UI should not be able to flatter the latency figures by
accident.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Query

from api import store

router = APIRouter(prefix="/api", tags=["metrics"])

WINDOWS = {"5m": 300, "1h": 3600, "24h": 86400, "all": 0}


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = max(0, min(len(s) - 1, int(q * len(s)) - 1))
    return round(s[i], 2)


def _overhead(lat: dict[str, Any]) -> float:
    """What the checker cost. Generation time is not ours to claim or hide."""
    return float(lat.get("t0", 0.0)) + float(lat.get("t1", 0.0)) + float(lat.get("decide", 0.0))


@router.get("/metrics/summary")
async def summary(
    window: str = Query("1h"),
    tenant: str | None = None,
) -> dict[str, Any]:
    q: dict[str, Any] = {}
    secs = WINDOWS.get(window, 3600)
    if secs:
        q["ts"] = {"$gte": time.time() - secs}
    if tenant:
        q["tenant"] = tenant

    ov: list[float] = []
    gen: list[float] = []
    tiers: dict[str, int] = {"0": 0, "1": 0, "2": 0, "3": 0}
    acts: dict[str, int] = {}
    dims: dict[str, int] = {"perf": 0, "resp": 0, "cost": 0}
    labels: dict[str, int] = {}
    tenants: dict[str, int] = {}
    cost = 0.0
    t1_n = 0

    cur = store.db()["traces"].find(
        q, {"_id": 0, "lat": 1, "tier": 1, "act": 1, "fnd": 1, "tenant": 1,
            "cost": 1, "risk": 1}
    ).sort("ts", -1).limit(2000)

    n = 0
    async for t in cur:
        n += 1
        lat = t.get("lat") or {}
        ov.append(_overhead(lat))
        if lat.get("gen"):
            gen.append(float(lat["gen"]))
        if lat.get("t1"):
            t1_n += 1
        tiers[str(t.get("tier", 0))] = tiers.get(str(t.get("tier", 0)), 0) + 1
        a = t.get("act", "allow")
        acts[a] = acts.get(a, 0) + 1
        tenants[t.get("tenant", "?")] = tenants.get(t.get("tenant", "?"), 0) + 1
        cost += float(t.get("cost", 0.0) or 0.0)
        for f in t.get("fnd") or []:
            dims[f.get("dim", "resp")] = dims.get(f.get("dim", "resp"), 0) + 1
            labels[f.get("label", "?")] = labels.get(f.get("label", "?"), 0) + 1

    flagged = sum(v for k, v in acts.items() if k != "allow")
    return {
        "n": n, "window": window, "tenant": tenant,
        "overhead": {"p50": _pct(ov, 0.50), "p95": _pct(ov, 0.95),
                     "max": round(max(ov), 2) if ov else 0.0},
        "gen": {"p50": _pct(gen, 0.50), "p95": _pct(gen, 0.95)},
        "tiers": tiers,
        "t1_rate": round(t1_n / n, 4) if n else 0.0,
        "acts": acts,
        "dims": dims,
        "labels": labels,
        "tenants": tenants,
        "alerts_per_100": round(100.0 * flagged / n, 2) if n else 0.0,
        "cost_usd": round(cost, 5),
    }


@router.get("/traces")
async def traces(
    limit: int = Query(50, le=200),
    tenant: str | None = None,
    act: str | None = None,
    sess: str | None = None,
    tier: int | None = None,
) -> dict[str, Any]:
    q: dict[str, Any] = {}
    for k, v in (("tenant", tenant), ("act", act), ("sess", sess), ("tier", tier)):
        if v is not None:
            q[k] = v
    rows = []
    cur = store.db()["traces"].find(
        q, {"_id": 0, "id": 1, "ts": 1, "tenant": 1, "geo": 1, "sess": 1,
            "prompt": 1, "resp": 1, "act": 1, "tier": 1, "risk": 1, "fnd": 1,
            "lat": 1, "pol_ver": 1, "tok_out": 1, "cost": 1, "ovr": 1}
    ).sort("ts", -1).limit(limit)
    async for t in cur:
        fnd = t.get("fnd") or []
        rows.append({
            "id": t["id"], "ts": t.get("ts"), "tenant": t.get("tenant"),
            "geo": t.get("geo"), "sess": t.get("sess"),
            "prompt": (t.get("prompt") or "")[:180],
            "resp": (t.get("resp") or "")[:180],
            "act": t.get("act"), "tier": t.get("tier"), "risk": t.get("risk"),
            "pol_ver": t.get("pol_ver"), "cost": t.get("cost"),
            "overhead": round(_overhead(t.get("lat") or {}), 2),
            "reviewed": bool(t.get("ovr")),
            "chips": sorted({(f.get("dim"), f.get("label")) for f in fnd}),
            "nfnd": len(fnd),
        })
    return {"rows": rows, "n": len(rows)}


@router.get("/traces/{tid}")
async def trace(tid: str) -> dict[str, Any]:
    t = await store.get_trace(tid)
    return t or {"error": "not found", "id": tid}


@router.get("/metrics/series")
async def series(window: str = Query("1h"), buckets: int = Query(24)) -> dict[str, Any]:
    """Overhead over time. Used for the header sparkline."""
    secs = WINDOWS.get(window, 3600) or 3600
    now = time.time()
    step = secs / buckets
    out = [{"t": now - secs + i * step, "ov": [], "n": 0, "flagged": 0}
           for i in range(buckets)]
    cur = store.db()["traces"].find(
        {"ts": {"$gte": now - secs}},
        {"_id": 0, "ts": 1, "lat": 1, "act": 1}).sort("ts", 1)
    async for t in cur:
        i = min(buckets - 1, max(0, int((t["ts"] - (now - secs)) / step)))
        out[i]["ov"].append(_overhead(t.get("lat") or {}))
        out[i]["n"] += 1
        if t.get("act", "allow") != "allow":
            out[i]["flagged"] += 1
    return {"buckets": [
        {"t": b["t"], "n": b["n"], "flagged": b["flagged"],
         "p50": _pct(b["ov"], 0.5), "p95": _pct(b["ov"], 0.95)}
        for b in out]}
