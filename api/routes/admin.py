"""Policy CRUD and recalibration.

Recalibration moves a detector's confidence threshold, and that threshold lives
in the policy document under thr["det"], never in the detector service. So a
recalibration is an ordinary policy version: reviewable, replayable, and
reversible by making the old version effective again. Nothing gets restarted.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Body, Query

from api import policy, router as rt, store
from api.schemas import Policy

router = APIRouter(prefix="/api", tags=["admin"])

# a threshold move must not silently drop findings reviewers confirmed as real
KEEP_TP = float(0.95)
MIN_FB = 5           # feedback items on a label before we will move anything
# With no confirmed true positives, any cut trivially "keeps 100% of them", so
# a handful of complaints could silence a detector outright. Demand more
# evidence in that case, and cap how far the threshold may travel.
MIN_FB_NO_TP = 8
NO_TP_CAP = 0.75


@router.get("/policies")
async def policies(all_versions: bool = Query(False)) -> dict[str, Any]:
    if all_versions:
        rows = []
        cur = store.db()["policies"].find({}, {"_id": 0}).sort("effective_from", -1)
        async for d in cur:
            rows.append(d)
        return {"rows": rows, "n": len(rows)}
    return {"rows": [p.model_dump() for p in policy._cache.values()],
            "n": len(policy._cache)}


@router.get("/policies/{tenant}/{geo}")
async def get_policy(tenant: str, geo: str) -> dict[str, Any]:
    p = policy.get(tenant, geo)
    return {**p.model_dump(), "pol_ver": policy.ver(tenant, geo)}


@router.post("/policies")
async def put_policy(doc: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Publish a new policy version. The editor posts the whole document."""
    try:
        p = Policy(**doc)
    except Exception as e:
        return {"error": f"invalid policy: {type(e).__name__}: {e}"}
    if p.ver == policy.get(p.tenant, p.geo).ver:
        return {"error": f"version {p.ver} already active; bump ver to publish"}
    pv = await policy.put(p)
    return {"ok": True, "pol_ver": pv, "active": policy.get(p.tenant, p.geo).model_dump()}


@router.get("/router")
async def router_info() -> dict[str, Any]:
    return rt.meta()


@router.post("/recalibrate")
async def recalibrate(
    tenant: str | None = None,
    geo: str | None = None,
    dry: bool = Query(False),
    ver: str | None = None,
) -> dict[str, Any]:
    """Move detector thresholds away from what reviewers rejected.

    For each label we have feedback on, pick the lowest confidence cut that
    excludes the false positives while still keeping KEEP_TP of the findings
    reviewers confirmed. If no such cut exists, we leave the threshold alone and
    say so: silently trading away true positives to look good on false ones is
    the failure this whole project is about.
    """
    agg: dict[str, dict[str, list[float]]] = {}
    cur = store.db()["feedback"].find({}, {"_id": 0, "items": 1, "tenant": 1})
    async for d in cur:
        if tenant and d.get("tenant") != tenant:
            continue
        for it in d.get("items") or []:
            c = it.get("conf")
            if c is None:
                continue
            a = agg.setdefault(it["label"], {"tp": [], "fp": []})
            a["tp" if it["agree"] else "fp"].append(float(c))

    moves: dict[str, Any] = {}
    for label, a in agg.items():
        tp, fp = sorted(a["tp"]), sorted(a["fp"])
        if len(tp) + len(fp) < MIN_FB:
            moves[label] = {"skipped": "not enough feedback", "n": len(tp) + len(fp)}
            continue
        if not fp:
            moves[label] = {"skipped": "no false positives to remove", "tp": len(tp)}
            continue
        cand = sorted({round(c + 1e-6, 6) for c in fp} | {0.0})
        best = None
        for t in cand:
            kept_tp = sum(1 for c in tp if c >= t)
            kept_fp = sum(1 for c in fp if c >= t)
            rec = kept_tp / len(tp) if tp else 1.0
            if rec < KEEP_TP:
                continue
            if best is None or kept_fp < best["kept_fp"]:
                best = {"thr": t, "kept_fp": kept_fp, "recall_kept": round(rec, 4)}
        if best is None or best["thr"] <= 0.0:
            moves[label] = {"skipped": "no cut keeps enough true positives",
                            "tp": len(tp), "fp": len(fp)}
            continue

        warn = None
        thr_new = best["thr"]
        if not tp:
            if len(fp) < MIN_FB_NO_TP:
                moves[label] = {
                    "skipped": "no confirmed true positives yet; "
                               f"need {MIN_FB_NO_TP} rejections to move on FPs alone",
                    "fp": len(fp)}
                continue
            if thr_new > NO_TP_CAP:
                thr_new = NO_TP_CAP
                warn = f"capped at {NO_TP_CAP}: no confirmed true positives to protect"

        moves[label] = {"new_thr": thr_new, "fp_before": len(fp),
                        "fp_after": sum(1 for c in fp if c >= thr_new),
                        "tp_kept": best["recall_kept"], "n": len(tp) + len(fp),
                        **({"warn": warn} if warn else {})}

    applied = [m for m in moves.values() if "new_thr" in m]
    if dry or not applied:
        return {"dry": dry, "moves": moves, "published": []}

    published = []
    targets = [(t, g) for (t, g) in policy._cache
               if (not tenant or t == tenant) and (not geo or g == geo)]
    stamp = ver or f"recal-{int(time.time())}"
    for t, g in targets:
        cur_p = policy.get(t, g)
        new = cur_p.model_copy(deep=True)
        det = dict(new.thr.get("det") or {})
        for label, m in moves.items():
            if "new_thr" in m:
                det[label] = m["new_thr"]
        new.thr = {**new.thr, "det": det}
        new.ver = stamp
        new.effective_from = 0.0
        published.append(await policy.put(new))

    return {"dry": False, "moves": moves, "published": published,
            "det_thr": policy.get(*targets[0]).thr.get("det") if targets else {}}
