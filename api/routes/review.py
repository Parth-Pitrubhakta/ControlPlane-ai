"""Human review queue and per-finding feedback.

Reviewers agree or disagree with each finding separately, not with the response
as a whole. That granularity is the whole point: "this response was fine" tells
you nothing about which detector was wrong, and recalibration needs to know
which one to move.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from api import store

router = APIRouter(prefix="/api/review", tags=["review"])


class Item(BaseModel):
    idx: int                      # index into Trace.fnd
    agree: bool                   # was the detector right
    note: str = ""


class Verdict(BaseModel):
    reviewer: str = "reviewer"
    items: list[Item] = Field(default_factory=list)
    act_agree: bool | None = None     # was the resulting action right
    act_should: str | None = None     # what it should have been


@router.get("/queue")
async def queue(limit: int = Query(50, le=200), tenant: str | None = None,
                include_done: bool = False) -> dict[str, Any]:
    q: dict[str, Any] = {"act": {"$in": ["escalate", "block"]}}
    if not include_done:
        q["ovr"] = None
    if tenant:
        q["tenant"] = tenant
    rows = []
    cur = store.db()["traces"].find(q, {"_id": 0}).sort("ts", -1).limit(limit)
    async for t in cur:
        # tier 3 already looked at this one; show the reviewer what it thought
        j = await store.db()["probes"].find_one(
            {"kind": "judge", "trace": t["id"]}, {"_id": 0},
            sort=[("ts", -1)])
        rows.append({
            "judge": {k: j[k] for k in ("verdict", "should_be", "why")} if j else None,
            "id": t["id"], "ts": t.get("ts"), "tenant": t.get("tenant"),
            "geo": t.get("geo"), "act": t.get("act"), "tier": t.get("tier"),
            "risk": t.get("risk"), "pol_ver": t.get("pol_ver"),
            "prompt": t.get("prompt", ""), "resp": t.get("resp", ""),
            "fnd": t.get("fnd", []), "ovr": t.get("ovr"),
        })
    return {"rows": rows, "n": len(rows)}


@router.post("/{tid}")
async def submit(tid: str, v: Verdict) -> dict[str, Any]:
    t = await store.get_trace(tid)
    if not t:
        return {"error": "not found", "id": tid}

    fnd = t.get("fnd") or []
    items: list[dict[str, Any]] = []
    for it in v.items:
        if not (0 <= it.idx < len(fnd)):
            continue
        f = fnd[it.idx]
        items.append({
            "idx": it.idx, "agree": it.agree, "note": it.note,
            "label": f.get("label"), "dim": f.get("dim"), "det": f.get("det"),
            "conf": f.get("conf"), "sev": f.get("sev"), "evid": f.get("evid"),
        })

    doc = {
        "id": f"fb-{uuid.uuid4().hex[:10]}",
        "trace": tid, "ts": time.time(), "reviewer": v.reviewer,
        "tenant": t.get("tenant"), "geo": t.get("geo"),
        "pol_ver": t.get("pol_ver"), "act": t.get("act"),
        "act_agree": v.act_agree, "act_should": v.act_should,
        "items": items,
    }
    await store.db()["feedback"].insert_one(dict(doc))
    doc.pop("_id", None)

    ovr = {"reviewer": v.reviewer, "ts": doc["ts"], "feedback": doc["id"],
           "act_agree": v.act_agree, "act_should": v.act_should,
           "disagreed": [i["idx"] for i in items if not i["agree"]]}
    await store.db()["traces"].update_one({"id": tid}, {"$set": {"ovr": ovr}})
    return {"ok": True, "feedback": doc["id"], "ovr": ovr}


@router.get("/feedback")
async def feedback(limit: int = Query(200, le=1000)) -> dict[str, Any]:
    rows = []
    cur = store.db()["feedback"].find({}, {"_id": 0}).sort("ts", -1).limit(limit)
    async for d in cur:
        rows.append(d)
    return {"rows": rows, "n": len(rows)}


@router.get("/stats")
async def stats() -> dict[str, Any]:
    """Per-label agreement. This is what recalibration reads."""
    agg: dict[str, dict[str, Any]] = {}
    cur = store.db()["feedback"].find({}, {"_id": 0, "items": 1})
    n = 0
    async for d in cur:
        n += 1
        for it in d.get("items") or []:
            a = agg.setdefault(it["label"], {"n": 0, "agree": 0, "fp_conf": [],
                                             "tp_conf": []})
            a["n"] += 1
            if it["agree"]:
                a["agree"] += 1
                a["tp_conf"].append(it.get("conf"))
            else:
                a["fp_conf"].append(it.get("conf"))
    out = {}
    for k, a in agg.items():
        out[k] = {
            "n": a["n"], "agree": a["agree"],
            "fp": a["n"] - a["agree"],
            "fp_rate": round(1.0 - a["agree"] / a["n"], 4) if a["n"] else 0.0,
            "fp_conf_max": max([c for c in a["fp_conf"] if c is not None], default=None),
            "tp_conf_min": min([c for c in a["tp_conf"] if c is not None], default=None),
        }
    return {"verdicts": n, "by_label": out}
