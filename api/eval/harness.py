"""Replay ControlPlane-Bench through the real pipeline and score it.

The harness calls gateway.resolve(), the same function the live endpoint calls.
It does not reimplement the ladder: a benchmark that scores a parallel copy of
the logic measures the copy, not the product.

    python -m api.eval.harness --bench bench/ --out bench/report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import time
import uuid
from typing import Any

from api import gateway, policy, router as rt, store
from api.eval import report as rep
from api.schemas import Trace


async def clear_cache() -> bool:
    """Drop the detector's semantic cache.

    Both passes must start cold. The baseline replays the same rows the routed
    pass just ran, so with a warm cache every /check would hit it and the
    "always deep" baseline would come out cheaper than the routed run -- which
    is exactly the nonsense (341% of always-deep) that surfaced without this.
    """
    import httpx

    from api import detclient
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(f"{detclient.DET_URL}/cache/clear")
            return r.status_code == 200
    except Exception:
        return False


def _load(p: pathlib.Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


async def replay(row: dict[str, Any], sem: asyncio.Semaphore) -> dict[str, Any]:
    tr = Trace(
        id=f"bench-{uuid.uuid4().hex[:10]}",
        # a fresh session per row: the ledger deliberately raises the tier for a
        # session that has misbehaved, which would leak across bench rows
        sess=f"bench-{uuid.uuid4().hex[:8]}",
        tenant=row["tenant"], geo=row["geo"], ts=time.time(),
        prompt=row["prompt"], resp="", ctx=row.get("ctx") or [],
        tools=row.get("tools") or [],
        tok_in=max(1, len(row["prompt"]) // 4),
        tok_out=max(1, len(row["resp"]) // 4),
    )
    from api import tier0
    t0p = tier0.scan(tier0.norm(row["prompt"]), side="prompt")

    async with sem:
        t = time.perf_counter()
        body, audit = await gateway.resolve(tr, t0p, row["resp"], "bench-model")
        wall = (time.perf_counter() - t) * 1000

    lat = tr.lat or {}
    return {
        "id": row["id"], "set": row["set"], "src": row["src"],
        "task": row.get("task"),
        "tenant": row["tenant"], "geo": row["geo"],
        "gold": row["gold"],
        "pred": [{"span": list(f.span), "side": f.side, "dim": f.dim,
                  "label": f.label, "sev": f.sev, "conf": f.conf, "det": f.det}
                 for f in tr.fnd],
        "act": tr.act, "tier": tr.tier, "risk": tr.risk, "pol_ver": tr.pol_ver,
        # the checker's own cost, never the model's generation time
        "overhead_ms": round(float(lat.get("t0", 0)) + float(lat.get("t1", 0))
                             + float(lat.get("decide", 0)), 3),
        "wall_ms": round(wall, 2),
        "body_changed": body != row["resp"],
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="bench")
    ap.add_argument("--out", default="bench/report.json")
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--always-deep-ms", dest="deep", type=float, default=500.0,
                    help="assumed T2 cost; only used for the projected figure")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the measured always-deep pass")
    a = ap.parse_args()

    b = pathlib.Path(a.bench)
    sets = {"benign": _load(b / "benign.jsonl"),
            "adversarial": _load(b / "adversarial.jsonl")}
    if a.limit:
        sets = {k: v[: a.limit] for k, v in sets.items()}
    if not any(sets.values()):
        raise SystemExit(f"no bench files under {b}. run api.eval.build_bench first")

    await store.open_store()
    await gateway.open_gw()
    from api import detclient
    await detclient.open_det()
    await policy.open_policy()
    rt.load()

    sem = asyncio.Semaphore(a.conc)
    out: dict[str, Any] = {
        "ts": time.time(),
        "concurrency": a.conc,
        "router": rt.meta(),
        "policy": {f"{t}/{g}": p.ver for (t, g), p in sorted(policy._cache.items())},
        "match": {"criterion": f"same label, same side, IoU >= {rep.IOU_MIN}"},
        "sets": {},
    }

    out["cache_cleared"] = await clear_cache()

    replayed: dict[str, list[dict[str, Any]]] = {}
    for name, rows in sets.items():
        if not rows:
            continue
        await clear_cache()
        t = time.perf_counter()
        res = list(await asyncio.gather(*(replay(r, sem) for r in rows)))
        replayed[name] = res
        out["sets"][name] = rep.score_set(res, always_deep_ms=a.deep)
        out["sets"][name]["replay_secs"] = round(time.perf_counter() - t, 1)

    # Measured always-deep baseline: the same rows, same hardware, same
    # concurrency, but every one forced to tier 1. Comparing our routed cost
    # against an assumed 500 ms constant would not be a claim worth making.
    if not a.no_baseline:
        real_tier = rt.tier
        rt.tier = lambda risk_v, pol, floor=0: max(1, floor)   # force deep
        try:
            base: dict[str, Any] = {}
            for name, rows in sets.items():
                if not rows:
                    continue
                await clear_cache()
                res = list(await asyncio.gather(*(replay(r, sem) for r in rows)))
                spent = sum(x["overhead_ms"] for x in res)
                base[name] = {
                    "spent_ms": round(spent, 1),
                    "mean_ms": round(spent / max(1, len(res)), 2),
                    "p50_ms": rep.pct([x["overhead_ms"] for x in res], 0.5),
                }
                routed = out["sets"][name]["verification_compute"]["spent_ms"]
                base[name]["routed_spent_ms"] = routed
                base[name]["pct_of_measured_always_deep"] = (
                    round(100 * routed / spent, 2) if spent else None)
            out["always_deep_measured"] = {
                "note": "every row forced to tier 1 on the same hardware and "
                        "concurrency. T2 is not built, so this is the honest "
                        "deep baseline, not the spec's 500 ms assumption.",
                **base,
            }
        finally:
            rt.tier = real_tier

    # the false-positive reduction curve, per detector label that has volume
    curves = {}
    ben = replayed.get("benign", [])
    if ben:
        thr = [i / 20 for i in range(21)]
        for label in ("unverifiable", "contradicted", "bias", "pii"):
            has = sum(1 for r in ben for p in r["pred"] if p["label"] == label)
            if has >= 10:
                curves[label] = rep.fp_curve(ben, label, thr)
    out["fp_curve"] = curves

    pathlib.Path(a.out).write_text(json.dumps(out, indent=2) + "\n")

    # a short console summary; the file carries everything
    print(json.dumps({
        "sets": {k: {
            "n": v["n"],
            "by_dim": {d: {kk: m[kk] for kk in ("precision", "recall")}
                       for d, m in v["by_dim"].items()},
            "catch_rate_before_delivery": v["catch_rate_before_delivery"],
            "overhead_ms": v["overhead_ms"],
            "pct_of_always_deep_assumed": v["verification_compute"]["pct_of_always_deep"],
            "pct_of_always_deep_measured":
                (out.get("always_deep_measured", {}).get(k) or {})
                .get("pct_of_measured_always_deep"),
            "alerts_per_1000": v["alerts_per_1000"],
            "t1_rate": v["t1_rate"],
            "router_ece": v["router_ece"],
            "router_ece_noise_floor": v["router_ece_noise_floor"],
        } for k, v in out["sets"].items()},
        "out": a.out,
    }, indent=2))

    await policy.close_policy()
    await detclient.close_det()
    await gateway.close_gw()
    await store.close_store()


if __name__ == "__main__":
    asyncio.run(main())
