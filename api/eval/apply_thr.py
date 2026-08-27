"""Publish trained router thresholds as new policy versions.

Thresholds live in the policy document, never in code, so a recalibration ships
the same way any other policy change does: as a new version with its own
effective_from, leaving the old one intact for replay.

    python -m api.eval.apply_thr --ver v2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib

from api import policy, store


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="bench/router_thresholds.json")
    ap.add_argument("--ver", default="v2")
    ap.add_argument("--note", default="router calibration")
    a = ap.parse_args()

    thr = json.loads(pathlib.Path(a.inp).read_text())
    await store.open_store()
    await policy.seed()
    await policy.refresh()

    out = []
    for (tenant, geo), cur in sorted(policy._cache.items()):
        t = thr.get(tenant)
        if not t:
            continue
        if cur.ver == a.ver and cur.thr.get("med") == t["med"]:
            out.append({"tenant": tenant, "geo": geo, "skipped": "already current"})
            continue
        new = cur.model_copy(deep=True)
        new.ver = a.ver
        new.effective_from = 0.0
        new.thr = {"med": t["med"], "high": t["high"],
                   "recall_floor": t["recall_floor"], "note": a.note}
        pv = await policy.put(new)
        out.append({"pol_ver": pv, "thr": new.thr})

    await policy.refresh()
    print(json.dumps({"published": out,
                      "active": {f"{k[0]}/{k[1]}": f"{v.ver} med={v.thr.get('med')}"
                                 for k, v in sorted(policy._cache.items())}}, indent=2))
    await store.close_store()


if __name__ == "__main__":
    asyncio.run(main())
