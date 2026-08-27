"""Correct the ambiguous labels in the router training set.

The spec calls for hand-correcting 150 rows. No human is available inside this
build, so a stronger model adjudicates instead, and every row it touches is
stamped label_src="judge" so the substitution is visible in the data rather than
hidden in a number. Rows labelled by construction are never overwritten.

    python -m api.eval.adjudicate --in bench/router_train.jsonl --max 150
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
from typing import Any

import httpx

VLLM = "http://127.0.0.1:8000/v1"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

Q = """You are checking one answer against one policy statement.

POLICY: {gov}

ANSWER: {resp}

Does the ANSWER state something that directly contradicts the POLICY?
Reply with exactly one word: YES or NO."""


async def judge(c: httpx.AsyncClient, gov: str, resp: str) -> bool | None:
    try:
        r = await c.post("/chat/completions", json={
            "model": MODEL, "temperature": 0.0, "max_tokens": 4,
            "messages": [{"role": "user",
                          "content": Q.format(gov=gov, resp=resp)}]})
        r.raise_for_status()
        out = (r.json()["choices"][0]["message"]["content"] or "").strip().upper()
        if re.match(r"^\s*YES", out):
            return True
        if re.match(r"^\s*NO", out):
            return False
    except Exception:
        pass
    return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", dest="inp", default="bench/router_train.jsonl")
    ap.add_argument("--max", type=int, default=150)
    a = ap.parse_args()

    p = pathlib.Path(a.inp)
    rows: list[dict[str, Any]] = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    from api.eval.gen_router import FACTS
    gov = {f["topic"]: f["gov"] for f in FACTS}

    todo = [r for r in rows if r.get("ambiguous")][: a.max]
    sem = asyncio.Semaphore(12)
    flipped = 0

    async with httpx.AsyncClient(base_url=VLLM, timeout=120) as c:
        async def go(r: dict[str, Any]) -> None:
            nonlocal flipped
            async with sem:
                v = await judge(c, gov.get(r["topic"], ""), r["resp"])
            if v is None:
                return
            r["label_src"] = "judge"
            r["ambiguous"] = False
            if v and "contradicted" not in r["defects"]:
                r["defects"] = sorted(set(r["defects"] + ["contradicted"]))
                flipped += 1
            r["y"] = int(bool(r["defects"]))
        await asyncio.gather(*(go(r) for r in todo))

    with p.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(json.dumps({
        "rows": len(rows),
        "adjudicated": len(todo),
        "flipped_to_contradicted": flipped,
        "still_ambiguous": sum(1 for r in rows if r.get("ambiguous")),
        "positives": sum(r["y"] for r in rows),
        "rate": round(sum(r["y"] for r in rows) / max(1, len(rows)), 3),
        "by_label_src": {s: sum(1 for r in rows if r["label_src"] == s)
                         for s in {r["label_src"] for r in rows}},
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
