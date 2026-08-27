"""OpenAI-compatible proxy. Every request through here produces a Trace.

Phase 1 scope: forward, trace, run tier 0. Findings are recorded but no action
is enforced yet -- decide.py lands in phase 3, and until then act stays "allow".

Callers pass our extensions either as headers (X-CP-Tenant, X-CP-Geo,
X-CP-Session) or as a "cp" object in the request body, which is stripped before
the body is forwarded upstream so the request stays valid OpenAI JSON:

    {"model": ..., "messages": [...],
     "cp": {"ctx": [{"id": "pol-3", "text": "...", "score": 0.8}],
            "tools": ["crm.read"], "tenant": "CS-BOT", "geo": "IN"}}
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any, AsyncIterator

import httpx
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from api import store, tier0
from api.schemas import Finding, Trace

log = structlog.get_logger("gw")
router = APIRouter()

VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000/v1")
MOCK = os.getenv("MOCK_H200", "0") == "1"
GEN_TIMEOUT_S = float(os.getenv("GEN_TIMEOUT_S", "120"))
TENANTS = ("CS-BOT", "KB-COPILOT", "DECIDE")
# model name -> price tier, for the cost arithmetic in tier0
TIER_OF: dict[str, str] = {"Qwen/Qwen2.5-7B-Instruct": "small"}

_cli: httpx.AsyncClient | None = None


async def open_gw() -> None:
    global _cli
    _cli = httpx.AsyncClient(base_url=VLLM_URL, timeout=GEN_TIMEOUT_S)


async def close_gw() -> None:
    global _cli
    if _cli is not None:
        await _cli.aclose()
        _cli = None


# ------------------------------------------------------------------ helpers

def _txt(msgs: list[dict[str, Any]]) -> str:
    """Flatten chat messages to the prompt text tier 0 scans."""
    out: list[str] = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            out.extend(p.get("text", "") for p in c if isinstance(p, dict))
    return "\n".join(out)


def _est_tok(s: str) -> int:
    return max(1, len(s) // 4)


def _meta(req: Request, cp: dict[str, Any]) -> tuple[str, str, str]:
    tenant = req.headers.get("X-CP-Tenant") or cp.get("tenant") or "CS-BOT"
    if tenant not in TENANTS:
        tenant = "CS-BOT"
    geo = req.headers.get("X-CP-Geo") or cp.get("geo") or "IN"
    sess = req.headers.get("X-CP-Session") or cp.get("sess") or f"s-{uuid.uuid4().hex[:8]}"
    return tenant, geo, sess


async def _scan(txt: str, side: str) -> list[Finding]:
    return tier0.scan(tier0.norm(txt), side=side)


_bg_tasks: set["asyncio.Task[None]"] = set()


def _bg(coro: Any) -> None:
    """Run to completion outside the request. A client that hangs up mid-stream
    must still leave a trace behind, so the tail cannot be awaited inside the
    response generator -- closing it would cancel the save."""
    t = asyncio.ensure_future(coro)
    _bg_tasks.add(t)
    t.add_done_callback(_bg_tasks.discard)


async def _save(tr: Trace) -> None:
    try:
        await store.put_trace(tr)
    except Exception as e:
        log.warning("trace_save_failed", id=tr.id, err=str(e))


def _finish(
    tr: Trace,
    resp: str,
    t0p: list[Finding],
    t_gen: float,
    tok_in: int,
    tok_out: int,
    model: str,
) -> Trace:
    """Common tail: tier 0 on the response, cost math, latency stamps."""
    t = time.perf_counter()
    fnd = list(t0p) + tier0.scan(tier0.norm(resp), side="resp")
    fnd += tier0.cost(
        tr.tenant, tok_in, tok_out, len(tr.tools), TIER_OF.get(model, "mid"),
        span=(0, min(len(resp), 1)),
    )
    t0_ms = (time.perf_counter() - t) * 1000
    tr.resp = resp
    tr.fnd = fnd
    tr.tok_in, tr.tok_out = tok_in, tok_out
    tr.cost = tier0.cost_usd(tok_in, tok_out, TIER_OF.get(model, "mid"))
    tr.lat = {"gen": round(t_gen, 2), "t0": round(t0_ms, 3),
              "total": round(t_gen + t0_ms, 2)}
    return tr


# --------------------------------------------------------------- mock upstream

_MOCK_TXT = (
    "Refunds are issued to the original payment method. Our policy allows returns "
    "within 3 years of purchase for unopened items. Contact support at "
    "help@example.com if the credit has not appeared."
)


async def _mock_stream(model: str) -> AsyncIterator[str]:
    for w in _MOCK_TXT.split(" "):
        yield w + " "
        await asyncio.sleep(0.004)


# ------------------------------------------------------------------- endpoint

@router.post("/v1/chat/completions")
async def chat(req: Request) -> Any:
    t_req = time.perf_counter()
    body = await req.json()
    cp: dict[str, Any] = body.pop("cp", {}) or {}
    tenant, geo, sess = _meta(req, cp)
    msgs = body.get("messages", [])
    prompt = _txt(msgs)
    model = body.get("model", "Qwen/Qwen2.5-7B-Instruct")

    # tier 0 on the prompt runs while the model generates, so it costs nothing
    t0p_task = asyncio.create_task(_scan(prompt, "prompt"))

    tr = Trace(
        id=f"t-{uuid.uuid4().hex[:12]}",
        sess=sess,
        tenant=tenant,
        geo=geo,
        ts=time.time(),
        prompt=prompt,
        resp="",
        ctx=cp.get("ctx", []) or [],
        tools=cp.get("tools", []) or [],
    )

    if body.get("stream"):
        return StreamingResponse(
            _stream(req, body, tr, t0p_task, t_req, model),
            media_type="text/event-stream",
            headers={"X-CP-Trace": tr.id, "X-Accel-Buffering": "no"},
        )

    if MOCK:
        await asyncio.sleep(0.05)
        txt = _MOCK_TXT
        tok_in, tok_out = _est_tok(prompt), _est_tok(txt)
        up: dict[str, Any] = {
            "id": f"chatcmpl-{tr.id}", "object": "chat.completion",
            "created": int(tr.ts), "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": txt}}],
            "usage": {"prompt_tokens": tok_in, "completion_tokens": tok_out,
                      "total_tokens": tok_in + tok_out},
        }
    else:
        if _cli is None:
            return JSONResponse({"error": "gateway not opened"}, status_code=503)
        try:
            r = await _cli.post("/chat/completions", json=body)
            r.raise_for_status()
            up = r.json()
        except Exception as e:
            log.warning("upstream_failed", err=str(e), id=tr.id)
            tr.lat = {"gen": (time.perf_counter() - t_req) * 1000}
            await _save(tr)
            return JSONResponse(
                {"error": {"message": f"upstream: {type(e).__name__}",
                           "type": "upstream_error"}},
                status_code=502, headers={"X-CP-Trace": tr.id},
            )
        txt = (up.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        u = up.get("usage") or {}
        tok_in = int(u.get("prompt_tokens") or _est_tok(prompt))
        tok_out = int(u.get("completion_tokens") or _est_tok(txt))

    t0p = await t0p_task
    tr = _finish(tr, txt, t0p, (time.perf_counter() - t_req) * 1000, tok_in, tok_out, model)
    await _save(tr)
    log.info("traced", id=tr.id, tenant=tenant, nfnd=len(tr.fnd), lat=tr.lat)
    return JSONResponse(
        up,
        headers={"X-CP-Trace": tr.id, "X-CP-Action": tr.act,
                 "X-CP-Findings": str(len(tr.fnd))},
    )


async def _stream(
    req: Request,
    body: dict[str, Any],
    tr: Trace,
    t0p_task: "asyncio.Task[list[Finding]]",
    t_req: float,
    model: str,
) -> AsyncIterator[bytes]:
    """SSE passthrough that accumulates the response for tracing.

    Phase 2 inserts the 40-token shadow buffer between accumulation and yield;
    the accumulator is already here so that change stays local.
    """
    acc: list[str] = []
    tok_in = tok_out = 0
    done = False   # upstream already sent [DONE]; do not send a second one
    try:
        if MOCK:
            async for w in _mock_stream(model):
                acc.append(w)
                tok_out += 1
                ch = {"id": f"chatcmpl-{tr.id}", "object": "chat.completion.chunk",
                      "created": int(tr.ts), "model": model,
                      "choices": [{"index": 0, "delta": {"content": w},
                                   "finish_reason": None}]}
                yield f"data: {json.dumps(ch)}\n\n".encode()
            tok_in = _est_tok(tr.prompt)
        else:
            if _cli is None:
                yield b'data: {"error": "gateway not opened"}\n\n'
                return
            body = {**body, "stream_options": {"include_usage": True}}
            async with _cli.stream("POST", "/chat/completions", json=body) as r:
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    yield f"{line}\n\n".encode()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        done = True
                        continue
                    try:
                        ch = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    for c in ch.get("choices") or []:
                        d = (c.get("delta") or {}).get("content")
                        if d:
                            acc.append(d)
                    u = ch.get("usage") or {}
                    if u:
                        tok_in = int(u.get("prompt_tokens") or 0)
                        tok_out = int(u.get("completion_tokens") or 0)
        if not done:
            yield b"data: [DONE]\n\n"
    finally:
        _bg(_tail(tr, "".join(acc), t0p_task, t_req, tok_in, tok_out, model))


async def _tail(
    tr: Trace,
    txt: str,
    t0p_task: "asyncio.Task[list[Finding]]",
    t_req: float,
    tok_in: int,
    tok_out: int,
    model: str,
) -> None:
    """Post-stream trace write. Runs detached, so a disconnect still traces."""
    t0p = await t0p_task
    tr = _finish(tr, txt, t0p, (time.perf_counter() - t_req) * 1000,
                 tok_in or _est_tok(tr.prompt), tok_out or _est_tok(txt), model)
    await _save(tr)
    log.info("traced_stream", id=tr.id, tenant=tr.tenant,
             nfnd=len(tr.fnd), lat=tr.lat)
