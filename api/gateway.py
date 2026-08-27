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
import re
import time
import uuid
from typing import Any, AsyncIterator

import httpx
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from api import detclient, store, tier0
from api.schemas import Finding, Trace

log = structlog.get_logger("gw")
router = APIRouter()

VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000/v1")
MOCK = os.getenv("MOCK_H200", "0") == "1"
GEN_TIMEOUT_S = float(os.getenv("GEN_TIMEOUT_S", "120"))
TENANTS = ("CS-BOT", "KB-COPILOT", "DECIDE")
# model name -> price tier, for the cost arithmetic in tier0
TIER_OF: dict[str, str] = {"Qwen/Qwen2.5-7B-Instruct": "small"}
SHADOW_TOKENS = int(os.getenv("SHADOW_TOKENS", "40"))
T1_ON = os.getenv("T1", "1") == "1"
T1_NEED = ["nli", "safety", "bias"]
SHADOW_ON = os.getenv("SHADOW", "1") == "1"

# sentence boundary used to cut the shadow buffer into checkable windows
_END = re.compile(r"[.!?](?:[\"\')\]]+)?(?:\s|$)|\n")

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


async def _t1(tr: Trace, resp: str) -> tuple[list[Finding], dict[str, Any]]:
    """Tier 1 on the H200.

    Phase 2 runs this on every response. Phase 3's router decides which
    responses are worth it, which is the entire point of the tier ladder -- so
    the call site stays here and only the gate in front of it changes.
    """
    if not T1_ON or not resp.strip():
        return [], {}
    return await detclient.check(resp, tr.ctx, T1_NEED)


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
    resp_fnd: list[Finding] | None = None,
) -> Trace:
    """Common tail: tier 0 on the response, cost math, latency stamps.

    resp_fnd is supplied when the shadow buffer already checked the text window
    by window; rescanning the whole response would just duplicate its findings.
    """
    t = time.perf_counter()
    fnd = list(t0p) + (resp_fnd if resp_fnd is not None
                       else tier0.scan(tier0.norm(resp), side="resp"))
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


# --------------------------------------------------------------- shadow buffer

def _cut(pend: str) -> int | None:
    """End offset of the first complete sentence in pend, or None."""
    m = _END.search(pend)
    return m.end() if m else None


def _hold(fnd: list[Finding]) -> bool:
    """Whether a checked window must not reach the client.

    Phase 2 placeholder: a deliberately narrow rule covering the cases tier 0
    can prove. decide.py replaces this in phase 3, where the policy rulebook and
    the per-tenant floors make the call. Note what is absent: `unverifiable`
    never holds, per invariant 2.
    """
    return any(f.sev >= 2 and f.label in ("pii", "unsafe", "inject") for f in fnd)


class Shadow:
    """Holds back the tail of a stream until it has been checked.

    A guardrail that inspects the response after the user has read it is not a
    guardrail. Text leaves here only once a detector has seen it: complete
    sentences are checked and released, and if generation stalls mid-sentence
    the buffer is force-checked once it exceeds SHADOW_TOKENS so a slow model
    cannot deadlock the stream.
    """

    def __init__(self, tenant: str, need: list[str] | None = None) -> None:
        self.pend = ""
        self.ntok = 0
        self.tenant = tenant
        self.need = need or []
        self.fnd: list[Finding] = []
        self.held = False
        self.base = 0        # char offset of pend[0] within the full response
        self.lat: list[float] = []

    async def _check(self, seg: str, off: int) -> list[Finding]:
        if not seg.strip():
            return []       # a whitespace window is not a claim, not a defect
        t = time.perf_counter()
        fnd = tier0.scan(tier0.norm(seg), side="resp")
        if self.need:
            t1, _ = await detclient.check(seg, [], self.need)
            fnd += t1
        self.lat.append((time.perf_counter() - t) * 1000)
        for f in fnd:                      # rebase spans onto the full response
            f.span = (f.span[0] + off, f.span[1] + off)
        return fnd

    async def feed(self, delta: str) -> str:
        """Take one token, return whatever is now safe to release."""
        self.pend += delta
        self.ntok += 1
        out = ""
        while True:
            c = _cut(self.pend)
            if c is None:
                break
            seg, self.pend = self.pend[:c], self.pend[c:]
            fnd = await self._check(seg, self.base)
            self.fnd += fnd
            self.base += len(seg)
            if _hold(fnd):
                self.held = True
                return out
            out += seg
            self.ntok = max(0, self.ntok - 1)
        if self.ntok > SHADOW_TOKENS:
            # no sentence end in sight; check and release all but the shadow tail
            keep = SHADOW_TOKENS * 4
            if len(self.pend) > keep:
                seg, self.pend = self.pend[:-keep], self.pend[-keep:]
                fnd = await self._check(seg, self.base)
                self.fnd += fnd
                self.base += len(seg)
                if _hold(fnd):
                    self.held = True
                    return out
                out += seg
                self.ntok = SHADOW_TOKENS
        return out

    async def drain(self) -> str:
        """Flush whatever is left when generation ends."""
        if not self.pend or self.held:
            return ""
        seg, self.pend = self.pend, ""
        fnd = await self._check(seg, self.base)
        self.fnd += fnd
        self.base += len(seg)
        if _hold(fnd):
            self.held = True
            return ""
        return seg


# --------------------------------------------------------------- mock upstream

# Deliberately clean at tier 0 so the offline rehearsal has a baseline where
# nothing fires. The "3 years" claim is the tier-1 contradiction against the
# governed returns policy, which needs ctx to surface.
_MOCK_TXT = (
    "Refunds are issued to the original payment method. Our policy allows returns "
    "within 3 years of purchase for unopened items. Reply to this message if the "
    "credit has not appeared after five business days."
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
    t1f, t1l = await _t1(tr, txt)
    tr = _finish(tr, txt, t0p, (time.perf_counter() - t_req) * 1000, tok_in, tok_out, model)
    if t1f or t1l:
        tr.fnd += t1f
        tr.tier = max(tr.tier, 1)
        tr.lat["t1"] = round(float(t1l.get("t1", 0.0)), 2)
        tr.lat["total"] = round(tr.lat["total"] + tr.lat["t1"], 2)
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
    """SSE stream, re-emitted through the shadow buffer.

    This is not a passthrough. Deltas go into Shadow and only come out once a
    detector has seen them, so the client receives sentence-sized chunks rather
    than raw tokens. That is the cost of being able to stop a bad span before it
    is read, and it is the whole point of the buffer.
    """
    sh = Shadow(tr.tenant) if SHADOW_ON else None
    acc: list[str] = []
    tok_in = tok_out = 0
    fin: str | None = None

    def ch(txt: str, finish: str | None = None) -> bytes:
        d = {"id": f"chatcmpl-{tr.id}", "object": "chat.completion.chunk",
             "created": int(tr.ts), "model": model,
             "choices": [{"index": 0, "delta": ({"content": txt} if txt else {}),
                          "finish_reason": finish}]}
        return f"data: {json.dumps(d)}\n\n".encode()

    async def deltas() -> AsyncIterator[str]:
        nonlocal tok_in, tok_out
        if MOCK:
            async for w in _mock_stream(model):
                tok_out += 1
                yield w
            tok_in = _est_tok(tr.prompt)
            return
        if _cli is None:
            return
        b2 = {**body, "stream_options": {"include_usage": True}}
        async with _cli.stream("POST", "/chat/completions", json=b2) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for c in obj.get("choices") or []:
                    d = (c.get("delta") or {}).get("content")
                    if d:
                        tok_out += 1
                        yield d
                u = obj.get("usage") or {}
                if u:
                    tok_in = int(u.get("prompt_tokens") or 0)
                    tok_out = int(u.get("completion_tokens") or tok_out)

    try:
        async for d in deltas():
            acc.append(d)
            if sh is None:
                yield ch(d)
                continue
            rel = await sh.feed(d)
            if rel:
                yield ch(rel)
            if sh.held:
                fin = "content_filter"
                yield ch("", finish=fin)
                break
        if sh is not None and not sh.held:
            rel = await sh.drain()
            if rel:
                yield ch(rel)
        if fin is None:
            yield ch("", finish="stop")
        yield b"data: [DONE]\n\n"
    finally:
        _bg(_tail(tr, "".join(acc), t0p_task, t_req, tok_in, tok_out, model, sh))


async def _tail(
    tr: Trace,
    txt: str,
    t0p_task: "asyncio.Task[list[Finding]]",
    t_req: float,
    tok_in: int,
    tok_out: int,
    model: str,
    sh: "Shadow | None" = None,
) -> None:
    """Post-stream trace write. Runs detached, so a disconnect still traces."""
    t0p = await t0p_task
    # The shadow buffer already ran tier 0 window by window, which is what can
    # stop a span mid-stream. Tier 1 costs tens of milliseconds per call, so on
    # a stream it runs once over the finished text rather than per sentence.
    t1f, t1l = await _t1(tr, txt)
    tr = _finish(tr, txt, t0p, (time.perf_counter() - t_req) * 1000,
                 tok_in or _est_tok(tr.prompt), tok_out or _est_tok(txt), model,
                 resp_fnd=sh.fnd if sh is not None else None)
    if t1f or t1l:
        tr.fnd += t1f
        tr.tier = max(tr.tier, 1)
        tr.lat["t1"] = round(float(t1l.get("t1", 0.0)), 2)
    if sh is not None:
        tr.lat["shadow"] = round(sum(sh.lat), 3)
        tr.lat["shadow_n"] = len(sh.lat)
        if sh.held:
            # the buffer actually stopped the span. decide.py takes this over in
            # phase 3; recording it here keeps the trace honest about what ran.
            tr.act = "block"
    await _save(tr)
    log.info("traced_stream", id=tr.id, tenant=tr.tenant, act=tr.act,
             nfnd=len(tr.fnd), lat=tr.lat)
