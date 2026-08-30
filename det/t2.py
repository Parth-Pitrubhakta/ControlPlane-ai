"""Tier 2: claim decomposition, per-claim retrieval, self-consistency.

Tier 1 scores every response sentence against every retrieved chunk and takes
the maximum contradiction. On ControlPlane-Bench that produced 364 false
positives against 11 true ones, because with a dozen chunks in play some pair
always scores high. More context made a contradiction MORE likely, which is
backwards.

Tier 2 attacks that directly:

  decompose   split the response into atomic claims, each quoted verbatim so we
              can still point at real character offsets
  retrieve    pick the evidence for each claim separately, so a claim about
              warranty length is never judged against the shipping policy
  vote        where NLI is not confident, ask the model k times and take the
              majority, which costs a lot and is why this is not tier 1

It is slower by design. The router decides who is worth it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import httpx
import torch

from det import embed, nli
from det.schema import Finding

VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000/v1")
# where the NLI model actually lives. Empty means it is in this process.
NLI_URL = os.getenv("T2_NLI_URL", "")
JUDGE_MODEL = os.getenv("T2_JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
VER = "t2-claim-decomp-v1"

TOPK = int(os.getenv("T2_TOPK", "3"))          # chunks retrieved per claim
K = int(os.getenv("T2_SELF_CONSISTENCY", "3"))  # votes per uncertain claim
MAX_CLAIMS = int(os.getenv("T2_MAX_CLAIMS", "12"))
TIMEOUT_S = float(os.getenv("T2_TIMEOUT_S", "60"))

# NLI is trusted outright only when it is very sure. Everything between these
# goes to the vote, which is the expensive part and the point of the tier.
SURE_CONTRA = float(os.getenv("T2_SURE_CONTRA", "0.90"))
SURE_ENTAIL = float(os.getenv("T2_SURE_ENTAIL", "0.75"))

import logging

log = logging.getLogger("det.t2")

_cli: httpx.AsyncClient | None = None

DECOMP = """Split the passage into its separate factual claims.

Rules:
- Copy each claim word for word from the passage. Do not paraphrase or reword.
- One self-contained factual statement per claim.
- Skip greetings, offers of help, questions and pleasantries.
- Reply with a JSON array of strings and nothing else.

PASSAGE:
{resp}"""

JUDGE = """Decide whether the EVIDENCE supports the CLAIM.

EVIDENCE:
{ev}

CLAIM: {claim}

Answer with exactly one word:
SUPPORTED if the evidence states the claim
CONTRADICTED if the evidence states something incompatible with the claim
UNKNOWN if the evidence neither states nor contradicts it"""


async def open_t2() -> None:
    global _cli
    _cli = httpx.AsyncClient(base_url=VLLM_URL, timeout=TIMEOUT_S)


async def close_t2() -> None:
    global _cli
    if _cli is not None:
        await _cli.aclose()
        _cli = None


def ready() -> bool:
    return _cli is not None


def info() -> dict[str, Any]:
    return {"model": JUDGE_MODEL, "ver": VER, "ready": ready(),
            "topk": TOPK, "k": K, "vllm": VLLM_URL}


async def score_pairs(pairs: list[tuple[str, str]]) -> list[tuple[float, float]]:
    """(entailment, contradiction) per (premise, hypothesis) pair."""
    if not pairs:
        return []
    if nli._bat is not None:
        return await nli._bat.submit_many(pairs)
    if not NLI_URL:
        raise RuntimeError("tier 2 has no NLI: set T2_NLI_URL or load nli locally")
    assert _cli is not None
    r = await _cli.post(f"{NLI_URL}/pairs", json={"pairs": [list(p) for p in pairs]})
    r.raise_for_status()
    return [(float(a), float(b)) for a, b in r.json()["scores"]]


async def _ask(prompt: str, temp: float, maxtok: int) -> str:
    assert _cli is not None
    r = await _cli.post("/chat/completions", json={
        "model": JUDGE_MODEL, "temperature": temp, "max_tokens": maxtok,
        "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def _parse_claims(out: str) -> list[str]:
    """Pull the claim list out of whatever shape the model replied in.

    Qwen answers this prompt with one JSON array per line rather than a single
    array, so a greedy [.*] match spans them all and parses as nothing. Collect
    every non-nested array instead, then fall back to bullets and bare lines.
    """
    claims: list[str] = []
    for m in re.finditer(r"\[[^\[\]]*\]", out, re.S):
        try:
            got = json.loads(m.group())
        except json.JSONDecodeError:
            continue
        if isinstance(got, list):
            claims += [str(x) for x in got if isinstance(x, (str, int, float))]
    if claims:
        return claims
    try:
        got = json.loads(out)
        if isinstance(got, list):
            return [str(x) for x in got if isinstance(x, (str, int, float))]
    except json.JSONDecodeError:
        pass
    for line in out.splitlines():
        t = line.strip().lstrip("-*0123456789. ").strip().strip('"')
        if len(t) > 12:
            claims.append(t)
    return claims


def _locate(resp: str, claim: str) -> tuple[int, int] | None:
    """Find the claim's real offsets in the response.

    The decomposer is told to quote verbatim but does not always obey, so fall
    back to the sentence that shares the most words. A finding whose span we
    cannot place is worse than no finding: the UI would highlight the wrong text
    and an edit would cut the wrong words.
    """
    c = claim.strip().strip('"').strip()
    if not c:
        return None
    i = resp.find(c)
    if i >= 0:
        return (i, i + len(c))
    norm = re.sub(r"\s+", " ", c).lower()
    i = re.sub(r"\s+", " ", resp).lower().find(norm)
    if i >= 0 and len(norm) > 12:
        j = min(len(resp), i + len(norm))
        return (i, j)
    want = {w for w in re.findall(r"[\w']+", c.lower()) if len(w) > 3}
    if not want:
        return None
    best, best_ov = None, 0.0
    for sp in nli.sents(resp):
        have = {w for w in re.findall(r"[\w']+", resp[sp[0]:sp[1]].lower()) if len(w) > 3}
        ov = len(want & have) / len(want)
        if ov > best_ov:
            best, best_ov = sp, ov
    return best if best_ov >= 0.6 else None


async def decompose(resp: str) -> list[tuple[str, tuple[int, int]]]:
    """Atomic claims with the offsets they occupy in the response."""
    try:
        out = await _ask(DECOMP.format(resp=resp), 0.0, 700)
    except Exception:
        out = ""
    claims = _parse_claims(out)
    if not claims:
        # Decomposition failed. Fall back to sentences so tier 2 still gets the
        # per-claim retrieval benefit, but say so: this used to fail silently and
        # tier 2 quietly became tier 1 with extra latency.
        log.warning('{"event": "t2_decompose_fallback", "chars": %d, "raw": %r}',
                    len(resp), out[:120])
        return [(resp[a:b], (a, b)) for a, b in nli.sents(resp)][:MAX_CLAIMS]

    out_pairs: list[tuple[str, tuple[int, int]]] = []
    for c in claims[:MAX_CLAIMS]:
        sp = _locate(resp, c)
        if sp:
            out_pairs.append((c.strip(), sp))
    return out_pairs


async def _evidence(claims: list[str], ctx: list[dict[str, Any]]) -> list[list[int]]:
    """Top-k chunks per claim, chosen by embedding similarity."""
    if not ctx:
        return [[] for _ in claims]
    if len(ctx) <= TOPK:
        return [list(range(len(ctx))) for _ in claims]
    vs = await embed._bat.submit_many(
        [embed.Q_PREFIX + c for c in claims] + [c.get("text", "") for c in ctx])
    cv = torch.tensor(vs[len(claims):])
    out = []
    for v in vs[: len(claims)]:
        sim = cv @ torch.tensor(v)
        out.append(sim.topk(min(TOPK, len(ctx))).indices.tolist())
    return out


async def _vote(claim: str, ev: str) -> tuple[str, float]:
    """Ask k times at non-zero temperature and take the majority."""
    async def once() -> str:
        try:
            t = (await _ask(JUDGE.format(ev=ev, claim=claim), 0.7, 6)).upper()
        except Exception:
            return "UNKNOWN"
        if "CONTRADICT" in t:
            return "CONTRADICTED"
        if "SUPPORT" in t:
            return "SUPPORTED"
        return "UNKNOWN"

    votes = await asyncio.gather(*(once() for _ in range(K)))
    tally: dict[str, int] = {}
    for v in votes:
        tally[v] = tally.get(v, 0) + 1
    top = max(tally, key=lambda k: tally[k])
    return top, tally[top] / len(votes)


async def check(resp: str, ctx: list[dict[str, Any]],
                prompt: str = "") -> list[Finding]:
    if not resp.strip():
        return []
    claims = await decompose(resp)
    if not claims:
        return []

    texts = [c for c, _ in claims]
    spans = [s for _, s in claims]

    if not ctx:
        # no retrieval means no evidence either way, for every claim
        return [Finding(span=sp, dim="perf", label="unverifiable", sev=1,
                        conf=0.5, evid=None, det=VER) for sp in spans]

    sel = await _evidence(texts, ctx)

    pairs: list[tuple[str, str]] = []
    owner: list[tuple[int, int]] = []
    for i, c in enumerate(texts):
        for j in sel[i]:
            pairs.append((ctx[j].get("text", ""), c))
            owner.append((i, j))
    res = await score_pairs(pairs)

    best_e = [0.0] * len(texts)
    best_c = [0.0] * len(texts)
    src_c: list[str | None] = [None] * len(texts)
    for k, (e, c) in enumerate(res):
        i, j = owner[k]
        if e > best_e[i]:
            best_e[i] = e
        if c > best_c[i]:
            best_c[i], src_c[i] = c, ctx[j].get("id")

    out: list[Finding] = []
    unsure: list[int] = []
    for i, sp in enumerate(spans):
        if best_c[i] >= SURE_CONTRA:
            out.append(Finding(span=sp, dim="perf", label="contradicted", sev=2,
                               conf=round(best_c[i], 4), evid=src_c[i], det=VER))
        elif best_e[i] >= SURE_ENTAIL:
            continue                       # confidently grounded
        else:
            unsure.append(i)

    if unsure:
        async def judge(i: int) -> tuple[int, str, float]:
            ev = "\n\n".join(ctx[j].get("text", "") for j in sel[i])
            v, agree = await _vote(texts[i], ev)
            return i, v, agree

        for i, verdict, agree in await asyncio.gather(*(judge(i) for i in unsure)):
            if verdict == "CONTRADICTED":
                out.append(Finding(span=spans[i], dim="perf", label="contradicted",
                                   sev=2, conf=round(agree, 4), evid=src_c[i],
                                   det=f"{VER}-vote"))
            elif verdict == "UNKNOWN":
                out.append(Finding(span=spans[i], dim="perf", label="unverifiable",
                                   sev=1, conf=round(agree, 4), evid=None,
                                   det=f"{VER}-vote"))
            # SUPPORTED emits nothing, which is the whole point
    return out
