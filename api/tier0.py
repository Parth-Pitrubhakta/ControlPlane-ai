"""Tier 0: everything we can check on CPU in single-digit milliseconds.

Runs on 100% of traffic, both directions. Pure and synchronous by design -- no
IO, no awaits, no model calls -- so it can hold a 2-5 ms budget. Every function
returns list[Finding], never a score and never a bool.

Label vocabulary emitted here: pii, inject, unsafe, format, cost_anom.
"format" extends the schema comment's list; it covers structural defects in the
response (empty, truncated code fence, leaked template) that are not injection.

Invariant 4: cost math lives here because it is free arithmetic, but a cost
finding must never reach a blocking action. decide.py enforces that ceiling.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

from api.schemas import Finding, Side

T0_PII = os.getenv("T0_PII", "regex")   # regex | presidio
DET = "t0-v1"

# ---------------------------------------------------------------- PII (regex)

_PII_PAT: list[tuple[str, re.Pattern[str], int]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"), 2),
    ("card", re.compile(r"\b(?:\d[ -]?){13,19}\b"), 3),
    ("aadhaar", re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b"), 3),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), 3),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 3),
    ("phone", re.compile(r"(?<![\w.])(?:\+\d{1,3}[ -]?)?(?:\(\d{2,4}\)[ -]?)?\d{5}[ -]?\d{5}(?![\w.])"), 2),
    ("ip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), 1),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), 3),
]


def _luhn(s: str) -> bool:
    d = [int(c) for c in s if c.isdigit()]
    if not 13 <= len(d) <= 19:
        return False
    tot, alt = 0, False
    for x in reversed(d):
        if alt:
            x *= 2
            if x > 9:
                x -= 9
        tot += x
        alt = not alt
    return tot % 10 == 0


def pii(txt: str) -> list[Finding]:
    """Regex PII with a Luhn gate on card numbers to keep precision up."""
    out: list[Finding] = []
    for name, pat, sev in _PII_PAT:
        for m in pat.finditer(txt):
            if name == "card" and not _luhn(m.group()):
                continue
            if name == "ip":
                p = m.group().split(".")
                if any(int(x) > 255 for x in p):
                    continue
            out.append(
                Finding(
                    span=(m.start(), m.end()),
                    dim="resp",
                    label="pii",
                    sev=sev,
                    conf=0.9 if name in ("email", "card", "ssn", "pan") else 0.65,
                    evid=name,
                    det=f"{DET}-pii-regex",
                )
            )
    return _dedupe(out)


# ------------------------------------------------------------------ injection

_OVERRIDE = re.compile(
    r"(?i)\b(?:ignore|disregard|forget|override)\b[^.\n]{0,30}\b"
    r"(?:previous|prior|above|earlier|all)\b[^.\n]{0,20}"
    r"\b(?:instruction|prompt|rule|direction|context)s?\b"
)
_PERSONA = re.compile(
    r"(?i)(?:you are now|from now on,? you|act as (?:if|an?)|pretend (?:to be|you)|"
    r"developer mode|do anything now|dan mode)"
)
_EXFIL = re.compile(
    r"(?i)(?:reveal|print|repeat|show|output|reproduce)\b[^.\n]{0,25}\b"
    r"(?:system prompt|initial instruction|your instruction|the rules above)"
)
_ROLE = re.compile(
    r"(?:<\|im_(?:start|end)\|>|<\|(?:system|user|assistant|endoftext)\|>|"
    r"\[/?INST\]|<<SYS>>|^\s*###\s*(?:System|Instruction)\b)",
    re.M,
)
_B64 = re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b")
# bidi overrides, zero-width, and tag characters used to smuggle instructions
_ODD = re.compile(r"[‪-‮⁦-⁩​-‏⁠﻿\U000e0000-\U000e007f]")

_INJ: list[tuple[str, re.Pattern[str], int, float]] = [
    ("override", _OVERRIDE, 3, 0.85),
    ("persona", _PERSONA, 2, 0.7),
    ("exfil", _EXFIL, 3, 0.8),
    ("role_marker", _ROLE, 2, 0.75),
    ("b64_blob", _B64, 1, 0.4),
    ("odd_unicode", _ODD, 2, 0.8),
]


def inject(txt: str) -> list[Finding]:
    out: list[Finding] = []
    for name, pat, sev, conf in _INJ:
        for m in pat.finditer(txt):
            out.append(
                Finding(
                    span=(m.start(), m.end()),
                    dim="resp",
                    label="inject",
                    sev=sev,
                    conf=conf,
                    evid=name,
                    det=f"{DET}-inject",
                )
            )
    return _dedupe(out)


# ------------------------------------------------------- denylist and format

DENY: tuple[str, ...] = tuple(
    x for x in os.getenv("T0_DENY", "").split(",") if x
) or (
    "guaranteed returns",
    "risk-free investment",
    "i am a licensed",
    "this is legal advice",
    "this is medical advice",
)

_FENCE = re.compile(r"```")
_TMPL = re.compile(r"\{\{\s*\w+\s*\}\}|<<[A-Z_]{3,}>>")


def deny(txt: str) -> list[Finding]:
    low = txt.lower()
    out: list[Finding] = []
    for p in DENY:
        i = low.find(p)
        while i != -1:
            out.append(
                Finding(
                    span=(i, i + len(p)),
                    dim="resp",
                    label="unsafe",
                    sev=2,
                    conf=0.95,
                    evid=f"deny:{p}",
                    det=f"{DET}-deny",
                )
            )
            i = low.find(p, i + 1)
    return out


def fmt(txt: str) -> list[Finding]:
    """Structural defects. Cheap, and they catch truncation the model hides."""
    out: list[Finding] = []
    n = len(txt)
    if not txt.strip():
        return [Finding(span=(0, 0), dim="resp", label="format", sev=1, conf=1.0,
                        evid="empty", det=f"{DET}-fmt")]
    if len(_FENCE.findall(txt)) % 2 == 1:
        m = list(_FENCE.finditer(txt))[-1]
        out.append(Finding(span=(m.start(), n), dim="resp", label="format", sev=1,
                           conf=0.9, evid="unclosed_fence", det=f"{DET}-fmt"))
    for m in _TMPL.finditer(txt):
        out.append(Finding(span=(m.start(), m.end()), dim="resp", label="format", sev=1,
                           conf=0.85, evid="unfilled_template", det=f"{DET}-fmt"))
    return out


# ----------------------------------------------------------------- cost math

# per-tenant baselines, refreshed off the request path. Pure arithmetic here.
BASE: dict[str, dict[str, float]] = {
    "CS-BOT": {"tok_out": 220.0, "tools": 1.0, "cost": 0.0016},
    "KB-COPILOT": {"tok_out": 420.0, "tools": 2.0, "cost": 0.0030},
    "DECIDE": {"tok_out": 600.0, "tools": 3.0, "cost": 0.0060},
}
COST_MULT = float(os.getenv("T0_COST_MULT", "3.0"))
# USD per 1k tokens, per model tier
PRICE: dict[str, tuple[float, float]] = {
    "small": (0.0002, 0.0006),
    "mid": (0.0010, 0.0030),
    "large": (0.0050, 0.0150),
}


def cost_usd(tok_in: int, tok_out: int, tier: str = "mid") -> float:
    pi, po = PRICE.get(tier, PRICE["mid"])
    return round(tok_in / 1000 * pi + tok_out / 1000 * po, 6)


def cost(
    tenant: str,
    tok_in: int,
    tok_out: int,
    ntool: int,
    tier: str,
    base: dict[str, float] | None = None,
    span: tuple[int, int] = (0, 0),
) -> list[Finding]:
    """Emits cost_anom only on anomaly. dim="cost" can never block (invariant 4)."""
    b = base or BASE.get(tenant, BASE["CS-BOT"])
    c = cost_usd(tok_in, tok_out, tier)
    hits: list[str] = []
    if tok_out > b["tok_out"] * COST_MULT:
        hits.append(f"tok_out={tok_out} vs base {b['tok_out']:.0f}")
    if ntool > max(b["tools"] * COST_MULT, b["tools"] + 2):
        hits.append(f"tools={ntool} vs base {b['tools']:.0f}")
    if c > b["cost"] * COST_MULT:
        hits.append(f"usd={c:.4f} vs base {b['cost']:.4f}")
    if tier == "large" and b["cost"] < PRICE["large"][1]:
        hits.append("tier=large above tenant baseline tier")
    if not hits:
        return []
    return [
        Finding(
            span=span,
            dim="cost",
            label="cost_anom",
            sev=1,
            conf=0.8,
            evid="; ".join(hits),
            det=f"{DET}-cost",
        )
    ]


# ------------------------------------------------------------------ entrypoints

def _dedupe(fnd: list[Finding]) -> list[Finding]:
    """Drop spans fully contained in another finding with the same label."""
    out: list[Finding] = []
    for f in sorted(fnd, key=lambda x: (x.span[0], -(x.span[1] - x.span[0]))):
        if any(
            g.label == f.label and g.span[0] <= f.span[0] and f.span[1] <= g.span[1]
            for g in out
        ):
            continue
        out.append(f)
    return out


def scan(txt: str, *, side: Side = "resp") -> list[Finding]:
    """All tier-0 text checks. side="prompt" skips response-only structure checks.

    Stamps `side` on every finding so spans resolve against the right text.
    """
    out = pii(txt) + inject(txt) + deny(txt)
    if side == "resp":
        out += fmt(txt)
    if side != "resp":
        for f in out:
            f.side = side
    return out


def norm(txt: str) -> str:
    """NFKC fold. Call before scan when comparing against denylists."""
    return unicodedata.normalize("NFKC", txt)


def summary(fnd: list[Finding]) -> dict[str, Any]:
    """Compact feature view the router will consume in phase 3."""
    return {
        "n": len(fnd),
        "pii": sum(1 for f in fnd if f.label == "pii"),
        "inj": sum(1 for f in fnd if f.label == "inject"),
        "sev_max": max((f.sev for f in fnd), default=0),
    }
