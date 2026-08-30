"""Tool reversibility.

The question that matters for an agent is not what a tool is called but whether
its effect can be taken back. Reading a record is free. Updating one is
recoverable. Issuing a refund is not: once the money moves, no policy decision
downstream can undo it, so that is where a human belongs.
"""

from __future__ import annotations

import re
from typing import Iterable, Literal

Cls = Literal["ro", "rev", "irrev"]

# Matched against the tool name, most specific first. A tool we do not
# recognise is treated as reversible rather than read-only: assuming a side
# effect exists is the safer error.
_RULES: list[tuple[str, Cls]] = [
    (r"\b(?:read|get|list|search|fetch|lookup|query|find|view|describe|check)\b", "ro"),
    (r"\b(?:refund|charge|pay|transfer|remit|payout|settle|disburse)\b", "irrev"),
    (r"\b(?:delete|purge|destroy|drop|revoke|terminate|deprovision|wipe)\b", "irrev"),
    (r"\b(?:send|email|sms|notify|publish|post|dispatch|submit|escalate)\b", "irrev"),
    (r"\b(?:deploy|release|rollout|migrate|provision)\b", "irrev"),
    (r"\b(?:cancel|close|suspend|ban|block_user|deactivate)\b", "irrev"),
    (r"\b(?:create|update|set|write|patch|upsert|assign|tag|draft|save)\b", "rev"),
]
_PATS: list[tuple[re.Pattern[str], Cls]] = [
    (re.compile(p, re.I), c) for p, c in _RULES
]

DEFAULT: Cls = "rev"


def cls(name: str) -> Cls:
    """Classify one tool. Namespaced names like crm.contact.delete work."""
    n = name.replace(".", " ").replace("_", " ").replace("-", " ")
    for pat, c in _PATS:
        if pat.search(n):
            return c
    return DEFAULT


def classify(names: Iterable[str]) -> dict[str, Cls]:
    return {n: cls(n) for n in names}


def has_irrev(names: Iterable[str]) -> bool:
    return any(cls(n) == "irrev" for n in names)


def all_ro(names: Iterable[str]) -> bool:
    """Read-only calls auto-allow: there is nothing to take back."""
    ns = list(names)
    return bool(ns) and all(cls(n) == "ro" for n in ns)
