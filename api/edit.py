"""The four permitted edits. Deterministic, and provably non-generative.

Everything this module can emit is either a slice of the input or one of the
frozen constants below. There is no model call and no string interpolation of
model output anywhere in here, which is what lets us claim an edited response
contains nothing the system invented (invariant 5).

If a fix would need new prose, that is a regenerate, and a regenerate is an
escalate, not an edit.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from api.schemas import Finding

Op = Literal["redact", "delete", "cite", "hedge"]

# frozen output vocabulary: the complete set of characters this module can add
REDACTION = "[redacted]"
HEDGE = (
    " Note: this answer could not be verified against the source documents, "
    "so please confirm it before relying on it."
)
CITE_OPEN = " [source: "
CITE_CLOSE = "]"

_WS = re.compile(r"[ \t]+")


def _clean(s: str) -> str:
    """Tidy the seams left by deletion. Whitespace only, never words."""
    s = _WS.sub(" ", s)
    s = re.sub(r" +([.,;:!?])", r"\1", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _append(out: str, frag: str) -> str:
    """Append a frozen fragment. Never leaves the result starting with a space."""
    return (out + frag) if out else frag.lstrip()


def plan(fnd: list[Finding], act: str) -> list[dict[str, Any]]:
    """Which edits the findings justify. Pure, so it is easy to test and audit."""
    ops: list[dict[str, Any]] = []
    if act != "edit":
        return ops

    for f in fnd:
        if f.side != "resp":
            continue          # spans into the prompt are not ours to rewrite
        if f.label == "pii":
            ops.append({"op": "redact", "span": list(f.span), "why": f.evid or "pii"})
        elif f.label == "contradicted":
            ops.append({"op": "delete", "span": list(f.span), "why": f.evid or "contradicted"})

    cites = sorted({f.evid for f in fnd
                    if f.label == "contradicted" and f.evid})
    for c in cites:
        ops.append({"op": "cite", "doc": c})

    if any(f.label == "unverifiable" for f in fnd):
        ops.append({"op": "hedge"})

    return ops


def apply(resp: str, ops: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Run the plan. Span edits go right to left so earlier offsets stay valid."""
    out = resp
    done: list[dict[str, Any]] = []

    span_ops = [o for o in ops if o["op"] in ("redact", "delete")]
    span_ops.sort(key=lambda o: o["span"][0], reverse=True)

    last_start = len(resp) + 1
    for o in span_ops:
        a, b = o["span"]
        if not (0 <= a < b <= len(out)) or a >= last_start:
            continue          # out of range, or overlaps an edit already applied
        rep = REDACTION if o["op"] == "redact" else ""
        out = out[:a] + rep + out[b:]
        last_start = a
        done.append({**o, "text": resp[a:b]})

    out = _clean(out)

    # deleting every sentence leaves nothing to deliver. Appending a citation to
    # an empty body would dress up a non-answer as an answer, so say so instead
    # and let the caller escalate: a response that needs rewriting is a
    # regenerate, and a regenerate is not an edit (invariant 5).
    if not out and any(o["op"] == "delete" for o in done):
        done.append({"op": "empty"})

    for o in ops:
        if o["op"] == "cite":
            frag = CITE_OPEN + str(o["doc"]) + CITE_CLOSE
            if frag not in out:
                out = _append(out, frag)
                done.append(o)

    if any(o["op"] == "hedge" for o in ops) and HEDGE.strip() not in out:
        out = _append(out, HEDGE)
        done.append({"op": "hedge"})

    return out, done


def emptied(ops: list[dict[str, Any]]) -> bool:
    """True when the edit removed the whole response. Caller should escalate."""
    return any(o["op"] == "empty" for o in ops)


def run(resp: str, fnd: list[Finding], act: str) -> tuple[str, list[dict[str, Any]]]:
    return apply(resp, plan(fnd, act))
