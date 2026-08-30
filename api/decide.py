"""Findings in, one action out.

No scalar risk score reaches this module (invariant 3). The router's float
picks a verification tier and is otherwise for offline analysis; the action is
resolved here from the findings themselves against the policy rulebook, by
most-restrictive-wins.

Two ceilings are enforced before the max is taken, and they are the reason this
is a function rather than a lookup:

  unverifiable  can never exceed annotate (invariant 2). We found no evidence
                either way, and blocking a true-but-unretrieved claim is the
                failure mode that gets guardrails switched off.
  dim == cost   can never exceed annotate (invariant 4). A response is never
                blocked or edited for being expensive.
"""

from __future__ import annotations

from api import tools
from api.schemas import RANK, Act, Finding, Policy

# per-label ceilings, applied to the policy floor before most-restrictive-wins
CEIL: dict[str, Act] = {"unverifiable": "annotate"}
CEIL_DIM: dict[str, Act] = {"cost": "annotate"}

DEFAULT_FLOOR: Act = "annotate"


def _cap(a: Act, ceiling: Act) -> Act:
    return a if RANK[a] <= RANK[ceiling] else ceiling


def act_for(f: Finding, pol: Policy) -> Act:
    """The action one finding argues for, after its ceilings."""
    a: Act = pol.floors.get(f.label, DEFAULT_FLOOR)
    if f.label in CEIL:
        a = _cap(a, CEIL[f.label])
    if f.dim in CEIL_DIM:
        a = _cap(a, CEIL_DIM[f.dim])
    return a


def decide(
    fnd: list[Finding],
    pol: Policy,
    tool_names: list[str] | None = None,
) -> Act:
    """Resolve a list of findings to a single action.

    tool_names is passed separately rather than read off the findings: tool
    reversibility is a property of the call the agent wants to make, not of
    anything a detector found in the text.
    """
    acts = [act_for(f, pol) for f in fnd]
    a: Act = max(acts, key=lambda x: RANK[x]) if acts else "allow"

    # severity 3 is "a human should see this", except on the cost dimension,
    # which never gates delivery
    if any(f.sev >= 3 and f.dim != "cost" for f in fnd):
        a = "escalate"

    if pol.escalate_if.get("irrev_tool") and tool_names and tools.has_irrev(tool_names):
        # an irreversible call with any open finding is not ours to wave through
        if any(f.dim != "cost" for f in fnd):
            a = "escalate"

    # belt and braces: the per-finding ceiling above already prevents this, but
    # the rule is important enough to state twice
    if a == "block" and fnd and all(f.label == "unverifiable" for f in fnd):
        a = "annotate"

    return a


def why(
    fnd: list[Finding],
    pol: Policy,
    tool_names: list[str] | None = None,
    final: Act | None = None,
) -> dict:
    """The decision, with its reasoning. Feeds the audit record and the UI.

    `final` is the action actually taken, which can outrank what the rulebook
    alone returned -- an edit that empties the response becomes an escalate. The
    audit has to report what happened, not what would have happened.
    """
    a = decide(fnd, pol, tool_names)
    per = [
        {"label": f.label, "dim": f.dim, "sev": f.sev, "conf": f.conf,
         "floor": pol.floors.get(f.label, DEFAULT_FLOOR), "act": act_for(f, pol),
         "span": list(f.span), "side": f.side, "det": f.det, "evid": f.evid}
        for f in fnd
    ]
    driver = None
    for p in per:
        if p["act"] == a:
            driver = p["label"]
            break

    reason = "findings"
    if any(f.sev >= 3 and f.dim != "cost" for f in fnd):
        reason = "severity_3"
    if (pol.escalate_if.get("irrev_tool") and tool_names
            and tools.has_irrev(tool_names) and any(f.dim != "cost" for f in fnd)):
        reason, driver = "irreversible_tool", driver or "irrev_tool"

    if final is not None and final != a:
        reason = "edit_emptied_response" if a == "edit" else "overridden"
        a = final
        driver = driver or reason

    return {
        "act": a,
        "reason": reason,
        "driver": driver,
        "findings": per,
        "tools": tools.classify(tool_names or []),
        "pol_ver": f"{pol.tenant}:{pol.geo}:{pol.ver}",
    }
