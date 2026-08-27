"""The rulebook. These tests are the enforcement point for invariants 2, 3 and 4."""

import pytest

from api import policy
from api.decide import act_for, decide, why
from api.schemas import Finding, Policy


def pol(**kw) -> Policy:
    base = dict(
        tenant="CS-BOT", geo="IN", ver="test", effective_from=0.0,
        lat_budget_ms=150, thr={"med": 0.3, "high": 0.7},
        floors={"contradicted": "edit", "unverifiable": "annotate", "pii": "edit",
                "bias": "annotate", "unsafe": "block", "inject": "block",
                "cost_anom": "annotate", "format": "annotate"},
        escalate_if={"sev": 3, "irrev_tool": False},
        sample={"t2": 0.05, "t3": 0.01}, retention_days=90,
    )
    base.update(kw)
    return Policy(**base)


def f(label, sev=1, dim=None, conf=0.9, span=(0, 5), side="resp") -> Finding:
    d = dim or ("perf" if label in ("contradicted", "unverifiable")
                else "cost" if label == "cost_anom" else "resp")
    return Finding(span=span, side=side, dim=d, label=label, sev=sev,
                   conf=conf, det="test-v1")


# ---------------------------------------------------------------- basics

def test_no_findings_allows():
    assert decide([], pol()) == "allow"


def test_single_finding_takes_its_floor():
    assert decide([f("pii")], pol()) == "edit"
    assert decide([f("unsafe")], pol()) == "block"


def test_most_restrictive_wins():
    a = decide([f("bias"), f("pii"), f("unsafe")], pol())
    assert a == "block"


def test_order_does_not_matter():
    fs = [f("unsafe"), f("bias"), f("pii")]
    assert decide(fs, pol()) == decide(list(reversed(fs)), pol())


def test_unknown_label_falls_back_to_annotate_not_crash():
    assert decide([f("some_new_detector_label")], pol()) == "annotate"


# ------------------------------------------------- invariant 2: unverifiable

def test_unverifiable_alone_annotates():
    assert decide([f("unverifiable")], pol()) == "annotate"


def test_unverifiable_cannot_block_even_if_policy_says_so():
    p = pol(floors={**pol().floors, "unverifiable": "block"})
    assert decide([f("unverifiable")], p) == "annotate"


def test_unverifiable_cannot_escalate_even_if_policy_says_so():
    p = pol(floors={**pol().floors, "unverifiable": "escalate"})
    assert decide([f("unverifiable")], p) == "annotate"


def test_unverifiable_ceiling_holds_when_mixed_with_other_findings():
    # a misconfigured floor must not leak through via the max
    p = pol(floors={**pol().floors, "unverifiable": "block", "bias": "annotate"})
    assert decide([f("unverifiable"), f("bias")], p) == "annotate"


def test_unverifiable_does_not_suppress_a_real_block():
    assert decide([f("unverifiable"), f("unsafe")], pol()) == "block"


def test_high_severity_unverifiable_still_cannot_block():
    # sev 3 escalates, which is a human looking, not a silent block
    assert decide([f("unverifiable", sev=3)], pol()) == "escalate"


# -------------------------------------------------------- invariant 4: cost

def test_cost_finding_never_blocks():
    p = pol(floors={**pol().floors, "cost_anom": "block"})
    assert decide([f("cost_anom")], p) == "annotate"


def test_cost_finding_never_edits():
    p = pol(floors={**pol().floors, "cost_anom": "edit"})
    assert decide([f("cost_anom")], p) == "annotate"


def test_severity_3_on_cost_does_not_escalate():
    assert decide([f("cost_anom", sev=3)], pol()) == "annotate"


def test_cost_does_not_mask_a_real_finding():
    assert decide([f("cost_anom", sev=3), f("unsafe")], pol()) == "block"


# ------------------------------------------------------------- severity

def test_severity_3_escalates():
    assert decide([f("pii", sev=3)], pol()) == "escalate"


def test_severity_3_beats_a_lower_floor():
    assert decide([f("bias", sev=3)], pol()) == "escalate"


# ------------------------------------------------------- irreversible tools

def test_irrev_tool_escalates_under_decide_policy():
    p = pol(tenant="DECIDE", escalate_if={"sev": 3, "irrev_tool": True})
    assert decide([f("unverifiable")], p, ["billing.refund"]) == "escalate"


def test_irrev_tool_with_no_findings_does_not_escalate():
    p = pol(tenant="DECIDE", escalate_if={"sev": 3, "irrev_tool": True})
    assert decide([], p, ["billing.refund"]) == "allow"


def test_irrev_tool_ignored_when_policy_does_not_ask_for_it():
    assert decide([f("bias")], pol(), ["billing.refund"]) == "annotate"


def test_read_only_tools_never_escalate():
    p = pol(tenant="DECIDE", escalate_if={"sev": 3, "irrev_tool": True})
    assert decide([f("bias")], p, ["crm.contact.read", "kb.search"]) == "annotate"


def test_cost_finding_alone_does_not_gate_an_irreversible_tool():
    p = pol(tenant="DECIDE", escalate_if={"sev": 3, "irrev_tool": True})
    assert decide([f("cost_anom")], p, ["billing.refund"]) == "annotate"


# ------------------------------------------------------------ geo behaviour

def test_geo_flip_changes_the_action_on_the_same_findings():
    """Demo step 5: same response, IN vs EU, different action."""
    pols = {(p.tenant, p.geo): p for p in policy.defaults()}
    fnd = [f("pii", sev=2)]
    a_in = decide(fnd, pols[("CS-BOT", "IN")])
    a_eu = decide(fnd, pols[("CS-BOT", "EU")])
    assert a_in == "edit"
    assert a_eu == "block"
    assert a_in != a_eu


def test_decide_tenant_is_strictest_on_contradiction():
    pols = {(p.tenant, p.geo): p for p in policy.defaults()}
    fnd = [f("contradicted", sev=2)]
    assert decide(fnd, pols[("KB-COPILOT", "IN")]) == "annotate"
    assert decide(fnd, pols[("CS-BOT", "IN")]) == "edit"
    assert decide(fnd, pols[("DECIDE", "IN")]) == "escalate"


def test_every_default_policy_respects_the_two_ceilings():
    # the ceiling is a maximum, not a fixed value: a permissive tenant is free
    # to allow unverifiable outright, it just may never exceed annotate
    from api.schemas import RANK
    for p in policy.defaults():
        assert RANK[act_for(f("unverifiable"), p)] <= RANK["annotate"], p.tenant
        assert RANK[act_for(f("cost_anom"), p)] <= RANK["annotate"], p.tenant


# ------------------------------------------------------------------- why

def test_why_reports_the_driving_finding():
    d = why([f("bias"), f("unsafe")], pol())
    assert d["act"] == "block"
    assert d["driver"] == "unsafe"
    assert d["pol_ver"] == "CS-BOT:IN:test"
    assert len(d["findings"]) == 2


def test_why_classifies_tools():
    d = why([f("bias")], pol(), ["crm.read", "billing.refund"])
    assert d["tools"] == {"crm.read": "ro", "billing.refund": "irrev"}


def test_no_scalar_score_is_accepted_by_decide():
    """Invariant 3: the signature takes findings, never a risk float."""
    import inspect
    params = list(inspect.signature(decide).parameters)
    assert params == ["fnd", "pol", "tool_names"]


def test_why_reports_the_action_actually_taken_not_the_rulebook_one():
    # an edit that empties the response is escalated; the audit must say so
    d = why([f("contradicted", sev=2)], pol(), final="escalate")
    assert d["act"] == "escalate"
    assert d["reason"] == "edit_emptied_response"


def test_why_names_the_irreversible_tool_as_the_reason():
    p = pol(tenant="DECIDE", escalate_if={"sev": 3, "irrev_tool": True})
    d = why([f("unverifiable")], p, ["billing.refund"])
    assert d["act"] == "escalate"
    assert d["reason"] == "irreversible_tool"


def test_why_names_severity_as_the_reason():
    d = why([f("pii", sev=3)], pol())
    assert d["act"] == "escalate" and d["reason"] == "severity_3"


def test_why_reason_is_findings_for_an_ordinary_decision():
    assert why([f("bias")], pol())["reason"] == "findings"


async def test_republished_policy_actually_takes_effect():
    """A document posted back from the editor carries the old effective_from.
    Without restamping, the new version loses the sort and never activates."""
    from api import store
    await store.open_store()
    try:
        await policy.seed()
        await policy.refresh()
        cur = policy.get("CS-BOT", "US")
        new = cur.model_copy(deep=True)
        new.ver = "restamp-test"
        new.floors = {**cur.floors, "bias": "block"}
        # keep the old timestamp exactly as the editor would post it back
        new.effective_from = cur.effective_from
        await policy.put(new)
        assert policy.get("CS-BOT", "US").ver == "restamp-test"
        assert policy.get("CS-BOT", "US").floors["bias"] == "block"
    finally:
        await store.close_store()
