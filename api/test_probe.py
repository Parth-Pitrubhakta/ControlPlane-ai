"""Sampled probes and tier 3. Both run off the response path, and must stay there."""

import pytest

from api import probe
from api.schemas import Finding, Policy, Trace


@pytest.fixture(autouse=True)
def no_io(monkeypatch):
    """These tests pin which probes get scheduled, not what the probes do.
    Without this they fire real GPU and network calls that outlive the loop."""
    fired: list[str] = []
    monkeypatch.setattr(probe, "_bg", lambda coro: (coro.close(), fired.append("x")))
    return fired


def pol(t2=0.0, t3=0.0) -> Policy:
    return Policy(tenant="CS-BOT", geo="IN", ver="t", effective_from=0.0,
                  lat_budget_ms=150, thr={"med": 0.3, "high": 0.7}, floors={},
                  escalate_if={}, sample={"t2": t2, "t3": t3}, retention_days=30)


def tr(resp="The warranty is 3 years.", tier=0) -> Trace:
    t = Trace(id="t-x", sess="s", tenant="CS-BOT", geo="IN", ts=0.0,
              prompt="How long?", resp=resp)
    t.tier = tier
    return t


def test_no_probe_when_the_sample_rate_is_zero():
    assert probe.maybe(tr(), pol(t2=0.0, t3=0.0), "allow", rnd=0.5) == []


def test_deep_probe_fires_below_the_sample_rate():
    assert "deep" in probe.maybe(tr(tier=0), pol(t2=0.5), "allow", rnd=0.1)


def test_deep_probe_does_not_fire_above_the_sample_rate():
    assert "deep" not in probe.maybe(tr(tier=0), pol(t2=0.05), "allow", rnd=0.9)


def test_deep_probe_is_pointless_when_the_router_already_went_deep():
    """Probing tier 2 with tier 2 measures nothing."""
    assert "deep" not in probe.maybe(tr(tier=2), pol(t2=1.0), "allow", rnd=0.0)


def test_escalation_always_gets_a_second_opinion():
    """A human is about to spend time on this, so the judge is worth it
    regardless of the sample rate."""
    assert "judge" in probe.maybe(tr(), pol(t3=0.0), "escalate", rnd=0.99)


def test_judge_fires_on_the_sampled_slice_too():
    assert "judge" in probe.maybe(tr(), pol(t3=0.5), "allow", rnd=0.1)


def test_empty_response_is_never_probed():
    assert probe.maybe(tr(resp="   "), pol(t2=1.0, t3=1.0), "allow", rnd=0.0) == []


def test_probe_can_be_switched_off_entirely(monkeypatch):
    monkeypatch.setattr(probe, "ON", False)
    assert probe.maybe(tr(), pol(t2=1.0, t3=1.0), "escalate", rnd=0.0) == []


def test_maybe_never_returns_an_action():
    """Invariant: a probe informs analysis and the reviewer. It cannot change
    what the user already received."""
    out = probe.maybe(tr(), pol(t2=1.0, t3=1.0), "allow", rnd=0.0)
    assert isinstance(out, list)
    assert all(isinstance(x, str) for x in out)
    assert set(out) <= {"deep", "judge"}
