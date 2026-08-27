"""Session risk accumulation. The score raises the tier floor, never the action."""

import pytest

from api import ledger, store
from api.schemas import Finding


def f(label, sev, dim="resp"):
    return Finding(span=(0, 3), dim=dim, label=label, sev=sev, conf=0.9, det="t")


def test_weight_sums_severity():
    assert ledger.weight([f("pii", 2), f("bias", 1)]) == 3.0


def test_cost_findings_never_accumulate():
    """Invariant 4: an expensive session is not a risky one."""
    assert ledger.weight([f("cost_anom", 3, dim="cost")]) == 0.0


def test_floor_thresholds():
    assert ledger.floor(0.0) == 0
    assert ledger.floor(ledger.T1_AT) == 1
    assert ledger.floor(ledger.T2_AT) == 2
    assert ledger.floor(ledger.T2_AT + 100) == 2


def test_decay_halves_over_the_halflife():
    v = ledger._decay(8.0, ledger.HALFLIFE_S)
    assert abs(v - 4.0) < 1e-6
    assert ledger._decay(8.0, 0.0) == 8.0


@pytest.fixture
async def redis():
    """Open the real store. Without this, bump() silently takes its exception
    path and the assertions below pass while testing nothing."""
    await store.open_store()
    try:
        await store.rd().ping()
    except Exception:
        await store.close_store()
        pytest.skip("redis not running")
    yield
    await store.close_store()


async def test_bump_accumulates_then_clears(redis):
    sess = "test-ledger-session"
    await ledger.clear(sess)
    assert await ledger.score(sess) == 0.0
    a = await ledger.bump(sess, [f("pii", 2)])
    b = await ledger.bump(sess, [f("unsafe", 3)])
    assert b > a
    assert abs(b - 5.0) < 0.1, "2 + 3 should accumulate, not overwrite"
    assert abs(await ledger.score(sess) - b) < 0.1, "score must survive the write"
    assert ledger.floor(b) >= 1
    await ledger.clear(sess)
    assert await ledger.score(sess) == 0.0


async def test_score_decays_between_turns(redis, monkeypatch):
    sess = "test-ledger-decay"
    await ledger.clear(sess)
    await ledger.bump(sess, [f("unsafe", 3)])
    monkeypatch.setattr(ledger, "HALFLIFE_S", 0.001)   # force the decay forward
    import asyncio
    await asyncio.sleep(0.02)
    assert await ledger.score(sess) < 3.0
    await ledger.clear(sess)


async def test_a_quiet_session_never_raises_the_floor(redis):
    sess = "test-ledger-quiet"
    await ledger.clear(sess)
    for _ in range(5):
        await ledger.bump(sess, [])
    assert ledger.floor(await ledger.score(sess)) == 0
    await ledger.clear(sess)
