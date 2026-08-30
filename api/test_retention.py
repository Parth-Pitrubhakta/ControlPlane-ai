"""Retention. The policy promises a deletion window; this is what keeps it."""

import time

import pytest

from api import policy, retention, store


@pytest.fixture
async def db():
    await store.open_store()
    await policy.seed()
    await policy.refresh()
    yield store.db()
    await store.db()["traces"].delete_many({"id": {"$regex": "^ret-test-"}})
    await store.close_store()


def _trace(i, tenant, geo, age_days):
    return {"id": f"ret-test-{tenant}-{geo}-{i}", "sess": "ret", "tenant": tenant,
            "geo": geo, "ts": time.time() - age_days * 86400, "prompt": "x",
            "resp": "y", "fnd": [], "act": "allow", "tier": 0, "risk": 0.0,
            "lat": {}, "pol_ver": "t"}


async def test_deletes_only_what_is_past_its_own_window(db):
    await db["traces"].delete_many({"id": {"$regex": "^ret-test-"}})
    # EU keeps 30 days, IN keeps 90. Both rows are 40 days old.
    await db["traces"].insert_many(
        [_trace(i, "CS-BOT", "EU", 40) for i in range(3)]
        + [_trace(i, "CS-BOT", "IN", 40) for i in range(3)])
    await retention.sweep(dry=False)
    assert await db["traces"].count_documents({"id": {"$regex": "^ret-test-CS-BOT-EU-"}}) == 0
    assert await db["traces"].count_documents({"id": {"$regex": "^ret-test-CS-BOT-IN-"}}) == 3


async def test_dry_run_deletes_nothing(db):
    await db["traces"].delete_many({"id": {"$regex": "^ret-test-"}})
    await db["traces"].insert_many([_trace(i, "CS-BOT", "EU", 40) for i in range(3)])
    r = await retention.sweep(dry=True)
    assert r["dry"] is True
    assert r["total_deleted"] == 0
    assert await db["traces"].count_documents({"id": {"$regex": "^ret-test-"}}) == 3


async def test_recent_traces_are_never_touched(db):
    await db["traces"].delete_many({"id": {"$regex": "^ret-test-"}})
    await db["traces"].insert_many([_trace(i, "CS-BOT", "EU", 1) for i in range(3)])
    await retention.sweep(dry=False)
    assert await db["traces"].count_documents({"id": {"$regex": "^ret-test-"}}) == 3


async def test_feedback_expires_with_the_trace_it_describes(db):
    """Feedback quotes the finding text, so it carries the same data and must
    not outlive the trace."""
    await db["traces"].delete_many({"id": {"$regex": "^ret-test-"}})
    await db["traces"].insert_one(_trace(0, "CS-BOT", "EU", 40))
    await db["feedback"].insert_one({"id": "fb-ret-test", "trace": "ret-test-CS-BOT-EU-0",
                                     "ts": time.time() - 40 * 86400, "items": []})
    await retention.sweep(dry=False)
    assert await db["feedback"].count_documents({"id": "fb-ret-test"}) == 0


async def test_every_seeded_policy_declares_a_window():
    for p in policy.defaults():
        assert p.retention_days > 0, f"{p.tenant}/{p.geo} has no retention window"
