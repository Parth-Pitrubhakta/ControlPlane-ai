"""Dashboard, review and admin routes.

The recalibration logic gets real coverage because it rewrites policy; the
read-only aggregates get light coverage, as the spec allows.
"""

import os

import pytest

os.environ.setdefault("MOCK_H200", "1")

import httpx

from api import policy, store
from api.main import create_app


@pytest.fixture
async def cli():
    app = create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            yield c


async def _seed_trace(cli, prompt="Card 4111 1111 1111 1111 please.", tenant="CS-BOT"):
    r = await cli.post("/v1/chat/completions",
                       headers={"X-CP-Tenant": tenant},
                       json={"model": "Qwen/Qwen2.5-7B-Instruct",
                             "messages": [{"role": "user", "content": prompt}]})
    return r.headers["x-cp-trace"]


# ------------------------------------------------------------------ metrics

async def test_summary_shape(cli):
    await _seed_trace(cli)
    d = (await cli.get("/api/metrics/summary?window=24h")).json()
    for k in ("n", "overhead", "tiers", "acts", "dims", "t1_rate", "alerts_per_100"):
        assert k in d
    assert set(d["overhead"]) == {"p50", "p95", "max"}


async def test_summary_overhead_excludes_generation(cli):
    """Generation time is the model's, not the checker's. Claiming it would
    make our overhead look worse, and hiding t1 would make it look better."""
    from api.routes.metrics import _overhead
    assert _overhead({"gen": 900.0, "t0": 1.0, "t1": 50.0, "decide": 0.5}) == 51.5


async def test_traces_list_and_detail(cli):
    tid = await _seed_trace(cli)
    lst = (await cli.get("/api/traces?limit=10")).json()
    assert lst["n"] >= 1
    assert all("chips" in r for r in lst["rows"])
    one = (await cli.get(f"/api/traces/{tid}")).json()
    assert one["id"] == tid and "fnd" in one


async def test_traces_filter_by_tenant(cli):
    await _seed_trace(cli, tenant="DECIDE")
    d = (await cli.get("/api/traces?tenant=DECIDE&limit=20")).json()
    assert all(r["tenant"] == "DECIDE" for r in d["rows"])


async def test_series_returns_requested_buckets(cli):
    d = (await cli.get("/api/metrics/series?window=1h&buckets=8")).json()
    assert len(d["buckets"]) == 8


# ------------------------------------------------------------------- review

async def test_verdict_writes_feedback_and_marks_the_trace(cli):
    tid = await _seed_trace(cli)
    t = await store.get_trace(tid)
    if not t["fnd"]:
        pytest.skip("seed produced no findings")
    r = (await cli.post(f"/api/review/{tid}", json={
        "reviewer": "t", "act_agree": False,
        "items": [{"idx": 0, "agree": False, "note": "wrong"}]})).json()
    assert r["ok"] and r["feedback"].startswith("fb-")
    again = await store.get_trace(tid)
    assert again["ovr"]["reviewer"] == "t"
    assert 0 in again["ovr"]["disagreed"]


async def test_verdict_ignores_out_of_range_finding_index(cli):
    tid = await _seed_trace(cli)
    r = (await cli.post(f"/api/review/{tid}", json={
        "reviewer": "t", "items": [{"idx": 999, "agree": False}]})).json()
    assert r["ok"]
    assert r["ovr"]["disagreed"] == []


async def test_verdict_on_missing_trace_is_not_fatal(cli):
    r = (await cli.post("/api/review/nope", json={"reviewer": "t", "items": []})).json()
    assert r["error"] == "not found"


# -------------------------------------------------------------- recalibrate

async def test_recalibrate_needs_enough_feedback(cli):
    await store.db()["feedback"].delete_many({})
    await store.db()["feedback"].insert_one({
        "tenant": "CS-BOT", "items": [
            {"label": "bias", "agree": False, "conf": 0.9}]})
    d = (await cli.post("/api/recalibrate?dry=true")).json()
    assert "not enough feedback" in d["moves"]["bias"]["skipped"]


async def test_recalibrate_will_not_silence_a_detector_on_fp_only_evidence(cli):
    """Without a confirmed true positive every cut trivially keeps 100% of
    them, so a handful of complaints could disable a detector outright."""
    await store.db()["feedback"].delete_many({})
    await store.db()["feedback"].insert_many([
        {"tenant": "CS-BOT", "items": [{"label": "bias", "agree": False, "conf": 0.9}]}
        for _ in range(6)])
    d = (await cli.post("/api/recalibrate?dry=true")).json()
    assert "no confirmed true positives" in d["moves"]["bias"]["skipped"]


async def test_recalibrate_caps_the_move_without_true_positives(cli):
    from api.routes.admin import NO_TP_CAP
    await store.db()["feedback"].delete_many({})
    await store.db()["feedback"].insert_many([
        {"tenant": "CS-BOT", "items": [{"label": "bias", "agree": False, "conf": 0.99}]}
        for _ in range(10)])
    d = (await cli.post("/api/recalibrate?dry=true")).json()
    assert d["moves"]["bias"]["new_thr"] == NO_TP_CAP
    assert "capped" in d["moves"]["bias"]["warn"]


async def test_recalibrate_protects_confirmed_true_positives(cli):
    await store.db()["feedback"].delete_many({})
    docs = [{"tenant": "CS-BOT", "items": [{"label": "bias", "agree": True, "conf": 0.95}]}
            for _ in range(10)]
    docs += [{"tenant": "CS-BOT", "items": [{"label": "bias", "agree": False, "conf": 0.40}]}
             for _ in range(6)]
    await store.db()["feedback"].insert_many(docs)
    d = (await cli.post("/api/recalibrate?dry=true")).json()
    m = d["moves"]["bias"]
    assert m["new_thr"] <= 0.95, "must not cut above the confirmed true positives"
    assert m["tp_kept"] >= 0.95
    assert m["fp_after"] < m["fp_before"]


async def test_recalibrate_dry_run_publishes_nothing(cli):
    before = (await cli.get("/api/policies/CS-BOT/IN")).json()["ver"]
    await cli.post("/api/recalibrate?dry=true")
    after = (await cli.get("/api/policies/CS-BOT/IN")).json()["ver"]
    assert before == after


# ------------------------------------------------------------------ policy

async def test_policy_editor_rejects_a_duplicate_version(cli):
    cur = (await cli.get("/api/policies/CS-BOT/IN")).json()
    cur.pop("pol_ver", None)
    r = (await cli.post("/api/policies", json=cur)).json()
    assert "already active" in r["error"]


async def test_policy_editor_rejects_malformed_documents(cli):
    r = (await cli.post("/api/policies", json={"tenant": "CS-BOT"})).json()
    assert "invalid policy" in r["error"]


async def test_policy_editor_publishes_and_takes_effect(cli):
    import uuid
    tag = uuid.uuid4().hex[:6]      # versions are permanent, so never reuse one
    cur = (await cli.get("/api/policies/CS-BOT/IN")).json()
    cur.pop("pol_ver", None)
    was = cur["floors"].get("bias")
    cur["ver"] = f"ui-test-{tag}-a"
    cur["floors"] = {**cur["floors"], "bias": "block"}
    r = (await cli.post("/api/policies", json=cur)).json()
    assert r.get("ok"), r
    assert policy.get("CS-BOT", "IN").floors["bias"] == "block"
    assert policy.get("CS-BOT", "IN").ver == f"ui-test-{tag}-a"
    # put it back so the rest of the suite sees the original posture
    cur["ver"] = f"ui-test-{tag}-b"
    cur["floors"] = {**cur["floors"], "bias": was}
    await cli.post("/api/policies", json=cur)
    assert policy.get("CS-BOT", "IN").floors["bias"] == was
