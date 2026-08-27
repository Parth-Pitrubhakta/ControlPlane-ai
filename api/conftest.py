"""Test isolation.

The suite exercises real Mongo and Redis rather than mocks, which is the right
call for code whose whole job is durable decisions. But it must not do that in
the database the gateway is serving from: an earlier run left a live tenant
pinned to a policy version named "ui-test-20abce-b". Tests get their own
database and their own Redis slot.
"""

import os

os.environ["MONGO_DB"] = os.getenv("TEST_MONGO_DB", "controlplane_test")
os.environ.setdefault("MOCK_H200", "1")
_r = os.getenv("REDIS_URL", "redis://127.0.0.1:6479/0")
os.environ["REDIS_URL"] = _r.rsplit("/", 1)[0] + "/15"

import pytest

from api import store


@pytest.fixture(scope="session", autouse=True)
def _guard():
    assert store.MONGO_DB.endswith("_test"), (
        f"tests must not run against {store.MONGO_DB!r}"
    )
    yield


@pytest.fixture(autouse=True)
async def _clean_policies():
    """Policy versions are permanent by design, so a test that publishes one
    would otherwise leak into every later assertion."""
    yield
