"""Mongo and Redis accessors. Single place that knows connection details."""

from __future__ import annotations

import os
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from redis.asyncio import Redis

from api.schemas import Trace

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "controlplane")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_mc: AsyncIOMotorClient | None = None
_rd: Redis | None = None


def db() -> AsyncIOMotorDatabase:
    if _mc is None:
        raise RuntimeError("store not opened")
    return _mc[MONGO_DB]


def rd() -> Redis:
    if _rd is None:
        raise RuntimeError("store not opened")
    return _rd


async def open_store() -> None:
    global _mc, _rd
    _mc = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=2000, tz_aware=False)
    _rd = Redis.from_url(REDIS_URL, decode_responses=True)


async def close_store() -> None:
    global _mc, _rd
    if _mc is not None:
        _mc.close()
        _mc = None
    if _rd is not None:
        await _rd.aclose()
        _rd = None


async def ensure_idx() -> None:
    """Indexes the dashboard and review queue read through."""
    t = db()["traces"]
    await t.create_index([("sess", 1), ("ts", -1)])
    await t.create_index([("tenant", 1), ("ts", -1)])
    await t.create_index([("act", 1), ("ts", -1)])
    await t.create_index([("id", 1)], unique=True)


async def put_trace(tr: Trace) -> None:
    await db()["traces"].replace_one({"id": tr.id}, tr.model_dump(), upsert=True)


async def get_trace(tid: str) -> dict[str, Any] | None:
    return await db()["traces"].find_one({"id": tid}, {"_id": 0})


async def ping() -> dict[str, Any]:
    """Component health, never raises."""
    out: dict[str, Any] = {"mongo": "down", "redis": "down"}
    try:
        if _mc is not None:
            await _mc.admin.command("ping")
            out["mongo"] = "up"
    except Exception as e:
        out["mongo"] = f"down: {type(e).__name__}"
    try:
        if _rd is not None:
            await _rd.ping()
            out["redis"] = "up"
    except Exception as e:
        out["redis"] = f"down: {type(e).__name__}"
    return out
