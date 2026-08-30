"""FastAPI app factory, lifespan, route mounting."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import detclient, gateway, policy, router, store
from api.routes import admin, metrics, review

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def setup_log() -> None:
    logging.basicConfig(format="%(message)s", level=getattr(logging, LOG_LEVEL, logging.INFO))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, LOG_LEVEL, logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await store.open_store()
    await detclient.open_det()
    await gateway.open_gw()
    await policy.open_policy()
    if not router.load():
        log.warning('router_untrained', note='every request falls back to the tier floor')
    try:
        await store.ensure_idx()
    except Exception as e:
        # a Mongo outage must not stop the gateway booting
        log.warning("idx_failed", err=str(e))
    log.info("boot", mock_h200=detclient.MOCK, det_url=detclient.DET_URL)
    yield
    await policy.close_policy()
    await gateway.close_gw()
    await detclient.close_det()
    await store.close_store()


def create_app() -> FastAPI:
    setup_log()
    app = FastAPI(title="ControlPlane.ai gateway", version="0.1.0", lifespan=lifespan)
    # the dashboard runs on its own dev-server port
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", **await store.ping(), **await detclient.health(),
                "router": router.meta(),
                "policies": len(policy._cache)}

    app.include_router(gateway.router)
    app.include_router(metrics.router)
    app.include_router(review.router)
    app.include_router(admin.router)
    return app


app = create_app()
