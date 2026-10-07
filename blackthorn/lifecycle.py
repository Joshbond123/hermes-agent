"""Startup / shutdown hooks (called from the dashboard's lifespan)."""

from __future__ import annotations

import asyncio
import logging

from blackthorn import http as bt_http
from blackthorn import sessions
from blackthorn.d1 import D1Error, d1
from blackthorn.gpu import controller

log = logging.getLogger("blackthorn.lifecycle")


async def startup() -> None:
    """Never raises: a Blackthorn hiccup must not take the whole dashboard down."""
    try:
        closed = await sessions.recover_interrupted()
        if closed:
            log.info("closed %d turn(s) that were interrupted by the last restart", closed)
    except (D1Error, Exception) as exc:  # noqa: BLE001
        log.warning("could not recover interrupted turns: %s", exc)
    try:
        await controller.start()
    except Exception as exc:  # noqa: BLE001
        log.warning("GPU supervisor did not start: %s", exc)


async def shutdown() -> None:
    from blackthorn import api

    try:
        if api._engine is not None:
            await api._engine.shutdown()
        await controller.stop()
        await d1.aclose()
        await bt_http.aclose_all()
    except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001
        log.debug("blackthorn shutdown: %s", exc)
