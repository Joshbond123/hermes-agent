"""Shared pooled HTTP clients (one per event loop)."""

from __future__ import annotations

import asyncio
from typing import Dict

import httpx

_clients: Dict[str, httpx.AsyncClient] = {}
_loops: Dict[str, asyncio.AbstractEventLoop] = {}


def client(name: str = "tunnel") -> httpx.AsyncClient:
    """Keep-alive client for the GPU tunnel.

    Reads are allowed to idle for a long time because the model may spend a while on
    prompt processing before the first token; the engine adds its own idle watchdog.
    """
    loop = asyncio.get_running_loop()
    existing = _clients.get(name)
    if existing is None or existing.is_closed or _loops.get(name) is not loop:
        _clients[name] = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=240.0, write=60.0, pool=10.0),
            limits=httpx.Limits(max_keepalive_connections=8, keepalive_expiry=60.0),
            headers={"User-Agent": "blackthorn/1.0"},
        )
        _loops[name] = loop
    return _clients[name]


async def aclose_all() -> None:
    for c in list(_clients.values()):
        if not c.is_closed:
            await c.aclose()
    _clients.clear()
    _loops.clear()
