"""Durable GPU relay: a stable endpoint for the Kaggle gateway without any tunnel service.

The notebook opens no inbound ports; instead its relay client long-polls ``/api/kaggle-relay/pull``
for work, executes each request against the local gateway and streams the response back through
``/api/kaggle-relay/push/{id}``. Agent traffic enters at ``/gpu-relay/{path}`` and is piped through,
so the published GPU URL never changes and a dropped connection cannot orphan a request.

This is the fallback path when no Cloudflare tunnel token is configured; a named tunnel
(``CLOUDFLARED_TUNNEL_TOKEN``) still takes precedence on the notebook side.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

PULL_WAIT_S = 25.0             # notebook long-poll window (below every proxy timeout)
HEADERS_TIMEOUT_S = 240.0      # a warm model answers far faster; queued jobs start instantly
IDLE_FAST_FAIL_S = 75.0        # no notebook seen for this long -> fail fast instead of hanging
KEY_CACHE_TTL_S = 30.0          # ride out D1 blips without 401ing a healthy notebook
DROP_HEADERS = {"host", "content-length", "connection", "accept-encoding", "transfer-encoding",
                "x-blackthorn-key", "x-relay-status", "x-relay-content-type"}


@dataclass
class _Job:
    id: str
    method: str
    path: str
    headers: Dict[str, str]
    body: bytes
    chunks: "asyncio.Queue[Optional[bytes]]" = field(default_factory=lambda: asyncio.Queue(maxsize=256))
    headers_ready: "asyncio.Event" = field(default_factory=asyncio.Event)
    status: int = 200
    content_type: str = "application/octet-stream"
    pushed: bool = False


class RelayHub:
    """One GPU, one queue. Multiple notebooks are not a supported topology (and not needed)."""

    def __init__(self) -> None:
        self.pending: "asyncio.Queue[_Job]" = asyncio.Queue()
        self.inflight: Dict[str, _Job] = {}
        self.last_seen: float = time.monotonic()
        self.waiting_pulls = 0

    def alive(self) -> bool:
        return self.waiting_pulls > 0 or (time.monotonic() - self.last_seen) < IDLE_FAST_FAIL_S

    def submit(self, method: str, path: str, headers: Dict[str, str], body: bytes) -> _Job:
        job = _Job(id=uuid.uuid4().hex, method=method, path=path, headers=headers, body=body)
        self.inflight[job.id] = job
        self.pending.put_nowait(job)
        return job

    def finish(self, job: _Job) -> None:
        self.inflight.pop(job.id, None)
        job.headers_ready.set()          # release any waiter even if the notebook vanished


CHANNELS = ("model", "computer")


def _hub(request: Request, channel: str = "model") -> RelayHub:
    """One RelayHub per role channel so the model host and the computer host never
    steal each other's jobs."""
    services = request.app.state.bt
    hubs = getattr(services, "relay_hubs", None)
    if hubs is None:
        hubs = {}
        services.relay_hubs = hubs
    hub = hubs.get(channel)
    if hub is None:
        hub = RelayHub()
        hubs[channel] = hub
    return hub


_KEY_CACHE = {"t": 0.0, "v": ()}


async def _row_keys(request: Request) -> tuple:
    """The per-boot keys the notebooks publish to D1 (model row AND computer row).
    Each host authenticates with its own key; a D1 read blip must not 401 a healthy
    notebook, so both keys are cached briefly.
    """
    now = time.monotonic()
    if _KEY_CACHE["v"] and now - _KEY_CACHE["t"] < KEY_CACHE_TTL_S:
        return tuple(_KEY_CACHE["v"])
    keys = set()
    try:
        store = request.app.state.bt.store
        for getter in ("gpu_row", "computer_row"):
            try:
                row = await getattr(store, getter)()
                value = str(row.get("api_key") or "")
                if value:
                    keys.add(value)
            except Exception:  # noqa: BLE001 - one row missing must not lose the other
                pass
        if keys:
            _KEY_CACHE.update(t=now, v=tuple(keys))
        return tuple(keys)
    except Exception:
        return tuple(_KEY_CACHE["v"] or ())


def _authed(request: Request, expected: tuple) -> bool:
    got = request.headers.get("x-blackthorn-key") or ""
    return bool(got) and got in set(expected)


# --------------------------------------------------------------------------- control plane (notebook side)
async def relay_pull(request: Request, channel: str = "model"):
    """Long-poll: hand the next queued request to the notebook, or 204 when quiet."""
    if channel not in CHANNELS:
        channel = "model"
    hub = _hub(request, channel)
    if not _authed(request, await _row_keys(request)):
        raise HTTPException(status_code=401, detail={"code": "relay_auth", "message": "unknown relay key"})
    hub.last_seen = time.monotonic()
    hub.waiting_pulls += 1
    try:
        # poll instead of wait_for(queue.get()): a cancelled getter can swallow a job that
        # arrived in the same instant as the timeout, and the request would then wait forever
        job = None
        deadline = time.monotonic() + PULL_WAIT_S
        while job is None:
            try:
                job = hub.pending.get_nowait()
            except asyncio.QueueEmpty:
                if time.monotonic() >= deadline:
                    return JSONResponse({"quiet": True}, status_code=204)
                await asyncio.sleep(0.25)
    finally:
        hub.waiting_pulls -= 1
    hub.last_seen = time.monotonic()
    return JSONResponse({
        "id": job.id, "method": job.method, "path": job.path,
        "headers": job.headers,
        "body_b64": base64.b64encode(job.body).decode() if job.body else "",
    })


async def relay_push(request: Request, job_id: str):
    """Stream the gateway's response back; the body is forwarded chunk for chunk."""
    services = request.app.state.bt
    hubs = getattr(services, "relay_hubs", {}) or {}
    hub = None
    for candidate in hubs.values():
        if job_id in candidate.inflight:
            hub = candidate
            break
    if hub is None:
        hub = _hub(request)
    if not _authed(request, await _row_keys(request)):
        raise HTTPException(status_code=401, detail={"code": "relay_auth", "message": "unknown relay key"})
    hub.last_seen = time.monotonic()
    job = hub.inflight.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={"code": "relay_gone", "message": "no such relay request"})
    try:
        job.status = int(request.headers.get("x-relay-status") or 200)
        job.content_type = request.headers.get("x-relay-content-type") or "application/octet-stream"
        job.pushed = True
        job.headers_ready.set()
        async for chunk in request.stream():
            if hub.inflight.get(job_id) is not job:
                # the consumer is gone (cancelled run, timeout): tell the notebook to stop
                # streaming into a dead job - its upstream call then closes and the model frees
                return JSONResponse({"dropped": True}, status_code=410)
            if chunk:
                await job.chunks.put(chunk)
    except Exception:
        # network blip mid-push: end the stream so the agent sees a clean error instead of a hang
        pass
    finally:
        await job.chunks.put(None)
        hub.finish(job)
    return JSONResponse({"ok": True})


# --------------------------------------------------------------------------- data plane (agent side)
async def relay_proxy(request: Request, path: str = ""):
    """Any method, any path under /gpu-relay/: pipe it to the notebook gateway and stream back.

    ``/gpu-relay/computer/...`` selects the computer channel (the leading segment is
    stripped before forwarding); everything else rides the model channel for
    backward compatibility with the published model URL.
    """
    channel = "model"
    for candidate in CHANNELS:
        if path == candidate or path.startswith(candidate + "/"):
            channel = candidate
            path = path[len(candidate):].lstrip("/")
            break
    if path.startswith("api/kaggle-relay"):
        # the control plane lives on the app origin; asking for it through the data plane
        # would queue a request for the very client that is asking
        raise HTTPException(status_code=404, detail={"code": "relay_path", "message": "control plane is /api/kaggle-relay/*"})
    hub = _hub(request, channel)
    if not hub.alive():
        raise HTTPException(status_code=502,
                            detail={"code": "gpu_unreachable",
                                    "message": "The GPU relay is not connected. Wait for the GPU to finish starting, then try again."})
    body = await request.body()
    headers = {k.lower(): v for k, v in request.headers.items()
               if k.lower() not in DROP_HEADERS and not k.lower().startswith("x-relay-")}
    job = hub.submit(request.method.upper(), path, headers, body)

    try:
        await asyncio.wait_for(job.headers_ready.wait(), timeout=HEADERS_TIMEOUT_S)
    except asyncio.TimeoutError:
        hub.finish(job)
        raise HTTPException(status_code=502,
                            detail={"code": "gpu_unreachable",
                                    "message": "The GPU did not answer through the relay (timed out)."})
    if not job.pushed:
        hub.finish(job)
        raise HTTPException(status_code=502,
                            detail={"code": "gpu_unreachable",
                                    "message": "The GPU relay lost the notebook before it answered."})

    async def stream() -> AsyncIterator[bytes]:
        try:
            while True:
                chunk = await job.chunks.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            hub.finish(job)

    return StreamingResponse(stream(), status_code=job.status, media_type=job.content_type,
                             headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


def register(router: APIRouter) -> None:
    router.add_api_route("/api/kaggle-relay/pull", relay_pull, methods=["GET"])
    router.add_api_route("/api/kaggle-relay/push/{job_id}", relay_push, methods=["POST"])
    router.add_api_route("/gpu-relay/{path:path}", relay_proxy,
                         methods=["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    router.add_api_route("/gpu-relay", relay_proxy, methods=["GET", "HEAD", "POST"], include_in_schema=False)
