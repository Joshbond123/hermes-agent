"""Durable GPU relay: a stable endpoint for the Kaggle gateway without any tunnel service.

The notebook opens no inbound ports; instead its relay client long-polls ``/api/kaggle-relay/pull``
for work, executes each request against the local gateway and streams the response back through
``/api/kaggle-relay/push/{id}``. Agent traffic enters at ``/gpu-relay/{path}`` and is piped through,
so the published GPU URL never changes and a dropped connection cannot orphan a request.

Failure rules (each one is a bug that was reproduced in production before it was fixed here):

* A request whose caller has gone (timeout, cancelled run, dropped client) is **abandoned**: it is
  never handed to the notebook afterwards. Before, abandoned jobs stayed in the queue and the GPU
  spent minutes answering nobody, which is what made later requests time out.
* A notebook push must never block forever. When the consumer disappears, the queue is closed and the
  push is answered with 410 at once, so the notebook worker is released. Before, a full chunk queue
  with no consumer left the worker blocked for good; after three of them every request hung.
* Relay health is recorded per request (answers, header timeouts) so the controller can report a stalled
  link instead of "online".

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
from fastapi.responses import JSONResponse, Response, StreamingResponse

PULL_WAIT_S = 25.0             # notebook long-poll window (below every proxy timeout)
HEADERS_TIMEOUT_S = 90.0       # fail faster when the notebook is stuck; warm model answers in seconds
IDLE_FAST_FAIL_S = 40.0        # no notebook seen for this long -> fail fast instead of hanging
KEY_CACHE_TTL_S = 5.0          # refresh cheaply at each long poll; avoid stale boot keys
CONSUMER_IDLE_ABANDON_S = 120.0  # a full chunk queue with no consumer read for this long -> abandon the job
PUSH_PUT_SLICE_S = 1.0         # how often a blocked push re-checks whether its consumer is gone
STALL_AFTER_TIMEOUTS = 2       # consecutive header timeouts that mark the link as stalled
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
    notebook_key: str = ""               # pinned when assigned; rotation must not break a live stream
    closed: bool = False                 # the consumer is gone or the job was abandoned: never deliver it
    last_consumed: float = field(default_factory=time.monotonic)


class RelayHub:
    """One GPU, one queue. Multiple notebooks are not a supported topology (and not needed)."""

    def __init__(self, channel: str = "model") -> None:
        self.channel = channel
        self.pending: "asyncio.Queue[_Job]" = asyncio.Queue()
        self.inflight: Dict[str, _Job] = {}
        self.last_seen: float = time.monotonic()
        self.waiting_pulls = 0
        # health: what the controller needs to tell "busy" from "not answering"
        self.consecutive_timeouts = 0
        self.last_answer_at: float = 0.0
        self.last_timeout_at: float = 0.0
        self.abandoned = 0

    def alive(self) -> bool:
        return self.waiting_pulls > 0 or (time.monotonic() - self.last_seen) < IDLE_FAST_FAIL_S

    def stalled(self) -> bool:
        return self.consecutive_timeouts >= STALL_AFTER_TIMEOUTS

    def health(self) -> Dict[str, Any]:
        now = time.monotonic()
        return {
            "alive": self.alive(),
            "stalled": self.stalled(),
            "consecutive_timeouts": self.consecutive_timeouts,
            "seconds_since_answer": round(now - self.last_answer_at, 1) if self.last_answer_at else None,
            "seconds_since_notebook": round(now - self.last_seen, 1),
            "abandoned_jobs": self.abandoned,
            "queued": self.pending.qsize(),
        }

    def submit(self, method: str, path: str, headers: Dict[str, str], body: bytes) -> _Job:
        job = _Job(id=uuid.uuid4().hex, method=method, path=path, headers=headers, body=body)
        self.inflight[job.id] = job
        self.pending.put_nowait(job)
        return job

    def finish(self, job: _Job) -> None:
        """Retire a job from the consumer side. Nothing else may deliver it afterwards."""
        job.closed = True
        self.inflight.pop(job.id, None)
        job.headers_ready.set()          # release any waiter even if the notebook vanished

    def abandon(self, job: _Job) -> None:
        self.abandoned += 1
        self.finish(job)

    def deliverable(self, job: _Job) -> bool:
        return (not job.closed) and self.inflight.get(job.id) is job


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
        hub = RelayHub(channel)
        hubs[channel] = hub
    return hub


_KEY_CACHE: Dict[str, tuple[float, str]] = {}


async def _row_keys(request: Request, channel: str = "model", *, force: bool = False) -> tuple[str, ...]:
    """Only the D1 key of this role may poll its channel. Do not mix both hosts.

    A short cache tolerates transient D1 failures; the job itself is additionally
    pinned to the agent's bearer key before it leaves the queue.
    """
    channel = channel if channel in CHANNELS else "model"
    if channel == "computer" and __import__("os").environ.get("PANDASTACK_API_KEY"):
        return ()  # old Kaggle computer cannot poll after the provider cutover
    now = time.monotonic()
    cached_at, cached = _KEY_CACHE.get(channel, (0.0, ""))
    if not force and cached and now - cached_at < KEY_CACHE_TTL_S:
        return (cached,)
    try:
        store = request.app.state.bt.store
        row = await (store.computer_row() if channel == "computer" else store.gpu_row())
        key = str(row.get("api_key") or "")
        if key:
            _KEY_CACHE[channel] = (now, key)
            return (key,)
        return ()                            # an empty D1 row must revoke old access
    except Exception:  # noqa: BLE001 - only a read failure permits the last known key
        return (cached,) if cached and now - cached_at < 60.0 else ()


def _authed(request: Request, expected: tuple) -> bool:
    got = request.headers.get("x-blackthorn-key") or ""
    return bool(got) and got in set(expected)


async def _put_chunk(job: _Job, item: Optional[bytes], hub: RelayHub) -> bool:
    """Queue one chunk for the consumer. Never blocks forever: gives up once the consumer is gone."""
    while True:
        if job.closed:
            return False
        if time.monotonic() - job.last_consumed > CONSUMER_IDLE_ABANDON_S and job.chunks.full():
            hub.abandon(job)            # nobody has read for minutes and the queue is full: the caller is gone
            return False
        try:
            await asyncio.wait_for(job.chunks.put(item), timeout=PUSH_PUT_SLICE_S)
            return True
        except asyncio.TimeoutError:
            continue


# --------------------------------------------------------------------------- control plane (notebook side)
async def relay_pull(request: Request, channel: str = "model"):
    """Long-poll: hand the next live queued request to the notebook, or 204 when quiet."""
    if channel not in CHANNELS:
        channel = "model"
    hub = _hub(request, channel)
    if not _authed(request, await _row_keys(request, channel)) and not _authed(
            request, await _row_keys(request, channel, force=True)):
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
                candidate = hub.pending.get_nowait()
            except asyncio.QueueEmpty:
                if time.monotonic() >= deadline:
                    return Response(status_code=204)  # HTTP 204 MUST NOT carry a JSON body
                await asyncio.sleep(0.25)
                continue
            if hub.deliverable(candidate):
                # The D1 cache may still admit an old notebook briefly after a
                # key rotation. Never hand the new agent's job to that host.
                bearer = candidate.headers.get("authorization", "")
                if bearer.lower().startswith("bearer ") and bearer[7:].strip() != request.headers.get("x-blackthorn-key", ""):
                    hub.pending.put_nowait(candidate)
                    raise HTTPException(status_code=401, detail={"code": "relay_auth"})
                candidate.notebook_key = request.headers.get("x-blackthorn-key", "")
                job = candidate
            # else: the caller already gave up on it - executing it now would only burn GPU time
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
    job = hub.inflight.get(job_id)
    if job is None or not _authed(request, (job.notebook_key,)):
        raise HTTPException(status_code=401, detail={"code": "relay_auth", "message": "unknown relay key"})
    hub.last_seen = time.monotonic()
    try:
        job.status = int(request.headers.get("x-relay-status") or 200)
        job.content_type = request.headers.get("x-relay-content-type") or "application/octet-stream"
        job.pushed = True
        job.headers_ready.set()
        async for chunk in request.stream():
            if not hub.deliverable(job):
                # the consumer is gone (cancelled run, timeout): tell the notebook to stop
                # streaming into a dead job - its upstream call then closes and the model frees
                return JSONResponse({"dropped": True}, status_code=410)
            if chunk and not await _put_chunk(job, chunk, hub):
                return JSONResponse({"dropped": True}, status_code=410)
    except Exception:
        # network blip mid-push: end the stream so the agent sees a clean error instead of a hang
        pass
    finally:
        # The consumer owns the job's lifecycle (it calls finish); the push only signals end-of-body.
        if not job.closed:
            await _put_chunk(job, None, hub)
    return JSONResponse({"ok": True})


# Render can buffer a chunked POST body until completion. This broke both long
# computer commands and slow model token streams (the agent saw 502 despite a
# healthy model). Both channels use acknowledged headers and bounded chunks;
# the old push endpoint remains available for notebooks not yet restarted.
def _find_inflight(request: Request, job_id: str):
    hubs = getattr(request.app.state.bt, "relay_hubs", {}) or {}
    for hub in hubs.values():
        job = hub.inflight.get(job_id)
        if job is not None:
            return hub, job
    return None, None


async def relay_headers(request: Request, job_id: str):
    hub, job = _find_inflight(request, job_id)
    if job is not None and not _authed(request, (job.notebook_key,)):
        raise HTTPException(status_code=401, detail={"code": "relay_auth"})
    if job is None or job.closed or hub is None:
        return JSONResponse({"dropped": True}, status_code=410)
    job.status = int(request.headers.get("x-relay-status") or 200)
    job.content_type = request.headers.get("x-relay-content-type") or "application/octet-stream"
    job.pushed = True
    job.headers_ready.set()
    hub.last_seen = time.monotonic()
    return {"ok": True}


async def relay_chunk(request: Request, job_id: str):
    hub, job = _find_inflight(request, job_id)
    if job is not None and not _authed(request, (job.notebook_key,)):
        raise HTTPException(status_code=401, detail={"code": "relay_auth"})
    if job is None or job.closed or not job.pushed or hub is None:
        return JSONResponse({"dropped": True}, status_code=410)
    chunk = await request.body()
    if len(chunk) > 1_000_000:
        raise HTTPException(status_code=413, detail={"code": "relay_chunk_too_large"})
    if chunk and not await _put_chunk(job, chunk, hub):
        return JSONResponse({"dropped": True}, status_code=410)
    hub.last_seen = time.monotonic()
    return {"ok": True}


async def relay_done(request: Request, job_id: str):
    hub, job = _find_inflight(request, job_id)
    if job is not None and not _authed(request, (job.notebook_key,)):
        raise HTTPException(status_code=401, detail={"code": "relay_auth"})
    if job is None or job.closed or hub is None:
        return JSONResponse({"dropped": True}, status_code=410)
    await _put_chunk(job, None, hub)
    hub.last_seen = time.monotonic()
    return {"ok": True}


# --------------------------------------------------------------------------- data plane (agent side)
async def relay_proxy(request: Request, path: str = ""):
    """Any method, any path under /gpu-relay/: pipe it to the notebook gateway and stream back.

    Channel selection rules (important):
    - Explicit ``model/...`` prefix → model channel, prefix stripped.
    - Paths under ``computer/...`` use the dedicated *computer* relay channel ONLY when
      that notebook is actually connected. Otherwise they stay on the *model* channel
      with the full ``/computer/...`` path so a single primary host can serve tools.
    - Never strip ``computer/`` into a bare path like ``exec`` — the local gateway routes
      are mounted at ``/computer/exec``, not ``/exec``.
    """
    raw = (path or "").lstrip("/")
    channel = "model"
    forward = raw
    if raw == "model" or raw.startswith("model/"):
        channel = "model"
        forward = raw[len("model"):].lstrip("/")
    elif raw == "computer" or raw.startswith("computer/"):
        if __import__("os").environ.get("PANDASTACK_API_KEY"):
            raise HTTPException(status_code=410,
                                detail={"code": "computer_relay_retired", "message": "Computer tools use PandaStack, not the Kaggle relay."})
        # Prefer a live dedicated computer host; otherwise the model host answers /computer/*.
        services = request.app.state.bt
        hubs = getattr(services, "relay_hubs", None) or {}
        comp = hubs.get("computer")
        if comp is None or not comp.alive():
            # The computer host may not have pulled yet (its hub appears on the first
            # pull) or may be between long-polls; give it a moment before falling back
            # to the model host (which also serves /computer/*).
            for _ in range(6):
                await asyncio.sleep(0.5)
                comp = hubs.get("computer")
                if comp is not None and comp.alive():
                    break
        # Never route a configured dedicated computer's request to the model host.
        # The model's /computer/* endpoints use a DIFFERENT ephemeral filesystem:
        # a fallback here can pass a health check and then silently lose user files.
        try:
            enrolled = bool(await services.store.computer_row())
        except Exception:
            raise HTTPException(status_code=503,
                                detail={"code": "computer_state_unknown", "message": "Computer routing state unavailable."})
        channel = "computer" if enrolled or (comp is not None and comp.alive()) else "model"
        # The published computer base is /gpu-relay/computer, so gateway paths arrive either
        # doubled (computer/computer/info) or single (computer/exec). The local gateway mounts
        # /computer/* — normalise BOTH forms to exactly one "computer/" prefix.
        rest = raw[len("computer"):].lstrip("/")
        if rest == "computer" or rest.startswith("computer/"):
            forward = rest
        else:
            forward = "computer/" + rest if rest else "computer"
    path = forward
    if path.startswith("api/kaggle-relay"):
        # the control plane lives on the app origin; asking for it through the data plane
        # would queue a request for the very client that is asking
        raise HTTPException(status_code=404, detail={"code": "relay_path", "message": "control plane is /api/kaggle-relay/*"})
    if path and not path.startswith("/"):
        path = "/" + path
    hub = _hub(request, channel)
    if not hub.alive() and not (channel == "computer" and enrolled):
        # Legacy model-only computer may be between long-polls; do not wait on
        # an explicitly enrolled computer that has no dedicated puller.
        for _ in range(20):
            await asyncio.sleep(0.5)
            if hub.alive():
                break
        if not hub.alive():
            raise HTTPException(status_code=502,
                                detail={"code": "gpu_unreachable",
                                        "message": "The GPU relay is not connected. Wait for the GPU to finish starting, then try again."})
    body = await request.body()
    headers = {k.lower(): v for k, v in request.headers.items()
               if k.lower() not in DROP_HEADERS and not k.lower().startswith("x-relay-")}
    if channel == "model":
        # The relay is the authority for the model credential.
        #
        # The notebook only accepts the key it published for this boot. Forwarding the caller's
        # own copy let a stale (cached) credential reach the notebook and come back as a 401 the
        # agent could only answer by refreshing - which returns the same stale key whenever the
        # notebook could not republish, producing the endless "GPU gateway kept rejecting the
        # refreshed credential" loop. Instead: authenticate the caller against the published key
        # here, and forward the authoritative one.
        authoritative = await _row_keys(request, "model", force=True)
        if not authoritative:
            raise HTTPException(status_code=503,
                                detail={"code": "gpu_key_unavailable",
                                        "message": "The model has not published an access key yet."})
        presented = (request.headers.get("authorization") or "")
        presented = presented[7:].strip() if presented.lower().startswith("bearer ") else presented.strip()
        if presented and presented not in set(authoritative):
            # Distinct, retryable code: the caller's route is stale, nothing ran, no work was lost.
            raise HTTPException(status_code=401,
                                detail={"code": "stale_key",
                                        "message": "This request used an out-of-date model key. Refresh the route and retry."})
        headers["authorization"] = "Bearer " + authoritative[0]
    job = hub.submit(request.method.upper(), path, headers, body)

    # Between long-polls waiting_pulls can be 0 while the notebook is healthy; use alive().
    wait_s = HEADERS_TIMEOUT_S if hub.alive() else min(45.0, HEADERS_TIMEOUT_S)
    try:
        await asyncio.wait_for(job.headers_ready.wait(), timeout=wait_s)
    except asyncio.TimeoutError:
        hub.abandon(job)                 # never hand this request to the notebook after we gave up
        hub.consecutive_timeouts += 1
        hub.last_timeout_at = time.monotonic()
        raise HTTPException(status_code=502,
                            detail={"code": "gpu_unreachable",
                                    "message": "The GPU did not answer through the relay (timed out)."})
    if not job.pushed:
        hub.abandon(job)
        hub.consecutive_timeouts += 1
        hub.last_timeout_at = time.monotonic()
        raise HTTPException(status_code=502,
                            detail={"code": "gpu_unreachable",
                                    "message": "The GPU relay lost the notebook before it answered."})
    hub.consecutive_timeouts = 0
    hub.last_answer_at = time.monotonic()

    async def stream() -> AsyncIterator[bytes]:
        try:
            while True:
                chunk = await job.chunks.get()
                job.last_consumed = time.monotonic()
                if chunk is None:
                    break
                yield chunk
        finally:
            # client gone or finished: release the queue so a blocked push is answered at once
            hub.finish(job)

    return StreamingResponse(stream(), status_code=job.status, media_type=job.content_type,
                             headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


def register(router: APIRouter) -> None:
    router.add_api_route("/api/kaggle-relay/pull", relay_pull, methods=["GET"])
    router.add_api_route("/api/kaggle-relay/push/{job_id}", relay_push, methods=["POST"])
    router.add_api_route("/api/kaggle-relay/headers/{job_id}", relay_headers, methods=["POST"])
    router.add_api_route("/api/kaggle-relay/chunk/{job_id}", relay_chunk, methods=["POST"])
    router.add_api_route("/api/kaggle-relay/done/{job_id}", relay_done, methods=["POST"])
    router.add_api_route("/gpu-relay/{path:path}", relay_proxy,
                         methods=["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    router.add_api_route("/gpu-relay", relay_proxy, methods=["GET", "HEAD", "POST"], include_in_schema=False)
