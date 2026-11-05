"""The command channel between the app and the agent running inside a codespace.

The container cannot be dialled from the internet safely, so the connection is opened outwards:
the agent long-polls ``/api/codespaces/pull``, the app hands it one job at a time, and the agent
posts the result back to ``/api/codespaces/result/{id}``. A pull is held open for up to
``pull_wait`` seconds, so a command issued at any moment reaches the container within one round
trip, and a container that restarts simply reconnects and carries on.

What this buys, compared with publishing an HTTP port:

* nothing inside the codespace has to be reachable from the internet, so there is no port to
  expose and no forwarded-port visibility to re-apply after every restart;
* a stopped codespace costs nothing, and a restarted one resumes on its own;
* a job delivered to a container that dies before answering is re-queued once (``requeue``), so a
  restart mid-command is retried instead of surfacing as a random failure.

Registration proves the container is ours: the agent sends the ``GITHUB_TOKEN`` that GitHub itself
injects into the codespace, and the app checks with the GitHub API that the token belongs to the
configured owner and can read that very codespace. Only then is the agent's session token stored.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

log = logging.getLogger("blackthorn.codespace.channel")

PULL_WAIT = 25.0
REQUEUE_SECONDS = 45.0
MAX_REQUEUE = 1


class Job:
    __slots__ = ("id", "path", "body", "future", "created", "delivered", "requeues", "name")

    def __init__(self, path: str, body: Dict[str, Any]):
        self.id = uuid.uuid4().hex
        self.path = path
        self.body = body
        self.created = time.time()
        self.delivered = 0.0
        self.requeues = 0
        self.name = ""
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()


class CodespaceChannel:
    """Job queue + presence for every codespace agent this app talks to."""

    def __init__(self, store: Any = None, *, pull_wait: float = PULL_WAIT):
        self.store = store
        self.pull_wait = pull_wait
        self.SESSION_KEY = "codespace_agent_session"
        self._queue: "asyncio.Queue[Job]" = asyncio.Queue()
        self._results: Dict[str, Dict[str, Any]] = {}
        self._inflight: Dict[str, Job] = {}
        self._tokens: Dict[str, str] = {}
        self._presence: Dict[str, Dict[str, Any]] = {}
        self._inflight_count = 0
        self._registered_at = 0.0

    # ------------------------------------------------------------------ tokens
    async def _load_tokens(self) -> None:
        if not self._tokens and self.store is not None:
            try:
                raw = await self.store.get_setting(self.SESSION_KEY)
                self._tokens = json.loads(raw) if raw else {}
            except Exception:  # noqa: BLE001 - a missing store must not break the channel
                self._tokens = {}

    async def _save_tokens(self) -> None:
        if self.store is None:
            return
        try:
            await self.store.set_setting(self.SESSION_KEY, json.dumps(self._tokens))
        except Exception:  # noqa: BLE001
            pass

    async def token_for(self, name: str) -> str:
        await self._load_tokens()
        return self._tokens.get(name, "")

    def known(self, name: str) -> bool:
        return bool(name) and name in self._tokens

    async def accept_registration(self, name: str, token: str) -> None:
        await self._load_tokens()
        self._tokens[name] = token
        self._registered_at = time.time()
        await self._save_tokens()
        log.info("codespace agent registered: %s", name)

    async def forget(self, name: str) -> None:
        await self._load_tokens()
        if self._tokens.pop(name, None) is not None:
            await self._save_tokens()

    # ---------------------------------------------------------------- presence
    def touch(self, name: str, *, version: str = "", port: int = 0) -> None:
        entry = self._presence.setdefault(name, {})
        entry.update({"last_seen": time.time(), "version": version, "port": port})
        entry["pulls"] = int(entry.get("pulls") or 0) + 1

    def present(self, name: str, *, fresh_s: float = 90.0) -> bool:
        entry = self._presence.get(name) or {}
        return bool(entry) and (time.time() - float(entry.get("last_seen") or 0)) < fresh_s

    def agents(self, *, fresh_s: float = 90.0) -> List[str]:
        """Names of codespace agents that are polling right now.

        Used to recognise the machine we are running on after a restart: the container dials in,
        so it announces itself even when this process has forgotten which codespace is ours.
        """
        now = time.time()
        return sorted(name for name, entry in self._presence.items()
                      if (now - float(entry.get("last_seen") or 0)) < fresh_s)

    def presence(self, name: str) -> Dict[str, Any]:
        entry = dict(self._presence.get(name) or {})
        last = float(entry.get("last_seen") or 0)
        entry["connected"] = bool(last) and (time.time() - last) < 90.0
        entry["last_seen_s"] = int(time.time() - last) if last else -1
        entry["registered"] = bool(name) and name in self._tokens
        return entry

    @property
    def inflight(self) -> int:
        return self._inflight_count

    # ------------------------------------------------------------------- calls
    async def call(self, path: str, body: Dict[str, Any], *, timeout: float = 120.0,
                   name: str = "") -> Dict[str, Any]:
        """Queue one job for the agent and wait for its result."""
        if not name:
            raise RuntimeError("codespace channel call without a codespace name")
        job = Job(path, body or {})
        job.name = name
        self._inflight_count += 1
        try:
            await self._queue.put(job)
            try:
                return await asyncio.wait_for(job.future, timeout=max(10.0, timeout))
            except asyncio.TimeoutError:
                self._inflight.pop(job.id, None)
                raise TimeoutError(f"the codespace did not answer {path} within {int(timeout)}s")
        finally:
            self._inflight_count = max(0, self._inflight_count - 1)
            self._inflight.pop(job.id, None)

    async def pull(self, name: str, *, version: str = "", port: int = 0,
                   wait: Optional[float] = None) -> Dict[str, Any]:
        """Long-poll entry point used by the agent: hand it the next job (or nothing)."""
        self.touch(name, version=version, port=port)
        # A job handed to a container that died before answering is retried once.
        self._requeue_stale(name)
        try:
            job = await asyncio.wait_for(self._queue.get(), timeout=wait or self.pull_wait)
        except asyncio.TimeoutError:
            return {"ok": True, "job": None}
        job.delivered = time.time()
        job.name = job.name or name
        self._inflight[job.id] = job
        return {"ok": True, "job": {"id": job.id, "path": job.path, "body": job.body}}

    def submit_result(self, job_id: str, payload: Dict[str, Any]) -> bool:
        """The agent's answer for a job: complete the caller's future."""
        job = self._inflight.pop(job_id, None)
        if job is None:
            # Late answer for a job the caller already gave up on: keep it for inspection.
            self._results[job_id] = payload
            return False
        if not job.future.done():
            job.future.set_result(payload.get("result") if isinstance(payload.get("result"), dict) else payload)
        return True

    def _requeue_stale(self, name: str) -> None:
        now = time.time()
        for job_id, job in list(self._inflight.items()):
            if job.name != name or job.requeues >= MAX_REQUEUE:
                continue
            if now - (job.delivered or job.created) < REQUEUE_SECONDS:
                continue
            self._inflight.pop(job_id, None)
            job.requeues += 1
            self._queue.put_nowait(job)
            log.warning("re-queued %s after the codespace stopped answering", job.path)

    def summary(self) -> Dict[str, Any]:
        return {"queued": self._queue.qsize(), "inflight": self._inflight_count,
                "agents": [{"name": k, **v} for k, v in self._presence.items()],
                "registered_at": self._registered_at, "sessions": len(self._tokens)}
