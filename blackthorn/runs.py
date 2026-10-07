"""In-memory registry of agent runs.

A run is *detached* from the HTTP request that started it: a page refresh or a dropped connection does not
abort the generation, and a client can re-attach (``subscribe(after=<last seq>)``) and receive exactly the
events it missed — no lost and no duplicated tokens. Only an explicit cancel stops a run.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional


class Run:
    def __init__(self, session_id: str, assistant_uid: str):
        self.id = "run_" + uuid.uuid4().hex[:14]
        self.session_id = session_id
        self.assistant_uid = assistant_uid
        self.created = time.time()
        self.events: List[Dict[str, Any]] = []
        self.cancel_event = asyncio.Event()
        self.task: Optional[asyncio.Task] = None
        self.done = False
        self.status = "running"
        self.finished_at: Optional[float] = None
        self._waiters: List["asyncio.Future[None]"] = []

    # ---- producer side ----------------------------------------------------------------
    def emit(self, type_: str, **data: Any) -> Dict[str, Any]:
        event = {"seq": len(self.events) + 1, "type": type_, **data}
        self.events.append(event)
        self._wake()
        return event

    def finish(self, status: str) -> None:
        self.done = True
        self.status = status
        self.finished_at = time.time()
        self._wake()

    def _wake(self) -> None:
        waiters, self._waiters = self._waiters, []
        for fut in waiters:
            if not fut.done():
                fut.set_result(None)

    # ---- consumer side ----------------------------------------------------------------
    async def subscribe(self, after: int = 0, heartbeat: float = 12.0) -> AsyncIterator[Optional[Dict[str, Any]]]:
        """Yield every event with seq > ``after``; yield ``None`` as a heartbeat when idle; stop after the last event."""
        index = max(0, int(after))
        while True:
            while index < len(self.events):
                yield self.events[index]
                index += 1
            if self.done:
                return
            fut: "asyncio.Future[None]" = asyncio.get_running_loop().create_future()
            self._waiters.append(fut)
            try:
                await asyncio.wait_for(fut, timeout=heartbeat)
            except asyncio.TimeoutError:
                yield None
            finally:
                if fut in self._waiters:
                    self._waiters.remove(fut)

    def info(self) -> Dict[str, Any]:
        return {"run_id": self.id, "session_id": self.session_id, "assistant_id": self.assistant_uid,
                "status": self.status, "last_seq": len(self.events), "done": self.done}


class RunManager:
    def __init__(self, *, ttl: float = 900.0, max_active: int = 4):
        self._runs: Dict[str, Run] = {}
        self._ttl = ttl
        self._max_active = max_active

    def create(self, session_id: str, assistant_uid: str) -> Run:
        self.gc()
        run = Run(session_id, assistant_uid)
        self._runs[run.id] = run
        return run

    def get(self, run_id: str) -> Optional[Run]:
        return self._runs.get(run_id)

    def live_for_session(self, session_id: str) -> Optional[Run]:
        for run in self._runs.values():
            if run.session_id == session_id and not run.done:
                return run
        return None

    def active_count(self) -> int:
        return sum(1 for r in self._runs.values() if not r.done)

    def at_capacity(self) -> bool:
        return self.active_count() >= self._max_active

    def cancel(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        if run is None or run.done:
            return False
        run.cancel_event.set()
        if run.task is not None and not run.task.done():
            run.task.cancel()
        return True

    def gc(self) -> None:
        now = time.time()
        for rid in [rid for rid, r in self._runs.items() if r.done and r.finished_at and now - r.finished_at > self._ttl]:
            del self._runs[rid]

    async def shutdown(self) -> None:
        tasks = []
        for run in self._runs.values():
            if not run.done and run.task is not None:
                run.cancel_event.set()
                run.task.cancel()
                tasks.append(run.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
