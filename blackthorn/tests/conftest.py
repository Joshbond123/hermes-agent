from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncIterator, Dict, List

import httpx
import pytest
import pytest_asyncio

from blackthorn.app import build_services, create_app
from blackthorn.config import Settings
from blackthorn.sql import SqliteExecutor

from .fakes import FakeBackend, ServerThread

pytest_plugins = ("pytest_asyncio",)


class Stack:
    """The real Blackthorn app + the fake model/computer backend, both on real sockets."""

    def __init__(self, backend: FakeBackend, backend_srv: ServerThread, app_srv: ServerThread, services, executor):
        self.backend, self.backend_srv, self.app_srv = backend, backend_srv, app_srv
        self.services, self.executor = services, executor
        self.url = app_srv.url

    async def set_gpu(self, status: str = "HEARTBEAT_ONLINE", url: str | None = None, key: str = "test-key") -> None:
        await self.executor.query("DELETE FROM kaggle_gpu_state")
        await self.executor.query(
            "INSERT INTO kaggle_gpu_state (id, status, tunnel_url, api_key, model, gpu_info, updated_at) VALUES ('primary', ?, ?, ?, 'fake-model', 'T4', ?)",
            [status, self.backend_srv.url if url is None else url, key, time.time()])
        self.services.resolver.invalidate()

    async def stream(self, payload: Dict, client: httpx.AsyncClient | None = None) -> "Streamed":
        """POST /api/chat/stream and collect every SSE event with its arrival time."""
        out = Streamed()
        own = client is None
        client = client or httpx.AsyncClient(timeout=30)
        try:
            async with client.stream("POST", f"{self.url}/api/chat/stream", json=payload) as resp:
                out.status = resp.status_code
                if resp.status_code != 200:
                    out.error_body = json.loads((await resp.aread()).decode() or "{}")
                    return out
                out.headers = dict(resp.headers)
                await out.consume(resp)
        finally:
            if own:
                await client.aclose()
        return out


class Streamed:
    def __init__(self) -> None:
        self.status = 0
        self.headers: Dict[str, str] = {}
        self.events: List[Dict] = []
        self.times: List[float] = []
        self.error_body: Dict = {}
        self.t0 = time.perf_counter()
        self.pings = 0

    async def consume(self, resp: httpx.Response, stop_after: int | None = None) -> None:
        if getattr(self, "_lines", None) is None:
            self._lines = resp.aiter_lines().__aiter__()       # one iterator per response, so reads can resume
        while True:
            try:
                line = await self._lines.__anext__()
            except StopAsyncIteration:
                return
            if line.startswith(":"):
                self.pings += 1
            if line.startswith("data:"):
                ev = json.loads(line[5:].strip())
                self.events.append(ev)
                self.times.append(time.perf_counter() - self.t0)
                if stop_after is not None and len(self.events) >= stop_after:
                    return

    def of(self, type_: str) -> List[Dict]:
        return [e for e in self.events if e["type"] == type_]

    @property
    def text(self) -> str:
        return "".join(e["text"] for e in self.of("text.delta"))

    @property
    def end(self) -> Dict:
        ends = self.of("run.end")
        return ends[-1] if ends else {}

    @property
    def session_id(self) -> str:
        return self.of("run.start")[0]["session_id"]


@pytest_asyncio.fixture
async def stack(tmp_path) -> AsyncIterator[Stack]:
    backend = FakeBackend()
    backend_srv = ServerThread(backend.app).start()
    executor = SqliteExecutor(":memory:")
    settings = Settings(store="sqlite", heartbeat_s=0.5, tool_timeout_s=5.0)
    services = build_services(settings, executor=executor, static_dir=tmp_path / "static")
    (tmp_path / "static" / "assets").mkdir(parents=True)
    (tmp_path / "static" / "index.html").write_text("<!doctype html><title>t</title>")
    app = create_app(settings, services=services, gpu_daemon=False)
    app_srv = ServerThread(app).start()
    s = Stack(backend, backend_srv, app_srv, services, executor)
    await services.store.ensure_schema()
    await s.set_gpu()
    try:
        yield s
    finally:
        app_srv.stop()
        backend_srv.stop()
