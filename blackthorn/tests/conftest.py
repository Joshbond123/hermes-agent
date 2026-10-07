from __future__ import annotations

import asyncio
import time
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


# ======================================================================================================
# Real-browser fixtures: the production UI build served by the real app, a scripted GPU controller.
# ======================================================================================================
class FakeD1:
    """Stand-in for ``cloudflare_d1_client`` with a deterministic boot sequence (one stage per status poll)."""

    SEQUENCE = ["BOOTING_KAGGLE_GPU", "CHECKING_ENVIRONMENT", "CHECKING_CACHE", "CACHE_HIT", "STARTING_OLLAMA", "LOADING_MODEL", "WARMING_GPU"]

    def __init__(self, status: str = "GPU_STOPPED_SAVING_QUOTA"):
        self.status = status
        self.stage = 0
        self.calls: list = []
        self.auto_off = 0
        self.boot_started = None

    def _state(self) -> dict:
        s = self.status
        booting = s in self.SEQUENCE
        ready = s == "HEARTBEAT_ONLINE"
        return {"active": ready, "booting": booting, "status": s, "tunnel_url": "https://hidden.example" if (ready or booting) else "",
                "api_key": "must-not-leak", "display_status": "Kaggle Ready" if ready else ("Starting Kaggle" if booting else "Kaggle Offline"),
                "engine_state": "ready" if ready else ("starting" if booting else "off"), "model": "Fake-Model", "model_loaded": True if ready else None,
                "gpu_info": "Tesla T4 + Tesla T4" if (ready or booting) else ("FAILED: Kaggle refused the kernel push" if s == "BOOT_FAILED" else ""), "progress_step": s.replace("_", " ").title() if booting else "",
                "quota": {"used_hours": 19.8, "total_hours": 30.0, "remaining_hours": 10.2, "used_pct": 66.0, "refresh_time": "2026-10-10T00:00:00Z"},
                "cloudflare_d1": {"account_id": "acct", "database_id": "db"}, "busy": False, "worker_status": "RUNNING" if ready else "OFF"}

    def get_kaggle_gpu_status(self, refresh=False):
        if self.status in self.SEQUENCE:
            self.stage += 1
            self.status = self.SEQUENCE[self.stage] if self.stage < len(self.SEQUENCE) else "HEARTBEAT_ONLINE"
        return self._state()

    def public_gpu_status(self, state):
        import cloudflare_d1_client as real
        return real.public_gpu_status(state)

    def turn_on_kaggle_gpu(self, blocking=False):
        self.calls.append("on")
        self.status, self.stage = self.SEQUENCE[0], 0
        self.boot_started = time.time()
        return self._state()

    def turn_off_kaggle_gpu(self):
        self.calls.append("off")
        self.status = "GPU_STOPPED_SAVING_QUOTA"
        return self._state()

    def activity_snapshot(self): return {"idle_seconds": 3.0, "active_tasks": 0, "busy": False}
    def auto_off_decision(self): return {"enabled": self.auto_off > 0, "minutes": self.auto_off, "should_stop": False}
    def get_auto_off_minutes(self): return self.auto_off
    def set_auto_off_minutes(self, m):
        if m not in (0, 5, 10, 15, 30, 60):
            raise ValueError("bad choice")
        self.auto_off = m
        return m
    def _gateway_api_key(self): return "test-key"


UI_DIR = Path(__file__).resolve().parents[1] / "static"


@pytest_asyncio.fixture
async def ui_stack() -> AsyncIterator[Stack]:
    from blackthorn.gpu import GpuService
    backend = FakeBackend()
    backend_srv = ServerThread(backend.app).start()
    executor = SqliteExecutor(":memory:")
    settings = Settings(store="sqlite", heartbeat_s=0.5, tool_timeout_s=5.0)
    services = build_services(settings, executor=executor, static_dir=UI_DIR)
    fake_d1 = FakeD1("HEARTBEAT_ONLINE")
    services.gpu = GpuService(services.store, fake_d1)
    app = create_app(settings, services=services, gpu_daemon=False)
    app_srv = ServerThread(app).start()
    s = Stack(backend, backend_srv, app_srv, services, executor)
    s.d1 = fake_d1  # type: ignore[attr-defined]
    await services.store.ensure_schema()
    await s.set_gpu()
    try:
        yield s
    finally:
        app_srv.stop()
        backend_srv.stop()


@pytest_asyncio.fixture
async def browser():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--no-sandbox"])
        try:
            yield b
        finally:
            await b.close()


class Pages:
    def __init__(self, browser, url):
        self.browser, self.url = browser, url
        self.contexts = []

    async def new(self, *, mobile: bool = False, width: int | None = None, height: int | None = None, scheme: str = "dark", clipboard: bool = False):
        vp = {"width": width or (390 if mobile else 1280), "height": height or (844 if mobile else 800)}
        ctx = await self.browser.new_context(viewport=vp, is_mobile=mobile, has_touch=mobile, color_scheme=scheme,
                                             permissions=["clipboard-read", "clipboard-write"] if clipboard else [])
        self.contexts.append(ctx)
        page = await ctx.new_page()
        page.problems = []  # console errors + page errors collected for assertions
        page.requests = []
        page.on("console", lambda m: page.problems.append(f"{m.type}: {m.text}") if m.type == "error" else None)
        page.on("pageerror", lambda e: page.problems.append(f"pageerror: {e}"))
        page.on("request", lambda r: page.requests.append(r.url))
        await page.goto(self.url)
        await page.get_by_test_id("composer-input").wait_for(timeout=15000)
        return page

    async def close(self):
        for c in self.contexts:
            await c.close()


@pytest_asyncio.fixture
async def pages(browser, ui_stack):
    p = Pages(browser, ui_stack.url)
    try:
        yield p
    finally:
        await p.close()
