from __future__ import annotations

import asyncio
import time
import json
import time
import os
from pathlib import Path
from typing import AsyncIterator, Dict, List

import httpx
import pytest
import pytest_asyncio

from blackthorn.app import build_services, create_app
from blackthorn.config import Settings
from blackthorn.sql import SqliteExecutor

from .fake_codespaces import BackendBridge, FakeChannel, FakeCodespaceAPI
from .fakes import FakeBackend, ServerThread

pytest_plugins = ("pytest_asyncio",)


class Stack:
    """The real Blackthorn app + the fake model/computer backend, both on real sockets."""

    def __init__(self, backend: FakeBackend, backend_srv: ServerThread, app_srv: ServerThread, services, executor,
                 codespaces=None):
        self.backend, self.backend_srv, self.app_srv = backend, backend_srv, app_srv
        self.services, self.executor = services, executor
        self.codespaces = codespaces
        self.url = app_srv.url

    async def set_model(self, url: str | None = None, key: str = "test-key", model: str = "nvidia/nemotron-3-ultra-550b-a55b") -> None:
        """Point the NVIDIA route at the fake OpenAI-compatible backend (or at nothing at all)."""
        os.environ["NVIDIA_BASE_URL"] = self.backend_srv.url if url is None else url
        os.environ["NVIDIA_API_KEY"] = key
        os.environ["BLACKTHORN_MODEL"] = model
        self.services.resolver.invalidate()

    def computer(self):
        """The fake codespace the app is wired to."""
        return self.codespaces

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


def fake_codespace(services, *, backend_url: str = "", state: str = "ON", name: str = "blackthorn-test-abc123"):
    """Attach a fake codespace computer to a Services instance (no network, real manager code)."""
    from .fake_codespaces import attach_fake_codespace
    return attach_fake_codespace(services, backend_url, name=name, state=state)


@pytest_asyncio.fixture
async def stack(tmp_path) -> AsyncIterator[Stack]:
    saved_env = {k: os.environ.get(k) for k in ("NVIDIA_BASE_URL", "NVIDIA_API_KEY", "BLACKTHORN_MODEL")}
    backend = FakeBackend()
    backend_srv = ServerThread(backend.app).start()
    executor = SqliteExecutor(":memory:")
    settings = Settings(store="sqlite", heartbeat_s=0.5, tool_timeout_s=5.0)
    services = build_services(settings, executor=executor, static_dir=tmp_path / "static")
    (tmp_path / "static" / "assets").mkdir(parents=True)
    (tmp_path / "static" / "index.html").write_text("<!doctype html><title>t</title>")
    app = create_app(settings, services=services)
    app_srv = ServerThread(app).start()
    s = Stack(backend, backend_srv, app_srv, services, executor)
    s.codespaces = fake_codespace(services, backend_url=backend_srv.url)
    s.services.computer = s.codespaces            # the tools must talk to the codespace, not the network
    await services.store.ensure_schema()
    await s.set_model()
    try:
        yield s
    finally:
        app_srv.stop()
        backend_srv.stop()
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# ======================================================================================================
# Real-browser fixtures: the production UI build served by the real app and a fake codespace.
# ======================================================================================================
UI_DIR = Path(__file__).resolve().parents[1] / "static"


@pytest_asyncio.fixture
async def ui_stack() -> AsyncIterator[Stack]:
    saved_env = {k: os.environ.get(k) for k in ("NVIDIA_BASE_URL", "NVIDIA_API_KEY", "BLACKTHORN_MODEL")}
    backend = FakeBackend()
    backend_srv = ServerThread(backend.app).start()
    executor = SqliteExecutor(":memory:")
    settings = Settings(store="sqlite", heartbeat_s=0.5, tool_timeout_s=5.0)
    services = build_services(settings, executor=executor, static_dir=UI_DIR)
    app = create_app(settings, services=services)
    app_srv = ServerThread(app).start()
    s = Stack(backend, backend_srv, app_srv, services, executor)
    s.codespaces = fake_codespace(services, backend_url=backend_srv.url)
    s.services.computer = s.codespaces
    await services.store.ensure_schema()
    await s.set_model()
    try:
        yield s
    finally:
        app_srv.stop()
        backend_srv.stop()
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest_asyncio.fixture
async def browser():
    # Browser tests need playwright + chromium; skip cleanly where they are not installed instead of
    # reporting a false failure of the product.
    playwright = pytest.importorskip("playwright.async_api")
    async_playwright = playwright.async_playwright
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
