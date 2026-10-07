"""Browser end-to-end tests for the Blackthorn chat (Playwright, real Chromium).

Run against any deployment:

    BT_BASE_URL=http://127.0.0.1:9119 pytest tests/blackthorn_e2e -p no:cacheprovider

* Tests that need the GPU/model are marked ``gpu`` and only run with ``BT_E2E_GPU=1``.
* Test data is created through D1 (not through the model) under the session source the server
  is configured with (``BLACKTHORN_SESSION_SOURCE``), and every created chat is deleted at the end.
* Secrets come from the environment only (CLOUDFLARE_*), never from files.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

import pytest
from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

BASE = os.environ.get("BT_BASE_URL", "http://127.0.0.1:9119").rstrip("/")
SOURCE = os.environ.get("BLACKTHORN_SESSION_SOURCE", "blackthorn-studio")
GPU_TESTS = os.environ.get("BT_E2E_GPU") == "1"


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a running GPU/model (set BT_E2E_GPU=1)")


def pytest_collection_modifyitems(config, items):
    if GPU_TESTS:
        return
    skip = pytest.mark.skip(reason="needs the GPU (BT_E2E_GPU=1)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


# --------------------------------------------------------------------------- D1 (test data)
def _d1(statements: List[tuple]) -> List[Dict[str, Any]]:
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CLOUDFLARE_ACCOUNT_ID']}"
        f"/d1/database/{os.environ['CLOUDFLARE_D1_DATABASE_ID']}/query"
    )
    body: Dict[str, Any]
    if len(statements) == 1:
        body = {"sql": statements[0][0], "params": list(statements[0][1])}
    else:
        body = {"batch": [{"sql": s, "params": list(p)} for s, p in statements]}
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {os.environ['CLOUDFLARE_API_TOKEN']}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    if not data.get("success"):
        raise RuntimeError(f"D1 error: {data.get('errors')}")
    return data["result"]


class Seeder:
    """Creates / removes conversations directly in D1 (independent of the GPU)."""

    def __init__(self) -> None:
        self.ids: List[str] = []

    def chat(self, title: str, messages: List[Dict[str, Any]], *, pinned: bool = False, archived: bool = False,
             age_seconds: float = 0) -> str:
        sid = f"studio-e2e{uuid.uuid4().hex[:8]}"
        now = time.time() - age_seconds
        stmts: List[tuple] = [(
            "INSERT INTO sessions (id, source, created_source, session_key, display_name, title, title_source, model, "
            "started_at, last_activity_at, message_count, tool_call_count, pinned, archived) "
            "VALUES (?, ?, ?, ?, ?, ?, 'user', 'e2e', ?, ?, ?, 0, ?, ?)",
            [sid, SOURCE, SOURCE, sid, title, title, now, now, len(messages), 1 if pinned else 0, 1 if archived else 0])]
        for i, m in enumerate(messages):
            meta = json.dumps({"parts": m["parts"]}) if m.get("parts") else None
            stmts.append((
                "INSERT INTO messages (session_id, role, content, timestamp, active, _compressed_summary, finish_reason, "
                "display_metadata, message_uid) VALUES (?, ?, ?, ?, 1, 0, ?, ?, ?)",
                [sid, m["role"], m["content"], now + i * 0.01, m.get("finish_reason"), meta, uuid.uuid4().hex]))
        _d1(stmts)
        self.ids.append(sid)
        return sid

    def cleanup(self) -> None:
        for sid in self.ids:
            try:
                _d1([("DELETE FROM messages WHERE session_id = ?", [sid]), ("DELETE FROM sessions WHERE id = ?", [sid])])
            except Exception:  # noqa: BLE001
                pass
        self.ids.clear()

    @staticmethod
    def row(sid: str) -> Optional[Dict[str, Any]]:
        res = _d1([("SELECT id, title, pinned, archived, source FROM sessions WHERE id = ?", [sid])])
        rows = res[0]["results"]
        return rows[0] if rows else None

    @staticmethod
    def message_count(sid: str) -> int:
        return _d1([("SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", [sid])])[0]["results"][0]["n"]


@pytest.fixture()
def seeder():
    s = Seeder()
    yield s
    s.cleanup()


# --------------------------------------------------------------------------- browser
@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


def _new_page(browser: Browser, *, width: int, height: int, mobile: bool, scheme: str = "dark") -> tuple[BrowserContext, Page, List[str]]:
    ctx = browser.new_context(viewport={"width": width, "height": height}, is_mobile=mobile, has_touch=mobile,
                              color_scheme=scheme, permissions=["clipboard-read", "clipboard-write"] if not mobile else [])
    page = ctx.new_page()
    errors: List[str] = []
    page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text[:300]}") if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {str(e)[:300]}"))
    page.on("requestfailed", lambda r: errors.append(f"requestfailed: {r.url[:120]}") if "/api/" in r.url else None)
    return ctx, page, errors


@pytest.fixture()
def desktop(browser):
    ctx, page, errors = _new_page(browser, width=1440, height=900, mobile=False)
    page.errors = errors  # type: ignore[attr-defined]
    yield page
    ctx.close()


@pytest.fixture()
def mobile(browser):
    ctx, page, errors = _new_page(browser, width=390, height=844, mobile=True)
    page.errors = errors  # type: ignore[attr-defined]
    yield page
    ctx.close()


def open_chat(page: Page, path: str = "/") -> None:
    page.goto(BASE + path, wait_until="domcontentloaded")
    page.wait_for_selector(".bt-chat-root", timeout=30000)


def api(page: Page, method: str, path: str, body: Any = None) -> Dict[str, Any]:
    """Call the app's own API from the page (same origin, same session token as the UI)."""
    return page.evaluate(
        """async ([method, path, body]) => {
            const r = await fetch(path, {method, headers: {'X-Hermes-Session-Token': window.__HERMES_SESSION_TOKEN__ || '',
                                   ...(body ? {'Content-Type': 'application/json'} : {})}, body: body ? JSON.stringify(body) : undefined});
            let j = null; try { j = await r.json(); } catch (e) {}
            return {status: r.status, json: j};
        }""",
        [method, path, body],
    )


def sidebar_titles(page: Page) -> List[str]:
    return [t.strip() for t in page.locator(".bt-row-title").all_inner_texts()]


def no_horizontal_overflow(page: Page) -> bool:
    return page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1")


def accessible_name_missing(page: Page) -> List[str]:
    return page.evaluate(
        """() => [...document.querySelectorAll('.bt-chat-root button, .bt-chat-root a, .bt-chat-root input, .bt-chat-root textarea')]
            .filter(e => e.offsetParent !== null && !(e.getAttribute('aria-label') || e.getAttribute('title') || (e.innerText||'').trim()
                      || e.getAttribute('placeholder') || (e.labels && e.labels.length)))
            .map(e => e.outerHTML.slice(0, 120))"""
    )
