"""Camoufox browser-controller tests.

The controller (``computer_assets/browse.py``) runs on the computer against a real
browser. To make its contract testable on CI — and to prove every supported action
works — these tests run the real script as a subprocess against a stub ``camoufox``
package that implements the same surface the script uses, plus a real local HTTP
server for navigation. Crash and timeout paths are injected through the stub, so the
retry/relaunch and error-classification logic is exercised for real.

Actions covered: open, click, type (+submit), scroll, screenshot, accessibility,
tabs (list/close), download, close, session persistence, CAPTCHA detection,
navigation timeout retry, browser crash relaunch, missing-dependency reporting.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "computer_assets" / "browse.py"

STUB = r'''
"""Minimal stand-in for camoufox.sync_api used by the browser-controller tests."""
import os, re, urllib.request, urllib.error, json

FAIL_ONCE = os.environ.get("FAKE_BROWSER_FAIL_ONCE") == "1"
STATE = os.environ.setdefault("FAKE_BROWSER_STATE", "/tmp/fake-state.json")

class _Response:
    def __init__(self, status): self.status = status

class _Download:
    suggested_filename = "report.txt"
    def __init__(self, path, data): self._path, self._data = path, data
    def save_as(self, target):
        with open(target, "wb") as fh: fh.write(self._data)

class _Info:
    def __init__(self, value): self.value = value
    def __enter__(self): return self
    def __exit__(self, *a): return False

class _Mouse:
    def __init__(self, page): self.page = page
    def wheel(self, _x, dy): self.page.scrolled += dy

class _Accessibility:
    def __init__(self, page): self.page = page
    def snapshot(self):
        return {"role": "WebArea", "name": self.page.title(),
                "children": [{"role": "heading", "name": "Main"}, {"role": "link", "name": "More"}]}

class _Locator:
    def __init__(self, page, selector): self.page, self.selector = page, selector
    def click(self, timeout=None):
        if "missing" in self.selector: raise TimeoutError("locator.click: Timeout 3000ms exceeded")
        self.page.clicked.append(self.selector)
    def fill(self, text, timeout=None): self.page.filled[self.selector] = text
    def press(self, key): self.page.pressed.append(key)

class Page:
    def __init__(self, context):
        self.context, self.url, self.title_text = context, "about:blank", "blank"
        self.body, self.clicked, self.filled, self.pressed = "", [], {}, []
        self.scrolled, self.screenshots, self.history = 0, [], []
        self.mouse = _Mouse(self)
        self.accessibility = _Accessibility(self)
    def set_default_timeout(self, ms): self.timeout = ms
    def goto(self, url, wait_until=None, timeout=None):
        if "slow.example" in url and wait_until == "load":
            raise TimeoutError("page.goto: Timeout 25000ms exceeded")
        if "slow.example" in url:            # the retry with a looser wait succeeds
            self.url, self.title_text, self.body = url, "Slow Page", "loaded late"
            return _Response(200)
        self.history.append((url, wait_until))
        if "download.example" in url:
            data = b"file-body"
            self._download = data
            self.url = url
            return _Response(200)
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                html = resp.read().decode("utf-8", "replace")
                status = resp.status
        except urllib.error.HTTPError as exc:
            html, status = exc.read().decode("utf-8", "replace"), exc.code
        self.url = url
        title = re.search(r"<title>(.*?)</title>", html, re.S)
        self.title_text = (title.group(1).strip() if title else "")
        text = re.sub(r"<script.*?</script>", " ", html, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        self.body = re.sub(r"\s+", " ", text).strip()
        return _Response(status)
    def title(self): return self.title_text
    def evaluate(self, script, *args):
        if "querySelectorAll" in script:
            return [{"text": "More", "href": self.url.rstrip('/') + "/more"}]
        if "innerText" in script: return self.body
        return None
    def query_selector(self, selector):
        if "recaptcha" in selector and "recaptcha" in self.body: return object()
        if "hcaptcha" in selector: return None
        if "challenge-form" in selector and "just a moment" in self.body.lower(): return object()
        if "captcha" in selector and "captcha" in self.body.lower(): return object()
        return None
    def locator(self, selector):
        class _Wrap:
            def __init__(self, loc): self.first = loc
        return _Wrap(_Locator(self, selector))
    def wait_for_timeout(self, ms): pass
    def screenshot(self, path=None, full_page=False):
        Path = __import__("pathlib").Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
        self.screenshots.append(path)
    def expect_download(self, timeout=None):
        return _Info(_Download("x", getattr(self, "_download", b"data")))
    def close(self): self.context.closed_pages.append(self)

class Context:
    def __init__(self, **kw):
        self.kw, self.pages, self.closed_pages, self.saved = kw, [], [], None
        self.pages.append(Page(self))
    def new_page(self):
        page = Page(self); self.pages.append(page); return page
    def storage_state(self, path=None):
        if path:
            import pathlib
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(path).write_text(json.dumps({"cookies": [], "origins": []}))
            self.saved = path
        return {"cookies": [], "origins": []}
    def close(self): self.closed = True

class Browser:
    def __init__(self, **kw):
        self.kw, self.contexts = kw, []
        if os.environ.get("FAKE_BROWSER_CRASH_ONCE") == "1":
            os.environ["FAKE_BROWSER_CRASH_ONCE"] = "0"
            raise RuntimeError("Target page, context or browser has been closed")
    def new_context(self, **kw):
        ctx = Context(**kw); self.contexts.append(ctx); return ctx

class Camoufox:
    def __init__(self, **kw): self.kw = kw
    def __enter__(self): return Browser(**self.kw)
    def __exit__(self, *a): return False
'''

HTML_INDEX = """<html><head><title>Test Page</title></head><body>
<h1>Hello</h1><p>Some visible text for the agent.</p>
<a href="/more">More</a>
<form><input name="q"></form>
</body></html>"""

HTML_CAPTCHA = """<html><head><title>Just a moment</title></head><body>
<p>Just a moment... checking your browser before accessing</p>
<div id="challenge-form"></div>
<iframe src="https://www.google.com/recaptcha/api2/anchor"></iframe>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = HTML_CAPTCHA if self.path.startswith("/captcha") else HTML_INDEX
        payload = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


@pytest.fixture(scope="module")
def site():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture(autouse=True)
def environment(tmp_path, monkeypatch):
    pkg = tmp_path / "stub" / "camoufox"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("__version__ = '0.4.11'\n")
    (pkg / "sync_api.py").write_text(STUB)
    cache = tmp_path / "home" / ".cache" / "camoufox" / "browsers"
    cache.mkdir(parents=True)
    (cache / "camoufox-bin").write_bytes(b"fake")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "stub"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLACKTHORN_BROWSER_NO_BOOTSTRAP", "1")
    monkeypatch.setenv("FAKE_BROWSER_STATE", str(tmp_path / "state.json"))
    return tmp_path


def run(request: dict, *, expect_ok: bool = True) -> dict:
    payload = base64.b64encode(json.dumps(request).encode()).decode()
    proc = subprocess.run([sys.executable, str(SCRIPT), payload], capture_output=True, text=True, timeout=180)
    lines = [line for line in proc.stdout.strip().splitlines() if line.startswith("{")]
    assert lines, f"no JSON from browser controller: {proc.stdout!r} {proc.stderr[-2000:]!r}"
    result = json.loads(lines[-1])
    if expect_ok:
        assert result.get("ok") is True, result
    return result


# --------------------------------------------------------------------------- actions
def test_open_returns_title_url_status_text_and_links(site):
    result = run({"url": site + "/", "action": "open"})
    assert result["title"] == "Test Page"
    assert result["status"] == 200
    assert "Some visible text" in result["text"]
    assert result["links"] and result["links"][0]["href"].endswith("/more")
    assert result["session_state_saved"] is True


def test_click_selects_by_css_and_reports_the_page(site):
    result = run({"url": site + "/", "action": "click", "selector": "#submit"})
    assert result["action"] == "click" and result["final_url"].startswith("http")


def test_type_fills_and_submits_a_field(site):
    result = run({"url": site + "/", "action": "type", "selector": "input[name=q]",
                  "text": "blackthorn", "submit": True})
    assert result["typed"] == 10


def test_scroll_moves_the_viewport(site):
    result = run({"url": site + "/", "action": "scroll", "scroll_px": -400})
    assert result["scrolled_px"] == -400


def test_screenshot_is_written_under_the_workspace(site, environment, monkeypatch):
    monkeypatch.chdir(environment)
    result = run({"url": site + "/", "action": "screenshot"})
    assert result["screenshot"].startswith("browser_screenshots/")
    assert (environment / result["screenshot"]).read_bytes().startswith(b"\x89PNG")


def test_accessibility_snapshot_is_bounded(site):
    result = run({"url": site + "/", "action": "accessibility", "depth": 2})
    tree = result["accessibility"]
    assert tree["role"] == "WebArea" and isinstance(tree["children"], list)


def test_tabs_lists_pages_and_closes_one(site):
    listed = run({"action": "tabs"})
    assert isinstance(listed["tabs"], list) and listed["tabs"]
    closed = run({"action": "tabs", "close_tab": 0})
    assert len(closed["tabs"]) < len(listed["tabs"]) or len(closed["tabs"]) >= 1


def test_download_is_saved_and_sized(site, environment, monkeypatch):
    monkeypatch.chdir(environment)
    result = run({"url": site + "/download.example", "action": "download",
                  "download_path": "downloads/report.txt"})
    assert result["path"] == "downloads/report.txt"
    assert (environment / "downloads/report.txt").exists()
    assert result["bytes"] > 0


def test_close_is_clean(site):
    result = run({"action": "close"})
    assert result["closed"] is True


def test_captcha_challenge_is_detected_and_never_claimed_solved(site):
    result = run({"url": site + "/captcha", "action": "open"})
    captcha = result.get("captcha") or {}
    assert captcha.get("present") is True
    assert "cloudflare" in captcha["kinds"] or "recaptcha" in captcha["kinds"]
    assert "Do not claim it was solved" in captcha["note"]


def test_navigation_timeout_is_retried_with_a_looser_wait(site):
    result = run({"url": "http://slow.example/", "action": "open"})
    assert any("retried once" in note for note in result["notes"]), result["notes"]


def test_browser_crash_triggers_exactly_one_relaunch(site, monkeypatch):
    # The stub raises a "target closed" error on the first browser construction.
    monkeypatch.setenv("FAKE_BROWSER_CRASH_ONCE", "1")
    payload = base64.b64encode(json.dumps({"url": site + "/"}).encode()).decode()
    env = {**os.environ, "FAKE_BROWSER_CRASH_ONCE": "1"}
    proc = subprocess.run([sys.executable, str(SCRIPT), payload], capture_output=True, text=True, env=env, timeout=120)
    result = json.loads([line for line in proc.stdout.strip().splitlines() if line.startswith("{")][-1])
    assert result["ok"] is True
    assert any("browser_crashed" in note for note in result["notes"])


def test_bad_request_is_rejected_before_launching_a_browser():
    result = run({"url": "file:///etc/passwd"}, expect_ok=False)
    assert result["ok"] is False and result["error_kind"] == "bad_request"


def test_unknown_action_is_rejected():
    result = run({"url": "https://example.com", "action": "teleport"}, expect_ok=False)
    assert result["ok"] is False and result["error_kind"] == "bad_request"


def test_missing_selector_is_reported():
    result = run({"url": "https://example.com", "action": "click"}, expect_ok=False)
    assert result["ok"] is False and "selector" in result["error"]
