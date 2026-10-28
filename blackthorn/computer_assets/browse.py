#!/usr/bin/env python3
"""Camoufox browser controller for the Blackthorn computer. Runs ON THE COMPUTER.

One JSON request in, one JSON result out. The request is base64-encoded JSON so
nothing the model writes can break the shell quoting.

Actions
-------
open          navigate (default) and return title/url/status/text/links
click         click a CSS selector or text locator, then summarise the page
type          fill a field (``selector`` + ``text``); ``submit`` presses Enter
scroll        scroll the viewport (``scroll_px``, negative scrolls up)
screenshot    save a PNG (``screenshot`` path, default under browser_screenshots/)
accessibility return a bounded accessibility snapshot of the page
tabs          list open tabs (``close_tab`` to close one)
download      click/await a download and save it under the workspace
close         shut the browser down (resources are freed on every exit anyway)

Robustness contract
-------------------
* restart-once on a crashed/closed browser;
* one retry for navigation timeouts with a looser wait condition;
* every action is bounded by ``timeout_ms``;
* the browser and its context are always closed;
* browser deps are rehydrated after a sandbox replacement (``_ensure_runtime``).

CAPTCHA: the controller *detects and reports* challenges (reCAPTCHA, hCaptcha,
Cloudflare Turnstile/interstitial, generic "verify you are human" pages) and can
pause for a human or for Camoufox's stealth profile. It never claims to have
solved a challenge it did not solve.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import sys
import time

ALLOWED_SCHEMES = ("http://", "https://")
DEFAULT_STORAGE = "/tmp/blackthorn_browser/storage.json"
DEFAULT_SHOT_DIR = "browser_screenshots"
MAX_TEXT = 20000
ACTIONS = {"open", "click", "type", "scroll", "screenshot", "accessibility", "tabs", "download", "close"}

CAPTCHA_MARKERS = (
    ("recaptcha", "iframe[src*='recaptcha']"),
    ("hcaptcha", "iframe[src*='hcaptcha']"),
    ("turnstile", "iframe[src*='challenges.cloudflare.com']"),
    ("cloudflare", "#challenge-form, #cf-challenge-running, .cf-browser-verification"),
    ("generic", "form[action*='captcha' i], input[name*='captcha' i], #captcha"),
)
CAPTCHA_TEXT = ("verify you are human", "checking your browser", "are you a robot",
                "unusual traffic", "complete the security check", "just a moment")


def out(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def bad(kind: str, message: str) -> None:
    out({"ok": False, "error_kind": kind, "error": message})
    sys.exit(0)


def load_request() -> dict:
    raw = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        req = json.loads(base64.b64decode(raw.encode()).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        bad("bad_request", "request must be base64-encoded JSON")
    if not isinstance(req, dict):
        bad("bad_request", "request must be a JSON object")
    action = str(req.get("action") or "open").lower()
    if action not in ACTIONS:
        bad("bad_request", f"action must be one of {sorted(ACTIONS)}")
    url = str(req.get("url") or "")
    if action in ("open", "download") and url and not url.startswith(ALLOWED_SCHEMES):
        bad("bad_request", "url must start with http:// or https://")
    if action == "open" and not url:
        bad("bad_request", "open requires a url")
    return req


# ------------------------------------------------------------------ runtime
def _ensure_runtime(notes: list) -> None:
    """Rehydrate rebuildable browser dependencies after a sandbox replacement.

    Steps run only when something is actually missing, and each failure is recorded
    rather than silently swallowed: a missing GTK runtime surfaces as a dependency
    error from the browser launch, not as a mysterious crash.
    """
    import importlib.util
    import subprocess
    if os.environ.get("BLACKTHORN_BROWSER_NO_BOOTSTRAP") == "1":
        notes.append("runtime bootstrap skipped by configuration")
        return
    if importlib.util.find_spec("camoufox") is None:
        proc = subprocess.run([sys.executable, "-m", "pip", "install", "--no-input",
                               "--disable-pip-version-check", "camoufox"],
                              capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            raise RuntimeError("camoufox package could not be installed: " + (proc.stderr or "")[-200:])
        notes.append("restored Camoufox Python package")
    cache = pathlib.Path(os.path.expanduser("~/.cache/camoufox"))
    if not cache.exists() or not any(cache.rglob("camoufox-bin")):
        proc = subprocess.run([sys.executable, "-m", "camoufox", "fetch"],
                              capture_output=True, text=True, timeout=1200)
        if proc.returncode != 0:
            raise RuntimeError("camoufox browser download failed: " + (proc.stderr or "")[-200:])
        notes.append("downloaded Camoufox browser after replacement")
    if sys.platform.startswith("linux"):
        import ctypes.util
        if ctypes.util.find_library("gtk-3") is None:
            proc = subprocess.run(["apt-get", "install", "-y", "-qq", "libgtk-3-0t64"],
                                  capture_output=True, text=True, timeout=600)
            notes.append("restored GTK runtime for the browser" if proc.returncode == 0
                         else "GTK runtime install failed (browser launch may fail)")


# Backwards-compatible name used by older callers/tests.
_ensure_browser_runtime = _ensure_runtime


def _is_timeout(exc: Exception) -> bool:
    return type(exc).__name__ == "TimeoutError" or "timeout" in str(exc)[:300].lower()


def _is_crash(exc: Exception) -> bool:
    text = str(exc)[:400].lower()
    return ("target closed" in text or "browser has been closed" in text or "crashed" in text
            or "connection closed" in text or "browser closed" in text)


def _navigate(page, url: str, timeout_ms: int, notes: list):
    try:
        return page.goto(url, wait_until="load", timeout=timeout_ms)
    except Exception as exc:
        if not _is_timeout(exc):
            raise
        notes.append("navigation timed out at load; retried once at domcontentloaded")
        return page.goto(url, wait_until="domcontentloaded", timeout=min(60000, timeout_ms * 2))


# ------------------------------------------------------------------ page helpers
def _text(page, limit: int) -> str:
    try:
        value = page.evaluate("() => (document.body ? document.body.innerText : '')") or ""
    except Exception:
        value = ""
    return str(value)[:limit]


def _links(page, limit: int = 40) -> list:
    try:
        return page.evaluate(
            "(limit) => Array.from(document.querySelectorAll('a[href]')).slice(0, limit)"
            ".map(a => ({text: (a.innerText || '').trim().slice(0, 80), href: a.href}))", limit)
    except Exception:
        return []


def _detect_challenge(page) -> dict:
    """Report a bot challenge; never claim to have solved one."""
    found = []
    for name, selector in CAPTCHA_MARKERS:
        try:
            if page.query_selector(selector) is not None:
                found.append(name)
        except Exception:
            continue
    text = _text(page, 4000).lower()
    hint = next((phrase for phrase in CAPTCHA_TEXT if phrase in text), "")
    if not found and not hint:
        return {}
    return {"present": True, "kinds": found or ["textual"], "hint": hint,
            "note": "A bot challenge is present. Do not claim it was solved; the stealth profile may pass "
                    "automatically, otherwise a human or an external solving service is required."}


def _locator(page, selector: str):
    return page.locator(selector).first


def _snapshot(page, req: dict, notes: list, started: float, status=None) -> dict:
    limit = max(500, min(MAX_TEXT, int(req.get("max_chars") or 6000)))
    text = _text(page, limit)
    result = {"ok": True, "action": str(req.get("action") or "open"), "title": page.title(),
              "final_url": page.url, "status": status, "text": text,
              "text_truncated": len(text) >= limit, "links": _links(page), "notes": notes,
              "elapsed_s": round(time.monotonic() - started, 2)}
    challenge = _detect_challenge(page)
    if challenge:
        result["captcha"] = challenge
    return result


def _save_screenshot(page, req: dict, result: dict) -> None:
    target = str(req.get("screenshot") or "")
    if not target:
        os.makedirs(DEFAULT_SHOT_DIR, exist_ok=True)
        target = f"{DEFAULT_SHOT_DIR}/shot-{time.time_ns()}.png"
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    page.screenshot(path=target, full_page=bool(req.get("full_page")))
    result["screenshot"] = target


# ------------------------------------------------------------------ session
def run_session(req: dict, notes: list, started: float) -> dict:
    _ensure_runtime(notes)
    from camoufox.sync_api import Camoufox
    action = str(req.get("action") or "open").lower()
    timeout_ms = max(3000, min(90000, int(req.get("timeout_ms") or 25000)))
    storage = str(req.get("storage") or DEFAULT_STORAGE)
    cookies = storage if os.path.exists(storage) else None
    with Camoufox(headless=bool(req.get("headless", True)), humanize=False) as browser:
        context = browser.new_context(storage_state=cookies,
                                      accept_downloads=True,
                                      viewport={"width": int(req.get("width") or 1280),
                                                "height": int(req.get("height") or 800)})
        page = context.new_page()
        page.set_default_timeout(timeout_ms)
        status = None
        try:
            if action == "open":
                response = _navigate(page, str(req["url"]), timeout_ms, notes)
                status = response.status if response else None
                page.wait_for_timeout(int(req.get("wait_ms") or 500))
                result = _snapshot(page, req, notes, started, status)
            elif action == "click":
                if req.get("url"):
                    response = _navigate(page, str(req["url"]), timeout_ms, notes)
                    status = response.status if response else None
                selector = str(req.get("selector") or "").strip()
                if not selector:
                    bad("bad_request", "click requires a CSS selector")
                _locator(page, selector).click(timeout=timeout_ms)
                page.wait_for_timeout(int(req.get("wait_ms") or 400))
                result = _snapshot(page, req, notes, started, status)
            elif action == "type":
                selector = str(req.get("selector") or "").strip()
                if not selector:
                    bad("bad_request", "type requires a CSS selector")
                if req.get("url"):
                    response = _navigate(page, str(req["url"]), timeout_ms, notes)
                    status = response.status if response else None
                field = _locator(page, selector)
                field.click(timeout=timeout_ms)
                field.fill(str(req.get("text") or ""), timeout=timeout_ms)
                if req.get("submit"):
                    field.press("Enter")
                    page.wait_for_timeout(int(req.get("wait_ms") or 800))
                result = _snapshot(page, req, notes, started, status)
                result["typed"] = len(str(req.get("text") or ""))
            elif action == "scroll":
                if req.get("url"):
                    response = _navigate(page, str(req["url"]), timeout_ms, notes)
                    status = response.status if response else None
                pixels = int(req.get("scroll_px") or 800)
                page.mouse.wheel(0, pixels)
                page.wait_for_timeout(int(req.get("wait_ms") or 400))
                result = _snapshot(page, req, notes, started, status)
                result["scrolled_px"] = pixels
            elif action == "screenshot":
                if req.get("url"):
                    response = _navigate(page, str(req["url"]), timeout_ms, notes)
                    status = response.status if response else None
                    page.wait_for_timeout(int(req.get("wait_ms") or 400))
                result = _snapshot(page, req, notes, started, status)
                _save_screenshot(page, req, result)
            elif action == "accessibility":
                if req.get("url"):
                    response = _navigate(page, str(req["url"]), timeout_ms, notes)
                    status = response.status if response else None
                snapshot = page.accessibility.snapshot() or {}
                result = _snapshot(page, req, notes, started, status)
                result["accessibility"] = _trim_tree(snapshot, depth=int(req.get("depth") or 4))
            elif action == "tabs":
                tabs = [{"index": i, "url": p.url, "title": _safe_title(p)} for i, p in enumerate(context.pages)]
                if req.get("close_tab") is not None:
                    index = int(req["close_tab"])
                    if 0 <= index < len(context.pages) and len(context.pages) > 1:
                        context.pages[index].close()
                        tabs = [{"index": i, "url": p.url, "title": _safe_title(p)} for i, p in enumerate(context.pages)]
                result = {"ok": True, "action": "tabs", "tabs": tabs, "notes": notes,
                          "elapsed_s": round(time.monotonic() - started, 2)}
            elif action == "download":
                if not req.get("url"):
                    bad("bad_request", "download requires a url")
                target = str(req.get("download_path") or os.path.join("downloads", f"download-{time.time_ns()}"))
                os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
                with page.expect_download(timeout=timeout_ms) as info:
                    try:
                        page.goto(str(req["url"]), wait_until="domcontentloaded", timeout=timeout_ms)
                    except Exception:
                        pass
                download = info.value
                download.save_as(target)
                result = {"ok": True, "action": "download", "path": target,
                          "suggested_filename": download.suggested_filename,
                          "bytes": os.path.getsize(target) if os.path.exists(target) else 0,
                          "notes": notes, "elapsed_s": round(time.monotonic() - started, 2)}
            else:  # close
                result = {"ok": True, "action": "close", "closed": True, "notes": notes,
                          "elapsed_s": round(time.monotonic() - started, 2)}
            if result.get("ok") and req.get("screenshot") and "screenshot" not in result and action != "screenshot":
                _save_screenshot(page, req, result)
            # Persist session state so a later call keeps cookies/logins.
            try:
                os.makedirs(os.path.dirname(storage) or ".", exist_ok=True)
                context.storage_state(path=storage)
                result["session_state_saved"] = True
            except Exception:
                result["session_state_saved"] = False
            return result
        finally:
            try:
                context.close()
            except Exception:
                pass


def _safe_title(page) -> str:
    try:
        return page.title()
    except Exception:
        return ""


def _trim_tree(node, depth: int) -> dict:
    """Bound an accessibility snapshot so it cannot flood the model context."""
    if not isinstance(node, dict) or depth <= 0:
        return {}
    trimmed = {key: node.get(key) for key in ("role", "name", "value", "description", "pressed", "checked")
               if node.get(key) not in (None, "")}
    children = node.get("children") or []
    if children:
        trimmed["children"] = [_trim_tree(child, depth - 1) for child in children[:25]]
    return trimmed


def main() -> None:
    started = time.monotonic()
    req = load_request()
    notes: list = []
    attempts = 0
    while True:
        attempts += 1
        try:
            out(run_session(req, notes, started))
            return
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            kind = ("timeout" if _is_timeout(exc) else
                    "browser_crashed" if _is_crash(exc) else
                    "dependency" if type(exc).__name__ in ("ModuleNotFoundError", "ImportError", "FileNotFoundError")
                    else "browser_error")
            if attempts < 2 and kind in ("browser_crashed", "timeout"):
                notes.append(f"{kind}: relaunched the browser once")
                continue
            out({"ok": False, "error_kind": kind, "error": str(exc)[:400], "notes": notes,
                 "attempts": attempts, "elapsed_s": round(time.monotonic() - started, 2)})
            return


if __name__ == "__main__":
    main()
