#!/usr/bin/env python3
"""Camoufox browser helper for the Blackthorn computer. Runs ON THE COMPUTER.

Usage: browse.py <base64-encoded JSON request>
Request keys:
  url          (required) http or https URL
  action       "open" (default) | "screenshot"
  timeout_ms   navigation timeout, 5000..60000 (default 25000)
  screenshot   output path for a PNG (optional, under the workspace directory)
  max_chars    text cap (default 6000, max 20000)
  storage      path of a cookie/storage JSON file, loaded if present and saved after the request

Prints one JSON object on stdout. Navigation timeouts get exactly one retry with a
looser wait condition. A crashed browser gets exactly one relaunch. Nothing else is retried.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time

ALLOWED_SCHEMES = ("http://", "https://")
DEFAULT_STORAGE = "/tmp/blackthorn_browser/storage.json"


def out(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def load_request() -> dict:
    raw = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        req = json.loads(base64.b64decode(raw.encode()).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        out({"ok": False, "error_kind": "bad_request", "error": "request must be base64 JSON"})
        sys.exit(0)
    if not isinstance(req, dict) or not str(req.get("url", "")).startswith(ALLOWED_SCHEMES):
        out({"ok": False, "error_kind": "bad_request", "error": "url must start with http:// or https://"})
        sys.exit(0)
    return req


def is_timeout(exc: Exception) -> bool:
    return type(exc).__name__ == "TimeoutError" or "Timeout" in str(exc)[:200]


def is_crash(exc: Exception) -> bool:
    text = str(exc)[:400].lower()
    return "target closed" in text or "browser has been closed" in text or "crashed" in text


def _navigate(page, url: str, timeout_ms: int, notes: list):
    try:
        return page.goto(url, wait_until="load", timeout=timeout_ms)
    except Exception as exc:
        if not is_timeout(exc):
            raise
        notes.append("navigation timed out at load; retried once at domcontentloaded")
        return page.goto(url, wait_until="domcontentloaded", timeout=min(60000, timeout_ms * 2))


def _ensure_browser_runtime(notes: list) -> None:
    """Rehydrate rebuildable browser deps after a Kaggle session restart.

    Workspace checkpoints retain the package inventory but exclude .cache, so
    the browser release must be fetched again. Never overwrite workspace data.
    """
    import importlib.util
    import pathlib
    import subprocess
    if importlib.util.find_spec("camoufox") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-input",
                        "--disable-pip-version-check", "camoufox"],
                       check=True, capture_output=True, timeout=180)
        notes.append("restored Camoufox Python package")
    cache = pathlib.Path(os.path.expanduser("~/.cache/camoufox/browsers"))
    if not cache.exists() or not any(cache.rglob("camoufox-bin")):
        subprocess.run([sys.executable, "-m", "camoufox", "fetch"],
                       check=True, capture_output=True, timeout=220)
        notes.append("downloaded Camoufox browser after restart")
    if sys.platform.startswith("linux"):
        import ctypes.util
        if ctypes.util.find_library("gtk-3") is None:
            subprocess.run(["apt-get", "install", "-y", "-qq", "libgtk-3-0t64"],
                           check=True, capture_output=True, timeout=180)
            notes.append("restored GTK runtime for browser")


def _session(req: dict, notes: list, started: float) -> dict:
    _ensure_browser_runtime(notes)
    from camoufox.sync_api import Camoufox
    url = req["url"]
    timeout_ms = max(5000, min(60000, int(req.get("timeout_ms") or 25000)))
    max_chars = max(500, min(20000, int(req.get("max_chars") or 6000)))
    storage = req.get("storage") or DEFAULT_STORAGE
    shot = req.get("screenshot")
    with Camoufox(headless=True, humanize=False) as browser:
        context = browser.new_context(storage_state=storage if os.path.exists(storage) else None)
        page = context.new_page()
        response = _navigate(page, url, timeout_ms, notes)
        page.wait_for_timeout(500)
        text = page.evaluate("() => (document.body ? document.body.innerText : '')") or ""
        links = page.evaluate("() => Array.from(document.querySelectorAll('a[href]')).slice(0, 40)"
                              ".map(a => ({text: (a.innerText || '').trim().slice(0, 80), href: a.href}))")
        result = {
            "ok": True, "title": page.title(), "final_url": page.url,
            "status": response.status if response else None,
            "text": text[:max_chars], "text_truncated": len(text) > max_chars,
            "links": links, "notes": notes,
            "elapsed_s": round(time.monotonic() - started, 2),
        }
        if shot:
            page.screenshot(path=shot, full_page=False)
            result["screenshot"] = shot
        os.makedirs(os.path.dirname(storage), exist_ok=True)
        context.storage_state(path=storage)
        return result


def run(req: dict) -> dict:
    """One session, plus at most one relaunch when the browser process has crashed."""
    notes: list = []
    started = time.monotonic()
    try:
        return _session(req, notes, started)
    except Exception as exc:
        if not is_crash(exc):
            raise
        notes.append("browser crashed; relaunched once")
    return _session(req, notes, started)


def main() -> None:
    req = load_request()
    try:
        result = run(req)
    except Exception as exc:  # single top-level handler: converts any failure into a reported error_kind
        kind = "timeout" if is_timeout(exc) else ("crash" if is_crash(exc) else "error")
        out({"ok": False, "error_kind": kind, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        return
    out(result)


if __name__ == "__main__":
    main()
