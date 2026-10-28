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
import subprocess
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
def _disk_dir(preferred: str, minimum_gb: float = 2.0) -> str:
    """Pick a directory with room for a ~1.5 GB browser download.

    A sandbox's ``/tmp`` is frequently a small tmpfs; downloading the browser there
    fails with ``No space left on device`` halfway through the fetch. Any candidate
    that does not have room is skipped in favour of one that does.
    """
    import shutil
    candidates = [preferred, os.environ.get("BLACKTHORN_BROWSER_TMP", ""),
                  "/workspace/.cache", os.path.join(os.path.expanduser("~"), ".cache"), "/var/tmp", "/tmp"]
    best, best_free = "", -1.0
    for candidate in candidates:
        if not candidate:
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
            free = shutil.disk_usage(candidate).free / (1024 ** 3)
        except OSError:
            continue
        if free >= minimum_gb:
            return candidate
        if free > best_free:
            best, best_free = candidate, free
    return best or "/tmp"


def _prepare_disk(notes: list) -> str:
    """Point pip/camoufox at disk-backed scratch space and return the cache root."""
    # An explicit override always wins (operators pin the cache to a mounted volume);
    # otherwise pick whichever candidate actually has room for the browser bundle.
    cache_root = os.environ.get("BLACKTHORN_BROWSER_CACHE", "") or _disk_dir("/workspace/.cache")
    tmp_root = os.environ.get("BLACKTHORN_BROWSER_TMP", "") or _disk_dir(cache_root)
    for target in (os.path.join(tmp_root, "tmp"), os.path.join(cache_root, "pip")):
        try:
            os.makedirs(target, exist_ok=True)
        except OSError:
            continue
    os.environ["TMPDIR"] = os.path.join(tmp_root, "tmp")
    os.environ["XDG_CACHE_HOME"] = cache_root
    os.environ.setdefault("PIP_NO_INPUT", "1")
    os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    notes.append(f"browser scratch at {os.environ['TMPDIR']} (cache {cache_root})")
    return cache_root


# Firefox needs these at load time. A bare microVM image ships almost none of them, and
# the failure without them is an opaque ``XPCOMGlueLoad error`` from inside the bundle.
LINUX_LIBS = (
    "libgtk-3-0t64", "libdbus-glib-1-2", "libxt6t64", "libasound2t64", "libx11-xcb1",
    "libxtst6", "libnss3", "libnspr4", "libgbm1", "libxrandr2", "libxcomposite1",
    "libxdamage1", "libxfixes3", "libatk1.0-0t64", "libatk-bridge2.0-0t64", "libcups2t64",
    "libpango-1.0-0", "libcairo2", "libatspi2.0-0t64", "libxkbcommon0", "libdrm2",
)


def _browser_binary(cache_root: str) -> str:
    for root in (pathlib.Path(cache_root) / "camoufox",
                 pathlib.Path(os.path.expanduser("~/.cache/camoufox"))):
        if not root.exists():
            continue
        for hit in sorted(root.rglob("camoufox-bin")):
            return str(hit)
    return ""


def _missing_libs(binary: str) -> list:
    """Shared objects the browser binary cannot resolve (precise, unlike a name guess)."""
    if not binary:
        return []
    try:
        proc = _run(["ldd", binary], 60)
    except (OSError, subprocess.SubprocessError):
        return []
    missing = []
    for line in (proc.stdout or "").splitlines():
        if "not found" in line:
            name = line.split("=>")[0].strip()
            if name and name not in missing:
                missing.append(name)
    return missing


def _install_linux_libs(binary: str) -> tuple:
    """Install the Firefox shared-library set. Returns (ok, detail)."""
    _run(["apt-get", "update", "-qq"], 600, env={"DEBIAN_FRONTEND": "noninteractive"})
    proc = _run(["apt-get", "install", "-y", "-qq", *LINUX_LIBS], 900,
                env={"DEBIAN_FRONTEND": "noninteractive"})
    remaining = _missing_libs(binary)
    if remaining:
        return False, "browser libraries still missing: " + ", ".join(remaining[:6]) + \
                      " (" + (proc.stderr or "")[-160:] + ")"
    return True, "restored the Firefox system libraries"


def _is_bundle_damage(exc: Exception) -> bool:
    """A half-extracted or self-repairing browser install, not a page or script error.

    Camoufox's launcher re-extracts its bundle when it believes the install is
    incomplete, which can collide with an existing directory (``OSError [Errno 39]
    Directory not empty .../fonts``) and then fails *every* later call. The only
    correct recovery is to discard the bundle and fetch it again once.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in (
        "camoufoxnotinstalled", "directory not empty", "errno 39", "no space left",
        "corrupted", "checksum", "not installed", "please run `camoufox fetch`",
        "xpcomglueload",
    ))


def _repair_bundle(cache_root: str) -> bool:
    """Delete a damaged bundle and download it once more. Returns True on success."""
    import shutil as _shutil
    browsers = pathlib.Path(cache_root) / "camoufox" / "browsers"
    for target in (browsers, pathlib.Path(os.path.expanduser("~/.cache/camoufox/browsers"))):
        try:
            if target.exists():
                _shutil.rmtree(target, ignore_errors=True)
        except OSError:
            pass
    try:
        proc = _run([sys.executable, "-m", "camoufox", "fetch"], 1800,
                    env={"XDG_CACHE_HOME": cache_root})
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and _browser_present(cache_root)


def _warm_browser(cache_root: str, notes: list) -> None:
    """Complete the expensive one-time browser start-up *before* a user action needs it.

    The first launch of a fresh bundle installs the fingerprint model and extracts the
    bundled addons, which can take minutes on a small microVM. Paying that inside the
    first ``open`` makes a normal tool call look like a timeout, so a stamped warm-up is
    done here: launch, load a blank page, close. A warm bundle costs one cheap launch.
    """
    import importlib.util
    stamp = pathlib.Path(cache_root) / "camoufox" / ".blackthorn-warm"
    if stamp.exists():
        return
    if importlib.util.find_spec("camoufox") is None:
        return
    try:
        from camoufox.sync_api import Camoufox
    except Exception as exc:  # a broken import surfaces at launch with a real message
        notes.append(f"browser warm-up skipped: {type(exc).__name__}")
        return
    for attempt in (1, 2):
        try:
            with Camoufox(headless=True, humanize=False) as browser:
                context = browser.new_context()
                page = context.new_page()
                page.goto("about:blank")
                context.close()
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == 1 and _is_bundle_damage(exc):
                notes.append(f"damaged browser bundle detected ({type(exc).__name__}); refetching once")
                if _repair_bundle(cache_root):
                    notes.append("browser bundle refetched")
                    continue
            notes.append(f"browser warm-up failed: {type(exc).__name__}: {str(exc)[:120]}")
            return
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(str(int(time.time())))
        notes.append("browser warmed once for this sandbox")
    except OSError:
        pass


def _browser_present(cache_root: str) -> bool:
    roots = [pathlib.Path(cache_root) / "camoufox",
             pathlib.Path(os.path.expanduser("~/.cache/camoufox"))]
    for root in roots:
        if root.exists() and any(root.rglob("camoufox-bin")):
            return True
    return False


def _run(cmd: list, timeout: int, env: dict | None = None):
    """Run a bootstrap command, capturing output (tests replace this hook)."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, **(env or {})})


def _ensure_runtime(notes: list) -> None:
    """Rehydrate rebuildable browser dependencies after a sandbox replacement.

    Steps run only when something is actually missing, and each failure is recorded
    rather than silently swallowed: a missing GTK runtime surfaces as a dependency
    error from the browser launch, not as a mysterious crash.
    """
    import importlib.util
    if os.environ.get("BLACKTHORN_BROWSER_NO_BOOTSTRAP") == "1":
        notes.append("runtime bootstrap skipped by configuration")
        return
    cache_root = _prepare_disk(notes)
    if importlib.util.find_spec("camoufox") is None:
        proc = _run([sys.executable, "-m", "pip", "install", "--no-input",
                     "--disable-pip-version-check", "camoufox"], 900)
        if proc.returncode != 0:
            raise RuntimeError("camoufox package could not be installed: " + (proc.stderr or "")[-200:])
        notes.append("restored Camoufox Python package")
    if not _browser_present(cache_root):
        proc = _run([sys.executable, "-m", "camoufox", "fetch"], 1800,
                    env={"XDG_CACHE_HOME": cache_root})
        if proc.returncode != 0:
            raise RuntimeError("camoufox browser download failed: " + ((proc.stderr or proc.stdout or "")[-300:]))
        notes.append("downloaded Camoufox browser after replacement")
    _warm_browser(cache_root, notes)
    if sys.platform.startswith("linux"):
        binary = _browser_binary(cache_root)
        missing = _missing_libs(binary)
        if missing:
            ok, detail = _install_linux_libs(binary)
            notes.append(detail)
            if not ok:
                raise RuntimeError("Camoufox cannot start: " + detail)


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


def _accessibility_tree(context, page, depth: int) -> tuple:
    """Bounded accessibility tree, with fallbacks for Playwright builds that dropped ``page.accessibility``."""
    try:
        if getattr(page, "accessibility", None) is not None:
            return _trim_tree(page.accessibility.snapshot() or {}, depth=depth), "playwright"
    except Exception:
        pass
    try:
        aria = page.locator("body").aria_snapshot(timeout=5000)
        if aria:
            return {"role": "document", "aria": str(aria)[:8000]}, "aria"
    except Exception:
        pass
    try:
        flat = page.evaluate(
            "(depth) => { const out = []; const walk = (el, d) => { if (!el || d > depth) return;"
            " const role = el.getAttribute && (el.getAttribute('role') || el.tagName.toLowerCase());"
            " const name = (el.getAttribute && (el.getAttribute('aria-label') || el.getAttribute('name') || ''))"
            " || (el.innerText ? String(el.innerText).trim().slice(0, 80) : '');"
            " if (role) out.push({role: role, name: name.slice(0, 80), depth: d});"
            " if (el.children) Array.from(el.children).slice(0, 30).forEach(c => walk(c, d + 1)); };"
            " walk(document.body, 0); return out.slice(0, 200); }", depth)
        return {"role": "document", "nodes": flat or []}, "dom"
    except Exception:
        return {}, "unavailable"


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
                tree, source = _accessibility_tree(context, page, int(req.get("depth") or 4))
                result = _snapshot(page, req, notes, started, status)
                result["accessibility"] = tree
                result["accessibility_source"] = source
            elif action == "tabs":
                if req.get("url"):
                    # Open a second tab so tab handling is meaningful: a session may need the
                    # original page kept open while a second one is inspected or scraped.
                    extra = context.new_page()
                    extra.set_default_timeout(timeout_ms)
                    try:
                        extra.goto(str(req["url"]), wait_until="domcontentloaded", timeout=timeout_ms)
                    except Exception as exc:
                        notes.append(f"secondary tab navigation failed: {type(exc).__name__}")
                tabs = [{"index": i, "url": p.url, "title": _safe_title(p)} for i, p in enumerate(context.pages)]
                if req.get("close_tab") is not None:
                    index = int(req["close_tab"])
                    if 0 <= index < len(context.pages) and len(context.pages) > 1:
                        try:
                            context.pages[index].close()
                        except Exception:
                            pass
                        tabs = [{"index": i, "url": p.url, "title": _safe_title(p)} for i, p in enumerate(context.pages)]
                result = {"ok": True, "action": "tabs", "tabs": tabs, "notes": notes,
                          "elapsed_s": round(time.monotonic() - started, 2)}
            elif action == "download":
                if not req.get("url"):
                    bad("bad_request", "download requires a url")
                target = str(req.get("download_path") or os.path.join("downloads", f"download-{time.time_ns()}"))
                os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
                # Direct fetch through the browser's request context first: a plain-text URL
                # renders inline, so waiting for a download event alone can hang until the
                # tool-level timeout. Fall back to the event path for URLs that need it.
                body, source = b"", "request"
                try:
                    response = context.request.get(str(req["url"]), timeout=timeout_ms)
                    if response.ok:
                        body = response.body()
                except Exception:
                    body = b""
                if not body:
                    source = "event"
                    with page.expect_download(timeout=min(30000, timeout_ms)) as info:
                        try:
                            page.goto(str(req["url"]), wait_until="domcontentloaded", timeout=timeout_ms)
                        except Exception:
                            pass
                    download = info.value
                    download.save_as(target)
                else:
                    with open(target, "wb") as fh:
                        fh.write(body)
                result = {"ok": True, "action": "download", "path": target, "source": source,
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
            damaged = _is_bundle_damage(exc)
            kind = ("timeout" if _is_timeout(exc) else
                    "browser_crashed" if _is_crash(exc) else
                    "dependency" if damaged or type(exc).__name__ in
                    ("ModuleNotFoundError", "ImportError", "FileNotFoundError")
                    else "browser_error")
            if damaged and not any("refetch" in note for note in notes):
                cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
                notes.append(f"dependency failure ({type(exc).__name__}); refetching the browser bundle once")
                if _repair_bundle(cache_root):
                    notes.append("browser bundle refetched")
                    attempts -= 1
                    continue
            if attempts < 2 and kind in ("browser_crashed", "timeout"):
                notes.append(f"{kind}: relaunched the browser once")
                continue
            out({"ok": False, "error_kind": kind, "error": str(exc)[:400], "notes": notes,
                 "attempts": attempts, "elapsed_s": round(time.monotonic() - started, 2)})
            return


if __name__ == "__main__":
    main()
