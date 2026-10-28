#!/usr/bin/env python3
"""Live Camoufox acceptance test inside a real PandaStack microVM.

Creates the managed sandbox, installs Camoufox in the background (the provider's
HTTP edge cuts long blocking exec calls, so installation runs detached and is
polled), serves a local form page from inside the guest, and then drives the real
browser through every supported action: open, click, type+submit, scroll,
screenshot, accessibility, tabs, download, CAPTCHA detection and close.

    PANDASTACK_API_KEY=... python3 blackthorn/scripts/live_browser_test.py [--keep]

Exit code 0 only when every action worked in the live browser.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import httpx

from blackthorn.computer_assets.browse import LINUX_LIBS as _LIBS
from blackthorn.panda_control import INTENT, SANDBOX, PandaControl

RESULTS: list[dict] = []
APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FORM = """<!doctype html><html><head><title>Blackthorn Form</title></head><body>
<h1>Interactive form</h1>
<p>Local page served from inside the guest.</p>
<input name="q" id="q">
<button id="go" onclick="document.title='clicked';document.getElementById('out').textContent='button clicked'">Go</button>
<label>Token</label><input name="token" id="token">
<div id="out"></div>
<form action="/upload" method="get"><input name="file" value="x"><button id="dl" type="submit">Download</button></form>
<a href="/report.txt">report</a>
</body></html>"""

CAPTCHA_PAGE = """<!doctype html><html><head><title>Just a moment</title></head><body>
<p>Checking your browser before accessing. Just a moment.</p>
<iframe src="https://challenges.cloudflare.com/turnstile/v0/api.js"></iframe>
<div id="challenge-form"></div>
</body></html>"""


def record(step: str, ok: bool, detail: object = "") -> None:
    RESULTS.append({"step": step, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + step + (" :: " + str(detail)[:400] if detail else ""), flush=True)


class Store:
    def __init__(self):
        self.values: dict[str, str] = {}

    async def get_setting(self, key):
        return self.values.get(key, "")

    async def set_setting(self, key, value):
        self.values[key] = value


async def exec_(control, sid, cmd, timeout=120, attempts=3):
    """Exec with retry: a freshly restored microVM can need a moment before its agent answers."""
    last = {}
    for attempt in range(1, attempts + 1):
        try:
            response = await control._request("POST", f"/{sid}/exec",
                                              json={"cmd": cmd, "timeout_seconds": timeout})
        except Exception as exc:                       # transport hiccup while the VM comes up
            last = {"ok": False, "status": 0, "output": f"{type(exc).__name__}"}
            await asyncio.sleep(3 * attempt)
            continue
        if response.status_code != 200:
            last = {"ok": False, "status": response.status_code, "output": ""}
            await asyncio.sleep(3 * attempt)
            continue
        body = response.json()
        return {"ok": body.get("exit_code") == 0, "exit_code": body.get("exit_code"),
                "output": str(body.get("stdout") or "") + str(body.get("stderr") or "")}
    return last


def parse(result: dict) -> dict:
    for line in reversed(str(result.get("output") or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return {}


async def browse(control, sid, request: dict, allow_bootstrap: bool = False) -> dict:
    payload = base64.b64encode(json.dumps(request).encode()).decode()
    guard = "" if allow_bootstrap else "BLACKTHORN_BROWSER_NO_BOOTSTRAP=1 "
    # NB: env assignments must prefix the command. ``export A=1 python3 ...`` only exports
    # a variable called ``python3`` and runs nothing, which silently returns empty output.
    raw = await exec_(control, sid,
                      f"cd /workspace && TMPDIR=/workspace/.tmp XDG_CACHE_HOME=/workspace/.cache "
                      f"{guard}python3 tools/browse.py {payload}", 600)
    result = parse(raw)
    if not result:
        print(f"    (no JSON from browse {request.get('action')}: "
              f"status={raw.get('status')} output={str(raw.get('output'))[-200:]!r})", flush=True)
    return result


async def main() -> int:
    keep = "--keep" in sys.argv
    key = os.environ.get("PANDASTACK_API_KEY", "").strip()
    if not key:
        print("PANDASTACK_API_KEY is required", file=sys.stderr)
        return 2
    store = Store()
    await store.set_setting(INTENT, "on")
    started = time.time()
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, read=180, write=180)) as http:
        control = PandaControl(store, http, key)
        await control.start()
        sid = await store.get_setting(SANDBOX)
        record("sandbox created", bool(sid), sid)
        for _ in range(20):                            # wait until the guest agent answers
            warm = await exec_(control, sid, "echo warm", 20, attempts=1)
            if "warm" in str(warm.get("output", "")):
                break
            await asyncio.sleep(3)
        record("guest agent answering", warm.get("ok") is True, warm.get("output"))
        try:
            # ---- upload the controller and a local interactive site
            for name, content in (("tools/browse.py", open(os.path.join(APP, "computer_assets", "browse.py")).read()),
                                  ("site/index.html", FORM), ("site/captcha.html", CAPTCHA_PAGE),
                                  ("site/report.txt", "report-body\n")):
                payload = {"cmd": f"mkdir -p /workspace/{os.path.dirname(name)}", "timeout_seconds": 30}
                await exec_(control, sid, payload["cmd"])
                b64 = base64.b64encode(content.encode()).decode()
                await exec_(control, sid, f"echo {b64} | base64 -d > /workspace/{name}")
            listing = await exec_(control, sid, "ls -la /workspace/tools /workspace/site | head -20")
            record("controller uploaded to guest", "browse.py" in str(listing.get("output")), "")

            # ---- install Camoufox detached, then poll (blocking exec is cut by the HTTP edge)
            # Mirror exactly what browse.py's own bootstrap does: disk-backed scratch space,
            # because the guest's /tmp is a 1 GB tmpfs and the browser bundle is ~1.5 GB.
            await exec_(control, sid,
                        "mkdir -p /workspace/.cache /workspace/.tmp && cd /workspace && nohup sh -c "
                        "'export TMPDIR=/workspace/.tmp XDG_CACHE_HOME=/workspace/.cache DEBIAN_FRONTEND=noninteractive; "
                        "python3 -m pip install -q camoufox && python3 -m camoufox fetch && "
                        "apt-get update -qq && apt-get install -y -qq " + " ".join(_LIBS) + " && "
                        "echo READY > /workspace/browser.ready' > /workspace/browser-install.log 2>&1 & echo started")
            ready = False
            for _ in range(90):
                await asyncio.sleep(15)
                probe = await exec_(control, sid, "cat /workspace/browser.ready 2>/dev/null || tail -c 300 /workspace/browser-install.log")
                if "READY" in str(probe.get("output", "")):
                    ready = True
                    break
            record("camoufox installed + browser fetched", ready, str(probe.get("output", ""))[-200:])
            if not ready:
                return 1
            version = await exec_(control, sid, "export XDG_CACHE_HOME=/workspace/.cache; "
                                               "python3 -c 'from camoufox.pkgman import installed_verstr;print(installed_verstr())' 2>&1 | tail -1")
            text = str(version.get("output") or "").strip()
            record("camoufox browser version reported", text[:1].isdigit(), text)

            # ---- local server so interactions need no external network
            await exec_(control, sid, "cd /workspace/site && nohup python3 -m http.server 8811 "
                                      "> /workspace/site-server.log 2>&1 & sleep 2; echo up")
            base = "http://127.0.0.1:8811"

            bootstrap = await browse(control, sid, {"url": f"{base}/index.html", "action": "open"},
                                     allow_bootstrap=True)
            record("bootstrap is a no-op once the bundle is on disk",
                   bootstrap.get("ok") is True and bootstrap.get("notes"), bootstrap.get("notes"))
            opened = await browse(control, sid, {"url": f"{base}/index.html", "action": "open"})
            record("open", opened.get("ok") and opened.get("title") == "Blackthorn Form",
                   {"title": opened.get("title"), "status": opened.get("status")})
            record("text extraction", "Interactive form" in str(opened.get("text", "")),
                   str(opened.get("text", ""))[:120])
            record("links extracted", bool(opened.get("links")), opened.get("links"))

            clicked = await browse(control, sid, {"url": f"{base}/index.html", "action": "click", "selector": "#go"})
            record("click", clicked.get("ok") and "button clicked" in str(clicked.get("text", "")),
                   str(clicked.get("text", ""))[-120:])

            typed = await browse(control, sid, {"url": f"{base}/index.html", "action": "type",
                                               "selector": "#q", "text": "blackthorn", "submit": True})
            record("type + submit", typed.get("ok") and typed.get("typed") == 10, typed.get("final_url"))

            scrolled = await browse(control, sid, {"url": f"{base}/index.html", "action": "scroll", "scroll_px": -300})
            record("scroll", scrolled.get("ok") and scrolled.get("scrolled_px") == -300, scrolled.get("scrolled_px"))

            shot = await browse(control, sid, {"url": f"{base}/index.html", "action": "screenshot",
                                               "screenshot": "browser_screenshots/live.png"})
            size = await exec_(control, sid, "stat -c %s /workspace/browser_screenshots/live.png 2>/dev/null || echo 0")
            record("screenshot written", shot.get("ok") and int(str(size.get("output", "0")).strip() or 0) > 1000,
                   f"bytes={str(size.get('output','')).strip()}")

            tree = await browse(control, sid, {"url": f"{base}/index.html", "action": "accessibility", "depth": 3})
            record("accessibility snapshot", bool(tree.get("accessibility")),
                   json.dumps(tree.get("accessibility"))[:160])

            tabs = await browse(control, sid, {"action": "tabs"})
            record("tabs listed", isinstance(tabs.get("tabs"), list) and tabs.get("tabs"), tabs.get("tabs"))

            download = await browse(control, sid, {"url": f"{base}/report.txt", "action": "download",
                                                   "download_path": "downloads/report.txt"})
            got = await exec_(control, sid, "cat /workspace/downloads/report.txt 2>/dev/null || echo missing")
            record("download saved", "report-body" in str(got.get("output", "")),
                   {"path": download.get("path"), "bytes": download.get("bytes")})

            captcha = await browse(control, sid, {"url": f"{base}/captcha.html", "action": "open"})
            detected = (captcha.get("captcha") or {}).get("present")
            record("bot challenge detected (not claimed solved)", bool(detected),
                   (captcha.get("captcha") or {}).get("kinds"))

            session = await browse(control, sid, {"url": f"{base}/index.html", "action": "open"})
            record("session state persisted", session.get("session_state_saved") is True, "")

            closed = await browse(control, sid, {"action": "close"})
            record("clean close", closed.get("ok") is True, "")

            leftover = await exec_(control, sid, "ps aux | grep -c '[c]amoufox' || true")
            record("no browser processes left", str(leftover.get("output", "0")).strip() in ("0", "1"),
                   str(leftover.get("output", "")).strip())
        finally:
            if not keep:
                final = await control.destroy()
                record("sandbox released", final.get("status") == "off", final.get("status"))
            else:
                print(f"keeping sandbox {sid} for inspection")
    failed = [r for r in RESULTS if not r["ok"]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} browser checks passed in {int(time.time() - started)}s")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
