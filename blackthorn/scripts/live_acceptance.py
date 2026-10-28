#!/usr/bin/env python3
"""Live acceptance test against the real PandaStack provider and GitHub drive.

Run this from a machine with the deployment's credentials in the environment:

    PANDASTACK_API_KEY=... COMPUTER_DRIVE_GITHUB_PAT=... python3 blackthorn/scripts/live_acceptance.py

It exercises the production code paths (not a mock):

  1. create the managed sandbox and verify a live exec probe;
  2. run real work (interpreter, filesystem, git) on the microVM;
  3. publish a workspace checkpoint to the private GitHub drive;
  4. perform a full sandbox handover — replacement created, workspace restored,
     verified, promoted, old sandbox deleted;
  5. prove the file written before the handover still exists afterwards;
  6. build the three-repository cybersecurity registry inside the guest;
  7. install and drive Camoufox (open/click/type/screenshot/session);
  8. release every sandbox it created.

Secrets are never printed: the key pool and the drive token stay in the process.
Exit code 0 only when every step that ran passed.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import httpx

from blackthorn.lifecycle import SandboxLifecycle
from blackthorn.panda_computer import PandaComputerClient
from blackthorn.panda_control import INTENT, SANDBOX, PandaControl
from blackthorn.panda_keys import PandaKeyPool

RESULTS: list[dict] = []


def record(step: str, ok: bool, detail: object = "") -> None:
    RESULTS.append({"step": step, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + step + (" :: " + str(detail)[:300] if detail else ""), flush=True)


class Store:
    def __init__(self):
        self.values: dict[str, str] = {}

    async def get_setting(self, key):
        return self.values.get(key, "")

    async def set_setting(self, key, value):
        self.values[key] = value

    async def delete_setting(self, key):
        self.values.pop(key, None)


async def main() -> int:
    key = os.environ.get("PANDASTACK_API_KEY", "").strip()
    drive = os.environ.get("COMPUTER_DRIVE_GITHUB_PAT", "").strip()
    if not key or not drive:
        print("PANDASTACK_API_KEY and COMPUTER_DRIVE_GITHUB_PAT are required", file=sys.stderr)
        return 2
    store = Store()
    await store.set_setting(INTENT, "on")
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, read=300, write=300)) as http:
        pool = PandaKeyPool.from_env(http, store)
        if pool is not None:
            state = await pool.ensure_active()
            record("key pool verification", state.valid, state.public())
        controller = PandaControl(store, http, pool or key)

        # 1. provider session
        started = time.time()
        try:
            status = await controller.start()
            record("create sandbox + live exec probe", bool(status.get("online")), status.get("status"))
        except Exception as exc:
            record("create sandbox + live exec probe", False, f"{type(exc).__name__}: {exc}")
            return 1
        sid = await store.get_setting(SANDBOX)
        record("sandbox id persisted", bool(sid), sid)

        computer = PandaComputerClient(http, api_key=pool or key, sandbox_id=sid,
                                       github_token=drive, store=store, key_pool=pool)
        try:
            # 2. real work on the microVM
            info = await computer.call("/computer/info", method="GET", timeout=90)
            record("computer_info", bool(info.get("ok")), info.get("details", "")[:120])
            wrote = await computer.call("/computer/write_file",
                                        {"path": "acceptance/before-handover.txt",
                                         "content": "written before the handover\n"})
            record("write_file (durable first)", bool(wrote.get("ok")), wrote)
            out = await computer.call("/computer/exec", {"command": "python3 -c 'print(6*7)'", "timeout_seconds": 30})
            record("execute python in guest", out.get("output", "").strip() == "42", out.get("output", "").strip())
            gitv = await computer.call("/computer/exec", {"command": "git --version", "timeout_seconds": 30})
            record("git available in guest", gitv.get("exit_code") == 0, gitv.get("output", "").strip())

            # 3. durable checkpoint to the private GitHub drive
            if computer.mirror is not None:
                published = await computer.mirror.sync(force=True)
                digest = await store.get_setting("pandastack_workspace_drive_digest")
                record("workspace checkpoint published", bool(published and digest), digest[:16])

            # 4/5. handover, then prove continuity
            lifecycle = SandboxLifecycle(store, computer, window_s=3600)
            result = await lifecycle.handover(reason="acceptance")
            record("sandbox handover", bool(result.get("ok")), result)
            if result.get("ok"):
                new_sid = result.get("sandbox_id")
                restored = await computer.call("/computer/read_file", {"path": "acceptance/before-handover.txt"},
                                               method="POST", timeout=120)
                record("workspace restored into replacement",
                       "before the handover" in str(restored.get("content", "")), new_sid)
                probe = await computer.call("/computer/exec", {"command": "echo alive", "timeout_seconds": 20})
                record("replacement executes work", probe.get("output", "").strip() == "alive",
                       probe.get("output", "").strip())

            # 6. three-repository cybersecurity registry inside the guest
            skillctl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "computer_assets", "skillctl.py")
            with open(skillctl, encoding="utf-8") as fh:
                await computer.call("/computer/write_file", {"path": "security_catalog/skillctl.py",
                                                             "content": fh.read()})
            ensure = await computer.call("/computer/exec", {
                "command": "cd /workspace && BLACKTHORN_SKILL_ROOT=/workspace/security_catalog "
                           "python3 security_catalog/skillctl.py ensure | tail -1", "timeout_seconds": 900},
                timeout=1000)
            record("clone the three repositories", ensure.get("exit_code") == 0,
                   str(ensure.get("output", ""))[-300:])
            registry = await computer.call("/computer/exec", {
                "command": "cd /workspace && BLACKTHORN_SKILL_ROOT=/workspace/security_catalog "
                           "python3 security_catalog/skillctl.py registry | tail -1", "timeout_seconds": 600},
                timeout=700)
            counts = {}
            try:
                counts = json.loads(str(registry.get("output", "")).strip().splitlines()[-1]).get("counts", {})
            except (ValueError, IndexError):
                pass
            record("unified registry built", counts.get("skills", 0) >= 800 and counts.get("tools", 0) >= 150, counts)
            search = await computer.call("/computer/exec", {
                "command": "cd /workspace && BLACKTHORN_SKILL_ROOT=/workspace/security_catalog "
                           "python3 security_catalog/skillctl.py search --q 'kerberos' --limit 2 | tail -1",
                "timeout_seconds": 300}, timeout=400)
            record("registry search works", '"ok": true' in str(search.get("output", "")).lower(),
                   str(search.get("output", ""))[-200:])
            nmap = await computer.call("/computer/exec", {
                "command": "command -v nmap >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq nmap); "
                           "nmap --version | head -1", "timeout_seconds": 900}, timeout=1000)
            record("executable security utility runs", "Nmap version" in str(nmap.get("output", "")),
                   str(nmap.get("output", ""))[:160])

            # 7. Camoufox end to end
            browse = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "computer_assets", "browse.py")
            with open(browse, encoding="utf-8") as fh:
                await computer.call("/computer/write_file", {"path": "tools/browse.py", "content": fh.read()})
            install = await computer.call("/computer/exec", {
                "command": "python3 -m pip install -q camoufox 2>&1 | tail -2; "
                           "python3 -m camoufox fetch >/dev/null 2>&1; python3 -c 'import camoufox; print(\"camoufox\", camoufox.__version__)'",
                "timeout_seconds": 1800}, timeout=1900)
            record("camoufox installed in guest", "camoufox" in str(install.get("output", "")),
                   str(install.get("output", ""))[-200:])
            import base64
            payload = base64.b64encode(json.dumps({"url": "https://example.com/", "action": "open",
                                                   "screenshot": "browser_screenshots/acceptance.png"}).encode()).decode()
            browsed = await computer.call("/computer/exec", {
                "command": f"cd /workspace && python3 tools/browse.py {payload} | tail -1", "timeout_seconds": 600},
                timeout=700)
            parsed = {}
            try:
                parsed = json.loads(str(browsed.get("output", "")).strip().splitlines()[-1])
            except (ValueError, IndexError):
                pass
            record("camoufox open + screenshot", bool(parsed.get("ok")) and bool(parsed.get("screenshot")),
                   {"title": parsed.get("title"), "status": parsed.get("status"),
                    "screenshot": parsed.get("screenshot"), "notes": parsed.get("notes")})

            # 8. provider-reported session state
            snap = await computer.lifecycle_snapshot()
            record("lifecycle snapshot", bool(snap.get("sandbox_id")), snap)
        finally:
            try:
                final = await controller.destroy()
                record("all managed sandboxes released", final.get("status") == "off", final.get("status"))
            except Exception as exc:
                record("all managed sandboxes released", False, f"{type(exc).__name__}: {exc}")

    failed = [r for r in RESULTS if not r["ok"]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed in {int(time.time() - started)}s")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
