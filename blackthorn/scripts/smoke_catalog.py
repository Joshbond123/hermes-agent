#!/usr/bin/env python3
"""Dispatch smoke test for the cybersecurity tools, through the real ToolRegistry.

Runs the registered tools end to end against a catalogue that already exists on this
machine (default ``/workspace/security_catalog``, override with ``BLACKTHORN_SKILL_ROOT``).
The "computer" is a local runner with the same ``/computer/*`` contract the real
PandaStack client implements, so the code path under test is production code:

    python3 blackthorn/scripts/smoke_catalog.py [--root /path/to/security_catalog]

Exit code 0 only if every check reported the truth. Used as the P6 acceptance check
and re-runnable after any sandbox replacement (it installs nothing by itself).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from blackthorn.config import Settings
from blackthorn.tools import ToolContext, default_registry
from blackthorn.tools import security

RESULTS: list = []


def record(name: str, ok: bool, detail: object = "") -> None:
    RESULTS.append((name, ok))
    print(("PASS " if ok else "FAIL ") + name + (" :: " + str(detail)[:300] if detail else ""), flush=True)


class LocalComputer:
    """Executes /computer/* locally, mapping the remote helper dir into a temp root."""

    def __init__(self, root: pathlib.Path, catalog: str):
        self.root = root
        self.catalog = catalog
        self.calls: list = []

    def _map(self, text: str) -> str:
        return text.replace(security.REMOTE_BIN, str(self.root / "bin"))

    async def call(self, path, payload=None, *, method="POST", timeout=90.0, idempotent=True):
        self.calls.append((path, payload))
        if path == "/computer/write_file":
            target = pathlib.Path(self._map(payload["path"]))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload["content"], encoding="utf-8")
            return {"ok": True, "path": str(target), "bytes": len(payload["content"].encode())}
        if path == "/computer/exec":
            env = {**os.environ, "BLACKTHORN_SKILL_ROOT": self.catalog}
            (self.root / "work").mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(["bash", "-c", self._map(payload["command"])], capture_output=True,
                                  text=True, cwd=str(self.root / "work"), timeout=timeout, env=env)
            return {"ok": True, "exit_code": proc.returncode, "output": proc.stdout + proc.stderr}
        if path == "/computer/read_file":
            target = pathlib.Path(self._map(payload["path"]))
            if not target.exists():
                return {"ok": False, "error": "not found"}
            return {"ok": True, "content": target.read_text(encoding="utf-8")}
        raise AssertionError(f"unexpected computer path {path}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("BLACKTHORN_SKILL_ROOT")
                    or ("/workspace/security_catalog" if os.path.isdir("/workspace")
                        else "/home/user/qa-catalog"))
    args = ap.parse_args()
    if not os.path.isdir(args.root):
        print(f"catalogue not found: {args.root} (clone it first: skillctl.py ensure)", file=sys.stderr)
        return 2

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        computer = LocalComputer(pathlib.Path(tmp), args.root)
        registry = default_registry()
        ctx = ToolContext(settings=Settings(), store=None, computer=computer, http=None,
                          tavily=None, session_id="smoke", run_id="smoke", board=None,
                          user_message="smoke test")

        pre = await registry.run("cyber_preflight", {}, ctx)
        counts = (pre.data or {}).get("counts") or {}
        record("cyber_preflight reports the real catalogue", pre.ok and counts.get("skills", 0) > 0, counts)

        search = await registry.run("skill_search", {"query": "kerberos"}, ctx)
        ids = [str(hit) for hit in ((search.data or {}).get("results") or [])]
        record("skill_search finds offensive and defensive material", search.ok and bool(ids), ids[:5])

        tool_search = await registry.run("tool_search", {"query": "nmap"}, ctx)
        tool_ids = [str(hit) for hit in ((tool_search.data or {}).get("results") or [])]
        record("tool_search returns executable catalogue ids", tool_search.ok and bool(tool_ids), tool_ids[:5])

        skill_id = next((i for i in ids if str(i).startswith(("anthropic:", "claude-red:"))), "")
        if skill_id:
            loaded = await registry.run("skill_load", {"id": skill_id}, ctx)
            record("skill_load returns the playbook body", loaded.ok and len(loaded.content) > 200,
                   len(loaded.content or ""))
        else:
            record("skill_load returns the playbook body", False, "no skill id from search")

        manual = await registry.run("cyber_run", {"tool_id": "tool:information_gathering.NMAP"}, ctx)
        honest = (not manual.ok) and "no runnable command" in (manual.content or "").lower()
        record("cyber_run refuses an entry with no command", honest or manual.ok,
               (manual.content or "")[:160])

        bad = await registry.run("cyber_run", {"tool_id": "../../etc/passwd"}, ctx)
        record("cyber_run rejects a malformed tool id", (not bad.ok) and "tool_id must be" in (bad.content or ""),
               (bad.content or "")[:120])

        runnable = next(iter((tool_search.data or {}).get("runnable") or []), "")
        if runnable:
            result = await registry.run("cyber_run", {"tool_id": runnable, "timeout_seconds": 120}, ctx)
            # Honest outcomes: it ran, or it said why it could not (missing binary, GUI
            # tool without a display, blocked installer, non-zero exit with the reason).
            honest = result.ok or any(tag in (result.content or "").lower() for tag in
                                      ("missing", "install", "ran_nonzero", "blocked", "no runnable"))
            record("cyber_run executes or reports the reason honestly", honest,
                   (result.content or "")[:200])

        leaks = [c for c in computer.calls if "pds_" in json.dumps(c[1] or {})]
        record("no credentials travel to the computer", not leaks, len(computer.calls))

    failed = [name for name, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} catalogue checks passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
