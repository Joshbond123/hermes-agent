#!/usr/bin/env python3
"""Live proof that the agent's computer really is a GitHub Codespace.

Run against the real GitHub API and the real deployed app (nothing is simulated):

    COMPUTER_DRIVE_GITHUB_PAT=... python3 blackthorn/scripts/live_codespaces.py

It proves, in order:

1. the codespace is found or created — and that exactly one exists for the repository (no duplicates);
2. a command really executes inside it (kernel, user, python version printed by the machine itself);
3. a file written there survives a stop → start cycle (so a task can be resumed, not restarted);
4. the power rules hold: a stop is refused while work is "in flight", allowed when idle;
5. the usage ledger reports a measured figure (never a placeholder).

Every codespace it creates is stopped at the end (stopping keeps the disk; deleting would throw away
the very state this script is meant to prove).
"""

from __future__ import annotations

import asyncio
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx

from blackthorn.codespace_channel import CodespaceChannel
from blackthorn.codespace_computer import CodespaceComputer
from blackthorn.codespaces import CodespaceAPI, CodespaceError

PROOF_FILE = f"proof/{int(time.time())}.txt"
MARKER = secrets.token_hex(8)


class Store:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def get_setting(self, key: str) -> str:
        return self.values.get(key, "")

    async def set_setting(self, key: str, value: str) -> None:
        self.values[key] = value

    async def delete_setting(self, key: str) -> None:
        self.values.pop(key, None)


results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)


async def main() -> int:
    token = (__import__("os").environ.get("COMPUTER_DRIVE_GITHUB_PAT") or "").strip()
    if not token:
        print("COMPUTER_DRIVE_GITHUB_PAT is required", file=sys.stderr)
        return 2
    repo = __import__("os").environ.get("CODESPACE_REPO", "jadeniran21/Computer-drive")
    async with httpx.AsyncClient(timeout=60) as http:
        api = CodespaceAPI(http, token, repo=repo, idle_timeout_minutes=30, retention_minutes=4320)
        channel = CodespaceChannel(Store())
        computer = CodespaceComputer(http, api, channel=channel, store=Store(), idle_stop_s=0)
        computer.channel = channel

        # -- 1. one codespace, created if absent ------------------------------------------------
        cs = await computer.find_or_create()
        name = computer._name
        check("a codespace exists for the repository", bool(name), name)
        others = [c for c in await api.list_all() if ((c.get("repository") or {}).get("full_name") == repo)]
        check("exactly one codespace (no duplicates)", len(others) <= 1, f"{len(others)} found")

        # -- 2. it starts and its agent attaches ------------------------------------------------
        started = await computer.ensure_running(timeout=600)
        check("the codespace started and its agent attached", started.get("state") == "ON",
              f"channel present={channel.present(name)} known={channel.known(name)}")

        # The live app must be reachable for registration; when it is not, the channel cannot have a
        # token, and the script says so instead of pretending the exec channel works.
        if not channel.known(name):
            check("agent registered with the app", False,
                  "no registration token: the deployed app has not seen this codespace yet "
                  "(its agent registers on start; check /api/codespaces/status on the app)")
        else:
            check("agent registered with the app", True)

        # -- 3. a command really runs there -----------------------------------------------------
        try:
            out = await computer.call("/computer/exec",
                                      {"command": "uname -s; id -un; python3 -V; pwd", "cwd": ".", "timeout_seconds": 60},
                                      timeout=120)
            text = str(out.get("output") or "")
            check("a command executed inside the codespace", out.get("ok") is True and "Linux" in text,
                  text.strip().replace("\n", " | ")[:160])
        except CodespaceError as exc:
            check("a command executed inside the codespace", False, f"{exc.code}: {exc.message}")

        # -- 4. a file written there survives a stop/start --------------------------------------
        wrote = await computer.call("/computer/write_file",
                                    {"path": PROOF_FILE, "content": f"marker={MARKER}\n"}, timeout=60)
        check("a file was written to the codespace disk", wrote.get("ok") is True, PROOF_FILE)

        await computer.stop(reason="manual", force=True)
        check("the codespace stopped (container kept, not deleted)", computer._state == "OFF")
        await asyncio.sleep(5)
        await computer.ensure_running(timeout=600)
        back = await computer.call("/computer/read_file", {"path": PROOF_FILE}, timeout=90)
        check("the file survived stop → start (work is resumable)", MARKER in str(back.get("content") or ""),
              str(back.get("content") or "").strip()[:80])

        # -- 5. the power rules ----------------------------------------------------------------
        computer._inflight = 1
        refused = await computer.stop()
        computer._inflight = 0
        check("a stop is refused while work is in flight", refused.get("stopped") is False, refused.get("reason", ""))

        computer._last_activity = time.time() - 10
        idle = await computer.maybe_stop()
        check("an idle computer stops itself", idle.get("stopped") is True, idle.get("reason", ""))

        # -- 6. usage is measured, not invented ------------------------------------------------
        seconds = await computer.ledger.seconds_this_month()
        check("usage hours are measured from observed intervals", seconds > 0, f"{seconds / 3600:.3f} h recorded")

        # leave the machine stopped: everything above is on its persistent disk
        print("\nThe codespace was left stopped (its disk holds the proof file).")
        print(f"codespace: {name}")

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
