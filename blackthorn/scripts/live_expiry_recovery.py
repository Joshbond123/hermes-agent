#!/usr/bin/env python3
"""Live reproduction of the production defect, against the real PandaStack API.

    "Provider VM expired while idle; the next computer tool call will create a
     replacement and restore the GitHub workspace."

The provider TTL cannot be fast-forwarded, so the expiry is reproduced the way the
provider performs it: the sandbox is deleted out from under the agent. What must then
happen without human help:

  1. the next computer tool call notices the machine is gone;
  2. a replacement sandbox is created and probed with real work;
  3. the workspace is restored from the private GitHub drive;
  4. the call that was in flight succeeds against the replacement;
  5. work done *before* the expiry is still there (no repeated work).

    PANDASTACK_API_KEY=... COMPUTER_DRIVE_GITHUB_PAT=... python3 blackthorn/scripts/live_expiry_recovery.py

Exit code 0 only when all of that is verified.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import httpx

from blackthorn.panda_computer import PandaComputerClient
from blackthorn.panda_control import INTENT, SANDBOX, PandaControl
from blackthorn.panda_keys import PandaKeyPool

RESULTS: list = []


def record(step: str, ok: bool, detail: object = "") -> None:
    RESULTS.append((step, bool(ok)))
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
    started = time.time()
    store = Store()
    await store.set_setting(INTENT, "on")
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, read=300, write=300)) as http:
        pool = PandaKeyPool.from_env(http, store)
        control = PandaControl(store, http, pool or key)
        await control.start()
        victim = await store.get_setting(SANDBOX)
        record("initial sandbox ready", bool(victim), victim)
        computer = PandaComputerClient(http, api_key=pool or key, sandbox_id=victim,
                                       github_token=drive, store=store, key_pool=pool)
        try:
            marker = "recovery/expiry-marker.txt"
            payload = f"written before the provider expired at {int(time.time())}\n"
            await computer.call("/computer/write_file", {"path": marker, "content": payload})
            record("work written and checkpointed", True, marker)

            # ---- the provider takes the machine away (TTL reap, host loss, ...)
            deleted = await control.delete(victim)
            record("provider-side expiry reproduced", deleted, victim)
            await asyncio.sleep(2)

            # ---- the next tool call must transparently recover
            recovered = await computer.call("/computer/read_file", {"path": marker}, timeout=180)
            replacement = await store.get_setting(SANDBOX)
            record("next call succeeded against a replacement",
                   str(recovered.get("content") or "").strip() == payload.strip(), replacement)
            record("a different sandbox was created", replacement != victim, f"{victim} -> {replacement}")

            probe = await computer.call("/computer/exec", {"command": "python3 -c 'print(6*7)'", "timeout_seconds": 60})
            record("replacement executes real work", probe.get("output", "").strip() == "42",
                   probe.get("output", "").strip())

            out = await computer.call("/computer/exec",
                                      {"command": f"cat /workspace/{marker}", "timeout_seconds": 60})
            record("workspace restored on the replacement",
                   payload.strip() in str(out.get("output") or ""), str(out.get("output"))[:120])

            snapshot = await computer.lifecycle_snapshot()
            record("lifecycle snapshot reports the live sandbox",
                   str(snapshot.get("sandbox_id") or "") == replacement, snapshot.get("sandbox_id"))
        finally:
            final = await control.destroy()
            record("sandbox released", final.get("status") == "off", final.get("status"))

    failed = [name for name, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed in {int(time.time() - started)}s")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
