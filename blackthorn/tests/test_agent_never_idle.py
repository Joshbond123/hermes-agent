"""P8 — the agent must never look idle while it is actually working.

A long tool call (a browser install, a scan, a build) runs for minutes between
``tool.start`` and ``tool.end``. Before this change that gap was total silence: the
stream, the UI and any watchdog could not tell "working" from "hung". The run now
reports elapsed time on a fixed cadence while the tool is in flight.
"""
from __future__ import annotations

import time

import pytest

from blackthorn.config import Settings
from .fakes import call, finish, say


def run_command(cmd: str, cid: str = "c1"):
    return call("run_command", {"command": cmd}, cid=cid)


@pytest.fixture()
def fast_progress(monkeypatch):
    """Report progress quickly so the test does not take a minute.

    Must be requested before ``stack``: the app builds its Settings at fixture time.
    """
    original = Settings.__init__

    def patched(self, *args, **kwargs):
        kwargs.setdefault("tool_progress_s", 0.2)
        original(self, *args, **kwargs)

    monkeypatch.setattr(Settings, "__init__", patched)


async def test_a_slow_tool_reports_progress_until_it_finishes(fast_progress, stack):
    stack.backend.exec_delay = 1.2
    stack.backend.queue([run_command("sleep 1")],
                        [*say("Finished the slow command."), finish("stop")])
    out = await stack.stream({"message": "run something slow"})

    progress = out.of("tool.progress")
    assert len(progress) >= 3, f"expected interim progress events, got {len(progress)}"
    assert all(p["id"] == "c1" for p in progress)
    assert [p["elapsed_s"] for p in progress] == sorted(p["elapsed_s"] for p in progress)
    # progress arrives *before* the tool finishes, not batched at the end
    ends = out.of("tool.end")
    assert ends and ends[0]["status"] == "ok"
    first_progress, tool_end = progress[0], ends[0]
    assert out.times[out.events.index(first_progress)] < out.times[out.events.index(tool_end)]
    assert out.text.strip()


async def test_progress_stops_when_the_tool_is_done(fast_progress, stack):
    stack.backend.exec_delay = 0.5
    stack.backend.queue([run_command("echo quick")],
                        [*say("Done."), finish("stop")])
    out = await stack.stream({"message": "quick"})
    lines = [e for e in out.events if e["type"] in ("tool.progress", "tool.end")]
    assert lines[-1]["type"] == "tool.end", "no progress heartbeat may follow completion"
    assert out.of("tool.progress") and out.of("tool.end")[0]["status"] == "ok"
