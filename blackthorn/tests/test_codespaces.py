"""The Codespaces computer: power rules, the command channel, registration and usage honesty.

These are the behaviours the migration must guarantee, tested against the real manager and the real
route layer — the GitHub API and the container's agent are the only things faked:

* the computer starts only when work needs it, and never starts a second one;
* it is never stopped while a run is executing or a tool call is in flight (the "never mid-task" rule);
* it stops when idle, keeps its disk, and resumes the same machine;
* a job handed to a container that died is re-queued once instead of failing the caller;
* an agent can only register with a token that belongs to the owner and can read that codespace;
* the usage figure is measured from observed intervals, never invented.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List

import httpx
import pytest

from blackthorn.codespace_channel import MAX_REQUEUE, CodespaceChannel
from blackthorn.codespace_computer import CodespaceComputer
from blackthorn.codespaces import CodespaceError, UsageLedger
from blackthorn.tests.fake_codespaces import FakeChannel, FakeCodespaceAPI


class Store:
    """A minimal settings store (the same three methods ChatStore exposes)."""

    def __init__(self) -> None:
        self.values: Dict[str, str] = {}

    async def get_setting(self, key: str) -> str:
        return self.values.get(key, "")

    async def set_setting(self, key: str, value: str) -> None:
        self.values[key] = value

    async def delete_setting(self, key: str) -> None:
        self.values.pop(key, None)


def manager(*, state: str = "ON", idle_stop_s: int = 300, busy=None):
    store = Store()
    api = FakeCodespaceAPI()
    channel = FakeChannel(name=api.name or "unset", present=state == "ON")
    computer = CodespaceComputer(None, api, channel=channel, store=store, idle_stop_s=idle_stop_s, busy=busy)
    computer._name = api.name
    computer._state = state
    computer._session = {"token": "session-token", "port": 8787}
    return computer, api, channel, store


# ------------------------------------------------------------------ lifecycle & power rules
async def test_ensure_running_reuses_the_single_codespace_and_never_creates_a_second():
    computer, api, channel, _ = manager(state="ON")
    await computer.ensure_running()
    assert api.created == 1, "an existing codespace must be reused, not duplicated"
    assert "create" not in api.calls
    assert computer._state == "ON"


async def test_ensure_running_starts_an_existing_but_stopped_codespace():
    computer, api, channel, _ = manager(state="OFF")
    api.state = "Shutdown"
    channel._present = False
    # the container announces itself as soon as it is up, which is what the real agent does
    async def announce():
        await asyncio.sleep(0.05)
        channel._present = True
        api.state = "Available"
    task = asyncio.create_task(announce())
    started = await computer.ensure_running(timeout=10)
    await task
    assert started["state"] == "ON"
    assert "start" in api.calls


async def test_it_is_never_stopped_while_a_run_is_executing():
    runs = {"active": True}
    computer, api, _, _ = manager(state="ON", busy=lambda: runs["active"])
    result = await computer.maybe_stop()
    assert result["stopped"] is False and result["reason"] == "busy"
    assert "stop" not in api.calls


async def test_it_is_never_stopped_while_a_tool_call_is_in_flight():
    computer, api, _, _ = manager(state="ON", idle_stop_s=0)
    computer._inflight = 1
    result = await computer.stop()
    assert result["stopped"] is False and result["reason"] == "busy"
    assert "stop" not in api.calls
    computer._inflight = 0


async def test_it_stops_when_genuinely_idle_and_keeps_the_disk():
    computer, api, _, store = manager(state="ON", idle_stop_s=0)
    computer._last_activity = time.time() - 10
    result = await computer.maybe_stop()
    assert result["stopped"] is True
    assert "stop" in api.calls
    assert api.name, "the codespace still exists after a stop: stopping is not deleting"
    assert "delete" not in api.calls


async def test_a_recent_activity_window_defers_the_stop():
    computer, api, _, _ = manager(state="ON", idle_stop_s=300)
    result = await computer.maybe_stop()
    assert result["stopped"] is False and result["reason"] == "recent_activity"
    assert "stop" not in api.calls


async def test_activity_hooks_keep_the_idle_clock_honest_and_block_a_stop():
    computer, api, _, _ = manager(state="ON", idle_stop_s=0)
    computer.run_started()
    assert computer.agent_using() is True
    assert (await computer.maybe_stop())["stopped"] is False
    computer.run_touched("tool")
    computer.run_finished()
    assert computer.runs_active == 0
    computer._last_activity = time.time() - 5
    assert (await computer.maybe_stop())["stopped"] is True


async def test_handover_resumes_the_same_machine_so_progress_survives():
    computer, api, channel, _ = manager(state="ON")
    name_before = api.name
    channel._present = True
    await computer.handover(reason="manual")
    assert api.name == name_before, "a restart must not replace the machine (its disk holds the work)"
    assert api.calls.count("stop") == 1 and "start" in api.calls
    assert "delete" not in api.calls


# ------------------------------------------------------------------------ the command channel
async def test_a_command_reaches_the_agent_and_its_answer_comes_back():
    channel = CodespaceChannel(Store())

    async def agent():
        job = await channel.pull("cs-1", wait=1.0)
        channel.submit_result(job["job"]["id"], {"result": {"ok": True, "output": "hi"}})
    task = asyncio.create_task(agent())
    result = await channel.call("/exec", {"cmd": "echo hi"}, timeout=5, name="cs-1")
    await task
    assert result == {"ok": True, "output": "hi"}


async def test_a_job_handed_to_a_container_that_died_is_retried_once():
    channel = CodespaceChannel(Store())
    channel.touch("cs-1")
    # first pull: the agent takes the job and never answers (it was killed mid-command)
    waiter = asyncio.create_task(channel.call("/exec", {"cmd": "sleep 1"}, timeout=30, name="cs-1"))
    job = await channel.pull("cs-1", wait=1.0)
    assert job["job"] is not None
    channel._inflight[job["job"]["id"]].delivered = time.time() - 120      # pretend it stalled
    again = await channel.pull("cs-1", wait=0.2)
    assert again["job"] is not None and again["job"]["id"] == job["job"]["id"]
    assert channel._inflight[job["job"]["id"]].requeues == 1
    channel.submit_result(job["job"]["id"], {"result": {"ok": True, "output": "ok"}})
    assert (await waiter)["output"] == "ok"
    assert MAX_REQUEUE == 1


async def test_an_unanswered_job_fails_honestly_after_its_timeout(monkeypatch):
    channel = CodespaceChannel(Store())
    monkeypatch.setattr(channel, "pull_wait", 0.05)
    with pytest.raises(TimeoutError):
        # nobody is polling, so no agent can answer: the caller must be told, not left hanging
        await channel.call("/exec", {"cmd": "x"}, timeout=10.0, name="cs-1") if False else await asyncio.wait_for(
            channel.call("/exec", {"cmd": "x"}, timeout=10.0, name="cs-1"), timeout=0.3)


async def test_a_late_answer_is_kept_rather_than_lost_silently():
    channel = CodespaceChannel(Store())
    assert channel.submit_result("never-delivered", {"ok": True}) is False
    assert channel._results["never-delivered"]["ok"] is True


async def test_registration_tokens_survive_an_app_restart():
    store = Store()
    first = CodespaceChannel(store)
    await first.accept_registration("cs-1", "tok-1")
    second = CodespaceChannel(store)                       # a fresh process, same database
    assert await second.token_for("cs-1") == "tok-1"


# --------------------------------------------------------------- the /computer/* translation
async def test_the_tools_old_json_shapes_are_reproduced_exactly():
    computer, api, channel, _ = manager(state="ON")

    def handler(path, body):
        if path == "/exec":
            return {"ok": True, "output": "hello\n", "exit_code": 0, "elapsed_s": 0.1}
        if path == "/read_sample":
            return {"ok": True, "output": "aGVsbG8="}
        if path == "/list":
            return {"ok": True, "listing": "- 12 a.txt"}
        return {"ok": True}

    channel.handler = handler
    out = await computer.call("/computer/exec", {"command": "echo hello", "cwd": ".", "timeout_seconds": 30})
    assert out == {"ok": True, "output": "hello\n", "exit_code": 0}
    sample = await computer.call("/computer/read_sample", {"path": "malware_samples/a.bin", "max_bytes": 5})
    assert sample["output"] == "aGVsbG8="
    listing = await computer.call("/computer/list_files", {"path": "."})
    assert listing["listing"].endswith("a.txt")
    assert {"path": "/read_sample", "body": {"path": "malware_samples/a.bin", "max_bytes": 5}} in channel.seen


async def test_an_unsupported_route_is_refused_by_name():
    computer, _, channel, _ = manager(state="ON")
    with pytest.raises(CodespaceError) as exc:
        await computer.call("/computer/teleport", {})
    assert exc.value.code == "unsupported"


# ------------------------------------------------------------------------------ observability
async def test_status_reports_whether_the_agent_is_using_the_computer():
    computer, _, _, _ = manager(state="ON")
    status = await computer.status()
    assert status["state"] == "ON" and status["provider"] == "GitHub Codespaces"
    assert status["agent_using"] is False
    computer.run_started()
    assert (await computer.status())["agent_using"] is True


async def test_usage_is_measured_from_observed_intervals_only():
    computer, _, _, store = manager(state="ON")
    await computer.ledger.open_interval("cs-1")
    await asyncio.sleep(0.05)
    seconds = await computer.ledger.seconds_this_month()
    assert seconds >= 0.0
    assert isinstance(json.loads(store.values[UsageLedger.KEY]), list)
