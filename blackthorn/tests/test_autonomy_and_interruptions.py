"""Autonomous capability use and interruption resistance.

Covers the two behaviours the operator asked for directly:

* the agent must be able to *decide* a task needs a skill or a tool, so the
  discovery and execution tools are always offered — never keyword-gated;
* a running task must not be destroyed by a computer rotation, an expiry or a
  dropped chat connection.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from blackthorn.agent import AgentRun as AgentRunner, Deps
from blackthorn.prompts import BASE_PROMPT


class FakeRun:
    def __init__(self):
        self.id = "run_test"
        self.session_id = "s1"
        self.cancel_event = asyncio.Event()

    def emit(self, *a, **k):
        return {}


@dataclass
class FakeSettings:
    context_tokens: int = 8192
    completion_reserve_tokens: int = 1200
    max_repeat_calls: int = 3
    tool_result_chars: int = 4000


def _runner(user_message: str, turns=None) -> AgentRunner:
    from blackthorn.tools import default_registry
    deps = Deps(settings=FakeSettings(), store=None, resolver=None, registry=default_registry(),
                http=None, tavily=None, computer=None)
    return AgentRunner(FakeRun(), deps, system="sys", history=[], user_message=user_message,
                       turns=turns or [])


def _offered(runner: AgentRunner) -> set:
    return {s["function"]["name"] for s in runner._active_schemas}


@pytest.mark.parametrize("message", [
    "Audit this server for weak SSH configuration and report what you find",
    "Here is a suspicious binary, tell me what it does",
    "Check whether this invoice email is a scam",
])
def test_discovery_tools_are_always_offered(message):
    offered = _offered(_runner(message))
    for tool in ("skill_search", "skill_load", "tool_search", "cyber_run"):
        assert tool in offered, f"{tool} must be available for the agent to choose it itself"


@pytest.mark.parametrize("message", [
    "Here is a suspicious binary, tell me what it does",
    "Set up a Windows characterisation run and execute the sample end to end",
])
def test_lab_tools_are_always_offered_without_the_word_malware(message):
    offered = _offered(_runner(message))
    for tool in ("lab_detonate", "lab_exec", "lab_os", "lab_close"):
        assert tool in offered


def test_a_plain_greeting_still_offers_discovery_but_not_a_flood():
    offered = _offered(_runner("hi"))
    assert "skill_search" in offered and "tool_search" in offered
    assert "browse" not in offered, "unrelated heavy tools stay undisclosed"


def test_the_prompt_tells_the_agent_to_pick_its_own_capabilities():
    assert "tool_search" in BASE_PROMPT and "skill_search" in BASE_PROMPT
    assert "Decide yourself" in BASE_PROMPT


# --------------------------------------------------------------------------- rotation safety
class BusyRuns:
    def __init__(self, busy: bool):
        self.busy = busy

    def active_count(self) -> int:
        return 1 if self.busy else 0


class FakeComputer:
    def __init__(self):
        self.handovers: List[str] = []

    async def handover(self, *, reason="manual"):
        self.handovers.append(reason)
        return {"ok": True, "sandbox_id": "new"}


def _snapshot(due=True):
    return {"sandbox_id": "sid-1", "status": "running", "seconds_left": 12.0, "handover_due": due}


@pytest.mark.asyncio
async def test_an_active_run_is_never_rotated_out_from_under_itself(monkeypatch):
    from blackthorn.lifecycle import SandboxLifecycle

    class Store:
        def __init__(self):
            self.values = {}

        async def get_setting(self, key):
            return self.values.get(key, "")

        async def set_setting(self, key, value):
            self.values[key] = value

    computer = FakeComputer()
    manager = SandboxLifecycle(Store(), computer)
    manager.runs = BusyRuns(True)
    monkeypatch.setattr(manager, "snapshot", lambda: _async(_snapshot()))
    manager._last_tick = -1e9
    result = await manager.tick()
    assert result["action"] == "deferred" and result["reason"] == "active_run"
    assert computer.handovers == [], "no handover while a turn is executing"


@pytest.mark.asyncio
async def test_the_same_lifecycle_still_rotates_when_the_agent_is_idle(monkeypatch):
    from blackthorn.lifecycle import SandboxLifecycle

    class Store:
        def __init__(self):
            self.values = {}

        async def get_setting(self, key):
            return self.values.get(key, "")

        async def set_setting(self, key, value):
            self.values[key] = value

    computer = FakeComputer()
    manager = SandboxLifecycle(Store(), computer)
    manager.runs = BusyRuns(False)
    monkeypatch.setattr(manager, "snapshot", lambda: _async(_snapshot()))
    monkeypatch.setattr(manager, "handover", lambda **kw: _async({"ok": True, "sandbox_id": "new"}))
    manager._last_tick = -1e9
    result = await manager.tick()
    assert result["action"] == "handover"


def _async(value):
    async def inner():
        return value
    return inner()
