"""Regression tests for the two production failures.

1. ``Provider VM expired while idle; the next computer tool call will create a
   replacement and restore the GitHub workspace.``
2. ``Stopped using tools (repeated identical tool calls); writing the answer from
   what was gathered.``

The old loop counted a *provider* failure against the duplicate-call budget, so a
recoverable VM expiry produced three "strikes" and forced the model to stop. These
tests drive ``AgentRun._run_tools`` directly (the exact code path) and assert that:

* an infrastructure failure is retried with backoff and never strikes;
* a call that succeeds after the computer recovers is reported as a success;
* an unrecovered infrastructure failure returns an actionable, *different* message;
* a genuinely repeated successful call still strikes and still gets suppressed;
* a long task is not truncated by infrastructure failures (the original bug).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from blackthorn import recovery
from blackthorn.agent import AgentRun, StepResult, Deps
from blackthorn.config import Settings
from blackthorn.runs import Run
from blackthorn.tools.base import ToolContext, ToolError, ToolResult, ToolSpec, ToolRegistry


# --------------------------------------------------------------------------- doubles
class MemoryStore:
    def __init__(self):
        self.values = {}

    async def get_setting(self, key):
        return self.values.get(key, "")

    async def set_setting(self, key, value):
        self.values[key] = value

    async def delete_setting(self, key):
        self.values.pop(key, None)

    async def finish_session_activity(self, *_a, **_k):
        return None


class ScriptedComputer:
    """Fails with a provider error the first ``failures`` times, then succeeds."""

    def __init__(self, failures: int = 2, error: Exception | None = None):
        self.failures = failures
        self.calls = []
        self.error = error or ToolError("offline", "Provider VM expired while idle; "
                                                   "the next computer tool call will create a replacement "
                                                   "and restore the GitHub workspace.")
        self.ensure_calls = 0

    async def call(self, path, payload=None, **_kw):
        self.calls.append((path, payload))
        if self.failures > 0:
            self.failures -= 1
            raise self.error
        return {"ok": True, "exit_code": 0, "output": "437\n", "path": path}

    async def ensure_ready(self):
        self.ensure_calls += 1


class Resolver:
    def __init__(self):
        self.invalidations = 0

    def invalidate(self):
        self.invalidations += 1


def _registry(computer):
    reg = ToolRegistry()

    async def run_command(args, ctx):
        data = await ctx.computer.call("/computer/exec", {"command": args["command"]})
        return ToolResult(ok=True, content=str(data.get("output")), summary="ran", data=data)

    async def write_file(args, ctx):
        return ToolResult(ok=True, content="wrote", summary="wrote", data={"bytes": len(args["content"])})

    reg.register(ToolSpec("run_command", "run", {"type": "object", "properties": {
        "command": {"type": "string"}}, "required": ["command"]}, run_command, kind="exec"))
    reg.register(ToolSpec("write_file", "write", {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
        write_file, kind="write"))
    return reg


def _agent(computer, settings=None):
    settings = settings or Settings(store="sqlite")
    deps = Deps(settings=settings, store=MemoryStore(), resolver=Resolver(),
                registry=_registry(computer), http=None, tavily=None, computer=computer)
    run = Run("session", "assistant")
    return AgentRun(run, deps, system="s", history=[], user_message="do the task"), run


def _call(name, args, cid="c1"):
    import json
    return {"id": cid, "name": name, "args": json.dumps(args)}


# --------------------------------------------------------------------------- policy
def test_classification_separates_infrastructure_from_logic():
    assert recovery.classify(ToolResult.failure("bad", data={"kind": "offline"})) == "infra"
    assert recovery.classify(ToolResult.failure("PandaStack sandbox no longer exists")) == "infra"
    assert recovery.classify(ToolResult.failure("invalid arguments for read_file: missing path")) == "invalid"
    assert recovery.classify(ToolResult.failure("duplicate call suppressed: x")) == "duplicate"
    assert recovery.classify(ToolResult.failure("file not found", data={"kind": "not_found"})) == "logic"
    assert recovery.classify(ToolResult(ok=True, content="fine")) == "ok"


def test_backoff_is_bounded_and_monotonic():
    policy = recovery.RetryPolicy(max_attempts=4, base_delay_s=1.0, max_delay_s=8.0, jitter=0.0)
    delays = [policy.delay(attempt) for attempt in range(1, 6)]
    assert delays[0] < delays[1] < delays[2]
    assert max(delays) <= 8.0
    assert policy.should_retry(1, "infra") and policy.should_retry(3, "infra")
    assert not policy.should_retry(4, "infra"), "attempts are bounded"
    assert not policy.should_retry(1, "logic"), "logic errors are never retried blindly"


def test_exhausted_deadline_stops_retrying():
    policy = recovery.RetryPolicy(max_attempts=9, total_budget_s=0.0)
    assert not policy.should_retry(1, "infra")


# --------------------------------------------------------------------------- retry path
@pytest.mark.asyncio
async def test_expired_vm_is_retried_and_completes_without_a_strike():
    computer = ScriptedComputer(failures=2)          # two provider failures, then success
    agent, _run = _agent(computer)
    messages = [{"role": "user", "content": "go"}]
    await agent._run_tools(StepResult(calls=[_call("run_command", {"command": "echo hi"})]), messages)

    assert len([c for c in computer.calls if c[0] == "/computer/exec"]) == 3, "the call is retried"
    assert agent.strikes == 0, "an infrastructure failure must never strike"
    assert agent.infra_failures == 2
    assert agent.tool_calls_total == 1
    tool_message = [m for m in messages if m.get("role") == "tool"][-1]
    assert tool_message["content"].strip() == "437", "the retried call's real result reaches the model"
    assert agent.deps.registry.get("run_command") is not None


@pytest.mark.asyncio
async def test_recovery_loop_is_bounded_and_reports_an_actionable_error():
    computer = ScriptedComputer(failures=50)          # never recovers
    agent, _run = _agent(computer)
    started = time.monotonic()
    messages = [{"role": "user", "content": "go"}]
    await agent._run_tools(StepResult(calls=[_call("run_command", {"command": "echo hi"})]), messages)
    elapsed = time.monotonic() - started

    assert len([c for c in computer.calls if c[0] == "/computer/exec"]) == agent_max_attempts()
    assert agent.strikes == 0, "still not a duplicate call"
    assert elapsed < 120, "backoff stays inside the budget"
    body = [m for m in messages if m.get("role") == "tool"][-1]["content"]
    assert body.startswith("ERROR:")
    assert "infrastructure failure" in body
    assert "attempt" in body and "switch approach" in body
    assert agent.ledger and agent.ledger[-1]["ok"] is False


def agent_max_attempts() -> int:
    from blackthorn.agent import COMPUTER_MAX_ATTEMPTS
    return COMPUTER_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_repeated_failures_produce_different_messages_not_duplicate_suppression():
    """The exact old bug: five identical VM failures used to be suppressed as duplicates."""
    computer = ScriptedComputer(failures=50)
    agent, _run = _agent(computer)
    messages = [{"role": "user", "content": "go"}]
    for _ in range(2):
        await agent._run_tools(StepResult(calls=[_call("run_command", {"command": "echo hi"})]), messages)
    bodies = [m["content"] for m in messages if m.get("role") == "tool"]
    assert len(bodies) == 2
    assert "duplicate call suppressed" not in "".join(bodies)
    assert bodies[0] != bodies[1], "each failure message is specific, so the model can adapt"
    assert agent.strikes == 0


@pytest.mark.asyncio
async def test_genuine_repeat_of_a_successful_call_still_strikes():
    computer = ScriptedComputer(failures=0)
    agent, _run = _agent(computer)
    messages = [{"role": "user", "content": "go"}]
    await agent._run_tools(StepResult(calls=[_call("run_command", {"command": "ls"})]), messages)
    await agent._run_tools(StepResult(calls=[_call("run_command", {"command": "ls"}, cid="c2")]), messages)
    await agent._run_tools(StepResult(calls=[_call("run_command", {"command": "ls"}, cid="c3")]), messages)

    assert agent.strikes >= 1, "the third identical successful call is a real stall"
    last = [m for m in messages if m.get("role") == "tool"][-1]["content"]
    assert "duplicate call suppressed" in last
    assert "Change strategy" in last


@pytest.mark.asyncio
async def test_a_write_advances_the_epoch_so_a_rerun_is_allowed():
    computer = ScriptedComputer(failures=0)
    agent, _run = _agent(computer)
    messages = [{"role": "user", "content": "go"}]
    await agent._run_tools(StepResult(calls=[_call("run_command", {"command": "pytest"})]), messages)
    await agent._run_tools(StepResult(calls=[_call("write_file", {"path": "a.py", "content": "x=1"})]), messages)
    await agent._run_tools(StepResult(calls=[_call("run_command", {"command": "pytest"}, cid="c9")]), messages)
    last = [m for m in messages if m.get("role") == "tool"][-1]["content"]
    assert "duplicate call suppressed" not in last
    assert agent.strikes == 0


@pytest.mark.asyncio
async def test_infrastructure_failures_do_not_consume_the_tool_call_budget():
    """A long task must not be cut short just because the provider expired twice."""
    computer = ScriptedComputer(failures=2)
    agent, _run = _agent(computer, Settings(store="sqlite", max_tool_calls=1))
    messages = [{"role": "user", "content": "go"}]
    await agent._run_tools(StepResult(calls=[_call("run_command", {"command": "echo hi"})]), messages)
    assert agent.tool_calls_total == 1           # one *real* call
    assert agent.infra_failures == 2             # retries are tracked separately
    assert agent.tool_calls_total < 2            # and the budget is not exhausted by retries
