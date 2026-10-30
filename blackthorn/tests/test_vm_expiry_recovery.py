"""End-to-end regression for the two production defects, over real HTTP/SSE.

Defect 1 — ``Provider VM expired while idle; the next computer tool call will
create a replacement and restore the GitHub workspace.``
    The provider answered 503/offline, the tool never ran, yet the failure was
    charged against the "repeated identical tool call" budget, so the agent
    aborted the task instead of reconnecting.

Defect 2 — ``Stopped using tools (repeated identical tool calls); writing the
answer from what was gathered.``
    A recoverable outage looked exactly like a stalled model.

These tests drive the real app: a scripted model server, the real tool registry,
the real agent loop, and a fake computer that can fail the way PandaStack does.
"""
from __future__ import annotations

import asyncio

import pytest

from .fakes import call, finish, say

STOP_NOTICE = "Stopped using tools"


def run_command(cmd: str, cid: str = "c1"):
    return call("run_command", {"command": cmd}, cid=cid)


async def test_expired_vm_is_recovered_without_truncating_the_task(stack):
    """Two provider failures, then success: the command's real output reaches the model."""
    stack.backend.computer_fail = [503, 503]
    stack.backend.queue(
        [run_command("uname -a")],
        [*say("The computer reports its kernel once the replacement is live."), finish("stop")],
    )
    out = await stack.stream({"message": "check the kernel"})

    notices = [e for e in out.of("notice") if STOP_NOTICE in str(e.get("text", ""))]
    assert notices == [], "an expired VM must not end the run"
    assert out.end.get("status") in (None, "stop", "complete", "done")

    exec_calls = [c for c in stack.backend.computer_calls if c["name"] == "exec"]
    assert len(exec_calls) >= 3, "the runtime must retry the call that never reached the tool"
    assert "kernel" in out.text.lower()


async def test_recovery_attempts_are_bounded_and_reported_honestly(stack):
    """When the provider stays down, the model is told it is infrastructure — with a way forward."""
    stack.backend.computer_fail = [503] * 12
    stack.backend.queue(
        [run_command("uname -a")],
        [*say("I could not reach the computer; here is the blocker and the next step."), finish("stop")],
    )
    out = await stack.stream({"message": "check the kernel"})

    notices = [e for e in out.of("notice") if STOP_NOTICE in str(e.get("text", ""))]
    assert notices == [], "persistent infrastructure failure is not a repeat-call stall"
    exec_calls = [c for c in stack.backend.computer_calls if c["name"] == "exec"]
    assert 1 < len(exec_calls) <= 6, "retries must be bounded, not unbounded"
    assert out.text.strip(), "the run must still produce an answer"


async def test_a_genuine_repeat_after_success_is_surfaced_and_bounded(stack):
    """The stall detector keeps working: identical successful calls are not free work.

    Two things must hold now that the old "Stopped using tools" path is gone:
      * real executions stay capped (a loop cannot burn the budget),
      * the stall is visible to the user and answered with a strategy change.
    """
    # One identical call per model step, seven steps in a row: a stalled loop.
    stack.backend.queue(*[[run_command("uname -a", cid=f"c{i}")] for i in range(1, 8)],
                        [*say("Done with what I have."), finish("stop")])
    out = await stack.stream({"message": "run it repeatedly"})
    noticed = [e["text"] for e in out.of("notice")]
    assert any("repeated identical tool calls" in t.lower() for t in noticed), \
        "a stalled step must be reported, not silently swallowed"
    exec_calls = [c for c in stack.backend.computer_calls if c["name"] == "exec"]
    assert len(exec_calls) <= 4, "the duplicate budget must cap real executions"
    assert out.text.strip(), "the run must still answer"


async def test_a_relentless_loop_still_ends_tool_use_after_the_strategy_rounds(stack):
    """Strategy change is bounded: a loop that ignores every hint is stopped, with a reason."""
    stack.backend.queue(*[[run_command("uname -a", cid=f"r{i}")] for i in range(1, 26)],
                        [*say("Blocked: the same call keeps returning the same thing."), finish("stop")])
    out = await stack.stream({"message": "loop forever"})
    notices = [e["text"] for e in out.of("notice")]
    assert any(STOP_NOTICE in t for t in notices), "an unbreakable loop must end tool use"
    exec_calls = [c for c in stack.backend.computer_calls if c["name"] == "exec"]
    assert len(exec_calls) <= 8, "strategy rounds must not turn into unlimited retries"


async def test_tool_results_after_recovery_are_marked_ok_in_the_ledger(stack):
    stack.backend.computer_fail = [503]
    stack.backend.queue(
        [run_command("uname -a")],
        [*say("Recovered and completed."), finish("stop")],
    )
    out = await stack.stream({"message": "check the kernel"})
    ends = out.of("tool.end")
    assert ends and all(e.get("status") == "ok" for e in ends), ends
    assert out.text.strip()


def test_a_live_model_row_is_never_reported_as_a_verified_stop(monkeypatch):
    """Kaggle's delete can be accepted while the notebook keeps heartbeating.

    A real turn-off did exactly that: the API answered 'stopped', the row was rewritten
    GPU_STOPPED, and the session went on answering requests. The verifier must read the
    notebook's own heartbeat instead of trusting the delete call.
    """
    import time as _time
    import cloudflare_d1_client as ctl

    fresh = {"status": "HEARTBEAT_ONLINE", "updated_at": _time.time() - 5, "kaggle_kernel": "black797/qwen"}
    monkeypatch.setattr(ctl, "d1_query", lambda *_a, **_k: [fresh])
    assert ctl._model_session_still_up() == (
        True, "the session row is HEARTBEAT_ONLINE (5s ago, kernel black797/qwen)")

    stale = dict(fresh, updated_at=_time.time() - 900)
    monkeypatch.setattr(ctl, "d1_query", lambda *_a, **_k: [stale])
    assert ctl._model_session_still_up()[0] is False, "an old heartbeat is not a live session"

    monkeypatch.setattr(ctl, "d1_query", lambda *_a, **_k: [dict(fresh, status="GPU_STOPPED_SAVING_QUOTA")])
    assert ctl._model_session_still_up()[0] is False


@pytest.mark.asyncio
async def test_turn_off_retries_a_transient_upstream_timeout(stack):
    """A ReadTimeout took a whole power-off down with 'gpu_control_failed' in live testing.

    Stopping is idempotent, so it is retried; a genuine refusal is still reported at once.
    """
    import httpx

    calls = {"n": 0}

    class Gpu:
        async def turn_off(self, role="all"):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("ReadTimeout: ")
            return {"turn_off_ok": True, "status": "GPU_STOPPED_SAVING_QUOTA", "turn_off_roles": [role], "fleet": {}}

    original = stack.services.gpu
    stack.services.gpu = Gpu()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f"{stack.url}/api/kaggle-gpu/turn-off",
                                         json={"role": "all", "confirm": True})
    finally:
        stack.services.gpu = original
    assert response.status_code == 200, response.text
    assert response.json()["turn_off_ok"] is True
    assert calls["n"] == 2, "the timeout must be retried, not surfaced as a failure"
