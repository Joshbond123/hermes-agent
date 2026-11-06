"""The four behaviours this round had to fix, pinned against the real loop.

Everything here runs the genuine ``AgentRun`` against a fake model endpoint: only the provider is
faked, so a passing test means the streaming, the retry/lifecycle logic and the transcript the UI
sees are the ones that ship.

1. A provider throttle (HTTP 429) never ends the task: the run waits the delay the provider asked
   for and continues without replaying a single tool call.
2. A transient fault mid-task is resumed from the checkpoint, not restarted.
3. Internal execution records stay in the prompt and never appear in the chat.
4. A conversation's model is a property of the conversation, and a running task keeps the model it
   started with.
"""

from __future__ import annotations

import json

import pytest

from blackthorn import agent as agent_module
from blackthorn import prompts


# ----------------------------------------------------------------------------- 1. rate limits
async def test_a_429_is_waited_out_and_the_task_continues_without_replaying_tools(stack):
    """The reported failure: "The model endpoint returned HTTP 429" ended the task.

    A throttle is the provider being busy, not the task failing. The run must keep its state,
    honour the wait the provider asked for, and continue — with the tool it already ran still done
    exactly once.
    """
    from .fakes import call, finish, say

    stack.backend.queue(
        [call("run_command", {"command": "echo once"}, cid="c1"), finish("tool_calls")],
        {"http_error": 429, "message": '{"status":429,"title":"Too Many Requests"}', "headers": {"Retry-After": "0"}},
        [*say("Both steps done."), finish("stop")],
    )
    out = await stack.stream({"message": "run the check and report"})
    assert out.end["status"] == "stop", out.end
    assert len(stack.backend.computer_calls) == 1, "the completed tool must not be replayed after a throttle"
    notices = [e.get("text", "") for e in out.of("notice")]
    assert any("throttl" in n.lower() for n in notices), notices
    assert not any("429" in n and "Too Many" in n for n in notices), "raw provider errors must not be shown"
    assert "Both steps done" in out.text


async def test_throttling_is_announced_with_the_wait_time_and_never_as_a_raw_api_error(stack):
    from .fakes import finish

    stack.backend.queue(*[{"http_error": 429, "message": "Too Many Requests", "headers": {"Retry-After": "0"}}]
                        * (agent_module.RATE_LIMIT_ATTEMPTS + 3))
    out = await stack.stream({"message": "anything"})
    assert out.end["status"] == "error"
    errors = out.of("error")
    assert errors, "a run that cannot continue must report why"
    message = str(errors[-1].get("message") or "")
    assert "throttl" in message.lower()
    assert "status\":429" not in message and "Too Many Requests" not in message, message


async def test_the_endpoint_gate_serialises_requests_to_one_model(stack):
    """Concurrent runs must not pile requests on one endpoint; the gate allows one at a time."""
    import asyncio

    from blackthorn.llm import gate_for

    gate = gate_for("https://example.invalid")
    active = 0
    peak = 0

    async def worker():
        nonlocal active, peak
        await gate.enter()
        try:
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
        finally:
            active -= 1
            gate.leave()

    await asyncio.gather(*[worker() for _ in range(4)])
    assert peak == 1, f"{peak} concurrent requests reached the endpoint"


def test_retry_after_is_read_from_the_header_and_the_body():
    import httpx

    from blackthorn.llm import backoff_delay, retry_after_seconds

    resp = httpx.Response(429, headers={"Retry-After": "17"}, content=b"{}")
    assert retry_after_seconds(resp, "") == 17.0
    resp2 = httpx.Response(429, content=json.dumps({"retry_after": 9}).encode())
    assert retry_after_seconds(resp2, '{"retry_after": 9}') == 9.0
    # The provider's own figure is respected, and the delay is always within the bounded window.
    assert backoff_delay(1, 17.0) >= 17.0
    assert 0 < backoff_delay(1) <= 120
    assert backoff_delay(20) <= 120, "backoff must stay bounded (no unbounded wait)"


# ------------------------------------------------------------------- 2. resume, not restart
async def test_a_transient_fault_mid_task_resumes_from_the_checkpoint(stack, monkeypatch):
    """The task is paused, not lost: it continues from the same state with no repeated work."""
    from .fakes import call, finish, say

    monkeypatch.setattr(agent_module, "RUN_RESUME_ATTEMPTS", 3)
    monkeypatch.setattr(agent_module, "RATE_LIMIT_ATTEMPTS", 1)   # one in-step attempt: the run-level supervisor is under test
    monkeypatch.setattr(agent_module, "backoff_delay", lambda attempt, retry_after=None: 0.0)
    stack.backend.queue(
        [call("run_command", {"command": "echo first"}, cid="c1"), finish("tool_calls")],
        {"http_error": 429, "message": "Too Many Requests", "headers": {"Retry-After": "0"}},
        [*say("Finished after resuming."), finish("stop")],
    )
    out = await stack.stream({"message": "do the two-step task"})
    assert out.end["status"] == "stop", (out.end, out.of("error"))
    assert len(stack.backend.computer_calls) == 1, "completed work must never be replayed on resume"
    assert out.of("internal.checkpoint"), "every resume is visible in the run's event stream"
    assert "Finished after resuming" in out.text


async def test_resumption_is_bounded_so_it_can_never_loop_forever(stack, monkeypatch):
    """Bounded attempts: a provider that never recovers ends the task honestly instead of hanging."""
    monkeypatch.setattr(agent_module, "RUN_RESUME_ATTEMPTS", 2)
    monkeypatch.setattr(agent_module, "RATE_LIMIT_ATTEMPTS", 1)
    monkeypatch.setattr(agent_module, "backoff_delay", lambda attempt, retry_after=None: 0.0)
    stack.backend.queue(*[{"http_error": 429, "message": "Too Many Requests",
                           "headers": {"Retry-After": "0"}} for _ in range(30)])
    out = await stack.stream({"message": "impossible"})
    assert out.end["status"] == "error"
    # 1 initial + 2 resumes, one in-step attempt each: a hard ceiling, not "retry until it works".
    assert len(stack.backend.requests) == 3, len(stack.backend.requests)


# ------------------------------------------------------- 3. execution records stay internal
async def test_the_execution_record_never_reaches_the_chat(stack):
    """Reported: the chat showed "[Execution record: …]" as the assistant's own answer."""
    from .fakes import call, finish, say

    stack.backend.queue(
        [call("read_file", {"path": "/workspace/pyproject.toml", "max_chars": 5000}, cid="r1"), finish("tool_calls")],
        [*say("The file declares the project metadata."), finish("stop")],
    )
    out = await stack.stream({"message": "look at the project file"})
    assert out.end["status"] == "stop"

    session_id = out.end["session_id"]
    async with __import__("httpx").AsyncClient(timeout=30) as client:
        response = await client.get(f"{stack.url}/api/chat/sessions/{session_id}")
    payload = response.text
    assert prompts.EXEC_HEADER not in payload, "server bookkeeping must never be served as chat"
    assert "Do not redo a step" not in payload
    for message in response.json()["messages"]:
        assert prompts.EXEC_HEADER not in (message.get("content") or "")


async def test_the_agent_still_receives_the_record_of_earlier_tool_calls(stack):
    """Keeping the record internal must not lose it: the model still learns what already ran."""
    turns = [
        {"role": "user", "content": "read the file"},
        {"role": "assistant", "content": "done", "status": "stop",
         "ledger": [{"n": "read_file", "a": {"path": "/workspace/pyproject.toml"}, "ok": True,
                     "o": "characters 0-5000 of 4591"}]},
    ]
    messages = prompts.compile_history(turns, 4000)[0]
    wire = json.dumps(messages, ensure_ascii=False)
    assert "read_file" in wire and "pyproject.toml" in wire
    record = [m for m in messages if m["role"] == "system"]
    assert record and prompts.EXEC_HEADER in record[0]["content"], "it belongs in the system layer"
    assistant = [m for m in messages if m["role"] == "assistant"]
    assert all(prompts.EXEC_HEADER not in m["content"] for m in assistant), \
        "it must never be written in the assistant's voice (that is what the model copied into answers)"


def test_a_model_that_echoes_the_record_cannot_put_it_in_the_transcript():
    from blackthorn.store import _public_message

    dirty = "Here is the answer.\n\n" + prompts.EXEC_HEADER + "\nAlready executed:\n- read_file(x) -> ok"
    clean = _public_message({"content": dirty, "parts": [{"type": "text", "text": dirty}]})
    assert clean["content"] == "Here is the answer."
    assert clean["parts"][0]["text"] == "Here is the answer."


# --------------------------------------------------------------- 4. switching model per chat
async def test_switching_the_model_keeps_the_conversation_and_marks_it_explicit(stack):
    from .fakes import finish, say

    stack.backend.queue([*say("Hello."), finish("stop")])
    out = await stack.stream({"message": "hello"})
    session_id = out.end["session_id"]

    async with __import__("httpx").AsyncClient(timeout=30) as client:
        catalogue = (await client.get(f"{stack.url}/api/models?session_id={session_id}")).json()
        assert any(m["id"] == "nvidia/nemotron-3-ultra-550b-a55b" for m in catalogue["models"])

        patched = await client.patch(f"{stack.url}/api/chat/sessions/{session_id}",
                                     json={"model": "nvidia/nemotron-3-ultra-550b-a55b"})
        assert patched.status_code == 200
        body = patched.json()
        assert body["session"]["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
        # The conversation is the same conversation: same id, same messages.
        after = (await client.get(f"{stack.url}/api/chat/sessions/{session_id}")).json()
    assert after["session"]["id"] == session_id
    assert after["messages"], "switching a model must not clear the transcript"

    refusal = None
    async with __import__("httpx").AsyncClient(timeout=30) as client:
        refusal = await client.patch(f"{stack.url}/api/chat/sessions/{session_id}", json={"model": "no/such-model"})
    assert refusal.status_code == 422, refusal.text
    assert refusal.json()["detail"]["code"] == "model_unknown"


def test_an_unconfigured_model_is_listed_with_its_reason_instead_of_being_offered():
    from blackthorn.route import model_options

    options = {o.label: o for o in model_options()}
    assert "Nemotron 3 Ultra" in options
    qwen = options["Qwen3.8 27B Uncensored"]
    # No QWEN_BASE_URL in the test environment: the picker must say why, not pretend it works.
    if not qwen.available:
        assert qwen.reason, "an unavailable model must explain itself"
        assert "endpoint" in qwen.reason or "key" in qwen.reason
