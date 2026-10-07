"""End-to-end behaviour of the chat pipeline over real HTTP/SSE against fake model + computer backends."""

import asyncio
import json
import time

import httpx
import pytest

from .fakes import call, finish, pause, say, think, usage


def sh(cmd: str, **kw):
    return call("run_command", {"command": cmd}, **kw)


async def session_messages(stack, sid):
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{stack.url}/api/chat/sessions/{sid}")
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------------------------------- streaming
async def test_streaming_is_incremental_not_buffered(stack):
    turn = []
    for word in ["alpha ", "beta ", "gamma ", "delta ", "epsilon"]:
        turn += [{"content": word}, pause(0.25)]
    stack.backend.queue([*turn, finish("stop")])
    out = await stack.stream({"message": "count"})
    deltas = [t for e, t in zip(out.events, out.times) if e["type"] == "text.delta"]
    assert out.text == "alpha beta gamma delta epsilon"
    assert len(deltas) == 5
    # the first token reached the browser long before the last: real incremental delivery
    assert deltas[-1] - deltas[0] > 0.8
    assert deltas[0] < deltas[-1] - 0.8
    assert "text/event-stream" in out.headers["content-type"]
    assert "no-transform" in out.headers["cache-control"]
    assert out.headers.get("x-accel-buffering") == "no"
    assert "content-encoding" not in out.headers


async def test_heartbeat_keeps_idle_connection_alive(stack):
    stack.backend.queue([pause(1.4), *say("late answer"), finish("stop")])
    out = await stack.stream({"message": "slow"})
    assert out.pings >= 1
    assert out.text == "late answer"


async def test_no_duplicate_or_lost_tokens_and_persisted_text_matches_stream(stack):
    text = "The quick brown fox jumps over the lazy dog. " * 6
    stack.backend.queue([*say(text, 40), finish("stop")])
    out = await stack.stream({"message": "long"})
    assert out.text == text
    seqs = [e["seq"] for e in out.events]
    assert seqs == list(range(1, len(seqs) + 1))
    data = await session_messages(stack, out.session_id)
    assistant = data["messages"][-1]
    assert assistant["content"] == text.strip() and assistant["status"] == "stop"   # persisted text is whitespace-normalised
    assert [m["role"] for m in data["messages"]] == ["user", "assistant"]


# ---------------------------------------------------------------------------------------------------- reasoning privacy
async def test_reasoning_and_think_blocks_never_leave_the_server(stack):
    secret = "TOP-SECRET-CHAIN-OF-THOUGHT"
    stack.backend.queue([think(secret), think(secret), {"content": "<think>"}, {"content": secret}, {"content": "</think>"},
                         *say("Final answer.", 2), finish("stop")])
    out = await stack.stream({"message": "hello"})
    assert out.text == "Final answer."
    blob = json.dumps(out.events)
    assert secret not in blob
    states = [e["state"] for e in out.of("thinking")]
    assert states[0] == "start" and states[-1] == "end"
    data = await session_messages(stack, out.session_id)
    assert secret not in json.dumps(data)


# ---------------------------------------------------------------------------------------------------- native tool calling
async def test_model_chooses_a_tool_result_is_fed_back_as_role_tool(stack):
    stack.backend.exec_output = "total 8\nnotes.txt\n"
    stack.backend.queue([think("need the listing"), *say("Let me look. ", 1), sh("ls", cid="call_1"), finish("tool_calls")],
                        [*say("You have one file: notes.txt."), usage(300, 12), finish("stop")])
    out = await stack.stream({"message": "what files do I have?"})
    assert [e["type"] for e in out.events if e["type"].startswith("tool.")] == ["tool.start", "tool.end"]
    end = out.of("tool.end")[0]
    assert end["status"] == "ok" and "exit code 0" in end["summary"]
    assert out.text == "Let me look. You have one file: notes.txt."
    # protocol: second model call carries the assistant tool_calls turn and a matching role=tool message
    second = stack.backend.requests[1]["messages"]
    assistant = [m for m in second if m["role"] == "assistant"][-1]
    tool_msg = [m for m in second if m["role"] == "tool"][-1]
    assert assistant["tool_calls"][0]["id"] == "call_1" == tool_msg["tool_call_id"]
    assert assistant["tool_calls"][0]["function"]["name"] == "run_command"
    assert "notes.txt" in tool_msg["content"]
    assert stack.backend.computer_calls[0]["name"] == "exec"
    data = await session_messages(stack, out.session_id)
    parts = data["messages"][-1]["parts"]
    assert [p["type"] for p in parts] == ["text", "tool", "text"]
    assert parts[1]["status"] == "ok"
    assert data["messages"][-1]["content"].startswith("Let me look.")


async def test_multi_step_tool_use(stack):
    stack.backend.files["a.txt"] = "alpha"
    stack.backend.queue([call("list_files", {"path": "."}, cid="c1"), finish("tool_calls")],
                        [call("read_file", {"path": "a.txt"}, cid="c2"), finish("tool_calls")],
                        [*say("a.txt contains: alpha"), finish("stop")])
    out = await stack.stream({"message": "read my file"})
    assert [e["name"] for e in out.of("tool.start")] == ["list_files", "read_file"]
    assert all(e["status"] == "ok" for e in out.of("tool.end"))
    assert out.text == "a.txt contains: alpha"
    assert out.end["steps"] == 3 and out.end["tool_calls"] == 2


async def test_parallel_tool_calls_in_one_step_are_all_executed(stack):
    stack.backend.queue([sh("echo one", cid="p1", index=0), sh("echo two", cid="p2", index=1), finish("tool_calls")],
                        [*say("both done"), finish("stop")])
    out = await stack.stream({"message": "run two things"})
    assert [e["id"] for e in out.of("tool.end")] == ["p1", "p2"]
    ids = [m["tool_call_id"] for m in stack.backend.requests[1]["messages"] if m["role"] == "tool"]
    assert ids == ["p1", "p2"]
    assert len(stack.backend.computer_calls) == 2


async def test_duplicate_tool_calls_do_not_loop_forever(stack):
    same = lambda i: [sh("whoami", cid=f"d{i}"), finish("tool_calls")]  # noqa: E731
    stack.backend.queue(same(1), same(2), same(3), same(4), same(5), [*say("I already have the result: root"), finish("stop")])
    out = await stack.stream({"message": "who am i"})
    executed = [c for c in stack.backend.computer_calls if c["name"] == "exec"]
    assert len(executed) == 2                      # third+ identical calls are suppressed, not executed
    assert out.end["status"] == "stop"
    assert "I already have the result" in out.text
    suppressed = [e for e in out.of("tool.end") if "duplicate" in (e.get("error") or "")]
    assert suppressed
    assert any("repeated identical tool calls" in n["text"] for n in out.of("notice"))
    last = stack.backend.requests[-1]
    assert "tools" not in last                      # the forced final pass offers no tools
    assert "Tool use is finished" in last["messages"][-1]["content"]


async def test_malformed_tool_arguments_are_reported_to_the_model_and_recovered(stack):
    stack.backend.queue([call("run_command", '{"command": "ls', cid="m1"), finish("tool_calls")],
                        [sh("ls", cid="m2"), finish("tool_calls")],
                        [*say("Fixed it."), finish("stop")])
    out = await stack.stream({"message": "list"})
    ends = out.of("tool.end")
    assert ends[0]["status"] == "error" and "valid JSON" in ends[0]["error"]
    assert ends[1]["status"] == "ok"
    tool_msgs = [m for m in stack.backend.requests[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs[0]["content"].startswith("ERROR:")
    assert len([c for c in stack.backend.computer_calls if c["name"] == "exec"]) == 1


async def test_invalid_schema_arguments_are_rejected_before_execution(stack):
    stack.backend.queue([call("run_command", {"timeout_seconds": 5}, cid="v1"), finish("tool_calls")], [*say("ok"), finish("stop")])
    out = await stack.stream({"message": "x"})
    end = out.of("tool.end")[0]
    assert end["status"] == "error" and "missing required argument 'command'" in end["error"]
    assert not stack.backend.computer_calls


async def test_unknown_tool_is_an_honest_error(stack):
    stack.backend.queue([call("launch_missiles", {}, cid="u1"), finish("tool_calls")], [*say("I cannot do that."), finish("stop")])
    out = await stack.stream({"message": "do it"})
    end = out.of("tool.end")[0]
    assert end["status"] == "error" and "unknown tool" in end["error"]
    assert out.text == "I cannot do that."


async def test_step_budget_forces_a_final_answer(stack):
    turns = [[sh(f"echo {i}", cid=f"s{i}"), finish("tool_calls")] for i in range(8)]   # max_steps=8 tool-capable steps
    turns.append([*say("Best answer from what I gathered."), finish("stop")])
    stack.backend.queue(*turns)
    out = await stack.stream({"message": "keep going"})
    assert out.end["status"] == "stop"
    assert out.text == "Best answer from what I gathered."
    assert out.end["steps"] <= 9
    assert any("tool-call limit" in n["text"] for n in out.of("notice"))


async def test_tool_failure_is_surfaced_honestly(stack):
    stack.backend.queue([call("read_file", {"path": "missing.txt"}, cid="f1"), finish("tool_calls")],
                        [*say("I could not find missing.txt."), finish("stop")])
    out = await stack.stream({"message": "read missing.txt"})
    end = out.of("tool.end")[0]
    assert end["status"] == "error" and "missing.txt" in end["error"]
    tool_msg = [m for m in stack.backend.requests[1]["messages"] if m["role"] == "tool"][0]
    assert tool_msg["content"].startswith("ERROR:")
    data = await session_messages(stack, out.session_id)
    assert data["messages"][-1]["parts"][0]["status"] == "error"


async def test_tool_output_is_bounded_for_the_model_and_ui(stack):
    stack.backend.exec_output = "x" * 60_000
    stack.backend.queue([sh("yes | head -c 60000", cid="big"), finish("tool_calls")], [*say("done"), finish("stop")])
    out = await stack.stream({"message": "big output"})
    content = [m for m in stack.backend.requests[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert len(content) < 3600 and "characters omitted" in content
    assert len(out.of("tool.end")[0]["output"]) <= 1300


async def test_denied_command_is_never_sent_to_the_computer(stack):
    stack.backend.queue([sh("rm -rf /", cid="x1"), finish("tool_calls")], [*say("I won't run that."), finish("stop")])
    out = await stack.stream({"message": "wipe it"})
    assert "refused" in out.of("tool.end")[0]["error"]
    assert not stack.backend.computer_calls


async def test_secrets_are_redacted_in_the_activity_ui(stack):
    cmd = 'curl -H "Authorization: Bearer sk-live-abcdefghijklmnopqrstuvwxyz0123" https://x.test'
    stack.backend.queue([sh(cmd, cid="r1"), finish("tool_calls")], [*say("done"), finish("stop")])
    out = await stack.stream({"message": "call api"})
    start = out.of("tool.start")[0]
    assert "abcdefghijklmnopqrstuvwxyz0123" not in json.dumps(out.events)
    assert "***" in start["summary"]


async def test_computer_offline_tool_fails_honestly_and_run_still_completes(stack):
    await stack.set_gpu(url="http://127.0.0.1:9")           # nothing listens there
    stack.services.resolver.invalidate()
    stack.backend.queue([sh("ls", cid="o1"), finish("tool_calls")], [*say("The computer is unreachable."), finish("stop")])
    # the model endpoint is the same dead URL, so point only the computer at it by queueing on a healthy model route:
    await stack.set_gpu()
    original = stack.services.computer.call

    async def dead(*a, **k):
        from blackthorn.tools.base import ToolError
        raise ToolError("offline", "The remote computer is offline.", hint="Turn the GPU on.")

    stack.services.computer.call = dead
    try:
        out = await stack.stream({"message": "ls"})
    finally:
        stack.services.computer.call = original
    end = out.of("tool.end")[0]
    assert end["status"] == "error" and "offline" in end["error"]
    assert out.end["status"] == "stop" and out.text == "The computer is unreachable."


# ---------------------------------------------------------------------------------------------------- errors
async def test_gpu_off_is_a_clear_409_and_nothing_is_persisted(stack):
    await stack.set_gpu("GPU_STOPPED_SAVING_QUOTA", url="")
    out = await stack.stream({"message": "hello"})
    assert out.status == 409
    assert out.error_body["detail"]["code"] == "gpu_off"
    async with httpx.AsyncClient() as c:
        listing = (await c.get(f"{stack.url}/api/chat/sessions")).json()
    assert listing["sessions"] == []


async def test_midstream_connection_drop_keeps_partial_and_reports_error(stack):
    stack.backend.queue([*say("Part one of the answer, ", 2), pause(0.05), {"drop": 1}])
    out = await stack.stream({"message": "hello"})
    assert out.end["status"] == "error"
    assert out.of("error")[0]["code"] == "gpu_dropped"
    data = await session_messages(stack, out.session_id)
    last = data["messages"][-1]
    assert last["status"] == "error" and last["content"].startswith("Part one")
    assert "error" in last["meta"]


async def test_retryable_upstream_failure_recovers_once(stack):
    stack.backend.queue({"http_error": 503, "message": "tunnel hiccup"}, [*say("Recovered."), finish("stop")])
    out = await stack.stream({"message": "hello"})
    assert out.text == "Recovered." and out.end["status"] == "stop"
    assert any("retrying" in n["text"] for n in out.of("notice"))


async def test_persistent_upstream_failure_ends_in_error_not_a_hang(stack):
    stack.backend.queue({"http_error": 503, "message": "down"}, {"http_error": 503, "message": "down"})
    out = await stack.stream({"message": "hello"})
    assert out.end["status"] == "error"
    assert out.of("error")[0]["retryable"] is True


async def test_empty_model_response_is_retried_once_invisibly(stack):
    stack.backend.queue([think("hmm"), finish("stop")], [*say("Here is the real answer."), finish("stop")])
    out = await stack.stream({"message": "hello"})
    assert out.end["status"] == "stop" and out.text == "Here is the real answer."
    assert len(stack.backend.requests) == 2                                   # one transparent second attempt
    assert any("asking once more" in n["text"] for n in out.of("notice"))
    data = await session_messages(stack, out.session_id)
    assert data["messages"][-1]["content"] == "Here is the real answer."       # nothing from the empty attempt was saved


async def test_two_empty_responses_end_in_an_honest_error_not_a_hang(stack):
    stack.backend.queue([think("hmm"), finish("stop")], [think("still nothing"), finish("stop")])
    out = await stack.stream({"message": "hello"})
    assert out.end["status"] == "error" and out.of("error")[0]["code"] == "empty_response" and out.of("error")[0]["retryable"] is True
    assert len(stack.backend.requests) == 2                                   # exactly one retry, never a loop


async def test_tools_then_silence_is_retried_then_reported(stack):
    stack.backend.queue([sh("ls", cid="s1"), finish("tool_calls")], [finish("stop")], [*say("Found a.txt."), finish("stop")])
    ok = await stack.stream({"message": "list files"})
    assert ok.end["status"] == "stop" and ok.text == "Found a.txt."
    stack.backend.queue([sh("ls", cid="s2"), finish("tool_calls")], [finish("stop")], [finish("stop")])
    bad = await stack.stream({"message": "list files again"})
    assert bad.end["status"] == "error" and bad.of("error")[0]["code"] == "no_answer"


async def test_context_overflow_is_recovered_by_trimming(stack):
    stack.backend.queue({"http_error": 400, "message": "the request exceeds the available context size"},
                        [*say("OK after trimming."), finish("stop")])
    async with httpx.AsyncClient(timeout=30) as c:
        first = await stack.stream({"message": "first question"}, c)
        stack.backend.queue([*say("first answer"), finish("stop")])
    # new turn in the same session: history exists, so trimming has something to drop
    stack.backend.script.clear()
    stack.backend.queue([*say("first answer"), finish("stop")])
    a = await stack.stream({"message": "q1"})
    stack.backend.queue({"http_error": 400, "message": "the request exceeds the available context size"},
                        [*say("OK after trimming."), finish("stop")])
    b = await stack.stream({"message": "q2", "session_id": a.session_id})
    assert b.text == "OK after trimming."
    assert any("trimmed" in n["text"] for n in b.of("notice"))
    retried = stack.backend.requests[-1]["messages"]
    assert [m["role"] for m in retried] == ["system", "user"]


# ---------------------------------------------------------------------------------------------------- cancel / resume
async def test_stop_cancels_the_backend_generation_and_keeps_the_partial(stack):
    slow = []
    for i in range(60):
        slow += [{"content": f"w{i} "}, pause(0.1)]
    stack.backend.queue([*slow, finish("stop")])
    async with httpx.AsyncClient(timeout=30) as c:
        async with c.stream("POST", f"{stack.url}/api/chat/stream", json={"message": "write a lot"}) as resp:
            from .conftest import Streamed
            got = Streamed()
            await got.consume(resp, stop_after=6)
            run_id = got.of("run.start")[0]["run_id"]
            t0 = time.perf_counter()
            r = await c.post(f"{stack.url}/api/chat/runs/{run_id}/cancel")
            assert r.status_code == 200 and r.json()["done"] is True and r.json()["status"] == "cancelled"
            await got.consume(resp)
    assert got.end["status"] == "cancelled"
    assert time.perf_counter() - t0 < 4
    for _ in range(40):                              # upstream request is closed → model stops generating
        if stack.backend.cancelled:
            break
        await asyncio.sleep(0.05)
    assert stack.backend.cancelled == 1 and stack.backend.completed == 0
    data = await session_messages(stack, got.of("run.start")[0]["session_id"])
    last = data["messages"][-1]
    assert last["status"] == "cancelled" and last["content"].startswith("w0 ")
    assert len(last["content"]) < 60 * 4              # nothing generated after the stop was kept
    # and the conversation is immediately usable again
    stack.backend.queue([*say("Still here."), finish("stop")])
    again = await stack.stream({"message": "hi again", "session_id": got.of("run.start")[0]["session_id"]})
    assert again.text == "Still here."


async def test_cancel_during_tool_execution_marks_the_tool_cancelled(stack):
    async def slow_exec(*a, **k):
        await asyncio.sleep(30)

    stack.services.computer.call = slow_exec
    stack.backend.queue([sh("sleep 100", cid="z1"), finish("tool_calls")])
    async with httpx.AsyncClient(timeout=30) as c:
        async with c.stream("POST", f"{stack.url}/api/chat/stream", json={"message": "sleep"}) as resp:
            from .conftest import Streamed
            got = Streamed()
            await got.consume(resp, stop_after=2)        # run.start + tool.start
            run_id = got.of("run.start")[0]["run_id"]
            await c.post(f"{stack.url}/api/chat/runs/{run_id}/cancel")
            await got.consume(resp)
    assert got.end["status"] == "cancelled"
    data = await session_messages(stack, got.of("run.start")[0]["session_id"])
    assert data["messages"][-1]["parts"][0]["status"] == "cancelled"


async def test_disconnect_does_not_kill_the_run_and_resume_has_no_loss_or_duplicates(stack):
    chunks = []
    for i in range(30):
        chunks += [{"content": f"t{i:02d} "}, pause(0.05)]
    stack.backend.queue([*chunks, finish("stop")])
    expected = "".join(f"t{i:02d} " for i in range(30)).strip()
    async with httpx.AsyncClient(timeout=30) as c:
        seen = []
        async with c.stream("POST", f"{stack.url}/api/chat/stream", json={"message": "stream"}) as resp:
            from .conftest import Streamed
            first = Streamed()
            await first.consume(resp, stop_after=8)       # then the "page refresh": we drop the connection here
            seen = list(first.events)
        run_id = seen[0]["run_id"]
        last_seq = seen[-1]["seq"]
        async with c.stream("GET", f"{stack.url}/api/chat/runs/{run_id}/events", params={"after": last_seq}) as resp2:
            second = Streamed()
            await second.consume(resp2)
    merged = seen + second.events
    assert [e["seq"] for e in merged] == list(range(1, len(merged) + 1))          # no gap, no duplicate
    text = "".join(e["text"] for e in merged if e["type"] == "text.delta")
    assert text.strip() == expected
    assert merged[-1]["type"] == "run.end" and merged[-1]["status"] == "stop"
    data = await session_messages(stack, seen[0]["session_id"])
    assert data["messages"][-1]["content"] == expected and data["live_run"] is None


async def test_resume_of_an_expired_run_is_a_clean_404(stack):
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{stack.url}/api/chat/runs/run_nope/events")
    assert r.status_code == 404 and r.json()["detail"]["code"] == "run_gone"


async def test_second_message_while_generating_is_rejected(stack):
    stack.backend.queue([pause(1.0), *say("slow"), finish("stop")])
    async with httpx.AsyncClient(timeout=30) as c:
        async with c.stream("POST", f"{stack.url}/api/chat/stream", json={"message": "first"}) as resp:
            from .conftest import Streamed
            got = Streamed()
            await got.consume(resp, stop_after=1)
            sid = got.of("run.start")[0]["session_id"]
            second = await stack.stream({"message": "second", "session_id": sid}, c)
            assert second.status == 409 and second.error_body["detail"]["code"] == "run_in_progress"
            await got.consume(resp)


async def test_progress_is_saved_while_streaming_and_page_reload_sees_live_run(stack):
    chunks = []
    for i in range(40):
        chunks += [{"content": f"p{i:02d} "}, pause(0.1)]
    stack.backend.queue([*chunks, finish("stop")])
    async with httpx.AsyncClient(timeout=30) as c:
        async with c.stream("POST", f"{stack.url}/api/chat/stream", json={"message": "long one"}) as resp:
            from .conftest import Streamed
            got = Streamed()
            await got.consume(resp, stop_after=30)
            sid = got.of("run.start")[0]["session_id"]
            mid = (await c.get(f"{stack.url}/api/chat/sessions/{sid}")).json()      # what a page reload would see
            assert mid["live_run"] and mid["live_run"]["run_id"] == got.of("run.start")[0]["run_id"]
            assert mid["messages"][-1]["status"] == "streaming"
            assert mid["messages"][-1]["content"].startswith("p00")                  # partial text already saved
            assert [m["role"] for m in mid["messages"]] == ["user", "assistant"]     # user message saved before the answer
            await c.post(f"{stack.url}/api/chat/runs/{got.of('run.start')[0]['run_id']}/cancel")


# ---------------------------------------------------------------------------------------------------- regenerate / attachments
async def test_regenerate_replaces_the_last_answer(stack):
    stack.backend.queue([*say("first try"), finish("stop")])
    a = await stack.stream({"message": "question"})
    stack.backend.queue([*say("second try"), finish("stop")])
    b = await stack.stream({"session_id": a.session_id, "regenerate": True})
    assert b.text == "second try"
    data = await session_messages(stack, a.session_id)
    assert [(m["role"], m["content"]) for m in data["messages"]] == [("user", "question"), ("assistant", "second try")]
    assert stack.backend.requests[-1]["messages"][-1]["content"] == "question"


async def test_failed_answer_can_be_retried(stack):
    stack.backend.queue([*say("partial "), {"drop": 1}])
    a = await stack.stream({"message": "question"})
    assert a.end["status"] == "error"
    stack.backend.queue([*say("full answer"), finish("stop")])
    b = await stack.stream({"session_id": a.session_id, "regenerate": True})
    data = await session_messages(stack, a.session_id)
    assert data["messages"][-1]["content"] == "full answer" and data["messages"][-1]["status"] == "stop"
    assert len(data["messages"]) == 2


async def test_attachments_are_uploaded_and_previewed(stack):
    stack.backend.queue([*say("I read it."), finish("stop")])
    out = await stack.stream({"message": "summarise", "attachments": [{"name": "../notes v1.txt", "content": "line1\nline2"}]})
    assert out.end["status"] == "stop"
    write = [c for c in stack.backend.computer_calls if c["name"] == "write_file"][0]
    assert write["body"]["path"] == "uploads/notes_v1.txt" and write["body"]["content"] == "line1\nline2"
    sent = stack.backend.requests[0]["messages"][-1]["content"]
    assert "uploads/notes_v1.txt" in sent and "line1" in sent
    data = await session_messages(stack, out.session_id)
    assert data["messages"][0]["content"] == "summarise"
    assert data["messages"][0]["attachments"][0]["name"] == "notes_v1.txt"


async def test_history_is_sent_to_the_model_for_follow_ups(stack):
    stack.backend.queue([*say("Paris."), finish("stop")])
    a = await stack.stream({"message": "capital of France?"})
    stack.backend.queue([*say("About 2 million."), finish("stop")])
    await stack.stream({"message": "and its population?", "session_id": a.session_id})
    roles = [(m["role"], m["content"]) for m in stack.backend.requests[-1]["messages"][1:]]
    assert roles == [("user", "capital of France?"), ("assistant", "Paris."), ("user", "and its population?")]
    assert stack.backend.requests[-1]["messages"][0]["role"] == "system"


async def test_system_prompt_is_short_and_has_no_keyword_rules(stack):
    stack.backend.queue([*say("hi"), finish("stop")])
    await stack.stream({"message": "hi"})
    system = stack.backend.requests[0]["messages"][0]["content"]
    assert len(system) < 1400
    for banned in ("NEVER call", "greetings:", "<tool_call>", "AUTONOMY"):
        assert banned not in system
    schemas = json.dumps(stack.backend.requests[0]["tools"])
    assert "minLength" not in schemas and "maximum" not in schemas       # validation-only keys stay server-side
