"""End-to-end tests of the turn engine against a real (fake) streaming model server."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import pytest

from blackthorn.agent.builtin_tools import Deps, SearchKeys, build_registry
from blackthorn.agent.engine import Context, Engine, EngineConfig, TurnRequest
from blackthorn.agent.kaggle_computer import KaggleComputer
from blackthorn.route import Route

pytestmark = pytest.mark.asyncio


class MemoryStore:
    def __init__(self, history: Optional[List[Dict[str, str]]] = None) -> None:
        self.history = history or []
        self.custom = ""
        self.memories: List[str] = []
        self.began: List[Dict[str, Any]] = []
        self.saves: List[Dict[str, Any]] = []
        self.last_user = {"id": 7, "content": "previous question"}
        self.dropped: List[int] = []
        self._next = 100

    async def begin_turn(self, session_id, user_text, *, model, started_at, create_user_message, user_meta):
        self._next += 2
        rec = {"session_id": session_id, "user_text": user_text, "create_user_message": create_user_message,
               "user_message_id": self._next if create_user_message else None, "assistant_message_id": self._next + 1,
               "user_meta": user_meta, "created": True}
        self.began.append(rec)
        return rec

    async def load_context(self, session_id, *, before_ts, before_id):
        return Context(history=list(self.history), custom_prompt=self.custom, memories=list(self.memories))

    async def save_assistant(self, message_id, session_id, *, content, finish_reason, metadata, token_count):
        self.saves.append({"message_id": message_id, "content": content, "finish_reason": finish_reason,
                           "metadata": metadata, "token_count": token_count})

    async def last_user_message(self, session_id):
        return self.last_user

    async def drop_after(self, session_id, message_id):
        self.dropped.append(message_id)
        return 1

    @property
    def final(self):
        return self.saves[-1]


def route_for(up):
    async def provider():
        return Route(url=up.url, api_key="test-key", model="qwen-test")
    return provider


async def build(up, http_client, *, store=None, route=None, **cfg):
    store = store or MemoryStore()
    provider = route or route_for(up)
    computer = KaggleComputer(provider)
    registry = build_registry(Deps(computer=computer, search_keys=SearchKeys(loader=_no_keys), http_client=lambda: http_client))
    config = EngineConfig(checkpoint_interval_s=0.05, **cfg)
    engine = Engine(route_provider=provider, registry=registry, store=store, client_factory=lambda: http_client,
                    config=config)
    return engine, store


async def _no_keys():
    return []


async def collect(turn, timeout=20.0):
    events = []

    async def run():
        async for ev in turn.subscribe(0):
            if ev is not None:
                events.append(ev)

    await asyncio.wait_for(run(), timeout)
    return events


def types(events):
    return [e.type for e in events]


def text_of(events):
    return "".join(e.data["text"] for e in events if e.type == "delta")


def say(text):
    return [{"content": text}, {"finish": "stop"}]


def call(name, args, cid="call_1", preamble=""):
    steps = [{"content": preamble}] if preamble else []
    steps.append({"tool_call": {"index": 0, "id": cid, "name": name, "arguments": json.dumps(args)}})
    steps.append({"finish": "tool_calls"})
    return steps


# ------------------------------------------------------------------------------------------
async def test_plain_answer_streams_incrementally_and_is_saved(fake_d1, upstream, http_client):
    upstream.script(say("Hello! How can I help you today?"))
    engine, store = await build(upstream, http_client)
    turn = await engine.start(TurnRequest(message="hi"))
    events = await collect(turn)
    assert types(events)[0] == "turn" and events[-1].type == "done"
    deltas = [e for e in events if e.type == "delta"]
    assert len(deltas) > 3, "text must arrive in many small increments, not one block"
    assert text_of(events) == "Hello! How can I help you today?"
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    done = events[-1].data
    assert done["finish_reason"] == "stop" and done["usage"]["completion_tokens"] == 7
    assert store.final["content"] == "Hello! How can I help you today?" and store.final["finish_reason"] == "stop"
    assert store.final["metadata"]["parts"] == [{"t": "text", "text": "Hello! How can I help you today?"}]


async def test_tools_are_offered_not_forced_and_greeting_uses_none(fake_d1, upstream, http_client):
    """No keyword gate: the *same* request shape (tools + auto) goes out for every message."""
    upstream.script(say("Hi!"), say("Canberra."))
    engine, _ = await build(upstream, http_client)
    for msg in ("hi", "What is the capital of Australia?"):
        await collect(await engine.start(TurnRequest(message=msg)))
    for body in upstream.requests:
        assert body["tool_choice"] == "auto"
        names = [t["function"]["name"] for t in body["tools"]]
        assert {"web_search", "terminal", "read_file", "write_file", "list_files", "fetch_url"} <= set(names)
        assert len(names) == len(set(names)), "no duplicate/alias tool names"
        assert body["reasoning_effort"] == "none" and body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
    assert not upstream.computer_calls


async def test_model_chooses_a_tool_and_result_goes_back_as_a_tool_message(fake_d1, upstream, http_client):
    upstream.exec_output = "/dev/root 25G 5G 19G 22% /"
    upstream.script(call("terminal", {"command": "df -h /"}, "call_df", preamble="Checking disk. "),
                    say("You have 19G free."))
    engine, store = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="how much disk is free?")))
    t = types(events)
    assert t.index("tool_start") < t.index("tool_result") < len(t) - 1 and t[-1] == "done"
    start = next(e for e in events if e.type == "tool_start").data
    result = next(e for e in events if e.type == "tool_result").data
    assert start["name"] == "terminal" and start["label"] == "Terminal" and start["args"]["command"] == "df -h /"
    assert result["ok"] is True and "19G" in result["output"]
    # the executed call really reached the computer
    assert upstream.computer_calls[0]["command"] == "df -h /"
    # protocol: assistant(tool_calls) + tool(role/tool_call_id) were sent back to the model
    second = upstream.requests[1]["messages"]
    assistant = [m for m in second if m["role"] == "assistant" and m.get("tool_calls")][0]
    tool_msg = [m for m in second if m["role"] == "tool"][0]
    assert assistant["tool_calls"][0]["id"] == "call_df" and tool_msg["tool_call_id"] == "call_df"
    assert "[exit 0]" in tool_msg["content"] and "19G" in tool_msg["content"]
    # one continuous message: text, tool, text — in order
    parts = store.final["metadata"]["parts"]
    assert [p["t"] for p in parts] == ["text", "tool", "text"]
    assert parts[1]["status"] == "ok" and parts[1]["name"] == "terminal"
    assert store.final["content"] == "Checking disk.\n\nYou have 19G free."


async def test_multi_step_tool_use_write_then_run(fake_d1, upstream, http_client):
    upstream.script(
        call("write_file", {"path": "/kaggle/working/blackthorn_workspace/fib.py", "content": "print(1)"}, "c1"),
        call("terminal", {"command": "python fib.py"}, "c2"),
        say("Done: it printed 1."),
    )
    engine, store = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="write and run fib.py")))
    assert types(events).count("tool_start") == 2 and events[-1].data["finish_reason"] == "stop"
    # the baseline bug: an absolute workspace path used to nest; now it is normalised
    assert upstream.computer_calls[0]["endpoint"] == "/computer/write_file"
    assert "fib.py" in upstream.files and upstream.files["fib.py"] == "print(1)"
    assert text_of(events) == "Done: it printed 1."


async def test_reasoning_text_never_reaches_events_or_storage(fake_d1, upstream, http_client):
    secret = "The user said hi so I should greet them and not use tools."
    upstream.script([{"reasoning": secret}, {"content": "Hello!"}, {"finish": "stop"}])
    engine, store = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="hi", thinking=True)))
    blob = json.dumps([e.data for e in events]) + json.dumps(store.final)
    assert "greet them" not in blob and "I should" not in blob
    assert text_of(events) == "Hello!" and store.final["content"] == "Hello!"
    states = [e.data for e in events if e.type == "reasoning"]
    assert states[0] == {"active": True} and states[1]["active"] is False and states[1]["duration_ms"] >= 0
    assert "reasoning_effort" not in upstream.requests[0], "thinking=True must not disable reasoning"


async def test_inline_think_tags_are_stripped_even_when_split(fake_d1, upstream, http_client):
    upstream.script([{"content": "<think>hidden chain"}, {"content": " of thought</th"}, {"content": "ink>Visible answer"},
                     {"finish": "stop"}])
    engine, store = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="q")))
    assert text_of(events) == "Visible answer" and "hidden" not in json.dumps(store.final)


async def test_text_form_tool_call_is_executed_and_never_shown(fake_d1, upstream, http_client):
    tag = '<tool_call>{"name": "list_files", "arguments": {"path": "."}}</tool_call>'
    upstream.script([{"content": "Looking. " + tag}, {"finish": "stop"}], say("Empty."))
    engine, store = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="ls")))
    assert "tool_call" not in text_of(events) and "<" not in text_of(events)
    assert types(events).count("tool_start") == 1 and upstream.computer_calls[0]["endpoint"] == "/computer/list_files"


async def test_invalid_arguments_come_back_to_the_model_which_can_recover(fake_d1, upstream, http_client):
    upstream.script(
        [{"tool_call": {"index": 0, "id": "bad", "name": "terminal", "arguments": '{"command": "ls"'}}, {"finish": "tool_calls"}],
        call("terminal", {"command": "ls"}, "good"),
        say("Listed."),
    )
    engine, store = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="ls")))
    results = [e.data for e in events if e.type == "tool_result"]
    assert results[0]["ok"] is False and "not valid JSON" in results[0]["output"]
    assert results[1]["ok"] is True and events[-1].data["finish_reason"] == "stop"
    assert len(upstream.computer_calls) == 1  # the malformed call never executed


async def test_schema_violations_are_reported_not_executed(fake_d1, upstream, http_client):
    upstream.script(call("terminal", {"command": "ls", "timeout_seconds": 99999, "nope": 1}, "c1"), say("ok"))
    engine, _ = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="x")))
    out = next(e.data for e in events if e.type == "tool_result")["output"]
    assert "unknown argument 'nope'" in out and "<= 300" in out and not upstream.computer_calls


async def test_unknown_tool_is_reported(fake_d1, upstream, http_client):
    upstream.script(call("rm_everything", {}, "c1"), say("sorry"))
    engine, _ = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="x")))
    out = next(e.data for e in events if e.type == "tool_result")["output"]
    assert "no tool named 'rm_everything'" in out and "terminal" in out


async def test_dangerous_command_is_refused_before_reaching_the_computer(fake_d1, upstream, http_client):
    upstream.script(call("terminal", {"command": "rm -rf /"}, "c1"), say("I won't do that."))
    engine, _ = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="wipe it")))
    result = next(e.data for e in events if e.type == "tool_result")
    assert result["ok"] is False and "refused" in result["output"] and not upstream.computer_calls


async def test_repeated_identical_calls_are_blocked_and_the_turn_still_finishes(fake_d1, upstream, http_client):
    same = call("terminal", {"command": "echo hi"})
    upstream.script(same, same, same, same, say("Final answer anyway."))
    engine, store = await build(upstream, http_client, repeat_limit=2, strike_limit=2)
    events = await collect(await engine.start(TurnRequest(message="loop")))
    ran = [c for c in upstream.computer_calls if c["endpoint"] == "/computer/exec"]
    assert len(ran) == 2, "the third identical call must not execute"
    outs = [e.data["output"] for e in events if e.type == "tool_result"]
    assert any("already made" in o for o in outs)
    assert any(e.type == "notice" and "Stopping tool use" in e.data["text"] for e in events)
    last_request = upstream.requests[-1]
    assert "tools" not in last_request, "after the guard trips, tools are withheld to force an answer"
    assert text_of(events).endswith("Final answer anyway.")
    assert events[-1].data["finish_reason"] == "incomplete"


async def test_step_budget_forces_a_final_answer(fake_d1, upstream, http_client):
    steps = [call("terminal", {"command": f"echo {i}"}, f"c{i}") for i in range(3)]
    upstream.script(*steps, say("Wrapping up."))
    engine, _ = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="go", max_steps=3)))
    assert types(events).count("tool_start") == 3
    assert events[-1].data["finish_reason"] == "incomplete" and text_of(events) == "Wrapping up."


async def test_failed_tool_is_reported_honestly_in_the_activity_and_to_the_model(fake_d1, upstream, http_client):
    upstream.exec_code = 2
    upstream.exec_output = "python: can't open file"
    upstream.script(call("terminal", {"command": "python nope.py"}), say("It failed: the file does not exist."))
    engine, store = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="run")))
    result = next(e.data for e in events if e.type == "tool_result")
    assert result["ok"] is False and result["exit_code"] == 2
    assert store.final["metadata"]["parts"][0]["status"] == "error"


async def test_secrets_in_tool_output_are_redacted_everywhere(fake_d1, upstream, http_client):
    upstream.exec_output = "token=ghp_abcdefghijklmnopqrstuvwxyz0123456789 done"
    upstream.script(call("terminal", {"command": "env"}), say("ok"))
    engine, store = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="env")))
    blob = json.dumps([e.data for e in events]) + json.dumps(store.final) + json.dumps(upstream.requests[1])
    assert "ghp_abcdefghijklmnop" not in blob and "[REDACTED]" in blob


# ---- cancellation ----------------------------------------------------------------------------
async def test_stop_cancels_the_upstream_stream_and_saves_the_partial_answer(fake_d1, upstream, http_client):
    upstream.token_delay = 0.05
    upstream.script([{"content": "word " * 400}, {"finish": "stop"}])
    engine, store = await build(upstream, http_client)
    turn = await engine.start(TurnRequest(message="long please"))
    seen = []

    async def watch():
        async for ev in turn.subscribe(0):
            if ev is not None:
                seen.append(ev)
                if ev.type == "delta" and len([e for e in seen if e.type == "delta"]) == 5:
                    assert await engine.cancel(turn.id) is True

    await asyncio.wait_for(watch(), 15)
    assert seen[-1].type == "done" and seen[-1].data["finish_reason"] == "cancelled"
    assert await asyncio.to_thread(upstream.disconnected.wait, 3.0), "upstream generation must be aborted"
    assert upstream.completed == 0
    partial = store.final
    assert partial["finish_reason"] == "cancelled" and 0 < len(partial["content"]) < 2000
    assert partial["content"] == "".join(e.data["text"] for e in seen if e.type == "delta").strip()


async def test_stop_during_a_tool_run_cancels_it(fake_d1, upstream, http_client):
    upstream.exec_delay = 30.0
    upstream.script(call("terminal", {"command": "sleep 100"}), say("never reached"))
    engine, store = await build(upstream, http_client)
    turn = await engine.start(TurnRequest(message="sleep"))
    started = time.monotonic()
    async for ev in turn.subscribe(0):
        if ev is not None and ev.type == "tool_start":
            await asyncio.sleep(0.2)
            await engine.cancel(turn.id)
        if ev is not None and ev.type == "done":
            assert ev.data["finish_reason"] == "cancelled"
            break
    assert time.monotonic() - started < 5, "cancel must not wait for the 30 s tool"
    assert store.final["metadata"]["parts"][-1]["status"] == "cancelled"
    assert len(upstream.requests) == 1, "the model must not be called again after Stop"


async def test_cancel_unknown_or_finished_turn_is_a_noop(fake_d1, upstream, http_client):
    engine, _ = await build(upstream, http_client)
    assert await engine.cancel("turn-nope") is False
    turn = await engine.start(TurnRequest(message="hi"))
    await collect(turn)
    assert await engine.cancel(turn.id) is False


# ---- reconnect / replay -------------------------------------------------------------------------
async def test_reattach_replays_exactly_the_missing_events(fake_d1, upstream, http_client):
    upstream.script(say("The quick brown fox jumps over the lazy dog."))
    engine, _ = await build(upstream, http_client)
    turn = await engine.start(TurnRequest(message="fox"))
    first: List[Any] = []
    async for ev in turn.subscribe(0):
        if ev is not None:
            first.append(ev)
        if len(first) == 6:
            break  # "refresh": the first client disappears mid-stream
    rest = await collect_after(turn, first[-1].seq)
    full = first + rest
    assert [e.seq for e in full] == list(range(1, len(full) + 1)), "no gaps and no duplicates"
    assert text_of(full) == "The quick brown fox jumps over the lazy dog." and full[-1].type == "done"
    again = await collect_after(turn, 0)
    assert [e.seq for e in again] == [e.seq for e in full], "late attach after completion replays everything"


async def collect_after(turn, after):
    out = []
    async for ev in turn.subscribe(after):
        if ev is not None:
            out.append(ev)
    return out


async def test_turn_survives_all_subscribers_leaving(fake_d1, upstream, http_client):
    upstream.token_delay = 0.01
    upstream.script(say("a fairly long answer that keeps going " * 5))
    engine, store = await build(upstream, http_client)
    turn = await engine.start(TurnRequest(message="go"))
    async for ev in turn.subscribe(0):
        break  # the browser vanished right away
    await asyncio.wait_for(turn.task, 15)
    assert store.final["finish_reason"] == "stop" and store.final["content"].startswith("a fairly long answer")


async def test_checkpoints_are_written_while_running(fake_d1, upstream, http_client):
    upstream.token_delay = 0.03
    upstream.script(say("slow answer " * 30))
    engine, store = await build(upstream, http_client)
    await collect(await engine.start(TurnRequest(message="slow")))
    running = [s for s in store.saves if s["finish_reason"] == "running"]
    assert running, "partial answers must be checkpointed so a crash/restart does not lose them"
    assert len(running[-1]["content"]) < len(store.final["content"]) or len(running) >= 1
    assert store.final["finish_reason"] == "stop"


# ---- failures ----------------------------------------------------------------------------------------
async def test_malformed_events_are_skipped(fake_d1, upstream, http_client):
    upstream.script([{"raw": "data: {not json"}, {"content": "Hel"}, {"raw": "data: [1,2]"}, {"raw": ": comment"},
                     {"content": "lo"}, {"finish": "stop"}])
    engine, _ = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="x")))
    assert text_of(events) == "Hello" and events[-1].data["finish_reason"] == "stop"


async def test_error_object_mid_stream_keeps_partial_text_and_reports(fake_d1, upstream, http_client):
    upstream.script([{"content": "Partial "}, {"error": "CUDA out of memory"}])
    engine, store = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="x")))
    err = next(e.data for e in events if e.type == "error")
    assert err["code"] == "model_error" and "out of memory" in err["message"]
    assert events[-1].data["finish_reason"] == "error" and store.final["content"] == "Partial"
    assert store.final["metadata"]["error"]["code"] == "model_error"


async def test_dropped_connection_mid_answer_is_an_honest_error(fake_d1, upstream, http_client):
    upstream.script([{"content": "Part one "}, {"drop": True}])
    engine, store = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="x")))
    assert events[-1].data["finish_reason"] == "error" and store.final["content"] == "Part one"


async def test_http_error_before_any_output_is_retried_once_then_reported(fake_d1, upstream, http_client):
    upstream.script([{"status": 500}], [{"status": 500}])
    engine, _ = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="x")))
    assert len(upstream.requests) == 2 and any(e.type == "notice" for e in events)
    assert next(e.data for e in events if e.type == "error")["code"] == "model_error"


async def test_http_error_then_success_recovers(fake_d1, upstream, http_client):
    upstream.script([{"status": 500}], say("Recovered."))
    engine, _ = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="x")))
    assert text_of(events) == "Recovered." and events[-1].data["finish_reason"] == "stop"


async def test_dead_tunnel_is_reported_as_gpu_unreachable(fake_d1, upstream, http_client):
    upstream.script([{"status": 530}])
    engine, store = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="x")))
    err = next(e.data for e in events if e.type == "error")
    assert err["code"] == "gpu_unreachable" and err["action"] == "check_gpu"
    assert "<html" not in json.dumps(err).lower()


async def test_idle_model_times_out_with_a_clear_error(fake_d1, upstream, http_client):
    upstream.script([{"hang": 5}], [{"hang": 5}])
    engine, _ = await build(upstream, http_client, idle_timeout_s=0.4)
    events = await collect(await engine.start(TurnRequest(message="x")), timeout=15)
    err = next(e.data for e in events if e.type == "error")
    assert err["code"] == "model_error" and "no output" in err["message"]


async def test_gpu_offline_keeps_the_user_message_and_offers_start_gpu(fake_d1, upstream, http_client):
    async def offline():
        return None

    engine, store = await build(upstream, http_client, route=offline)
    events = await collect(await engine.start(TurnRequest(message="hello?")))
    err = next(e.data for e in events if e.type == "error")
    assert err["code"] == "gpu_offline" and err["action"] == "start_gpu"
    assert store.began[0]["user_text"] == "hello?" and store.final["finish_reason"] == "error"
    assert not upstream.requests


async def test_empty_model_reply_is_nudged_once_then_reported(fake_d1, upstream, http_client):
    upstream.script([{"finish": "stop"}], [{"finish": "stop"}])
    engine, store = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="x")))
    assert len(upstream.requests) == 2
    assert events[-1].data["finish_reason"] == "incomplete" and "could not produce" in text_of(events)


async def test_empty_final_step_after_tools_is_not_mistaken_for_an_answer(fake_d1, upstream, http_client):
    upstream.script(call("terminal", {"command": "ls"}, preamble="Let me look. "), [{"finish": "stop"}], say("Here it is."))
    engine, store = await build(upstream, http_client)
    events = await collect(await engine.start(TurnRequest(message="x")))
    assert text_of(events).endswith("Here it is.") and events[-1].data["finish_reason"] == "stop"


# ---- history, regenerate, attachments, concurrency -----------------------------------------------------
async def test_history_custom_prompt_and_memories_reach_the_model(fake_d1, upstream, http_client):
    store = MemoryStore(history=[{"role": "assistant", "content": "newer"}, {"role": "user", "content": "older"}])
    store.custom = "You are a pirate."
    store.memories = ["User likes tea"]
    upstream.script(say("Arr."))
    engine, _ = await build(upstream, http_client, store=store)
    await collect(await engine.start(TurnRequest(message="ahoy")))
    msgs = upstream.requests[0]["messages"]
    assert msgs[0]["role"] == "system" and "You are a pirate." in msgs[0]["content"] and "User likes tea" in msgs[0]["content"]
    assert [m["content"] for m in msgs[1:]] == ["older", "newer", "ahoy"]  # chronological


async def test_history_is_trimmed_to_the_character_budget(fake_d1, upstream, http_client):
    rows = [{"role": "user", "content": "x" * 400} for _ in range(10)]
    store = MemoryStore(history=rows)
    upstream.script(say("ok"))
    engine, _ = await build(upstream, http_client, store=store, history_limit_chars=1000)
    await collect(await engine.start(TurnRequest(message="now")))
    assert len(upstream.requests[0]["messages"]) == 1 + 2 + 1  # 3x400 chars would exceed the 1000-char budget


async def test_regenerate_reuses_the_last_user_message_without_duplicating_it(fake_d1, upstream, http_client):
    upstream.script(say("Second try."))
    engine, store = await build(upstream, http_client)
    await collect(await engine.start(TurnRequest(session_id="studio-x", regenerate=True)))
    assert store.dropped == [7] and store.began[0]["create_user_message"] is False
    assert upstream.requests[0]["messages"][-1] == {"role": "user", "content": "previous question"}


async def test_attachments_are_shown_to_the_model_and_recorded(fake_d1, upstream, http_client):
    upstream.script(say("It says hello."))
    store = MemoryStore()

    async def loader(items):
        return [{"name": "note.txt", "path": "note.txt", "text": "HELLO FILE"}]

    engine, _ = await build(upstream, http_client, store=store)
    engine.attachment_loader = loader
    await collect(await engine.start(TurnRequest(message="read it", attachments=[{"path": "note.txt"}])))
    assert "HELLO FILE" in upstream.requests[0]["messages"][-1]["content"]
    assert store.began[0]["user_meta"] == {"attachments": ["note.txt"]}


async def test_new_turn_in_a_session_cancels_the_previous_one(fake_d1, upstream, http_client):
    upstream.token_delay = 0.05
    upstream.script([{"content": "slow " * 200}, {"finish": "stop"}], say("fast"))
    engine, store = await build(upstream, http_client)
    first = await engine.start(TurnRequest(session_id="studio-same", message="one"))
    for _ in range(50):
        if any(e.type == "delta" for e in first.events):
            break
        await asyncio.sleep(0.05)
    second = await engine.start(TurnRequest(session_id="studio-same", message="two"))
    await collect(second)
    assert first.finished and first.events[-1].data["finish_reason"] == "cancelled"
    assert text_of(second.events) == "fast"


async def test_unexpected_exception_becomes_a_clean_error(fake_d1, upstream, http_client):
    store = MemoryStore()

    async def boom(*a, **k):
        raise RuntimeError("db exploded")

    store.load_context = boom
    engine, _ = await build(upstream, http_client, store=store)
    events = await collect(await engine.start(TurnRequest(message="x")))
    err = next(e.data for e in events if e.type == "error")
    assert err["code"] == "internal" and "db exploded" not in json.dumps(err)
    assert events[-1].type == "done"
