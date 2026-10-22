"""Conversation memory + autonomous task execution.

These tests reproduce the reported failures — forgotten instructions, repeated work, invented
progress, work abandoned at a restart — against the real app (real HTTP/SSE, real store, real agent
loop) with only the model and the remote computer faked. Scripts are queued on ``FakeBackend``, one
list of events per model request, so every assertion is about what the server really sent and stored.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pytest

from blackthorn.app import build_services, create_app
from blackthorn.config import Settings
from blackthorn.sql import SqliteExecutor
from blackthorn.store import ChatStore
from blackthorn.tools import TaskBoard

from .fakes import FakeBackend, ServerThread, call, finish, pause, say, usage


def sh(cmd: str, **kw):
    return call("run_command", {"command": cmd}, **kw)


def todos(*items: Any, **kw):
    """A todo_list write. ``items`` are ``(id, content, status)`` tuples."""
    payload = [{"id": i, "content": c, "status": s} for i, c, s in items]
    return call("todo_list", {"todos": payload, "merge": kw.get("merge", True)}, cid=kw.get("cid", "todo"),
                index=kw.get("index", 0))


def blob(events: List[Dict]) -> str:
    return json.dumps(events, ensure_ascii=False)


async def persisted_ledger(stack, session_id: str) -> List[Dict[str, Any]]:
    """The ledger stored with the assistant row, as the next turn will read it."""
    from blackthorn.store import ledger_from_row
    res = await stack.services.store.db.query(
        "SELECT role, content, finish_reason, display_kind, display_metadata FROM messages "
        "WHERE session_id = ? AND role = 'assistant' ORDER BY id DESC LIMIT 1", [session_id])
    return ledger_from_row(dict(res.rows[0]))


def last_request(backend: FakeBackend) -> Dict[str, Any]:
    return backend.requests[-1]


def system_of(backend: FakeBackend) -> str:
    return str(last_request(backend)["messages"][0]["content"])


def prompt_text(backend: FakeBackend) -> str:
    return json.dumps(last_request(backend), ensure_ascii=False)


def check_wire_protocol(messages: List[Dict[str, Any]]) -> None:
    """The request must be one a real server accepts: strict alternation, no orphan tool results."""
    roles = [m["role"] for m in messages]
    assert roles[0] == "system"
    for a, b in zip(roles, roles[1:]):
        if a == b:
            assert a == "tool", f"two {a} messages in a row: {roles}"
    call_ids = {c["id"] for m in messages for c in (m.get("tool_calls") or [])}
    for m in messages:
        if m["role"] == "tool":
            assert m.get("tool_call_id") in call_ids, "orphan tool result"
    for m in messages:
        if m.get("tool_calls"):
            for c in m["tool_calls"]:
                assert any(x.get("tool_call_id") == c["id"] for x in messages if x["role"] == "tool"), "call without a result"


# =====================================================================================================
# 1. Long conversations: the instruction and the executed work must survive
# =====================================================================================================
async def test_next_turn_sees_what_the_tools_actually_did(stack):
    """The model's next turn must be able to see the previous turn's real tool results."""
    stack.backend.exec_output = "3 passed, 0 failed"
    stack.backend.queue([*say("Running the suite. ", 1), sh("pytest -q", cid="c1"), finish("tool_calls")],
                        [*say("3 tests passed."), finish("stop")])
    first = await stack.stream({"message": "run the test suite and tell me how many passed"})
    assert first.text.endswith("3 tests passed.")

    # Follow-up with no new tool work: the answer must come from the record, not from a re-run.
    stack.backend.queue([*say("Earlier, 3 tests passed."), finish("stop")])
    second = await stack.stream({"session_id": first.session_id, "message": "how many passed?"})
    assert second.text == "Earlier, 3 tests passed."
    assert second.end["tool_calls"] == 0, "the model re-ran the suite: the execution record did not reach it"
    system = system_of(stack.backend)
    assert "pytest -q" in system and "ok" in system        # server-side state layer carries the evidence
    check_wire_protocol(last_request(stack.backend)["messages"])


async def test_long_conversation_keeps_the_original_instruction(stack):
    """Ten turns with big outputs: the first request and the newest one both stay in the prompt."""
    first_request = "MIGRATION-ANCHOR-7741: rebuild the reporting service without downtime"
    stack.backend.exec_output = "x" * 3000
    stack.backend.queue([*say(f"Step {0} done. " + "y" * 1500), finish("stop")])
    out = await stack.stream({"message": first_request})
    sid = out.session_id
    for turn in range(1, 10):
        stack.backend.queue([sh(f"echo {turn}", cid=f"c{turn}"), finish("tool_calls")],
                            [*say(f"Step {turn} done. " + "z" * 1200), finish("stop")])
        await stack.stream({"session_id": sid, "message": f"continue with step {turn} of the MIGRATION-ANCHOR plan"})
    stack.backend.queue([*say("Carrying on."), finish("stop")])
    await stack.stream({"session_id": sid, "message": "where are we now?"})
    messages = last_request(stack.backend)["messages"]
    text = json.dumps(messages, ensure_ascii=False)
    assert "MIGRATION-ANCHOR-7741" in text, "the original instruction was lost in a long conversation"
    assert messages[-1]["role"] == "user" and "where are we now" in messages[-1]["content"]
    check_wire_protocol(messages)
    # every tool call has its result, and no result was left behind without its call
    assert text.count('"role": "tool"') == 0 or '"tool_calls"' in text


async def test_compaction_summarises_instead_of_deleting(stack):
    """When the history no longer fits, the dropped turns are replaced by a server-built summary."""
    from blackthorn import prompts
    turns: List[Dict[str, Any]] = []
    for i in range(12):
        turns.append({"role": "user", "content": f"request number {i}: install package {i}", "status": "", "ledger": []})
        turns.append({"role": "assistant", "content": f"installed package {i}",
                      "status": "stop",
                      "ledger": [{"n": "run_command", "a": f'{{"command":"pip install pkg{i}"}}', "ok": i != 4,
                                  "s": "ok" if i != 4 else "error", "e": "" if i != 4 else "boom", "o": "done"}]})
    messages, dropped, summary = prompts.compile_history(turns, budget_tokens=600, chars_per_token=3.2)
    assert dropped > 0
    assert summary is not None
    assert "request number 0" in summary, "the opening request must survive inside the summary"
    assert "pip install pkg4" in summary and "error" in summary, "failures must be summarised too"
    assert "installed package 11" in json.dumps(messages, ensure_ascii=False), "the newest turn must stay verbatim"
    check_wire_protocol([{"role": "system", "content": "s"}] + messages)


# =====================================================================================================
# 2. Multi-step autonomous tasks: the plan persists and the run continues until it is really done
# =====================================================================================================
async def test_multi_step_task_continues_until_the_list_is_empty(stack):
    """A model that stops with open tasks is asked to continue — bounded, then honest."""
    stack.backend.exec_output = "ok"
    stack.backend.queue(
        [todos(("1", "create the workspace", "in_progress"), ("2", "run the build", "pending"),
               ("3", "write the summary", "pending"), cid="t1", merge=False), finish("tool_calls")],
        [sh("mkdir -p build", cid="c1", index=0),
         todos(("1", "create the workspace", "completed"), ("2", "run the build", "in_progress"), cid="t2", index=1),
         finish("tool_calls")],
        [*say("The workspace is ready. I'll stop here."), finish("stop")],      # premature stop #1
        [sh("make build", cid="c2", index=0), todos(("2", "run the build", "completed"), cid="t3", index=1),
         finish("tool_calls")],
        [*say("Build done. All good!"), finish("stop")],                        # premature stop #2
        [todos(("3", "write the summary", "completed"), cid="t4"), finish("tool_calls")],
        [*say("All three steps are complete."), finish("stop")],
    )
    out = await stack.stream({"message": "set up the build, run it, and summarise"})
    assert out.end["status"] == "stop", out.end
    commands = [c["body"]["command"] for c in stack.backend.computer_calls if c["name"] == "exec"]
    assert commands == ["mkdir -p build", "make build"], f"work was repeated or skipped: {commands}"
    assert out.text.endswith("All three steps are complete.")
    notices = [n["text"] for n in out.of("notice")]
    assert any("still open" in n for n in notices), f"the continuation was not surfaced: {notices}"
    # the plan is persisted, and it is finished
    board = await TaskBoard.load(stack.services.store, out.session_id)
    assert board.items and all(i["status"] == "completed" for i in board.items), board.items
    assert not any("unfinished" in n for n in notices)


async def test_unfinished_tasks_are_reported_not_claimed(stack):
    """Nudges are bounded: after the budget the run ends and says plainly that it is unfinished."""
    stack.backend.queue(
        [todos(("1", "step one", "in_progress"), ("2", "step two", "pending"), cid="t1", merge=False),
         finish("tool_calls")],
        *[[*say("Stopping early."), finish("stop")]] * 6,
    )
    out = await stack.stream({"message": "do the two steps"})
    assert out.end["status"] == "stop"
    assert out.text.strip().startswith("Stopping early.")
    notices = [n["text"] for n in out.of("notice")]
    assert any("unfinished task" in n for n in notices), f"no honest close-out: {notices}"
    board = await TaskBoard.load(stack.services.store, out.session_id)
    assert len(board.pending()) == 2


async def test_task_list_survives_a_new_turn_and_stops_repeating_work(stack):
    """The plan written in turn 1 is still there (and still open) in turn 2."""
    stack.backend.queue(
        [todos(("1", "collect the data", "completed"), ("2", "train the model", "in_progress"),
               ("3", "evaluate", "pending"), cid="t1", merge=False), finish("tool_calls")],
        [*say("Collected. Training next."), finish("stop")],
    )
    first = await stack.stream({"message": "train and evaluate a model"})
    stack.backend.queue([*say("Resuming training."), finish("stop")])
    await stack.stream({"session_id": first.session_id, "message": "continue"})
    system = system_of(stack.backend)
    assert "train the model" in system and "[>]" in system, system
    assert "collect the data" in system


# =====================================================================================================
# 3. Failed tools: recorded honestly, recoverable
# =====================================================================================================
async def test_failed_tool_is_recorded_and_recovery_is_possible(stack):
    stack.backend.files["real.txt"] = "the real content"
    stack.backend.queue([call("read_file", {"path": "missing.txt"}, cid="f1"), finish("tool_calls")],
                        [call("read_file", {"path": "real.txt"}, cid="f2"), finish("tool_calls")],
                        [*say("The first path was wrong; real.txt says: the real content"), finish("stop")])
    out = await stack.stream({"message": "read the file"})
    assert out.end["status"] == "stop"
    ends = out.of("tool.end")
    assert [e["status"] for e in ends] == ["error", "ok"]
    system = system_of(stack.backend)
    assert "missing.txt" in system and "error" in system, "the failure must be in the record, not forgotten"
    assert "real.txt" in system and "-> ok" in system
    # the persisted record is the anti-hallucination evidence: failure first, then the successful retry
    stored = await stack.services.store.get_messages(out.session_id)
    assert [p["status"] for p in stored[-1]["parts"] if p["type"] == "tool"] == ["error", "ok"]
    ledger = await persisted_ledger(stack, out.session_id)
    assert [(e["n"], e["s"]) for e in ledger] == [("read_file", "error"), ("read_file", "ok")]


async def test_repeated_identical_read_is_suppressed(stack):
    """Unchanged input, unchanged answer: the safety valve still fires (no repeated work)."""
    stack.backend.queue([call("list_files", {"path": "."}, cid="d1"), finish("tool_calls")],
                        [call("list_files", {"path": "."}, cid="d2"), finish("tool_calls")],
                        [call("list_files", {"path": "."}, cid="d3"), finish("tool_calls")],
                        [*say("Done."), finish("stop")])
    out = await stack.stream({"message": "list the files"})
    statuses = [e["status"] for e in out.of("tool.end")]
    assert statuses == ["ok", "ok", "error"], statuses      # the third identical call is refused
    assert "duplicate call" in json.dumps(out.of("tool.end")[-1], ensure_ascii=False)
    runs = [c for c in stack.backend.computer_calls if c["name"] == "list_files"]
    assert len(runs) == 2, "the same listing was fetched more times than needed"


async def test_same_command_after_a_write_is_allowed(stack):
    """Re-running a command after a real change is legitimate — the old key blocked it."""
    stack.backend.queue([sh("pytest -q", cid="r1"), finish("tool_calls")],
                        [call("write_file", {"path": "a.py", "content": "x = 1"}, cid="w1"), finish("tool_calls")],
                        [sh("pytest -q", cid="r2"), finish("tool_calls")],
                        [*say("Both runs executed."), finish("stop")])
    out = await stack.stream({"message": "run the tests, edit the file, run them again"})
    assert [e["status"] for e in out.of("tool.end")] == ["ok", "ok", "ok"]
    assert len([c for c in stack.backend.computer_calls if c["name"] == "exec"]) == 2


# =====================================================================================================
# 4. Context pressure
# =====================================================================================================
async def test_context_pressure_keeps_the_request_and_the_protocol(stack):
    """Huge tool outputs: the run continues, nothing is orphaned, the request stays in the prompt."""
    stack.backend.exec_output = ("output line " * 400)
    stack.backend.queue([sh("big command one", cid="b1"), sh("big command two", cid="b2"), finish("tool_calls")],
                        [sh("big command three", cid="b3"), finish("tool_calls")],
                        [sh("big command four", cid="b4"), finish("tool_calls")],
                        [sh("big command five", cid="b5"), finish("tool_calls")],
                        [*say("Finished the REFACTOR-ANCHOR-3311 work."), finish("stop")])
    out = await stack.stream({"message": "REFACTOR-ANCHOR-3311: rebuild the parser and report"})
    assert out.end["status"] == "stop", out.end
    assert out.text.endswith("Finished the REFACTOR-ANCHOR-3311 work.")
    for req in stack.backend.requests:
        check_wire_protocol(req["messages"])
        assert "REFACTOR-ANCHOR-3311" in json.dumps(req["messages"], ensure_ascii=False)


async def test_server_context_overflow_compacts_instead_of_wiping(stack):
    """A 400 'context size' from the engine: history is compacted, the answer still arrives."""
    stack.backend.exec_output = "z" * 2500
    stack.backend.queue([sh("first command", cid="o1"), finish("tool_calls")],
                        [*say("First turn finished with a long answer. " * 20), finish("stop")])
    first = await stack.stream({"message": "FIRST-ANCHOR: start the migration"})
    stack.backend.script.clear()
    stack.backend.queue({"http_error": 400, "message": "the request exceeds the available context size"},
                        [sh("second command", cid="o2"), finish("tool_calls")],
                        [*say("Continuing after compaction."), finish("stop")])
    second = await stack.stream({"session_id": first.session_id, "message": "SECOND-ANCHOR: keep going"})
    assert second.end["status"] == "stop"
    assert "Continuing after compaction." in second.text
    notices = [n["text"] for n in second.of("notice")]
    assert any("trimmed" in n for n in notices), notices
    check_wire_protocol(last_request(stack.backend)["messages"])
    assert "SECOND-ANCHOR" in prompt_text(stack.backend), "the current request must survive an overflow"


# =====================================================================================================
# 5. Follow-ups, regenerate
# =====================================================================================================
async def test_short_followup_still_gets_tools(stack):
    stack.backend.queue([sh("echo start", cid="s1"), finish("tool_calls")], [*say("Started."), finish("stop")])
    first = await stack.stream({"message": "start the deployment please"})
    stack.backend.queue([sh("echo continue", cid="s2"), finish("tool_calls")], [*say("Continued."), finish("stop")])
    second = await stack.stream({"session_id": first.session_id, "message": "continue"})
    assert "tools" in last_request(stack.backend), "a follow-up lost its tools: it cannot finish the work"
    assert second.end["tool_calls"] >= 1

    # a brand-new greeting is still sent without schemas (prefill latency)
    other = await stack.stream({"message": "Hi"})
    assert "tools" not in stack.backend.requests[-1]


async def test_regenerate_keeps_the_record_of_the_discarded_attempt(stack):
    stack.backend.queue([sh("rm -rf /tmp/cache", cid="g1"), finish("tool_calls")],
                        [*say("Cache cleared (this answer is wrong)."), finish("stop")])
    first = await stack.stream({"message": "clear the cache"})
    stack.backend.queue([*say("The cache was already cleared in the previous attempt."), finish("stop")])
    await stack.stream({"session_id": first.session_id, "regenerate": True})
    system = system_of(stack.backend)
    assert "rm -rf /tmp/cache" in system, "the discarded attempt's side effects were forgotten"
    assert "discarded" in system.lower() or "do not repeat" in system.lower()


# =====================================================================================================
# 6. Restart / interruption (a second app instance on the same database)
# =====================================================================================================
class FileStack:
    """The real app on a file-backed SQLite database, so it can be killed and restarted."""

    def __init__(self, db_path: Path, static_dir: Path) -> None:
        self.backend = FakeBackend()
        self.backend_srv = ServerThread(self.backend.app).start()
        self.executor = SqliteExecutor(str(db_path))
        self.settings = Settings(store="sqlite", heartbeat_s=0.5, tool_timeout_s=5.0)
        self.services = build_services(self.settings, executor=self.executor, static_dir=static_dir)
        self.app_srv = ServerThread(create_app(self.settings, services=self.services, gpu_daemon=False)).start()
        self.url = self.app_srv.url

    async def ready(self) -> None:
        await self.services.store.ensure_schema()
        await self.executor.query("DELETE FROM kaggle_gpu_state")
        await self.executor.query(
            "INSERT INTO kaggle_gpu_state (id, status, tunnel_url, api_key, model, gpu_info, updated_at) "
            "VALUES ('primary', 'HEARTBEAT_ONLINE', ?, 'test-key', 'fake-model', 'T4', ?)",
            [self.backend_srv.url, time.time()])
        self.services.resolver.invalidate()

    async def stream(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", f"{self.url}/api/chat/stream", json=payload) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        events.append(json.loads(line[5:].strip()))
        return events

    def kill(self) -> None:
        self.app_srv.stop()
        self.backend_srv.stop()


@pytest.fixture
def file_db(tmp_path):
    (tmp_path / "static").mkdir(exist_ok=True)
    return tmp_path / "state.sqlite3", tmp_path / "static"


async def test_restart_resumes_from_the_saved_state(file_db):
    """Stop the process mid-run: the next process must resume from what actually ran."""
    db_path, static_dir = file_db
    one = FileStack(db_path, static_dir)
    await one.ready()
    one.backend.files["report.txt"] = "Q3 numbers"
    one.backend.queue([call("read_file", {"path": "report.txt"}, cid="k1"), finish("tool_calls")],
                      [pause(300), *say("this answer never arrives"), finish("stop")])

    session_id: Optional[str] = None
    saw_tool = False
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", f"{one.url}/api/chat/stream",
                                 json={"message": "read report.txt and summarise it"}) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:].strip())
                if event.get("type") == "run.start":
                    session_id = event["session_id"]
                if event.get("type") == "tool.end":
                    saw_tool = True
                    break
    assert saw_tool and session_id
    await asyncio.sleep(0.8)                       # let the forced post-tool flush land
    one.kill()                                     # the process dies before the answer is finished

    # ---- a new process on the same database
    two = FileStack(db_path, static_dir)
    try:
        await two.ready()
        async with httpx.AsyncClient(timeout=30) as client:
            data = (await client.get(f"{two.url}/api/chat/sessions/{session_id}")).json()
        last = data["messages"][-1]
        assert last["status"] in ("interrupted", "cancelled"), last["status"]
        assert [p["type"] for p in last["parts"]] == ["tool"], "the work that ran must be in the saved turn"

        two.backend.queue([*say("Continuing: report.txt contains the Q3 numbers."), finish("stop")])
        events = await two.stream({"session_id": session_id, "message": "continue"})
        end = [e for e in events if e["type"] == "run.end"][-1]
        assert end["status"] == "stop"
        assert "".join(e["text"] for e in events if e["type"] == "text.delta") == \
            "Continuing: report.txt contains the Q3 numbers."
        system = system_of(two.backend)
        assert "report.txt" in system and "-> ok" in system, "the pre-restart work was lost"
        joined = prompt_text(two.backend)
        assert "interrupted" in joined or "cancelled" in joined, "the unfinished turn must be marked unfinished"
        check_wire_protocol(last_request(two.backend)["messages"])
    finally:
        two.kill()


async def test_task_list_survives_a_restart(file_db):
    db_path, static_dir = file_db
    one = FileStack(db_path, static_dir)
    await one.ready()
    one.backend.queue([todos(("1", "download the dataset", "completed"), ("2", "train", "in_progress"),
                             ("3", "evaluate", "pending"), cid="t1", merge=False), finish("tool_calls")],
                      [*say("Training now."), finish("stop")])
    events = await one.stream({"message": "train and evaluate the model"})
    sid = [e for e in events if e["type"] == "run.start"][0]["session_id"]
    one.kill()

    two = FileStack(db_path, static_dir)
    try:
        await two.ready()
        board = await TaskBoard.load(two.services.store, sid)
        assert [i["status"] for i in board.items] == ["completed", "in_progress", "pending"], board.items
        two.backend.queue([*say("Picking up training."), finish("stop")])
        await two.stream({"session_id": sid, "message": "continue"})
        assert "train" in system_of(two.backend) and "[>]" in system_of(two.backend)
    finally:
        two.kill()


async def test_interrupted_turn_is_recorded_in_history(file_db):
    """A turn that never finished is stored as unfinished — never as a complete answer."""
    db_path, static_dir = file_db
    stack = FileStack(db_path, static_dir)
    await stack.ready()
    store: ChatStore = stack.services.store
    try:
        begin = await store.begin_turn(None, "do the migration", model="m")
        await store.update_assistant(begin["assistant_uid"], text="Half of the answer",
                                     parts=[{"type": "text", "text": "Half of the answer"}], status="streaming",
                                     ledger=[{"n": "run_command", "a": '{"command":"alembic upgrade head"}',
                                              "ok": True, "s": "ok", "e": "", "o": "ok"}])
        assert await store.sweep_interrupted() >= 1       # what start-up does after a crash
        nxt = await store.begin_turn(begin["session_id"], "carry on", model="m")
        turn = [t for t in nxt["turns"] if t["role"] == "assistant"][-1]
        assert turn["status"] == "interrupted"
        assert turn["ledger"] and turn["ledger"][0]["n"] == "run_command"
        messages = prompts_compile(nxt["turns"])
        joined = json.dumps(messages, ensure_ascii=False)
        assert "alembic upgrade head" in joined
        assert "interrupted" in joined
        check_wire_protocol([{"role": "system", "content": "s"}] + messages)
    finally:
        stack.kill()


def prompts_compile(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from blackthorn import prompts
    messages, _dropped, _summary = prompts.compile_history(turns, budget_tokens=100_000, chars_per_token=3.2)
    return messages


# =====================================================================================================
# 7. Storage-level guarantees
# =====================================================================================================
async def test_legacy_rows_without_a_ledger_are_still_readable(file_db):
    """Rows written before the ledger existed must not lose their tool history."""
    db_path, static_dir = file_db
    stack = FileStack(db_path, static_dir)
    await stack.ready()
    store = stack.services.store
    try:
        begin = await store.begin_turn(None, "old question", model="m")
        parts = [{"type": "text", "text": "old answer"},
                 {"type": "tool", "id": "x1", "name": "run_command", "status": "ok", "args": "old cmd",
                  "summary": "exit 0", "output": "fine"}]
        await store.update_assistant(begin["assistant_uid"], text="old answer", parts=parts, status="stop")
        nxt = await store.begin_turn(begin["session_id"], "new question", model="m")
        turn = [t for t in nxt["turns"] if t["role"] == "assistant"][-1]
        assert [e["n"] for e in turn["ledger"]] == ["run_command"], "historic conversations lost their record"
        assert turn["ledger"][0]["ok"] is True
    finally:
        stack.kill()


async def test_memories_from_another_surface_never_leak_in(file_db):
    db_path, static_dir = file_db
    stack = FileStack(db_path, static_dir)
    await stack.ready()
    store = stack.services.store
    try:
        await store.add_memory("Blackthorn runs on two T4 GPUs.")          # this product's memory
        await store.db.query(
            "INSERT INTO hermes_memories (id, target, content, memory_type, importance, user_id, session_id, "
            "profile, created_at, updated_at) VALUES ('foreign', 'memory', 'SECRET-FROM-ANOTHER-PROFILE', "
            "'factual', 1.0, 'someone-else', 'cli-9999', 'default', 1, 1)")
        await store.db.query(
            "INSERT INTO hermes_memories (id, target, content, memory_type, importance, user_id, session_id, "
            "profile, created_at, updated_at) VALUES ('foreign2', 'memory', 'SECRET-OTHER-SESSION', "
            "'factual', 1.0, 'default', 'cli-8888', 'default', 1, 1)")
        memories = await store.list_memories()
        assert any("two T4 GPUs" in m for m in memories)
        assert not any("SECRET" in m for m in memories), memories
        prompt = await stack.services.user_prompt() if hasattr(stack.services, "user_prompt") else ""
        assert "SECRET" not in prompt
    finally:
        stack.kill()


async def test_todo_persistence_round_trip(file_db):
    db_path, static_dir = file_db
    stack = FileStack(db_path, static_dir)
    await stack.ready()
    store = stack.services.store
    try:
        board = TaskBoard(store, "studio-test")
        board.write([{"content": "one"}, {"content": "two"}], merge=False)
        await board.save()
        restored = await TaskBoard.load(store, "studio-test")
        assert [i["content"] for i in restored.items] == ["one", "two"]
        assert restored.pending() and not restored.counts()["completed"]
        other = await TaskBoard.load(store, "studio-other-session")
        assert other.items == [], "task state leaked across sessions"
    finally:
        stack.kill()

async def test_task_origin_survives_long_history_and_restart(file_db):
    db_path, static_dir = file_db
    one = FileStack(db_path, static_dir)
    await one.ready()
    anchor = 'ORIGINAL-GOAL-6841: keep all three migration conditions and verify the final result'
    one.backend.queue([todos(('1', 'run checks', 'pending'), ('2', 'publish result', 'pending'),
                             cid='plan', merge=False), finish('tool_calls')],
                      [*say('Work in progress.'), finish('stop')])
    out = await one.stream({'message': anchor})
    sid = [e for e in out if e['type'] == 'run.start'][0]['session_id']
    for n in range(28):  # more messages than the 48-message history window
        one.backend.queue([*say('not yet'), finish('stop')])
        await one.stream({'session_id': sid, 'message': f'Progress check {n}'})
    one.kill()
    two = FileStack(db_path, static_dir)
    try:
        await two.ready()
        board = await TaskBoard.load(two.services.store, sid)
        assert board.goal == anchor
        two.backend.queue([*say('Continuing'), finish('stop')])
        await two.stream({'session_id': sid, 'message': 'continue the original task'})
        wire = json.dumps(two.backend.requests[-1]['messages'], ensure_ascii=False)
        assert anchor in wire
        assert 'ORIGINAL-GOAL-6841' in wire and 'publish result' in wire
    finally:
        two.kill()


async def test_task_save_failure_does_not_claim_saved(stack, monkeypatch):
    async def cannot_save(*args):
        raise OSError('storage write failed')
    monkeypatch.setattr(stack.services.store, 'save_task_state', cannot_save)
    stack.backend.queue([todos(('1', 'do durable work', 'pending'), merge=False), finish('tool_calls')],
                        [*say('Persistence failed; no task saved.'), finish('stop')])
    out = await stack.stream({'message': 'do durable work'})
    assert 'NOT saved' in json.dumps(out.of('tool.end')) or any(
        e.get('status') == 'error' and e.get('name') == 'todo_list' for e in out.of('tool.end'))
    board = await TaskBoard.load(stack.services.store, out.session_id)
    assert board.items == [] and board.goal == ''


async def test_unavailable_task_storage_blocks_model_not_forgets_tasks(stack, monkeypatch):
    async def cannot_read(*args):
        raise OSError('database unavailable')
    monkeypatch.setattr(stack.services.store, 'load_task_state', cannot_read)
    response = await stack.stream({'message': 'continue the saved migration'})
    assert response.status == 503
    assert response.error_body['detail']['code'] == 'task_storage_unavailable'
    assert stack.backend.requests == []


async def test_legacy_list_only_plan_loads_without_goal(file_db):
    db_path, static_dir = file_db
    stack = FileStack(db_path, static_dir)
    try:
        await stack.ready()
        await stack.services.store.set_setting('bt_todo:studio-old',
            json.dumps([{'id':'1','content':'restore old plan','status':'pending'}]))
        board = await TaskBoard.load(stack.services.store, 'studio-old')
        assert board.goal == '' and board.pending()[0]['content'] == 'restore old plan'
    finally:
        stack.kill()

async def test_session_search_returns_scoped_actual_messages_not_invented_results(file_db):
    from blackthorn.tools.memory import session_search
    from blackthorn.tools.base import ToolContext
    db_path, static_dir = file_db
    stack = FileStack(db_path, static_dir)
    try:
        await stack.ready()
        store = stack.services.store
        sid = (await store.begin_turn(None, 'MIGRATION-SEARCH-914: check tests and publish', model='m'))['session_id']
        await store.db.query("INSERT INTO sessions (id,source,title) VALUES ('other','foreign','private')")
        await store.db.query("INSERT INTO messages (session_id,role,content,active) VALUES ('other','user',?,1)",
                             ['MIGRATION-SEARCH-914: foreign source secret'])
        rows = await store.search_session_messages('MIGRATION-SEARCH-914')
        assert len(rows) == 1 and rows[0]['session_id'] == sid
        ctx = ToolContext(settings=stack.services.settings, store=store, computer=None, http=None, tavily=None)
        result = await session_search({'query':'MIGRATION-SEARCH-914'}, ctx)
        assert result.ok and sid in result.content and 'foreign source secret' not in result.content
        assert (await session_search({'query':'something-that-isnt-here'}, ctx)).content.startswith('No matching')
    finally:
        stack.kill()


async def test_corrupt_persisted_plan_is_not_silently_reset(file_db):
    db_path, static_dir = file_db
    stack = FileStack(db_path, static_dir)
    try:
        await stack.ready()
        await stack.services.store.set_setting('bt_todo:studio-broken', '{not JSON')
        with pytest.raises(ValueError):
            await TaskBoard.load(stack.services.store, 'studio-broken')
    finally:
        stack.kill()
