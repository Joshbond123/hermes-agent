"""History storage: every UI action must change real stored rows."""

from __future__ import annotations

import time

import pytest

from blackthorn import config, sessions

pytestmark = pytest.mark.asyncio


async def make_session(fake_d1, text="hello world", sid=None):
    sid = sid or sessions.new_session_id()
    rows = await sessions.begin_turn(sid, text, model="m", started_at=time.time())
    await sessions.save_assistant(
        rows["assistant_message_id"], sid, content="answer to " + text, finish_reason="stop",
        metadata={"parts": [{"t": "text", "text": "answer to " + text}], "usage": {"completion_tokens": 3}},
        token_count=3)
    return sid, rows


async def test_begin_turn_creates_session_user_and_assistant_rows_in_one_round_trip(fake_d1):
    before = fake_d1.calls
    sid = "studio-abc"
    rows = await sessions.begin_turn(sid, "What is 2+2?", model="qwen", started_at=1000.0)
    assert fake_d1.calls - before == 1  # one D1 request, not three
    assert rows["created"] and rows["user_message_id"] and rows["assistant_message_id"] > rows["user_message_id"]
    sess = fake_d1.rows("SELECT * FROM sessions WHERE id = ?", [sid])[0]
    assert sess["source"] == "blackthorn-test" and sess["title"] == "What is 2+2?" and sess["title_source"] == "auto"
    msgs = fake_d1.rows("SELECT role, content, finish_reason FROM messages WHERE session_id = ? ORDER BY id", [sid])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "What is 2+2?" and msgs[1]["finish_reason"] == sessions.RUNNING


async def test_second_turn_reuses_session(fake_d1):
    sid, _ = await make_session(fake_d1)
    rows = await sessions.begin_turn(sid, "again", model="m", started_at=time.time() + 1)
    assert rows["created"] is False
    assert fake_d1.rows("SELECT COUNT(*) AS n FROM sessions")[0]["n"] == 1


async def test_cannot_write_into_a_session_owned_by_another_app(fake_d1):
    fake_d1.rows("INSERT INTO sessions (id, source, started_at) VALUES ('tui-1', 'tui', 1.0)")
    with pytest.raises(sessions.SessionConflict):
        await sessions.begin_turn("tui-1", "hijack", model="m", started_at=2.0)
    assert fake_d1.rows("SELECT COUNT(*) AS n FROM messages WHERE session_id = 'tui-1'")[0]["n"] == 0


async def test_list_orders_pinned_first_then_most_recent_activity(fake_d1):
    a, _ = await make_session(fake_d1, "alpha")
    b, _ = await make_session(fake_d1, "bravo")
    c, _ = await make_session(fake_d1, "charlie")
    fake_d1.rows("UPDATE sessions SET last_activity_at = 100 WHERE id = ?", [a])
    fake_d1.rows("UPDATE sessions SET last_activity_at = 300 WHERE id = ?", [b])
    fake_d1.rows("UPDATE sessions SET last_activity_at = 200 WHERE id = ?", [c])
    assert [s["id"] for s in await sessions.list_sessions()] == [b, c, a]
    await sessions.update_session(a, pinned=True)
    listed = await sessions.list_sessions()
    assert [s["id"] for s in listed] == [a, b, c] and listed[0]["pinned"] is True


async def test_rename_persists_and_marks_user_title(fake_d1):
    sid, _ = await make_session(fake_d1, "original question")
    updated = await sessions.update_session(sid, title="  My   renamed\nchat ")
    assert updated["title"] == "My renamed chat"
    row = fake_d1.rows("SELECT title, display_name, title_source FROM sessions WHERE id = ?", [sid])[0]
    assert row == {"title": "My renamed chat", "display_name": "My renamed chat", "title_source": "user"}
    assert (await sessions.get_session(sid))["title"] == "My renamed chat"


async def test_rename_rejects_empty_and_clips_long_titles(fake_d1):
    sid, _ = await make_session(fake_d1)
    with pytest.raises(ValueError):
        await sessions.update_session(sid, title="   ")
    out = await sessions.update_session(sid, title="x" * 500)
    assert len(out["title"]) == sessions.TITLE_MAX


async def test_pin_and_unpin_round_trip(fake_d1):
    sid, _ = await make_session(fake_d1)
    assert (await sessions.update_session(sid, pinned=True))["pinned"] is True
    assert fake_d1.rows("SELECT pinned FROM sessions WHERE id = ?", [sid])[0]["pinned"] == 1
    assert (await sessions.update_session(sid, pinned=False))["pinned"] is False
    assert fake_d1.rows("SELECT pinned FROM sessions WHERE id = ?", [sid])[0]["pinned"] == 0


async def test_archive_unpins_hides_from_main_list_and_can_be_restored(fake_d1):
    sid, _ = await make_session(fake_d1, "to archive")
    await sessions.update_session(sid, pinned=True)
    out = await sessions.update_session(sid, archived=True)
    assert out["archived"] is True and out["pinned"] is False
    assert [s["id"] for s in await sessions.list_sessions()] == []
    assert [s["id"] for s in await sessions.list_sessions(archived=True)] == [sid]
    restored = await sessions.update_session(sid, archived=False)
    assert restored["archived"] is False
    assert [s["id"] for s in await sessions.list_sessions()] == [sid]
    assert await sessions.list_sessions(archived=True) == []


async def test_archived_session_messages_remain_readable(fake_d1):
    sid, _ = await make_session(fake_d1, "keep my words")
    await sessions.update_session(sid, archived=True)
    msgs = await sessions.list_messages(sid)
    assert [m["content"] for m in msgs] == ["keep my words", "answer to keep my words"]


async def test_delete_removes_session_and_all_messages(fake_d1):
    sid, _ = await make_session(fake_d1)
    other, _ = await make_session(fake_d1, "other")
    assert await sessions.delete_session(sid) is True
    assert fake_d1.rows("SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", [sid])[0]["n"] == 0
    assert await sessions.get_session(sid) is None
    assert await sessions.get_session(other) is not None
    assert await sessions.delete_session(sid) is False


async def test_delete_refuses_foreign_sessions(fake_d1):
    fake_d1.rows("INSERT INTO sessions (id, source, started_at) VALUES ('tui-1', 'tui', 1.0)")
    fake_d1.rows("INSERT INTO messages (session_id, role, content, timestamp) VALUES ('tui-1', 'user', 'x', 1.0)")
    assert await sessions.delete_session("tui-1") is False
    assert fake_d1.rows("SELECT COUNT(*) AS n FROM messages WHERE session_id = 'tui-1'")[0]["n"] == 1


async def test_update_unknown_session_raises(fake_d1):
    with pytest.raises(sessions.SessionNotFound):
        await sessions.update_session("studio-nope", title="x")
    with pytest.raises(sessions.SessionNotFound):
        await sessions.update_session("studio-nope", pinned=True)


async def test_search_matches_title_and_message_text_and_is_injection_safe(fake_d1):
    a, _ = await make_session(fake_d1, "kubernetes networking")
    b, _ = await make_session(fake_d1, "pasta recipe")
    assert [s["id"] for s in await sessions.list_sessions(query="kubern")] == [a]
    assert [s["id"] for s in await sessions.list_sessions(query="answer to pasta")] == [b]
    assert await sessions.list_sessions(query="100%") == []
    assert len(await sessions.list_sessions(query="'; DROP TABLE sessions; --")) == 0
    assert fake_d1.rows("SELECT COUNT(*) AS n FROM sessions")[0]["n"] == 2


async def test_messages_returned_in_order_with_parts_and_attachments(fake_d1):
    sid = "studio-parts"
    rows = await sessions.begin_turn(sid, "run it", model="m", started_at=10.0, user_meta={"attachments": ["a.txt"]})
    parts = [{"t": "text", "text": "Running."}, {"t": "tool", "id": "c1", "name": "terminal", "status": "ok"},
             {"t": "text", "text": "Done."}]
    await sessions.save_assistant(rows["assistant_message_id"], sid, content="Running.\n\nDone.",
                                  finish_reason="stop", metadata={"parts": parts, "thinking_ms": 1200}, token_count=5)
    msgs = await sessions.list_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["attachments"] == ["a.txt"]
    assert msgs[1]["parts"] == parts and msgs[1]["thinking_ms"] == 1200 and msgs[1]["finish_reason"] == "stop"


async def test_message_count_and_last_activity_follow_saved_answers(fake_d1):
    sid, _ = await make_session(fake_d1)
    row = fake_d1.rows("SELECT message_count, last_activity_at FROM sessions WHERE id = ?", [sid])[0]
    assert row["message_count"] == 2 and row["last_activity_at"] > 0


async def test_regenerate_drops_everything_after_the_user_message(fake_d1):
    sid, rows = await make_session(fake_d1, "question")
    last = await sessions.last_user_message(sid)
    assert last["content"] == "question"
    assert await sessions.drop_after(sid, last["id"]) == 1
    assert [m["role"] for m in await sessions.list_messages(sid)] == ["user"]


async def test_history_statement_excludes_the_turn_being_created(fake_d1):
    sid, _ = await make_session(fake_d1, "first")
    cutoff = time.time() + 5
    await sessions.begin_turn(sid, "second", model="m", started_at=cutoff)
    sql, params = sessions.history_statement(sid, cutoff)
    got = [r["content"] for r in fake_d1.rows(sql, params)]
    assert "second" not in got and "first" in got


async def test_running_rows_are_closed_on_restart(fake_d1):
    sid = "studio-r"
    await sessions.begin_turn(sid, "q", model="m", started_at=1.0)
    assert await sessions.recover_interrupted() == 1
    msgs = await sessions.list_messages(sid)
    assert msgs[-1]["finish_reason"] == "interrupted"
    assert await sessions.recover_interrupted() == 0


async def test_auto_title_shortens_on_word_boundary():
    assert sessions.auto_title("") == "New chat"
    long = "word " * 40
    t = sessions.auto_title(long)
    assert len(t) <= sessions.AUTO_TITLE_MAX + 1 and t.endswith("…")
    assert sessions.auto_title("  spaced\n out  ") == "spaced out"


async def test_test_source_isolates_from_production_source(fake_d1, monkeypatch):
    sid, _ = await make_session(fake_d1)
    monkeypatch.setenv("BLACKTHORN_SESSION_SOURCE", config.DEFAULT_SESSION_SOURCE)
    assert await sessions.list_sessions() == []
    assert await sessions.get_session(sid) is None
