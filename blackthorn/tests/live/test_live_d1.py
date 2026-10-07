"""Opt-in verification against the REAL Cloudflare D1 schema and the REAL GPU (``BLACKTHORN_LIVE=1`` + CLOUDFLARE_* env).

Rows are tagged ``source = 'blackthorn-livetest'`` so they never show up in the product and are deleted afterwards.
"""

import os

import httpx
import pytest
import pytest_asyncio

from blackthorn.app import build_services, create_app
from blackthorn.config import Settings
from blackthorn.sql import D1Executor
from blackthorn.store import ChatStore

from ..fakes import ServerThread

pytestmark = [pytest.mark.skipif(not os.environ.get("BLACKTHORN_LIVE"), reason="live checks are opt-in (BLACKTHORN_LIVE=1)"),
              pytest.mark.asyncio]
SRC = "blackthorn-livetest"


def d1() -> D1Executor:
    return D1Executor(os.environ["CLOUDFLARE_ACCOUNT_ID"], os.environ["CLOUDFLARE_D1_DATABASE_ID"], os.environ["CLOUDFLARE_API_TOKEN"])


async def cleanup(ex: D1Executor):
    await ex.batch([("DELETE FROM messages WHERE session_id IN (SELECT id FROM sessions WHERE source = ?)", [SRC]),
                    ("DELETE FROM sessions WHERE source = ?", [SRC])])


@pytest_asyncio.fixture
async def ex():
    e = d1()
    await cleanup(e)
    yield e
    await cleanup(e)
    await e.aclose()


async def test_store_semantics_on_the_real_schema(ex):
    store = ChatStore(ex, source=SRC, backend="d1")
    await store.ensure_schema()
    a = await store.begin_turn(None, "first question about D1", model="m")
    assert a["created"] and a["history"] == [] and a["session"]["title"].startswith("first question")
    await store.update_assistant(a["assistant_uid"], text="first answer", parts=[{"type": "text", "text": "first answer"}], status="stop",
                                 meta={"duration_ms": 5}, tokens=7)
    b = await store.begin_turn(a["session_id"], "second question", model="m")
    assert b["history"] == [{"role": "user", "content": "first question about D1"}, {"role": "assistant", "content": "first answer"}]
    await store.update_assistant(b["assistant_uid"], text="second answer", parts=[{"type": "text", "text": "second answer"}], status="stop")
    msgs = await store.get_messages(a["session_id"])
    assert [(m["role"], m["content"]) for m in msgs] == [("user", "first question about D1"), ("assistant", "first answer"),
                                                          ("user", "second question"), ("assistant", "second answer")]
    assert [m["seq"] for m in msgs] == sorted(m["seq"] for m in msgs)
    assert msgs[1]["meta"]["duration_ms"] == 5 and msgs[1]["status"] == "stop"
    # list / rename / pin / archive / search all hit the real columns
    assert [s["id"] for s in await store.list_sessions()] == [a["session_id"]]
    assert (await store.update_session(a["session_id"], title="  Renamed  "))["title"] == "Renamed"
    assert (await store.update_session(a["session_id"], pinned=True))["pinned"] is True
    assert [s["id"] for s in await store.list_sessions(q="second answer")] == [a["session_id"]]
    assert (await store.update_session(a["session_id"], archived=True))["archived"] is True
    assert await store.list_sessions() == [] and len(await store.list_sessions(archived=True)) == 1
    row = (await ex.query("SELECT title, pinned, archived, title_source, source, message_count FROM sessions WHERE id = ?", [a["session_id"]])).rows[0]
    assert row["title"] == "Renamed" and row["archived"] == 1 and row["pinned"] == 0 and row["title_source"] == "user" and row["source"] == SRC
    assert row["message_count"] == 4
    # un-archive via a new message; regenerate replaces the last answer
    c = await store.begin_turn(a["session_id"], "third", model="m")
    assert (await store.get_session(a["session_id"]))["archived"] is False
    await store.update_assistant(c["assistant_uid"], text="third answer", parts=[], status="error", meta={"error": {"code": "x"}})
    r = await store.begin_regeneration(a["session_id"], model="m")
    assert r["user_text"] == "third" and len(r["history"]) == 4
    assert [m["status"] for m in (await store.get_messages(a["session_id"]))[-1:]] == ["streaming"]
    assert await store.sweep_interrupted(a["session_id"]) == 1
    assert (await store.get_messages(a["session_id"]))[-1]["status"] == "interrupted"
    await store.delete_session(a["session_id"])
    assert (await ex.query("SELECT count(*) AS n FROM messages WHERE session_id = ?", [a["session_id"]])).rows[0]["n"] == 0


async def test_full_agent_turn_on_real_d1_real_gpu_real_tools(ex):
    settings = Settings(store="d1", session_source=SRC, heartbeat_s=5)
    services = build_services(settings)
    app = create_app(settings, services=services, gpu_daemon=False)
    srv = ServerThread(app).start()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(240, connect=20)) as c:
            events = []
            async with c.stream("POST", f"{srv.url}/api/chat/stream", json={"message": "Run `echo live-d1-check` on the remote computer and tell me what it printed."}) as r:
                assert r.status_code == 200, await r.aread()
                import json
                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        events.append(json.loads(line[5:]))
            end = [e for e in events if e["type"] == "run.end"][-1]
            text = "".join(e["text"] for e in events if e["type"] == "text.delta")
            sid = events[0]["session_id"]
            assert end["status"] == "stop", (end, [e for e in events if e["type"] == "error"])
            assert [e["name"] for e in events if e["type"] == "tool.start"] == ["run_command"]
            assert "live-d1-check" in text
            saved = (await c.get(f"{srv.url}/api/chat/sessions/{sid}")).json()
            assert saved["messages"][-1]["content"] == text.strip() and saved["messages"][-1]["status"] == "stop"
            assert saved["messages"][-1]["parts"][0]["type"] in ("tool", "text")
            assert (await c.delete(f"{srv.url}/api/chat/sessions/{sid}")).status_code == 200
    finally:
        srv.stop()
