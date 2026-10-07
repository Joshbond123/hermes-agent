"""The HTTP API, exercised over a real socket."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from typing import Any, Dict, List

import httpx
import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI

from blackthorn import api as bt_api
from blackthorn import http as bt_http
from blackthorn.agent.builtin_tools import Deps, SearchKeys, build_registry
from blackthorn.agent.engine import Engine, EngineConfig
from blackthorn.agent.kaggle_computer import KaggleComputer
from blackthorn.route import Route
from blackthorn.store import D1Store

pytestmark = pytest.mark.asyncio


class Server:
    def __init__(self, app):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error", lifespan="off"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started and time.time() < deadline:
            time.sleep(0.02)
        self.url = f"http://127.0.0.1:{self.port}"

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=5)


@pytest.fixture()
def server(fake_d1, upstream):
    async def provider():
        return Route(url=upstream.url, api_key="k", model="qwen-test")

    computer = KaggleComputer(provider)
    registry = build_registry(Deps(computer=computer, search_keys=SearchKeys(loader=_none), http_client=bt_http.client))
    engine = Engine(route_provider=provider, registry=registry, store=D1Store(), client_factory=bt_http.client,
                    config=EngineConfig(checkpoint_interval_s=0.05))
    bt_api.set_engine(engine)
    app = FastAPI()
    app.include_router(bt_api.router)
    srv = Server(app)
    yield srv
    srv.stop()
    bt_api.set_engine(None)


async def _none():
    return []


@pytest_asyncio.fixture()
async def client(server):
    async with httpx.AsyncClient(base_url=server.url, timeout=30.0) as c:
        yield c


def parse_sse(text: str) -> List[Dict[str, Any]]:
    events = []
    for block in text.split("\n\n"):
        ev: Dict[str, Any] = {}
        for line in block.split("\n"):
            if line.startswith("id: "):
                ev["id"] = int(line[4:])
            elif line.startswith("event: "):
                ev["event"] = line[7:]
            elif line.startswith("data: "):
                ev["data"] = json.loads(line[6:])
        if "event" in ev:
            events.append(ev)
    return events


async def stream_events(client, body):
    async with client.stream("POST", "/api/studio/agent/stream", json=body) as resp:
        assert resp.status_code == 200, await resp.aread()
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["x-accel-buffering"] == "no" and "no-transform" in resp.headers["cache-control"]
        text = (await resp.aread()).decode()
    return parse_sse(text)


def say(text):
    return [{"content": text}, {"finish": "stop"}]


# ---- chat over HTTP -----------------------------------------------------------------------
async def test_stream_endpoint_delivers_incrementally_with_ids_and_persists(server, client, upstream, fake_d1):
    upstream.token_delay = 0.05
    upstream.script(say("streaming works over a real socket, token by token"))
    arrivals = []
    async with client.stream("POST", "/api/studio/agent/stream", json={"message": "hi"}) as resp:
        t0 = time.monotonic()
        async for chunk in resp.aiter_raw():
            if b"event: delta" in chunk:
                arrivals.append(time.monotonic() - t0)
    assert len(arrivals) >= 6 and arrivals[-1] - arrivals[0] > 0.3, "deltas must be spread over time, not one burst"
    sess = fake_d1.rows("SELECT id, title FROM sessions")[0]
    msgs = fake_d1.rows("SELECT role, content, finish_reason FROM messages WHERE session_id = ? ORDER BY id", [sess["id"]])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "streaming works over a real socket, token by token" and msgs[1]["finish_reason"] == "stop"


async def test_event_sequence_and_ready_event_carry_ids(server, client, upstream):
    upstream.script(say("Hello"))
    events = await stream_events(client, {"message": "hi"})
    assert [e["id"] for e in events] == list(range(1, len(events) + 1))
    ready = next(e for e in events if e["event"] == "turn" and e["data"].get("state") == "ready")["data"]
    assert ready["session_id"].startswith("studio-") and ready["assistant_message_id"] > ready["user_message_id"]
    assert events[-1]["event"] == "done" and events[-1]["data"]["finish_reason"] == "stop"


async def test_follow_up_message_in_same_session_sees_history(server, client, upstream):
    upstream.script(say("Paris."), say("About 2 million."))
    first = await stream_events(client, {"message": "Capital of France?"})
    sid = first[0]["data"]["session_id"]
    await stream_events(client, {"message": "And its population?", "session_id": sid})
    contents = [m["content"] for m in upstream.requests[1]["messages"]]
    assert "Capital of France?" in contents and "Paris." in contents and contents[-1] == "And its population?"


async def test_validation_errors(server, client):
    assert (await client.post("/api/studio/agent/stream", json={"message": "   "})).status_code == 400
    assert (await client.post("/api/studio/agent/stream", json={"message": "x" * 40000})).status_code == 422
    assert (await client.post("/api/studio/agent/stream", json={"message": "hi", "session_id": "../etc"})).status_code == 400
    assert (await client.post("/api/studio/agent/stream", json={"regenerate": True})).status_code == 400
    assert (await client.post("/api/studio/agent/stream", json={"message": "hi", "temperature": 9})).status_code == 422


async def test_reattach_replays_after_a_given_sequence(server, client, upstream):
    upstream.script(say("one two three four five six seven"))
    events = await stream_events(client, {"message": "count"})
    turn_id = events[0]["data"]["turn_id"]
    cut = events[4]["id"]
    async with client.stream("GET", f"/api/studio/turns/{turn_id}/stream", params={"after": cut}) as resp:
        replay = parse_sse((await resp.aread()).decode())
    assert [e["id"] for e in replay] == [e["id"] for e in events if e["id"] > cut]
    async with client.stream("GET", f"/api/studio/turns/{turn_id}/stream", headers={"Last-Event-ID": str(cut)}) as resp:
        again = parse_sse((await resp.aread()).decode())
    assert [e["id"] for e in again] == [e["id"] for e in replay]
    assert (await client.get("/api/studio/turns/turn-nope/stream")).status_code == 404


async def test_cancel_endpoint_stops_the_turn_and_upstream(server, client, upstream, fake_d1):
    upstream.token_delay = 0.05
    upstream.script([{"content": "word " * 500}, {"finish": "stop"}])
    buf, cancelled = "", False
    async with client.stream("POST", "/api/studio/agent/stream", json={"message": "long"}) as resp:
        async for chunk in resp.aiter_text():
            buf += chunk
            if not cancelled and buf.count("event: delta") >= 4:
                turn_id = parse_sse(buf)[0]["data"]["turn_id"]
                r = await client.post(f"/api/studio/turns/{turn_id}/cancel")
                assert r.json() == {"ok": True, "cancelled": True}
                cancelled = True
    seen = parse_sse(buf)
    assert cancelled and seen[-1]["event"] == "done" and seen[-1]["data"]["finish_reason"] == "cancelled"
    assert await asyncio.to_thread(upstream.disconnected.wait, 3.0)
    row = fake_d1.rows("SELECT finish_reason, content FROM messages WHERE role='assistant'")[0]
    assert row["finish_reason"] == "cancelled" and 0 < len(row["content"]) < 2500


async def test_active_turn_is_reported_for_a_running_chat_and_disconnect_does_not_kill_it(server, client, upstream, fake_d1):
    upstream.token_delay = 0.04
    upstream.script(say("a long answer " * 20))
    sid = None
    async with client.stream("POST", "/api/studio/agent/stream", json={"message": "go"}) as resp:
        buf = ""
        async for chunk in resp.aiter_text():
            buf += chunk
            if "event: delta" in buf:
                sid = next(e for e in parse_sse(buf) if e["data"].get("session_id"))["data"]["session_id"]
                break
    # the browser is gone; the turn must still be running and visible
    detail = (await client.get(f"/api/studio/sessions/{sid}")).json()
    assert detail["active_turn"] and detail["active_turn"]["turn_id"].startswith("turn-")
    tid = detail["active_turn"]["turn_id"]
    async with client.stream("GET", f"/api/studio/turns/{tid}/stream", params={"after": 0}) as resp:
        replay = parse_sse((await resp.aread()).decode())
    text = "".join(e["data"]["text"] for e in replay if e["event"] == "delta")
    assert text == "a long answer " * 20 and replay[-1]["event"] == "done"
    msgs = (await client.get(f"/api/studio/sessions/{sid}/messages")).json()["messages"]
    assert msgs[-1]["content"] == ("a long answer " * 20).strip() and msgs[-1]["finish_reason"] == "stop"


async def test_regenerate_replaces_the_last_answer(server, client, upstream):
    upstream.script(say("first answer"), say("second answer"))
    first = await stream_events(client, {"message": "question"})
    sid = first[0]["data"]["session_id"]
    await stream_events(client, {"session_id": sid, "regenerate": True})
    msgs = (await client.get(f"/api/studio/sessions/{sid}/messages")).json()["messages"]
    assert [(m["role"], m["content"]) for m in msgs] == [("user", "question"), ("assistant", "second answer")]


# ---- history endpoints ------------------------------------------------------------------------
async def make_chat(client, upstream, text="hello"):
    upstream.script(say("ok"))
    events = await stream_events(client, {"message": text})
    return events[0]["data"]["session_id"]


async def test_rename_pin_archive_delete_persist_through_the_api(server, client, upstream, fake_d1):
    sid = await make_chat(client, upstream, "plan my trip")
    other = await make_chat(client, upstream, "other chat")
    r = await client.patch(f"/api/studio/sessions/{sid}", json={"title": "Trip plan"})
    assert r.status_code == 200 and r.json()["session"]["title"] == "Trip plan"
    r = await client.patch(f"/api/studio/sessions/{sid}", json={"pinned": True})
    assert r.json()["session"]["pinned"] is True
    listed = (await client.get("/api/studio/sessions")).json()["sessions"]
    assert [s["id"] for s in listed][0] == sid and listed[0]["title"] == "Trip plan" and listed[0]["pinned"]
    # a "page reload" = a fresh GET that reads D1 again
    again = (await client.get(f"/api/studio/sessions/{sid}")).json()["session"]
    assert again["title"] == "Trip plan" and again["pinned"] is True
    r = await client.patch(f"/api/studio/sessions/{sid}", json={"archived": True})
    assert r.json()["session"]["archived"] is True and r.json()["session"]["pinned"] is False
    assert sid not in [s["id"] for s in (await client.get("/api/studio/sessions")).json()["sessions"]]
    arch = (await client.get("/api/studio/sessions", params={"archived": True})).json()["sessions"]
    assert [s["id"] for s in arch] == [sid]
    assert (await client.get(f"/api/studio/sessions/{sid}/messages")).json()["messages"][0]["content"] == "plan my trip"
    r = await client.patch(f"/api/studio/sessions/{sid}", json={"archived": False})
    assert r.json()["session"]["archived"] is False
    assert (await client.delete(f"/api/studio/sessions/{other}")).json() == {"ok": True, "deleted": other}
    assert (await client.get(f"/api/studio/sessions/{other}")).status_code == 404
    assert fake_d1.rows("SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", [other])[0]["n"] == 0
    assert (await client.delete(f"/api/studio/sessions/{other}")).status_code == 404


async def test_patch_validation(server, client, upstream):
    sid = await make_chat(client, upstream)
    assert (await client.patch(f"/api/studio/sessions/{sid}", json={})).status_code == 400
    assert (await client.patch(f"/api/studio/sessions/{sid}", json={"title": "  "})).status_code == 400
    assert (await client.patch("/api/studio/sessions/studio-missing", json={"title": "x"})).status_code == 404
    assert (await client.patch("/api/studio/sessions/bad%20id!", json={"title": "x"})).status_code in (400, 404)
    assert (await client.get("/api/studio/sessions/studio-missing/messages")).status_code == 404


async def test_search_param(server, client, upstream):
    a = await make_chat(client, upstream, "docker networking help")
    await make_chat(client, upstream, "banana bread")
    found = (await client.get("/api/studio/sessions", params={"q": "docker"})).json()["sessions"]
    assert [s["id"] for s in found] == [a]


async def test_messages_include_tool_activity_for_reload(server, client, upstream):
    upstream.exec_output = "hello from the box"
    upstream.script(
        [{"content": "Running it. "}, {"tool_call": {"index": 0, "id": "c1", "name": "terminal",
                                                      "arguments": json.dumps({"command": "echo hi"})}},
         {"finish": "tool_calls"}],
        say("It printed hello."),
    )
    events = await stream_events(client, {"message": "run echo"})
    sid = events[0]["data"]["session_id"]
    msgs = (await client.get(f"/api/studio/sessions/{sid}/messages")).json()["messages"]
    parts = msgs[-1]["parts"]
    assert [p["t"] for p in parts] == ["text", "tool", "text"]
    assert parts[1]["name"] == "terminal" and parts[1]["status"] == "ok" and "hello from the box" in parts[1]["output"]


# ---- settings / gpu / version ------------------------------------------------------------------
async def test_system_prompt_round_trip_is_used_by_the_model(server, client, upstream):
    assert (await client.get("/api/system-prompt")).json()["prompt"] == ""
    assert (await client.put("/api/system-prompt", json={"prompt": "  Speak like a pirate.  "})).json()["ok"]
    assert (await client.get("/api/system-prompt")).json()["prompt"] == "Speak like a pirate."
    upstream.script(say("Arr"))
    await stream_events(client, {"message": "hi"})
    assert "Speak like a pirate." in upstream.requests[-1]["messages"][0]["content"]
    assert (await client.put("/api/system-prompt", json={"prompt": "x" * 9000})).status_code == 422
    await client.put("/api/system-prompt", json={"prompt": ""})
    assert (await client.get("/api/system-prompt")).json()["prompt"] == ""


async def test_version_endpoint_reports_commit_from_environment(server, client, monkeypatch):
    monkeypatch.setenv("RENDER_GIT_COMMIT", "abc123def")
    data = (await client.get("/api/blackthorn/version")).json()
    assert data["commit"] == "abc123def" and "uptime_s" in data


async def test_gpu_status_is_instant_and_validates_auto_off(server, client, fake_d1, monkeypatch):
    t0 = time.monotonic()
    r = await client.get("/api/kaggle-gpu/status")
    assert r.status_code == 200 and "state" in r.json()
    r = await client.post("/api/kaggle-gpu/auto-off", json={"minutes": 7})
    assert r.status_code == 400
