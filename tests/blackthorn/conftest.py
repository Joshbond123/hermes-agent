"""Test doubles for the Blackthorn layer.

* ``FakeD1``  — a SQLite database with the *real* D1 table definitions, plugged in where the
  production code talks to Cloudflare.  SQLite is D1's engine, so the real SQL strings run.
* ``FakeUpstream`` — a real HTTP server (uvicorn on a free port) that speaks the
  OpenAI streaming protocol and the Kaggle ``/computer/*`` API, with scriptable behaviour.

These live only in tests; the product never uses them.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

import httpx
import pytest
import pytest_asyncio
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route as StarRoute

from blackthorn import d1 as d1mod
from blackthorn.d1 import D1Result

DDL = """
CREATE TABLE sessions (
  id TEXT PRIMARY KEY, source TEXT NOT NULL, created_source TEXT, user_id TEXT, session_key TEXT,
  chat_id TEXT, chat_type TEXT, thread_id TEXT, display_name TEXT, origin_json TEXT,
  expiry_finalized INTEGER DEFAULT 0, model TEXT, model_config TEXT, system_prompt TEXT,
  system_prompt_hash TEXT, parent_session_id TEXT, started_at REAL NOT NULL, ended_at REAL, end_reason TEXT,
  message_count INTEGER DEFAULT 0, tool_call_count INTEGER DEFAULT 0, title TEXT, title_source TEXT,
  last_activity_at REAL, archived INTEGER NOT NULL DEFAULT 0, auto_archived INTEGER NOT NULL DEFAULT 0,
  pinned INTEGER NOT NULL DEFAULT 0, hidden INTEGER NOT NULL DEFAULT 0, last_read_at REAL
);
CREATE TABLE messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT,
  tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, timestamp REAL NOT NULL, token_count INTEGER,
  finish_reason TEXT, reasoning TEXT, _compressed_summary INTEGER NOT NULL DEFAULT 0,
  active INTEGER NOT NULL DEFAULT 1, display_kind TEXT, display_metadata TEXT, message_uid TEXT
);
CREATE INDEX idx_messages_session ON messages(session_id);
CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE kaggle_gpu_state (
  id TEXT PRIMARY KEY, status TEXT NOT NULL, tunnel_url TEXT NOT NULL DEFAULT '', api_key TEXT NOT NULL DEFAULT '',
  model TEXT NOT NULL DEFAULT 'm', gpu_info TEXT NOT NULL DEFAULT '', updated_at REAL NOT NULL,
  detail TEXT NOT NULL DEFAULT ''
);
CREATE TABLE hermes_memories (
  id TEXT PRIMARY KEY, target TEXT, content TEXT, memory_type TEXT, importance REAL, user_id TEXT,
  session_id TEXT, profile TEXT, created_at REAL, updated_at REAL
);
"""


class FakeD1:
    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(DDL)
        self.lock = threading.Lock()
        self.calls = 0

    def _run(self, statements: Sequence[Any]) -> List[D1Result]:
        out: List[D1Result] = []
        with self.lock:
            self.calls += 1
            for sql, params in statements:
                try:
                    cur = self.conn.execute(sql, list(params or []))
                except sqlite3.Error as exc:  # the real client surfaces SQL errors as D1Error
                    self.conn.rollback()
                    raise d1mod.D1Error(f"Cloudflare D1 error: {exc}") from exc
                rows = [dict(r) for r in cur.fetchall()] if cur.description else []
                out.append(D1Result(rows, {"changes": max(cur.rowcount, 0), "last_row_id": cur.lastrowid}))
            self.conn.commit()
        return out

    async def abatch(self, statements, **kw):
        return self._run(statements)

    def batch(self, statements, **kw):
        return self._run(statements)

    # convenience for assertions
    def rows(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, list(params)).fetchall()]


@pytest.fixture()
def fake_d1(monkeypatch):
    fake = FakeD1()
    monkeypatch.setattr(d1mod.d1, "abatch", fake.abatch)
    monkeypatch.setattr(d1mod.d1, "batch", fake.batch)
    monkeypatch.setenv("BLACKTHORN_SESSION_SOURCE", "blackthorn-test")
    return fake


# --------------------------------------------------------------------------- #
# fake model server + Kaggle computer
# --------------------------------------------------------------------------- #
class FakeUpstream:
    """Scriptable OpenAI-compatible streaming server.

    ``scenarios`` is a queue; each ``/v1/chat/completions`` request pops one.  A scenario is a
    list of actions: ``{"content": s}``, ``{"reasoning": s}``, ``{"tool_call": {...}}``,
    ``{"finish": "stop"}``, ``{"raw": line}``, ``{"error": msg}``, ``{"sleep": s}``,
    ``{"hang": s}``, ``{"drop": True}`` or ``{"status": 530}`` (whole-response failure).
    """

    def __init__(self) -> None:
        self.scenarios: List[List[Dict[str, Any]]] = []
        self.requests: List[Dict[str, Any]] = []
        self.computer_calls: List[Dict[str, Any]] = []
        self.exec_output = "[ok]"
        self.exec_code = 0
        self.exec_delay = 0.0
        self.verify_reply = "OK"
        self.disconnected = threading.Event()
        self.completed = 0
        self.token_delay = 0.005
        self.app = Starlette(routes=[
            StarRoute("/v1/chat/completions", self._chat, methods=["POST"]),
            StarRoute("/computer/exec", self._exec, methods=["POST"]),
            StarRoute("/computer/write_file", self._write, methods=["POST"]),
            StarRoute("/computer/read_file", self._read, methods=["POST"]),
            StarRoute("/computer/list_files", self._list, methods=["POST"]),
            StarRoute("/computer/fetch_url", self._fetch, methods=["POST"]),
            StarRoute("/computer/info", self._info, methods=["GET"]),
            StarRoute("/health", self._health, methods=["GET"]),
        ])
        self.files: Dict[str, str] = {}
        self.url = ""
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle
    def start(self) -> None:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        config = uvicorn.Config(self.app, host="127.0.0.1", port=port, log_level="error", lifespan="off")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.time() + 10
        while not self._server.started and time.time() < deadline:
            time.sleep(0.02)
        assert self._server.started, "fake upstream did not start"
        self.url = f"http://127.0.0.1:{port}"

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)

    def script(self, *scenarios: List[Dict[str, Any]]) -> None:
        self.scenarios.extend(scenarios)

    # -- handlers
    async def _chat(self, request: Request):
        body = await request.json()
        self.requests.append(body)
        if body.get("stream") is False:
            return JSONResponse({"choices": [{"message": {"content": self.verify_reply}}],
                                 "usage": {"completion_tokens": 2}})
        scenario = self.scenarios.pop(0) if self.scenarios else [{"content": "ok"}, {"finish": "stop"}]
        status = next((a["status"] for a in scenario if "status" in a), None)
        if status:
            return JSONResponse({"error": "boom"}, status_code=status, headers={"content-type": "text/html"}
                                ) if status in (502, 530) else JSONResponse({"error": {"message": "boom"}}, status_code=status)

        async def gen():
            finished = False
            try:
                for action in scenario:
                    if "sleep" in action:
                        await asyncio.sleep(action["sleep"])
                    elif "hang" in action:
                        await asyncio.sleep(action["hang"])
                    elif "drop" in action:
                        raise ConnectionResetError("dropped")
                    elif "raw" in action:
                        yield (action["raw"] + "\n\n").encode()
                    elif "error" in action:
                        yield f'data: {json.dumps({"error": {"message": action["error"]}})}\n\n'.encode()
                    elif "content" in action:
                        for ch in self._pieces(action["content"]):
                            yield self._sse({"delta": {"content": ch}})
                            await asyncio.sleep(self.token_delay)
                    elif "reasoning" in action:
                        for ch in self._pieces(action["reasoning"]):
                            yield self._sse({"delta": {"reasoning": ch}})
                            await asyncio.sleep(self.token_delay)
                    elif "tool_call" in action:
                        tc = action["tool_call"]
                        yield self._sse({"delta": {"tool_calls": [{
                            "index": tc.get("index", 0), "id": tc.get("id"), "type": "function",
                            "function": {"name": tc.get("name"), "arguments": tc.get("arguments", "")}}]}})
                    elif "finish" in action:
                        finished = True
                        yield self._sse({"delta": {}}, finish=action["finish"])
                        yield f'data: {json.dumps({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7}})}\n\n'.encode()
                yield b"data: [DONE]\n\n"
                self.completed += 1
            except asyncio.CancelledError:
                self.disconnected.set()
                raise

        return StreamingResponse(gen(), media_type="text/event-stream")

    @staticmethod
    def _pieces(text: str, size: int = 4) -> List[str]:
        return [text[i:i + size] for i in range(0, len(text), size)] or [""]

    @staticmethod
    def _sse(choice: Dict[str, Any], finish: Optional[str] = None) -> bytes:
        payload = {"choices": [{"index": 0, **choice, "finish_reason": finish}]}
        return f"data: {json.dumps(payload)}\n\n".encode()

    async def _exec(self, request: Request):
        body = await request.json()
        self.computer_calls.append({"endpoint": "/computer/exec", **body})
        if self.exec_delay:
            await asyncio.sleep(self.exec_delay)
        return JSONResponse({"ok": True, "exit_code": self.exec_code, "output": self.exec_output, "cwd": "/w"})

    async def _write(self, request: Request):
        body = await request.json()
        self.computer_calls.append({"endpoint": "/computer/write_file", **body})
        self.files[body["path"]] = body["content"]
        return JSONResponse({"ok": True, "path": "/w/" + body["path"], "bytes": len(body["content"])})

    async def _read(self, request: Request):
        body = await request.json()
        self.computer_calls.append({"endpoint": "/computer/read_file", **body})
        if body["path"] not in self.files:
            return JSONResponse({"ok": False, "error": body["path"] + " is not a file"})
        return JSONResponse({"ok": True, "path": "/w/" + body["path"], "content": self.files[body["path"]]})

    async def _list(self, request: Request):
        body = await request.json()
        self.computer_calls.append({"endpoint": "/computer/list_files", **body})
        return JSONResponse({"ok": True, "path": "/w", "listing": "\n".join("FILE " + n for n in self.files) or "(empty)"})

    async def _fetch(self, request: Request):
        body = await request.json()
        self.computer_calls.append({"endpoint": "/computer/fetch_url", **body})
        return JSONResponse({"ok": True, "url": body["url"], "text": "page text"})

    async def _info(self, request: Request):
        return JSONResponse({"hostname": "kaggle", "gpu": "Tesla T4 x2", "disk_free_gb": 100})

    async def _health(self, request: Request):
        return JSONResponse({"status": "online", "model_loaded": True})


@pytest.fixture()
def upstream():
    server = FakeUpstream()
    server.start()
    yield server
    server.stop()


@pytest_asyncio.fixture()
async def http_client():
    client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=30.0))
    yield client
    await client.aclose()
