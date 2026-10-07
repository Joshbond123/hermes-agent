"""Test doubles: a scripted OpenAI-compatible model server + a fake Kaggle computer, served by real uvicorn threads.

These are *test doubles for the test-suite only* — the product never uses them.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerThread:
    def __init__(self, app: FastAPI):
        self.port = free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error", lifespan="on"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "ServerThread":
        self.thread.start()
        for _ in range(200):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("test server did not start")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


# ---- event helpers for scripts ---------------------------------------------------------------------------
def say(text: str, pieces: int = 4) -> List[Dict[str, Any]]:
    n = max(1, min(pieces, len(text)))
    size = max(1, len(text) // n)
    chunks = [text[i:i + size] for i in range(0, len(text), size)]
    return [{"content": c} for c in chunks]


def think(text: str) -> Dict[str, Any]:
    return {"reasoning": text}


def call(name: str, args: Any, cid: Optional[str] = None, index: int = 0) -> Dict[str, Any]:
    return {"tool": (name, args if isinstance(args, str) else json.dumps(args), cid or f"call_{name}_{index}", index)}


def finish(reason: str = "stop") -> Dict[str, Any]:
    return {"finish": reason}


def pause(seconds: float) -> Dict[str, Any]:
    return {"sleep": seconds}


def usage(prompt: int = 100, completion: int = 20) -> Dict[str, Any]:
    return {"usage": (prompt, completion)}


class FakeBackend:
    """One process pretending to be the Kaggle gateway: model API + /computer/*."""

    def __init__(self, key: str = "test-key") -> None:
        self.key = key
        self.script: List[Any] = []
        self.requests: List[Dict[str, Any]] = []
        self.cancelled = 0
        self.completed = 0
        self.computer_calls: List[Dict[str, Any]] = []
        self.exec_output = "hello from the remote computer\n"
        self.exec_code = 0
        self.exec_delay = 0.0
        self.fetch_text = "Example Domain. This domain is for use in illustrative examples."
        self.files: Dict[str, str] = {}
        self.app = self._build()

    def queue(self, *turns: Any) -> None:
        self.script.extend(turns)

    def _build(self) -> FastAPI:
        app = FastAPI()

        def authed(request: Request) -> bool:
            return request.headers.get("authorization") == f"Bearer {self.key}"

        @app.get("/health")
        async def health():
            return {"status": "online", "model": "fake-model", "model_loaded": True}

        @app.post("/v1/chat/completions")
        async def chat(request: Request):
            if not authed(request):
                return JSONResponse({"detail": "Invalid or missing API Key."}, status_code=401)
            body = await request.json()
            self.requests.append(body)
            turn = self.script.pop(0) if self.script else [*say("(no script)"), finish("stop")]
            if isinstance(turn, dict) and "http_error" in turn:
                return JSONResponse({"error": {"message": turn["message"]}}, status_code=turn["http_error"])

            async def stream():
                try:
                    for ev in turn:
                        if "sleep" in ev:
                            await asyncio.sleep(ev["sleep"])
                            continue
                        if "drop" in ev:
                            raise RuntimeError("simulated connection drop")
                        delta: Dict[str, Any] = {}
                        choice: Dict[str, Any] = {"index": 0, "delta": delta, "finish_reason": None}
                        payload: Dict[str, Any] = {"id": "x", "object": "chat.completion.chunk", "choices": [choice]}
                        if "content" in ev:
                            delta["content"] = ev["content"]
                        elif "reasoning" in ev:
                            delta["reasoning"] = ev["reasoning"]
                        elif "tool" in ev:
                            name, args, cid, idx = ev["tool"]
                            delta["tool_calls"] = [{"index": idx, "id": cid, "type": "function", "function": {"name": name, "arguments": args}}]
                        elif "finish" in ev:
                            choice["finish_reason"] = ev["finish"]
                        elif "usage" in ev:
                            payload["choices"] = []
                            payload["usage"] = {"prompt_tokens": ev["usage"][0], "completion_tokens": ev["usage"][1], "total_tokens": sum(ev["usage"])}
                        yield f"data: {json.dumps(payload)}\n\n".encode()
                        await asyncio.sleep(0)
                    yield b"data: [DONE]\n\n"
                    self.completed += 1
                except asyncio.CancelledError:
                    self.cancelled += 1
                    raise

            return StreamingResponse(stream(), media_type="text/event-stream")

        async def computer(request: Request, name: str):
            if not authed(request):
                return JSONResponse({"detail": "Invalid or missing API Key."}, status_code=401)
            body = await request.json() if request.method == "POST" else {}
            self.computer_calls.append({"name": name, "body": body})
            if name == "exec":
                if self.exec_delay:
                    await asyncio.sleep(self.exec_delay)
                return {"ok": True, "exit_code": self.exec_code, "output": self.exec_output, "cwd": "/w"}
            if name == "list_files":
                return {"ok": True, "path": "/w", "listing": "FILE a.txt  3 B\nDIR sub"}
            if name == "read_file":
                path = body.get("path", "")
                if path not in self.files:
                    return {"ok": False, "error": f"{path} is not a file"}
                return {"ok": True, "path": f"/w/{path}", "content": self.files[path]}
            if name == "write_file":
                self.files[body["path"]] = body["content"]
                return {"ok": True, "path": f"/w/{body['path']}", "bytes": len(body["content"].encode())}
            if name == "fetch_url":
                return {"ok": True, "url": body.get("url"), "text": self.fetch_text}
            return JSONResponse({"detail": "unknown"}, status_code=404)

        def make(n: str):
            async def handler(request: Request):
                return await computer(request, n)
            return handler

        for name in ("exec", "list_files", "read_file", "write_file", "fetch_url"):
            app.add_api_route(f"/computer/{name}", make(name), methods=["POST"])

        @app.get("/computer/info")
        async def info(request: Request):
            if not authed(request):
                return JSONResponse({"detail": "no"}, status_code=401)
            return {"workspace": "/w", "hostname": "fake", "gpu": "Tesla T4\nTesla T4", "disk_free_gb": 3.2, "disk_total_gb": 20.9,
                    "cuda_available": True, "platform": "Linux", "python": "3.13"}

        return app
