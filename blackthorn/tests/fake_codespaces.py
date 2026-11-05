"""Test doubles for the Codespaces computer: the API, the channel and the manager.

They are deliberately thin: they implement only what ``api.py`` and the power loop actually call, so a
test that passes here is testing the real route and the real manager, not a second implementation.
"""

from __future__ import annotations

import time
import base64
from typing import Any, Dict, List, Optional

import httpx


class FakeCodespaceAPI:
    """The slice of the GitHub Codespaces API the app uses."""

    def __init__(self, name: str = "blackthorn-test-abc123", state: str = "Available",
                 repo: str = "jadeniran21/Computer-drive", owner: str = "jadeniran21"):
        self.repo, self.owner = repo, owner
        self.name, self.state = name, state
        self.created = 1
        self.calls: List[str] = []
        self.machine = "basicLinux32gb"

    def _row(self) -> Dict[str, Any]:
        return {"name": self.name, "state": self.state, "repository": {"full_name": self.repo},
                "owner": {"login": self.owner}, "machine": self.machine,
                "last_used_at": "2026-11-05T00:00:00Z", "created_at": "2026-11-01T00:00:00Z"}

    async def list_all(self):
        return [self._row()] if self.name else []

    async def find(self):
        self.calls.append("find")
        return self._row() if self.name else None

    async def duplicates(self):
        self.calls.append("duplicates")
        return []

    async def get(self, name: str):
        self.calls.append("get")
        return self._row()

    async def create(self):
        self.calls.append("create")
        self.created += 1
        self.name, self.state = f"blackthorn-test-{self.created:06d}", "Provisioning"
        return self._row()

    async def start(self, name: str):
        self.calls.append("start")
        self.state = "Available"
        return {}

    async def stop(self, name: str):
        self.calls.append("stop")
        self.state = "Shutdown"
        return {}

    async def delete(self, name: str):
        self.calls.append("delete")

    async def publish_port(self, name: str, port: int, visibility: str = "public"):
        self.calls.append("publish_port")
        return {}


class FakeChannel:
    """In-memory stand-in for ``CodespaceChannel``."""

    def __init__(self, name: str = "", token: str = "session-token", present: bool = True):
        self.name = name
        self.tokens: Dict[str, str] = {name: token} if name else {}
        self._present = present
        self.seen: List[Dict[str, Any]] = []
        self.queue: List[Dict[str, Any]] = []
        self.results: Dict[str, Dict[str, Any]] = {}
        self.connected_at = time.time()

    # -- presence ---------------------------------------------------------
    def touch(self, name: str, *, version: str = "", port: int = 0) -> None:
        self._present = True
        self.connected_at = time.time()

    def present(self, name: str, *, fresh_s: float = 90.0) -> bool:
        return bool(name) and self._present

    def known(self, name: str) -> bool:
        return bool(name) and name in self.tokens

    def presence(self, name: str) -> Dict[str, Any]:
        return {"connected": bool(name) and self._present, "last_seen_s": 1 if self._present else -1,
                "registered": self.known(name), "version": "test"}

    def summary(self) -> Dict[str, Any]:
        return {"queued": len(self.queue), "inflight": 0, "agents": [], "registered_at": self.connected_at,
                "sessions": len(self.tokens)}

    # -- registration -----------------------------------------------------
    async def token_for(self, name: str) -> str:
        return self.tokens.get(name, "")

    async def accept_registration(self, name: str, token: str) -> None:
        self.tokens[name] = token

    async def forget(self, name: str) -> None:
        self.tokens.pop(name, None)

    # -- jobs -------------------------------------------------------------
    async def call(self, path: str, body: Dict[str, Any], *, timeout: float = 120.0, name: str = ""):
        self.seen.append({"path": path, "body": body})
        if not hasattr(self, "handler"):
            return {"ok": True, "output": "", "exit_code": 0}
        result = self.handler(path, body)
        if hasattr(result, "__await__"):
            result = await result
        return result

    async def pull(self, name: str, *, version: str = "", port: int = 0, wait: Optional[float] = None):
        self.touch(name)
        return {"ok": True, "job": self.queue.pop(0) if self.queue else None}

    def submit_result(self, job_id: str, payload: Dict[str, Any]) -> bool:
        self.results[job_id] = payload
        return True


class BackendBridge:
    """Answer the agent's routes by calling the fake computer the older tests already script.

    The point is that the *real* manager does the translation (logical ``/computer/*`` →
    agent ``/exec`` etc.) and the fake only supplies the transport, exactly as the real agent does.
    """

    AGENT_TO_LOGICAL = {"/exec": "exec", "/write": "write_file", "/read": "read_file",
                        "/list": "list_files", "/fetch": "fetch_url"}

    def __init__(self, base_url: str, key: str = "test-key"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30, headers={"Authorization": f"Bearer {key}"})

    async def __call__(self, route: str, body: Dict[str, Any]) -> Dict[str, Any]:
        if route == "/info":
            resp = await self.client.get(f"{self.base_url}/computer/info")
            return resp.json()
        if route == "/checkpoint":
            return {"ok": True, "published": True}
        if route == "/read_sample":
            path = str(body.get("path") or "")
            resp = await self.client.post(f"{self.base_url}/computer/read_file", json={"path": path})
            data = resp.json()
            if not data.get("ok"):
                return data
            return {"ok": True, "output": base64.b64encode(str(data.get("content") or "").encode()).decode()}
        logical = self.AGENT_TO_LOGICAL.get(route)
        if logical is None:
            return {"ok": False, "error": f"unknown operation {route}"}
        payload: Dict[str, Any] = dict(body)
        if route == "/exec":
            payload = {"command": body.get("cmd") or body.get("command") or "",
                       "cwd": body.get("cwd") or ".", "timeout_seconds": body.get("timeout") or 60}
        resp = await self.client.post(f"{self.base_url}/computer/{logical}", json=payload)
        return resp.json()


def attach_fake_codespace(services, backend_url: str = "", *, name: str = "blackthorn-test-abc123",
                         state: str = "ON"):
    """Wire a fake codespace onto a Services instance (real manager, fake transport)."""
    from blackthorn.codespace_computer import CodespaceComputer
    api = FakeCodespaceAPI(name=name)
    channel = FakeChannel(name=name, present=state == "ON")
    if backend_url:
        channel.handler = BackendBridge(backend_url)
    computer = CodespaceComputer(None, api, channel=channel, store=services.store, idle_stop_s=300)
    computer._name, computer._state = name, state
    computer._session = {"token": "session-token", "port": 8787}
    services.codespaces = computer
    services.channel = channel
    services.computer = computer
    return computer
