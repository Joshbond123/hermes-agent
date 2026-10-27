"""PandaStack computer adapter for the existing agent /computer/* tool contract.

Opt-in transport only. A sandbox ID alone is NOT a durable workspace; write_file
persists through the GitHub drive, while arbitrary shell side effects require a
separate checkpoint strategy and must not be advertised as durable.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import shlex
import tempfile
import time
from typing import Any

import httpx

from .computer_drive import GitHubComputerDrive, _safe_path
from .tools.base import ToolError


class PandaComputerClient:
    def __init__(self, http: httpx.AsyncClient, *, api_key: str, sandbox_id: str = "",
                 github_token: str, repo: str = "jadeniran21/Computer-drive", store=None):
        if not api_key or not github_token or not (sandbox_id or store):
            raise ValueError("PandaStack, private drive and sandbox state are required")
        self.http = http
        self.key = api_key
        self.sandbox_id = sandbox_id
        self.github_token = github_token
        self.repo = repo
        self.store = store
        self.base = "https://api.pandastack.ai/v1/sandboxes/" + sandbox_id
        self._awake_checked_at = 0.0
        self._awake_lock = asyncio.Lock()

    async def _request(self, method: str, path: str, *, timeout: float = 60,
                       payload: Any = None, params: dict | None = None,
                       content: bytes | None = None) -> httpx.Response:
        if self.store is not None:
            from .panda_control import INTENT, SANDBOX
            if await self.store.get_setting(INTENT) != "on":
                raise ToolError("offline", "PandaStack computer is off")
            sandbox_id = await self.store.get_setting(SANDBOX)
            if not sandbox_id:
                raise ToolError("offline", "PandaStack computer has no verified sandbox")
            if sandbox_id != self.sandbox_id:
                self._awake_checked_at = 0.0
                self.sandbox_id = sandbox_id
            self.base = "https://api.pandastack.ai/v1/sandboxes/" + sandbox_id
            # Persistent sandboxes hibernate after inactivity even while the owner's
            # intent is ON. Check once per short window and wake before any tool call.
            async with self._awake_lock:
                if time.monotonic() - self._awake_checked_at >= 15:
                    try:
                        listed = await self.http.get("https://api.pandastack.ai/v1/sandboxes",
                            headers={"Authorization": "Bearer " + self.key}, timeout=35)
                        if listed.status_code != 200 or not isinstance(listed.json(), list):
                            raise ToolError("offline", "PandaStack computer state could not be verified")
                        item = next((x for x in listed.json() if x.get("id") == sandbox_id), None)
                        if item is None:
                            raise ToolError("offline", "PandaStack computer is missing")
                        if str(item.get("status") or "").lower() == "hibernated":
                            woke = await self.http.post(self.base + "/wake",
                                headers={"Authorization": "Bearer " + self.key}, json={}, timeout=35)
                            if woke.status_code not in (200, 201, 204):
                                raise ToolError("offline", "PandaStack computer could not be woken")
                        self._awake_checked_at = time.monotonic()
                    except httpx.HTTPError as exc:
                        raise ToolError("offline", "PandaStack computer state could not be checked") from exc
        try:
            response = await self.http.request(method, self.base + path,
                headers={"Authorization": "Bearer " + self.key},
                json=payload, params=params, content=content, timeout=timeout)
        except httpx.HTTPError as exc:
            raise ToolError("offline", "PandaStack computer could not be reached") from exc
        if response.status_code == 404 and path == "":
            raise ToolError("offline", "PandaStack sandbox no longer exists")
        if response.status_code in (401, 403):
            raise ToolError("auth", "PandaStack denied computer access")
        return response

    @staticmethod
    def _relative(value: str) -> str:
        return _safe_path(value.removeprefix("/workspace/") if value.startswith("/workspace/") else value)

    async def _exec(self, command: str, *, timeout: float = 60) -> dict:
        response = await self._request("POST", "/exec", timeout=timeout + 15,
            payload={"cmd": command, "timeout_seconds": min(3600, max(1, int(timeout)))})
        if response.status_code != 200:
            raise ToolError("rejected", f"PandaStack execution returned HTTP {response.status_code}")
        result = response.json()
        output = str(result.get("stdout") or "") + str(result.get("stderr") or "")
        code = result.get("exit_code")
        return {"ok": code == 0, "exit_code": code, "output": output}

    def _save(self, content: bytes, logical: str):
        with tempfile.TemporaryDirectory() as tmp:
            source = pathlib.Path(tmp) / "data"
            source.write_bytes(content)
            with GitHubComputerDrive(self.repo, self.github_token) as drive:
                return drive.upload(source, logical)

    def _load(self, logical: str) -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "data"
            with GitHubComputerDrive(self.repo, self.github_token) as drive:
                drive.download(logical, target)
            return target.read_bytes()

    async def call(self, path: str, payload: dict | None = None, *, method: str = "POST",
                   timeout: float = 90, idempotent: bool = True) -> dict:
        payload = payload or {}
        if path == "/computer/exec":
            cwd = str(payload.get("cwd") or ".")
            if cwd == ".":
                cwd = "/workspace"
            elif not cwd.startswith("/workspace/"):
                cwd = "/workspace/" + self._relative(cwd)
            command = "cd " + shlex.quote(cwd) + " && " + str(payload.get("command") or "")
            return await self._exec(command, timeout=float(payload.get("timeout_seconds") or timeout))
        if path == "/computer/write_file":
            logical = self._relative(str(payload.get("path") or ""))
            body = str(payload.get("content") or "").encode("utf-8")
            # Persist before acknowledging: a failed GitHub write MUST NOT look durable.
            try:
                await asyncio.to_thread(self._save, body, logical)
            except Exception as exc:
                raise ToolError("storage", "Private computer-drive did not confirm the write") from exc
            location = "/workspace/" + logical
            await self._exec("mkdir -p " + shlex.quote(str(pathlib.PurePosixPath(location).parent)))
            response = await self._request("PUT", "/fs", params={"path": location}, content=body)
            if response.status_code not in (200, 201, 204):
                raise ToolError("storage", f"PandaStack file write returned HTTP {response.status_code}")
            return {"ok": True, "bytes": len(body), "path": logical, "persistent": True}
        if path == "/computer/read_file":
            logical = self._relative(str(payload.get("path") or ""))
            location = "/workspace/" + logical
            response = await self._request("GET", "/fs", params={"path": location})
            if response.status_code == 404:
                # Replacement VM: fetch only this file on demand; no credential enters guest.
                try:
                    body = await asyncio.to_thread(self._load, logical)
                except FileNotFoundError as exc:
                    raise ToolError("not_found", "The file is absent from the private drive") from exc
                except Exception as exc:
                    raise ToolError("storage", "Private computer-drive retrieval failed") from exc
                await self._exec("mkdir -p " + shlex.quote(str(pathlib.PurePosixPath(location).parent)))
                written = await self._request("PUT", "/fs", params={"path": location}, content=body)
                if written.status_code not in (200, 201, 204):
                    raise ToolError("storage", "Restored file could not be staged")
            elif response.status_code == 200:
                body = response.content
            else:
                raise ToolError("rejected", f"PandaStack read returned HTTP {response.status_code}")
            try:
                text = body.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ToolError("binary", "Use the binary download interface for non-text files") from exc
            return {"ok": True, "path": logical, "content": text}
        if path == "/computer/list_files":
            root = str(payload.get("path") or ".")
            directory = "/workspace" if root == "." else "/workspace/" + self._relative(root)
            result = await self._exec("ls -la " + shlex.quote(directory))
            return {"ok": result["ok"], "path": root, "listing": result["output"]}
        if path == "/computer/info":
            result = await self._exec("python3 -V; uname -s; df -h /workspace | tail -1")
            return {"ok": result["ok"], "platform": "PandaStack Firecracker Linux",
                    "workspace": "/workspace (ephemeral; named writes backed by private GitHub releases)",
                    "gpu": "none", "details": result["output"][:400]}
        raise ToolError("unsupported", f"PandaStack adapter does not implement {path}")
