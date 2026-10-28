"""PandaStack computer adapter for the existing agent /computer/* tool contract.

A sandbox ID alone is NOT a durable workspace. Named writes are persisted as
private GitHub release assets, and foreground shell/file mutations checkpoint
/workspace through WorkspaceMirror before their tool call is acknowledged.
System packages and unfinished background work are not covered.
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
    def __init__(self, http: httpx.AsyncClient, *, api_key, sandbox_id: str = "",
                 github_token: str, repo: str = "jadeniran21/Computer-drive", store=None,
                 mirror: bool = True, snapshot_path: str = "_system/blackthorn-workspace.tar.gz",
                 key_pool=None):
        if not api_key or not github_token or not (sandbox_id or store):
            raise ValueError("PandaStack, private drive and sandbox state are required")
        self.http = http
        # ``api_key`` may be a string, a callable returning the current key, or a
        # PandaKeyPool. Rotation then happens without rebuilding the client.
        self.key_pool = key_pool
        self._key_source = api_key
        self.key = api_key
        self.sandbox_id = sandbox_id
        self.github_token = github_token
        self.repo = repo
        self.store = store
        self.snapshot_path = _safe_path(snapshot_path)
        self.base = "https://api.pandastack.ai/v1/sandboxes/" + sandbox_id
        self._awake_checked_at = 0.0
        self._awake_lock = asyncio.Lock()
        self._tool_lock = asyncio.Lock()
        # Drive-hosted skills/tools restored onto every new VM (set by build_services).
        self.capabilities = None
        self.capability_status: Dict[str, Any] = {}
        if store is not None and mirror:
            from .workspace_mirror import WorkspaceMirror
            self.mirror = WorkspaceMirror(self)
        else:
            self.mirror = None

    @property
    def key(self) -> str:  # type: ignore[override]
        source = self._key_source
        if callable(source):
            return str(source())
        if hasattr(source, "active_key"):
            return str(source.active_key())
        return str(source)

    @key.setter
    def key(self, value) -> None:
        self._key_source = value

    def invalidate(self) -> None:
        """Drop cached session/key health after a failure that never reached the guest."""
        pool = self.key_pool if self.key_pool is not None else (
            self._key_source if hasattr(self._key_source, "invalidate") else None)
        if pool is not None:
            try:
                pool.invalidate()
            except Exception:
                pass
        self._awake_checked_at = 0.0

    async def _ensure_capabilities(self) -> None:
        """Restore skills/tools from the drive on this VM, once per VM.

        Runs on the first real task of every VM (including a replacement created after
        expiry or a handover) and is a fast no-op when the catalog version already
        matches, so it can never be forgotten and never costs a turn.
        """
        if self.capabilities is None:
            return
        marker = f"{self.sandbox_id}:{getattr(self.capabilities, '_version', '')}"
        if getattr(self, "_capabilities_for", "") == marker and self.capability_status.get("ok"):
            return
        try:
            self.capability_status = await self.capabilities.ensure(self, kind="computer")
            self._capabilities_for = marker
        except Exception as exc:  # noqa: BLE001 - a catalog outage must not block the task
            self.capability_status = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:200]}

    async def _rotate_after(self, status_code: int, message: str = "") -> bool:
        """Switch to another authorized key after an auth/quota rejection."""
        pool = self.key_pool if self.key_pool is not None else (
            self._key_source if hasattr(self._key_source, "note_failure") else None)
        if pool is None:
            return False
        try:
            rotated = await pool.note_failure(self.key, status_code, message)
        except Exception:
            return False
        return rotated is not None

    async def lifecycle_snapshot(self) -> dict:
        """Provider-reported session timing, used for honest status reporting."""
        from .lifecycle import SandboxLifecycle
        if self.store is None:
            return {"sandbox_id": self.sandbox_id, "status": "unknown"}
        return await SandboxLifecycle(self.store, self).snapshot()

    async def handover(self, *, reason: str = "manual") -> dict:
        from .lifecycle import SandboxLifecycle
        if self.store is None:
            return {"ok": False, "phase": "no_store"}
        result = await SandboxLifecycle(self.store, self).handover(reason=reason)
        if result.get("ok") and result.get("sandbox_id"):
            self.sandbox_id = result["sandbox_id"]
            self.base = "https://api.pandastack.ai/v1/sandboxes/" + self.sandbox_id
        return result

    async def ensure_ready(self) -> None:
        """Reconcile an expired VM on an actual agent task, then restore its drive."""
        if self.store is None:
            return
        from .panda_control import PandaControl, INTENT, SANDBOX
        if await self.store.get_setting(INTENT) != "on":
            raise ToolError("offline", "The model/computer power control is off")
        async with self._awake_lock:
            try:
                control = PandaControl(self.store, self.http, self.key)
                owned = await control._owned()
                current = await control.current(owned)
                if current is None and owned:
                    # Ambiguous set (e.g. a sandbox leaked by a crashed worker): release the
                    # extras so exactly one production computer remains, then adopt it.
                    await control.reconcile_extras(owned)
                    owned = await control._owned()
                    current = await control.current(owned)
                if current is None or str(current.get("status") or "").lower() == "hibernated":
                    await control.ensure_for_task()
                sid = await self.store.get_setting(SANDBOX)
                if not sid:
                    raise ToolError("offline", "Computer recovery did not confirm a sandbox")
                if sid != self.sandbox_id:
                    self._awake_checked_at = 0.0
                    self.sandbox_id = sid
                self.base = "https://api.pandastack.ai/v1/sandboxes/" + sid
                self._awake_checked_at = time.monotonic()
                await self._ensure_capabilities()
            except ToolError:
                raise
            except Exception as exc:
                raise ToolError("offline", "Computer task recovery failed; no command was run") from exc
        if self.mirror is not None:
            await self.mirror.restore(self.sandbox_id)

    async def keepalive(self) -> None:
        """Keep a sandbox alive ONLY while a real agent run has a durable lease."""
        if self.store is None:
            return
        from .panda_control import INTENT
        if await self.store.get_setting(INTENT) != "on":
            return
        await self.ensure_ready()
        await self._exec("true", timeout=15)

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
            if response.status_code in (401, 402, 403, 429) and await self._rotate_after(response.status_code):
                # A rotated key must be used for exactly one retry; never loop.
                response = await self.http.request(method, self.base + path,
                    headers={"Authorization": "Bearer " + self.key},
                    json=payload, params=params, content=content, timeout=timeout)
        except httpx.HTTPError as exc:
            raise ToolError("offline", "PandaStack computer could not be reached") from exc
        if response.status_code == 404 and path == "":
            raise ToolError("session_expired", "PandaStack sandbox no longer exists")
        if response.status_code in (401, 403):
            raise ToolError("auth", "PandaStack denied computer access")
        if response.status_code == 402:
            raise ToolError("quota", "PandaStack quota is exhausted for the active key")
        if response.status_code == 429:
            raise ToolError("rate_limited", "PandaStack is rate limiting the active key")
        if response.status_code in (502, 503, 504, 522, 524):
            # 524 is the edge giving up on a long request while the guest keeps working;
            # it is a provider/timeout condition, never a tool-logic rejection.
            raise ToolError("provider", f"PandaStack is temporarily unavailable (HTTP {response.status_code})")
        return response

    @staticmethod
    def _relative(value: str) -> str:
        return _safe_path(value.removeprefix("/workspace/") if value.startswith("/workspace/") else value)

    async def _exec(self, command: str, *, timeout: float = 60) -> dict:
        # The guest is bounded by ``timeout`` itself: an HTTP timeout does not kill a
        # process inside the VM, so a runaway command would otherwise keep burning CPU.
        seconds = min(3600, max(1, int(timeout)))
        bounded = f"timeout -k 5 {seconds} sh -c {shlex.quote(command)}"
        response = await self._request("POST", "/exec", timeout=seconds + 15,
            payload={"cmd": bounded, "timeout_seconds": seconds})
        if response.status_code != 200:
            raise ToolError("rejected", f"PandaStack execution returned HTTP {response.status_code}")
        result = response.json()
        output = str(result.get("stdout") or "") + str(result.get("stderr") or "")
        code = result.get("exit_code")
        if code == 124:
            raise ToolError("timeout", f"Command exceeded its {seconds}s budget inside the computer")
        if code == 137:
            raise ToolError("timeout", "Command was killed by the sandbox after exceeding its resource budget")
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
        # Mutating computer calls are serialized until their GitHub checkpoint
        # commits. Parallel readers remain possible after the first restore.
        if path in ("/computer/exec", "/computer/write_file"):
            async with self._tool_lock:
                await self.ensure_ready()
                return await self._call(path, payload or {}, method=method, timeout=timeout)
        await self.ensure_ready()
        return await self._call(path, payload or {}, method=method, timeout=timeout)

    async def _call(self, path: str, payload: dict, *, method: str, timeout: float) -> dict:
        if path == "/computer/exec":
            cwd = str(payload.get("cwd") or ".")
            if cwd == ".":
                cwd = "/workspace"
            elif not cwd.startswith("/workspace/"):
                cwd = "/workspace/" + self._relative(cwd)
            # Package/cache paths live inside the mirrored workspace. APT/system
            # packages outside /workspace cannot be made durable by GitHub assets.
            setup = ("export HOME=/workspace PIP_TARGET=/workspace/.packages "
                     "PIP_CACHE_DIR=/workspace/.pip-cache NPM_CONFIG_CACHE=/workspace/.npm-cache "
                     "PYTHONPATH=/workspace/.packages:${PYTHONPATH:-} "
                     "PATH=/workspace/.packages/bin:$PATH; ")
            command = "cd " + shlex.quote(cwd) + " && " + setup + str(payload.get("command") or "")
            result = await self._exec(command, timeout=float(payload.get("timeout_seconds") or timeout))
            if self.mirror is not None:
                await self.mirror.sync()  # even a non-zero command may have changed files
            return result
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
            if self.mirror is not None:
                await self.mirror.sync()
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
        if path == "/computer/fetch_url":
            url = str(payload.get("url") or "")
            if not url.startswith(("https://", "http://")):
                raise ToolError("invalid", "fetch_url requires an HTTP or HTTPS URL")
            script = ("import sys,urllib.request; "
                      "response=urllib.request.urlopen(sys.argv[1],timeout=25); "
                      "print(response.read(100001).decode('utf-8',errors='replace')[:100000])")
            result = await self._exec("python3 -c " + shlex.quote(script) + " " + shlex.quote(url), timeout=35)
            return {"ok": result["ok"], "text": result["output"][:100000]}
        if path == "/computer/info":
            result = await self._exec("python3 -V; uname -s; df -h /workspace | tail -1")
            return {"ok": result["ok"], "platform": "PandaStack Firecracker Linux",
                    "workspace": "/workspace (ephemeral; named writes backed by private GitHub releases)",
                    "gpu": "none", "details": result["output"][:400]}
        if path == "/computer/checkpoint":
            # Durability barrier: publish /workspace to the private drive and only
            # report success when the upload was actually confirmed.
            if self.mirror is None:
                raise ToolError("unsupported", "No workspace mirror is configured, so a checkpoint cannot be published")
            try:
                published = bool(await self.mirror.sync(force=True))
            except Exception as exc:
                raise ToolError("storage", "Checkpoint upload failed: " + (str(exc) or type(exc).__name__)) from exc
            digest = ""
            if self.store is not None:
                try:
                    digest = str(await self.store.get_setting("pandastack_workspace_drive_digest") or "")
                except Exception:
                    digest = ""
            if not published:
                raise ToolError("storage", "The workspace was unchanged or the drive did not publish a new version")
            return {"ok": True, "published": True, "sandbox_id": self.sandbox_id, "digest": digest}
        raise ToolError("unsupported", f"PandaStack adapter does not implement {path}")
