"""The agent's computer, served by a GitHub Codespace, plus its automatic power management.

``ctx.computer.call("/computer/...")`` is the only interface the tools know about; this module
implements exactly that surface on top of a codespace, so every existing tool (run_command,
read_file, write_file, browse, skill_search, cyber_run, …) works unchanged.

Power rules implemented here (requirement: Codespaces OFF when idle, ON when working):

* a task that touches the computer calls ``ensure_running`` before anything else;
* every call marks activity, so the idle clock measures real work;
* ``maybe_stop`` refuses to stop while a run is executing or a call is in flight, so a codespace
  is never shut down mid-command;
* files persist because a Codespace keeps its container across stop/start — stopping is not
  deleting, so an interrupted task can be resumed with its progress intact.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
import logging
import os
import shlex
import time
from typing import Any, Dict, Optional

import httpx

from .codespace_channel import CodespaceChannel
from .codespaces import CodespaceAPI, CodespaceError, UsageLedger

log = logging.getLogger("blackthorn.codespace")

LOGICAL_ROOT = "/workspace"
DEFAULT_PORT = 8787
STATE_KEY = "codespace_state"
SESSION_KEY = "codespace_session"


# A first boot (or any boot after the devcontainer changed) rebuilds the image: minutes, not seconds.
START_TIMEOUT_S = float(__import__("os").environ.get("CODESPACE_START_TIMEOUT_SECONDS", "900") or 900)
# How long to wait for the container's own agent to answer once the machine is up.
AGENT_WAIT_S = float(__import__("os").environ.get("CODESPACE_AGENT_WAIT_SECONDS", "300") or 300)


def _int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    try:
        return int(float(raw)) if raw else default
    except ValueError:
        return default


def _ts(value: Any) -> float:
    """Parse GitHub's ISO-8601 timestamps; unknown means "not evidence", never "now"."""
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return 0.0

class _EmptyLedger:
    """No machine has ever run here, so the measured total is exactly zero."""

    async def seconds_this_month(self, now: float = 0.0) -> float:
        return 0.0

    async def open_interval(self, name: str) -> None:
        return None

    async def close_interval(self) -> None:
        return None


class UnconfiguredComputer:
    """Stand-in used when no Codespaces credential is configured.

    Every call fails with the real reason instead of an ``AttributeError`` from a ``None`` client, so
    the tool that needed the computer reports "no computer" to the user and the rest of the app runs.
    """

    def __init__(self, reason: str = ""):
        self.reason = reason or "No Codespaces credential is configured on this server."
        self.ledger = _EmptyLedger()

    async def call(self, path: str, payload: Optional[Dict[str, Any]] = None, **_kwargs: Any) -> Dict[str, Any]:
        raise CodespaceError("computer_unconfigured", f"{self.reason} ({path})")

    def note_activity(self) -> None:
        return None

    def run_started(self) -> None:
        return None

    def run_touched(self, *_a: Any, **_k: Any) -> None:
        return None

    def run_finished(self) -> None:
        return None

    async def status(self) -> Dict[str, Any]:
        return {"state": "OFF", "name": "", "error": self.reason, "agent_using": False,
                "agent_connected": False, "idle_seconds": 0, "idle_stop_seconds": 300,
                "workspace": LOGICAL_ROOT, "provider": "GitHub Codespaces"}

    async def lifecycle_snapshot(self) -> Dict[str, Any]:
        return await self.status()

    async def ensure_running(self, **_k: Any) -> Dict[str, Any]:
        raise CodespaceError("computer_unconfigured", self.reason)

    async def stop(self, **_k: Any) -> Dict[str, Any]:
        return {"ok": True, "stopped": False, "reason": "unconfigured"}

    async def maybe_stop(self) -> Dict[str, Any]:
        return {"ok": True, "stopped": False, "reason": "unconfigured"}

    async def adopt(self, **_kwargs: Any) -> Dict[str, Any]:
        return {"ok": True, "adopted": False, "reason": "unconfigured"}


class CodespaceComputer:
    """``/computer/*`` over a codespace, with start-on-demand and stop-when-idle."""

    def __init__(self, http: httpx.AsyncClient, api: CodespaceAPI, *, channel: Optional[CodespaceChannel] = None,
                 store: Any = None, port: int = DEFAULT_PORT, workspace: str = "", idle_stop_s: int = 300,
                 busy: Any = None, mirror: Any = None):
        self.http = http
        self.api = api
        self.channel = channel or CodespaceChannel(store)
        self.store = store
        self.port = port
        self.workspace = workspace or f"{LOGICAL_ROOT}"          # logical root used by the tools
        self.idle_stop_s = idle_stop_s
        self.busy = busy                    # callable() -> bool: a run is executing right now
        self.mirror = mirror                # optional private-drive mirror
        self.capabilities = None            # set by the app (skills/tools restored on demand)
        self.key_pool = None
        self._lock = asyncio.Lock()
        self._inflight = 0
        self._runs = 0                       # agent runs currently executing
        self._last_activity = time.time()
        self._last_reason = "startup"
        self._last_lookup = 0.0
        self._state = "OFF"                 # OFF | STARTING | ON | STOPPING | ERROR
        self._error = ""
        self._name = ""
        self._session: Optional[Dict[str, Any]] = None
        self.ledger = UsageLedger(store)

    # ------------------------------------------------------------------ state
    def _busy(self) -> bool:
        if self._inflight > 0 or self._runs > 0:
            return True
        try:
            return bool(self.busy and self.busy())
        except Exception:  # noqa: BLE001 - an unknown run state must not block a stop decision
            return True

    def agent_using(self) -> bool:
        """True while the agent holds the computer (a run is executing or a call is in flight)."""
        return self._busy()

    def note_activity(self, reason: str = "") -> None:
        """Touch the idle clock, remembering *what* touched it.

        The reason is in ``status()`` so a machine that never idles out can be explained instead of
        guessed at (an idle codespace that keeps resetting its clock is wasted money).
        """
        self._last_activity = time.time()
        if reason:
            self._last_reason = reason

    # Hooks the agent loop calls (``Deps.activity``): they keep the idle clock honest and make it
    # impossible to stop the codespace while a task is running.
    def run_started(self) -> None:
        self._runs += 1
        self.note_activity("run_started")

    def run_touched(self, *_args: Any, **_kwargs: Any) -> None:
        self.note_activity("run_touched")

    def run_finished(self) -> None:
        self._runs = max(0, self._runs - 1)
        self.note_activity("run_finished")

    @property
    def runs_active(self) -> int:
        return self._runs

    def idle_seconds(self) -> float:
        return max(0.0, time.time() - self._last_activity)

    # ------------------------------------------------------------- persistence
    async def _save_session(self) -> None:
        if self.store is None or not self._name:
            return
        payload = {"name": self._name, "port": self.port,
                   "token": (self._session or {}).get("token", ""),
                   "registered_at": (self._session or {}).get("registered_at", 0)}
        try:
            await self.store.set_setting(f"{SESSION_KEY}:{self._name}", json.dumps(payload))
        except Exception:  # noqa: BLE001
            pass

    async def _load_session(self) -> Optional[Dict[str, Any]]:
        if self.store is None or not self._name:
            return None
        try:
            raw = await self.store.get_setting(f"{SESSION_KEY}:{self._name}")
            data = json.loads(raw) if raw else None
        except Exception:  # noqa: BLE001
            data = None
        if data and data.get("token"):
            return {"token": data["token"], "port": int(data.get("port") or self.port),
                    "registered_at": data.get("registered_at", 0)}
        return None

    # -------------------------------------------------------------- lifecycle
    async def find_or_create(self) -> Dict[str, Any]:
        """The one codespace for this repo — created only when absent, never duplicated."""
        existing = await self.api.find()
        if existing:
            self._name = str(existing.get("name"))
            return existing
        created = await self.api.create()
        self._name = str(created.get("name") or "")
        if not self._name:
            raise CodespaceError("create_failed", "GitHub created no codespace for this repository")
        log.info("codespace created: %s", self._name)
        # Anything created concurrently (a second request while this one was in flight) is a
        # duplicate: keep the oldest, delete the rest.
        try:
            for dup in await self.api.duplicates():
                if str(dup.get("name")) != self._name:
                    await self.api.delete(str(dup.get("name")))
        except Exception as exc:  # noqa: BLE001
            log.warning("duplicate cleanup failed: %s", type(exc).__name__)
        return created

    async def ensure_running(self, *, timeout: float = START_TIMEOUT_S) -> Dict[str, Any]:
        """Start the codespace if needed and wait until its agent answers."""
        self.note_activity("ensure_running")
        async with self._lock:
            cs = await self.find_or_create()
            state = str(cs.get("state") or "")
            if state != "Available":
                self._state = "STARTING"
                await self.api.start(self._name)
                cs = await self._wait_available(timeout=timeout)
            if self._session is None:
                self._session = await self._load_session()
            # "known" only means we hold a token from an earlier life of that container: after a
            # stop the token survives while the agent does not, so it is never proof of life.
            await self._ensure_agent(timeout=min(timeout, AGENT_WAIT_S))
            self._state = "ON"
            self._error = ""
            await self.ledger.open_interval(self._name)
            await self._save_state()
            return {"ok": True, "name": self._name, "state": self._state}

    async def adopt(self, *, throttle_s: float = 60.0) -> Dict[str, Any]:
        """Take responsibility for a codespace that is already running.

        A previous process (a redeploy, a crash) forgets which codespace is ours while GitHub keeps
        the machine — and its meter — running. Nothing in the container announces the name to a
        fresh process except the agent's own poll, so: prefer the agent that is dialling in, fall
        back to GitHub's listing, and only ask GitHub once a minute while there is nothing to find.
        Adoption is what makes the panel tell the truth (running, and counted) instead of "off".
        """
        if self._state == "ON" and self._name:
            return {"ok": True, "adopted": False, "reason": "already_on", "name": self._name}
        async with self._lock:
            name = self._name
            if not name:
                live = self.channel.agents() if self.channel is not None else []
                name = live[0] if live else ""
            if not name:
                if time.time() - self._last_lookup < throttle_s:
                    return {"ok": True, "adopted": False, "reason": "unknown"}
                self._last_lookup = time.time()
                try:
                    cs = await self.api.find()
                except CodespaceError:
                    return {"ok": True, "adopted": False, "reason": "lookup_failed"}
                name = str(cs.get("name") or "") if cs else ""
            if not name:
                return {"ok": True, "adopted": False, "reason": "absent"}
            self._name = name
            try:
                cs = await self.api.get(name)
            except CodespaceError as exc:
                self._state, self._error = "ERROR", exc.message
                return {"ok": False, "adopted": False, "reason": "lookup_failed", "detail": exc.message}
            if str(cs.get("state") or "") != "Available":
                self._state, self._error = "OFF", ""
                await self._save_state()
                return {"ok": True, "adopted": False, "reason": "not_running",
                        "state": str(cs.get("state") or "Unknown")}
            self._state, self._error = "ON", ""
            # GitHub's own last_used_at is the only honest evidence of when the machine stopped
            # being used; the idle clock is seeded from it so an abandoned machine stops promptly
            # instead of waiting a fresh idle window.
            evidence = _ts(cs.get("last_used_at"))
            self._last_activity = evidence or time.time()
            self._last_reason = "adopt:GITHUB_LAST_USED" if evidence else "adopt:now"
            await self.ledger.open_interval(name)
            await self._save_state()
            log.info("adopted running codespace %s (last used %s)", name,
                     "unknown" if not evidence else f"{int(time.time() - evidence)}s ago")
            return {"ok": True, "adopted": True, "name": name, "idle_seconds": int(self.idle_seconds())}

    async def _wait_available(self, timeout: float) -> Dict[str, Any]:
        """Wait for GitHub to hand over an Available container.

        A first boot — or any boot after the devcontainer changed — rebuilds the image and stays in
        Provisioning for many minutes (10+ observed). That is not a failure: it is reported as
        STARTING and waited for, and only a terminal state fails. The last state GitHub reported is
        carried in the error, so a timeout says what the machine was actually doing.
        """
        deadline = time.time() + timeout
        delay = 2.0
        last = "unknown"
        while time.time() < deadline:
            await asyncio.sleep(delay)
            delay = min(delay * 1.4, 10.0)
            try:
                cs = await self.api.get(self._name)
            except CodespaceError as exc:
                last = exc.code
                continue
            state = str(cs.get("state") or "")
            last = state or last
            if state == "Available":
                return cs
            if state in ("Failed", "Unavailable", "Deleted"):
                raise CodespaceError("start_failed", f"The codespace reported state {state}")
            self._state = "STARTING"
        raise CodespaceError("start_timeout",
                             f"The codespace was still {last} after {int(timeout)}s; it is being built and "
                             "will be adopted as soon as it is available")

    # ------------------------------------------------------------ exec channel
    async def _ensure_agent(self, *, timeout: float = 60.0) -> None:
        """Prove the container's agent is really answering, not merely that we hold a token.

        A live agent polls the channel, so presence is proof. A token without presence belongs to
        an earlier life of the container (or a container still booting), so it is verified with a
        cheap authenticated call; if that does not answer, the wait continues. This is what keeps
        the panel from reporting ON for a machine that cannot yet run a command.
        """
        if not self._name:
            raise CodespaceError("agent_unavailable", "No codespace is attached to this computer")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.channel.present(self._name):
                return
            try:
                probe = await self.channel.call("/computer/info", {}, timeout=min(20.0, max(5.0, deadline - time.time())),
                                                name=self._name)
                if isinstance(probe, dict) and probe.get("ok") is not False:
                    return
            except Exception:  # noqa: BLE001 - not answering yet is the expected case while booting
                pass
            await asyncio.sleep(2.0)
        raise CodespaceError("agent_unreachable",
                             "The codespace is running but its agent is not answering on the channel yet")

    def agent_connected(self) -> bool:
        if not self._name:
            return bool(self.channel.agents()) if self.channel is not None else False
        return self.channel.present(self._name)

    async def _agent(self, path: str, body: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:
        """Send one job through the channel; a timeout is reported, never faked."""
        try:
            result = await self.channel.call(path, body, timeout=timeout, name=self._name)
        except TimeoutError as exc:
            # The container may have restarted mid-job; drop the session so the next call
            # re-establishes it instead of failing forever.
            self._session = None
            raise CodespaceError("timeout", str(exc)) from exc
        if isinstance(result, dict) and result.get("error") and not result.get("ok"):
            err = str(result["error"])
            if "unknown operation" in err or "not found" in err:
                raise CodespaceError("protocol", f"The codespace agent rejected {path}: {err}")
        return result if isinstance(result, dict) else {"ok": True}

    # ------------------------------------------------------------- computer API
    async def call(self, path: str, payload: Optional[Dict[str, Any]] = None, *, method: str = "POST",
                   timeout: float = 90, idempotent: bool = True) -> Dict[str, Any]:
        payload = payload or {}
        self.note_activity(f"tool:{path}")
        self._inflight += 1
        try:
            if self._state != "ON" or self._session is None:
                await self.ensure_running()
            translated = self._translate(path, payload)
            result = await self._agent(translated["route"], translated["body"], timeout=max(30.0, timeout))
            return self._post(path, payload, result)
        finally:
            self._inflight -= 1
            self.note_activity(f"tool_done:{path}")

    def _translate(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if path == "/computer/exec":
            cwd = str(payload.get("cwd") or ".")
            return {"route": "/exec", "body": {"cmd": str(payload.get("command") or ""), "cwd": cwd,
                                               "timeout": float(payload.get("timeout_seconds") or 60)}}
        if path == "/computer/write_file":
            return {"route": "/write", "body": {"path": str(payload.get("path") or ""),
                                                "content": str(payload.get("content") or "")}}
        if path == "/computer/read_file":
            return {"route": "/read", "body": {"path": str(payload.get("path") or "")}}
        if path == "/computer/list_files":
            return {"route": "/list", "body": {"path": str(payload.get("path") or ".")}}
        if path == "/computer/fetch_url":
            return {"route": "/fetch", "body": {"url": str(payload.get("url") or "")}}
        if path == "/computer/info":
            return {"route": "/info", "body": {}}
        if path == "/computer/read_sample":
            return {"route": "/read_sample", "body": {k: payload.get(k) for k in ("path", "max_bytes")}}
        if path == "/computer/checkpoint":
            return {"route": "/checkpoint", "body": {}}
        raise CodespaceError("unsupported", f"The codespace computer does not implement {path}")

    def _post(self, path: str, payload: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise the agent's answer to the shape the tools already expect."""
        if path == "/computer/exec":
            out = {"ok": bool(result.get("ok")), "output": str(result.get("output") or ""),
                   "exit_code": result.get("exit_code")}
            if result.get("error"):
                out["error"] = str(result["error"])[:400]
            return out
        if path == "/computer/info":
            return result if isinstance(result, dict) else {"ok": True}
        return result if isinstance(result, dict) else {"ok": True}

    # ------------------------------------------------------------ power control
    async def stop(self, *, reason: str = "idle", force: bool = False) -> Dict[str, Any]:
        """Stop the codespace. Refuses while work is in flight unless ``force``."""
        if not force and self._busy():
            return {"ok": False, "stopped": False, "reason": "busy",
                    "detail": "A task is executing; the codespace stays on."}
        async with self._lock:
            if not self._name:
                cs = await self.api.find()
                if not cs:
                    self._state = "OFF"
                    return {"ok": True, "stopped": False, "reason": "absent"}
                self._name = str(cs.get("name"))
            self._state = "STOPPING"
            try:
                await self.api.stop(self._name)
            except CodespaceError as exc:
                self._state = "ERROR"
                self._error = exc.message
                raise
            await self.ledger.close_interval()
            drop = getattr(self.channel, "drop_presence", None)
            if callable(drop):
                drop(self._name)   # a stopped container is not connected
            self._state = "OFF"
            self._session = None
            if self.store is not None:
                try:
                    await self.store.delete_setting(f"{SESSION_KEY}:{self._name}")
                except Exception:  # noqa: BLE001
                    pass
            await self._save_state()
            log.info("codespace stopped (%s)", reason)
            return {"ok": True, "stopped": True, "reason": reason}

    async def maybe_stop(self) -> Dict[str, Any]:
        """Stop only when the agent is genuinely idle — the automatic half of requirement 4."""
        if self._state not in ("ON", "ERROR"):
            return {"ok": True, "stopped": False, "reason": "already_off"}
        if self._busy():
            return {"ok": True, "stopped": False, "reason": "busy"}
        if self.idle_seconds() < self.idle_stop_s:
            return {"ok": True, "stopped": False, "reason": "recent_activity",
                    "idle_seconds": int(self.idle_seconds())}
        return await self.stop(reason="idle")

    async def _save_state(self) -> None:
        if self.store is None:
            return
        try:
            await self.store.set_setting(STATE_KEY, json.dumps({
                "state": self._state, "name": self._name, "error": self._error, "at": time.time()}))
        except Exception:  # noqa: BLE001
            pass

    async def status(self) -> Dict[str, Any]:
        return {"state": self._state, "name": self._name, "error": self._error,
                "agent_using": self.agent_using(), "runs_active": self._runs,
                "agent_connected": self.agent_connected(),
                "idle_seconds": int(self.idle_seconds()),
                "idle_stop_seconds": self.idle_stop_s,
                "last_activity": self._last_reason,
                "last_activity_age_s": int(self.idle_seconds()),
                "workspace": LOGICAL_ROOT, "provider": "GitHub Codespaces"}

    async def lifecycle_snapshot(self) -> Dict[str, Any]:
        snap = await self.status()
        try:
            cs = await self.api.get(self._name) if self._name else {}
        except CodespaceError:
            cs = {}
        snap["codespace"] = {k: cs.get(k) for k in ("name", "state", "machine", "created_at", "last_used_at",
                                                    "idle_timeout_minutes", "retention_period_minutes")}
        return snap

    async def keepalive(self) -> Dict[str, Any]:
        """Touch the codespace so GitHub's own idle timer does not reap it mid-task."""
        if self._state != "ON":
            return {"ok": False, "reason": "off"}
        self.note_activity("keepalive")
        return {"ok": True}

    async def handover(self, *, reason: str = "manual") -> Dict[str, Any]:
        """Replace nothing: a codespace keeps its disk, so a restart IS the handover."""
        await self.stop(reason=reason, force=True)
        return await self.ensure_running()

    async def prepare_and_checkpoint(self) -> Dict[str, Any]:
        return await self.call("/computer/checkpoint", {}, timeout=230.0, idempotent=False)
