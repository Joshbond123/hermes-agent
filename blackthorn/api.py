"""HTTP API of the Blackthorn app."""

from __future__ import annotations

import asyncio
import json
import os
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import prompts
from .agent import AgentRun, Deps
from .route import RouteError, RouteResolver
from .runs import Run, RunManager
from .store import ChatStore, NotFound
from .tools import TaskBoard, ToolError, ToolRegistry

log = logging.getLogger("blackthorn.api")
router = APIRouter()

MAX_MESSAGE_CHARS = 20_000
SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive", "X-Accel-Buffering": "no", "Content-Encoding": "identity"}


@dataclass
class Services:
    settings: Any
    store: ChatStore
    resolver: RouteResolver
    registry: ToolRegistry
    http: httpx.AsyncClient
    tavily: Any
    computer: Any
    runs: RunManager
    codespaces: Any             # the CodespaceComputer: the agent's own computer (power-managed)
    static_dir: Path
    channel: Any = None         # command channel to the in-codespace agent
    lab: Any = None             # dedicated malware/ransomware execution lab (own provider account)
    capabilities: Any = None    # drive-hosted skills/tools restored onto every machine
    _prompt_cache: tuple = field(default=(0.0, ""))
    _memory_cache: tuple = field(default=(0.0, ()))

    def deps(self) -> Deps:
        return Deps(settings=self.settings, store=self.store, resolver=self.resolver, registry=self.registry,
                    http=self.http, tavily=self.tavily, computer=self.computer, activity=self.codespaces,
                    lab=self.lab)

    async def user_prompt(self) -> str:
        ts, value = self._prompt_cache
        if time.monotonic() - ts < 30:
            return value
        try:
            value = await self.store.get_setting("blackthorn_system_prompt")
        except Exception as exc:
            log.warning("system prompt unavailable: %s", exc)
            value = value or ""
        self._prompt_cache = (time.monotonic(), value)
        return value

    def set_prompt_cache(self, value: str) -> None:
        self._prompt_cache = (time.monotonic(), value)

    async def memories(self) -> List[str]:
        ts, value = self._memory_cache
        if time.monotonic() - ts < 60:
            return list(value)
        try:
            value = tuple(await self.store.list_memories(6))
        except Exception as exc:
            log.warning("memories unavailable: %s", exc)
        self._memory_cache = (time.monotonic(), value)
        return list(value)


def svc(request: Request) -> Services:
    return request.app.state.bt


def _err(status: int, code: str, message: str, **extra: Any) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message, **extra})


# --------------------------------------------------------------------------- health / version
@router.api_route("/api/health", methods=["GET", "HEAD"])
async def health() -> Dict[str, Any]:
    """Liveness plus the deployed build identity, so a deploy can be verified externally.

    Only non-secret identifiers are exposed: version, commit, start time and uptime.
    """
    from . import __version__
    from .version import STARTED_AT, _git_commit
    return {"ok": True, "app": "blackthorn", "version": __version__, "commit": _git_commit(),
            "started_at": STARTED_AT, "uptime_seconds": int(time.time() - STARTED_AT)}


@router.get("/api/lab")
async def lab_state(request: Request) -> Dict[str, Any]:
    """Whether this deployment has the malware execution lab wired. No secrets, no provider call."""
    from .keys_role import describe as _describe_keys
    lab = getattr(svc(request), "lab", None)
    keys = _describe_keys()
    if lab is None:
        return {"available": False, "open": False, "keys": keys,
                "detail": "not configured (no dedicated PANDASTACK_LAB_API_KEY on this deployment)"}
    session = lab.session
    return {"available": True, "configured": lab.configured(),
            "open": session is not None,
            "sealed": bool(session.sealed) if session else False,
            "runs": len(session.runs) if session else 0,
            "ttl_s": lab.ttl_s, "idle_close_s": lab.idle_s,
            "dedicated_key": bool(getattr(lab, "key_is_dedicated", False)),
            "os": (session.os if session else {}),
            "held": int(getattr(lab, "holds", 0) or 0),
            "capabilities": getattr(lab, "capability_status", {}),
            "keys": keys,
            "tools": ["lab_open", "lab_status", "lab_detonate", "lab_exec", "lab_os", "lab_close", "lab_file"]}


@router.get("/api/version")
async def version(request: Request) -> Dict[str, Any]:
    from .version import build_info
    return build_info(svc(request).static_dir)


# --------------------------------------------------------------------------- remote analysis egress (authenticated by AuthMiddleware)
@router.get("/api/malware-analyzer/egress")
async def malware_analyzer_egress() -> Dict[str, Any]:
    """Prove the server can reach the public API via verified Proton exit; expose no IP."""
    from .malware_analyzer import API, EGRESS, EgressUnavailable
    try:
        client = await EGRESS.client()
        try:
            response = await client.get(API + "/v1/status")
            if response.is_redirect or response.status_code != 200:
                raise EgressUnavailable("provider status was not HTTP 200")
        finally:
            await client.aclose()
        return {"ok": True, "vpn_exit_distinct_from_host": True, "provider_status": response.status_code,
                "remote_sample_execution": "not tested by status check"}
    except (EgressUnavailable, httpx.HTTPError) as exc:
        reason = str(exc) if isinstance(exc, EgressUnavailable) else f"provider status unavailable ({type(exc).__name__})"
        # Only a fixed-category diagnostic; never return either IP, key or profile.
        raise _err(503, "vpn_unavailable", "VPN egress/provider status not verified; remote analysis is disabled.",
                   reason=reason[:120]) from exc


@router.get("/api/malware-analyzer/report/{sha256}")
async def malware_analyzer_public_report(sha256: str) -> Dict[str, Any]:
    """Authenticated, read-only diagnostic of a PUBLIC report via the verified exit."""
    from .malware_analyzer import EgressUnavailable, HASH, request_report
    if not HASH.fullmatch(sha256):
        raise _err(422, "invalid_hash", "A SHA-256 hash is required.")
    try:
        result = await request_report(sha256)
    except EgressUnavailable as exc:
        raise _err(503, "remote_report_unavailable", "The verified VPN report lookup did not complete.") from exc
    data = result.get("data", {}) if isinstance(result, dict) else {}
    attrs = data.get("attributes", {}) if isinstance(data, dict) else {}
    if not isinstance(attrs, dict):
        attrs = {}
    return {"ok": True, "sha256": sha256.lower(), "visibility": attrs.get("visibility"),
            "verdict": attrs.get("verdict"), "dynamicDegraded": attrs.get("dynamicDegraded")}


# --------------------------------------------------------------------------- sessions
class SessionPatch(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    pinned: Optional[bool] = None
    archived: Optional[bool] = None


@router.get("/api/chat/sessions")
async def list_sessions(request: Request, archived: bool = False, q: str = "", limit: int = Query(100, ge=1, le=300)):
    s = svc(request)
    try:
        return {"sessions": await s.store.list_sessions(archived=archived, q=q, limit=limit)}
    except Exception as exc:
        log.warning("list sessions failed: %s", exc)
        raise _err(502, "storage_unavailable", "Chat history is temporarily unavailable. Try again in a moment.")


@router.get("/api/chat/sessions/{session_id}")
async def get_session(session_id: str, request: Request):
    s = svc(request)
    try:
        session = await s.store.get_session(session_id)
        messages = await s.store.get_messages(session_id)
    except NotFound:
        raise _err(404, "not_found", "That conversation no longer exists.")
    except Exception as exc:
        log.warning("get session failed: %s", exc)
        raise _err(502, "storage_unavailable", "Could not load this conversation. Try again in a moment.")
    live = s.runs.live_for_session(session_id)
    if not live and any(m["role"] == "assistant" and m["status"] == "streaming" for m in messages):
        try:
            await s.store.sweep_interrupted(session_id)
        except Exception as exc:
            log.warning("sweep failed: %s", exc)
        for m in messages:
            if m["role"] == "assistant" and m["status"] == "streaming":
                m["status"] = "interrupted"
    return {"session": session, "messages": messages, "live_run": live.info() if live else None}


@router.patch("/api/chat/sessions/{session_id}")
async def patch_session(session_id: str, patch: SessionPatch, request: Request):
    s = svc(request)
    try:
        session = await s.store.update_session(session_id, title=patch.title, pinned=patch.pinned, archived=patch.archived)
    except NotFound:
        raise _err(404, "not_found", "That conversation no longer exists.")
    except ValueError as exc:
        raise _err(422, "invalid", str(exc))
    return {"session": session}


@router.delete("/api/chat/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    s = svc(request)
    live = s.runs.live_for_session(session_id)
    if live:
        s.runs.cancel(live.id)
        await asyncio.sleep(0.2)
    try:
        await s.store.delete_session(session_id)
    except NotFound:
        raise _err(404, "not_found", "That conversation no longer exists.")
    return {"ok": True, "deleted": session_id}


# --------------------------------------------------------------------------- streaming chat
class AttachmentIn(BaseModel):
    name: str = Field(max_length=200)
    content: str = ""


class StreamRequest(BaseModel):
    session_id: Optional[str] = None
    message: Optional[str] = None
    regenerate: bool = False
    attachments: List[AttachmentIn] = Field(default_factory=list)


def _sse_bytes(event: Dict[str, Any]) -> bytes:
    return f"id: {event['seq']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")


async def _sse(run: Run, after: int, heartbeat: float) -> AsyncIterator[bytes]:
    # Force proxy flush + immediately drain pre-emitted events (run.start / thinking).
    yield b"retry: 1000\n\n"
    yield b": " + (b" " * 16384) + b"\n\n"
    await asyncio.sleep(0)
    try:
        for event in list(getattr(run, "events", []) or []):
            seq = int(event.get("seq") or 0)
            if seq > after:
                after = seq
                yield _sse_bytes(event)
    except Exception:
        pass
    async for event in run.subscribe(after=after, heartbeat=heartbeat):
        yield b": ping\n\n" if event is None else _sse_bytes(event)





async def _stage_attachments(s: Services, items: List[AttachmentIn]) -> List[Dict[str, Any]]:
    staged: List[Dict[str, Any]] = []
    cfg = s.settings
    for item in items[: cfg.max_attachments]:
        name = prompts.safe_filename(item.name)
        content = item.content[: cfg.max_attachment_chars]
        entry: Dict[str, Any] = {"name": name, "chars": len(content), "preview": content[: cfg.attachment_preview_chars],
                                 "truncated": len(content) > cfg.attachment_preview_chars, "workspace_path": None}
        try:
            await s.computer.call("/computer/write_file", {"path": f"uploads/{name}", "content": content}, timeout=40.0)
            entry["workspace_path"] = f"uploads/{name}"
        except ToolError:
            entry["preview"] = content[: cfg.attachment_preview_chars * 2]
            entry["truncated"] = len(content) > len(entry["preview"])
        staged.append(entry)
    return staged


@router.post("/api/chat/stream")
async def chat_stream(body: StreamRequest, request: Request):
    s = svc(request)
    cfg = s.settings
    text = (body.message or "").strip()
    if body.regenerate:
        if not body.session_id:
            raise _err(422, "invalid", "regenerate needs a session_id")
    else:
        if not text:
            raise _err(422, "empty_message", "Type a message first.")
        if len(text) > MAX_MESSAGE_CHARS:
            raise _err(413, "too_long", f"Messages are limited to {MAX_MESSAGE_CHARS:,} characters.")
    if s.runs.at_capacity():
        raise _err(429, "busy", "The assistant is handling several requests right now. Try again in a moment.")
    if body.session_id and s.runs.live_for_session(body.session_id):
        raise _err(409, "run_in_progress", "This conversation is still generating a response.")
    # Parallelize route + prompt + memories so simple messages are not stuck on serial D1 reads.
    try:
        route, user_prompt, memories = await asyncio.gather(s.resolver.get(), s.user_prompt(), s.memories())
    except RouteError as exc:
        raise _err(409, exc.code, exc.message)
    system = prompts.system_prompt(user_prompt, memories)
    try:
        if body.regenerate:
            begin = await s.store.begin_regeneration(body.session_id or "", model=route.model, history_limit=cfg.history_messages)
            model_text = begin["user_text"]
        else:
            attachments = await _stage_attachments(s, body.attachments) if body.attachments else []
            display = [{"name": a["name"], "chars": a["chars"], "workspace_path": a["workspace_path"]} for a in attachments]
            begin = await s.store.begin_turn(body.session_id, text, model=route.model, attachments=display or None,
                                             history_limit=cfg.history_messages)
            model_text = prompts.user_message_with_attachments(text, attachments)
    except NotFound:
        raise _err(404, "not_found", "That conversation no longer exists.")
    except ValueError as exc:
        raise _err(422, "invalid", str(exc))
    except Exception as exc:
        log.error("could not start turn: %s", exc)
        raise _err(502, "storage_unavailable", "Could not save your message (history storage is unavailable). Nothing was sent; try again.")

    # The session's task list is restored before the run starts, so a follow-up ("continue"), a restart
    # or a compacted context all resume the same plan instead of restarting it.
    try:
        board = await TaskBoard.load(s.store, begin["session_id"])
    except Exception:
        # A missing D1 read must not silently turn an unfinished job into an empty plan.
        await s.store.update_assistant(begin["assistant_uid"], text="", parts=[], status="interrupted")
        raise _err(503, "task_storage_unavailable", "Could not restore the saved task list; nothing was sent to the model. Retry later.")
    if board.items and not board.is_open():
        # The previous plan is finished: drop it so a new request is not steered by a stale list.
        # The evidence of what ran is not lost — it lives in each turn's execution ledger.
        board.write([], merge=False)
        board.goal = ""
        try:
            await board.save()
        except Exception:
            await s.store.update_assistant(begin["assistant_uid"], text="", parts=[], status="interrupted")
            raise _err(503, "task_storage_unavailable", "Could not clear the completed task list; retry later.")
    run = s.runs.create(begin["session_id"], begin["assistant_uid"])
    run.emit("run.start", run_id=run.id, session_id=begin["session_id"], assistant_id=begin["assistant_uid"],
             user_id=begin.get("user_uid"), model=route.model, session=begin["session"], created=begin["created"])
    resume_hint = ""
    if begin.get("previous_attempt"):
        resume_hint = ("The previous answer to this request was discarded by the user. Steps it had already run are "
                       "listed above: do not repeat them unless a check shows they failed.")
    agent = AgentRun(run, s.deps(), system=system, history=begin["history"], user_message=model_text, model=route.model,
                     turns=begin.get("turns"), board=board, previous_attempt=begin.get("previous_attempt"),
                     resume_hint=resume_hint)
    run.task = asyncio.create_task(agent.execute(), name=f"blackthorn-{run.id}")
    return StreamingResponse(_sse(run, 0, cfg.heartbeat_s), media_type="text/event-stream", headers=SSE_HEADERS)


@router.get("/api/chat/runs/{run_id}/events")
async def run_events(run_id: str, request: Request, after: int = Query(0, ge=0)):
    s = svc(request)
    run = s.runs.get(run_id)
    if run is None:
        raise _err(404, "run_gone", "That response is no longer live. Reload the conversation to see what was saved.")
    return StreamingResponse(_sse(run, after, s.settings.heartbeat_s), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/api/chat/runs/{run_id}/cancel")
async def cancel_run(run_id: str, request: Request):
    s = svc(request)
    run = s.runs.get(run_id)
    if run is None:
        raise _err(404, "run_gone", "That response is no longer running.")
    cancelled = s.runs.cancel(run_id)
    if run.task is not None:
        try:
            await asyncio.wait_for(asyncio.shield(run.task), timeout=8)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass
    return {"ok": True, "cancelled": cancelled, "status": run.status, "done": run.done}


# --------------------------------------------------------------------------- system prompt
class PromptBody(BaseModel):
    prompt: str = Field(default="", max_length=8000)


@router.get("/api/system-prompt")
async def get_prompt(request: Request):
    s = svc(request)
    try:
        value = await s.store.get_setting("blackthorn_system_prompt")
    except Exception:
        raise _err(502, "storage_unavailable", "Could not load the system prompt.")
    # Always show a usable default in the header UI when the user has not saved one yet.
    if not (value or "").strip():
        from .prompts import BASE_PROMPT
        value = BASE_PROMPT
    return {"prompt": value}


@router.put("/api/system-prompt")
async def put_prompt(body: PromptBody, request: Request):
    s = svc(request)
    try:
        await s.store.set_setting("blackthorn_system_prompt", body.prompt)
    except Exception:
        raise _err(502, "storage_unavailable", "Could not save the system prompt (storage unavailable).")
    s.set_prompt_cache(body.prompt)
    return {"ok": True, "prompt": body.prompt}


# --------------------------------------------------------------------------- GPU
class OffBody(BaseModel):
    confirm: bool = False
    role: Literal["all", "model", "computer"] = "all"


class AutoOffBody(BaseModel):
    minutes: int




def _computer(request: Request) -> Any:
    computer = svc(request).codespaces
    if computer is None:
        raise _err(503, "computer_unconfigured", "No codespace computer is configured on this server.")
    return computer


# ----------------------------------------------------------------- codespaces: power
CODESPACE_CORE_HOURS_DEFAULT = 120.0     # GitHub's included Codespaces core-hours per month


def _cores(machine: str) -> int:
    """Core count from a machine type name (``basicLinux32gb`` -> 2, ``standardLinux32gb`` -> 4)."""
    table = {"basiclinux32gb": 2, "standardlinux32gb": 4, "premiumlinux64gb": 8}
    return table.get(str(machine or "").lower(), 2)


@router.get("/api/codespaces/status")
async def codespaces_status(request: Request, refresh: bool = False):
    """Everything the usage controller shows: power state, live link, and measured hours."""
    s = svc(request)
    computer = s.codespaces
    if computer is None:
        raise _err(503, "computer_unconfigured", "No codespace computer is configured on this server.")
    try:
        out = await computer.status()
    except Exception as exc:  # noqa: BLE001
        log.warning("codespace status failed: %s", exc)
        raise _err(502, "codespace_status_failed", f"{type(exc).__name__}: {str(exc)[:160]}") from exc
    out["link"] = s.channel.presence(out.get("name") or "") if s.channel is not None else {}
    out["channel"] = s.channel.summary() if s.channel is not None else {}
    if refresh:
        try:
            out["codespace"] = (await computer.lifecycle_snapshot()).get("codespace") or {}
        except Exception as exc:  # noqa: BLE001
            out["codespace"] = {"error": type(exc).__name__}
    return out


@router.post("/api/codespaces/start")
async def codespaces_start(request: Request):
    computer = _computer(request)
    try:
        return await computer.ensure_running()
    except Exception as exc:  # noqa: BLE001
        raise _err(502, "codespace_start_failed", f"{type(exc).__name__}: {str(exc)[:200]}") from exc


@router.post("/api/codespaces/stop")
async def codespaces_stop(request: Request, force: bool = False):
    computer = _computer(request)
    try:
        return await computer.stop(reason="manual", force=force)
    except Exception as exc:  # noqa: BLE001
        raise _err(502, "codespace_stop_failed", f"{type(exc).__name__}: {str(exc)[:200]}") from exc


@router.post("/api/codespaces/restart")
async def codespaces_restart(request: Request):
    """Resume unfinished work: stop and start the same codespace, keeping its disk."""
    computer = _computer(request)
    try:
        return await computer.handover(reason="manual")
    except Exception as exc:  # noqa: BLE001
        raise _err(502, "codespace_restart_failed", f"{type(exc).__name__}: {str(exc)[:200]}") from exc


@router.get("/api/codespaces/usage")
async def codespaces_usage(request: Request):
    """Monthly Codespaces hours: the allowance, what has been measured, what is left.

    The consumed figure is measured from the intervals the app itself observed (recorded while the
    codespace was running). When the GitHub token also carries the Plan permission, GitHub's own
    billing figure is returned instead and labelled as such — the panel never invents a number.
    """
    computer = _computer(request)
    status = await computer.status()
    state = str(status.get("state") or "OFF")
    machine = ((await computer.lifecycle_snapshot()).get("codespace") or {}).get("machine") or ""
    machine = machine if isinstance(machine, str) else str((machine or {}).get("name") or "")
    cores = _cores(machine)
    allowance_core_hours = float(os.environ.get("CODESPACES_CORE_HOURS") or CODESPACE_CORE_HOURS_DEFAULT)
    if os.environ.get("CODESPACES_MONTHLY_HOURS"):
        allowance_hours = float(os.environ["CODESPACES_MONTHLY_HOURS"])
        basis = "configured monthly hours (CODESPACES_MONTHLY_HOURS)"
    else:
        allowance_hours = allowance_core_hours / max(1, cores)
        basis = f"{allowance_core_hours:g} included core-hours per month / {cores} cores"
    measured_s = await computer.ledger.seconds_this_month()
    out: Dict[str, Any] = {
        "provider": "GitHub Codespaces",
        "state": state,
        "running": state == "ON",
        "agent_using": bool(status.get("agent_using")),
        "agent_connected": bool(status.get("agent_connected")),
        "last_activity": status.get("last_activity") or "",   # what last touched the idle clock
        "name": status.get("name") or "",
        "machine": machine or "basicLinux32gb",
        "cores": cores,
        "hours_total": round(allowance_hours, 2),
        "hours_used": round(measured_s / 3600.0, 2),
        "hours_basis": basis,
        "usage_source": "measured" if measured_s else "measured (nothing recorded yet this month)",
        "month": time.strftime("%Y-%m", time.gmtime()),
        "updated_at": int(time.time()),
        "idle_stop_seconds": status.get("idle_stop_seconds"),
    }
    out["hours_remaining"] = round(max(0.0, allowance_hours - out["hours_used"]), 2)
    out["percent_used"] = round(min(100.0, (out["hours_used"] / allowance_hours) * 100.0), 1) if allowance_hours else 0.0
    if state == "ON":
        # Live projection: what the month will cost if the current session keeps running.
        session_s = float(status.get("idle_seconds") or 0)
        out["projected_hours"] = round(out["hours_used"] + session_s / 3600.0, 2)
    return out


# ------------------------------------------------------- codespaces: agent channel
class RegisterBody(BaseModel):
    name: str
    port: int = 8787
    token: str
    version: str = ""


@router.post("/api/codespaces/register")
async def codespaces_register(request: Request, body: RegisterBody):
    """An agent inside a codespace announces itself, proving identity with its GitHub token.

    The token must belong to the configured owner *and* be able to read the very codespace that is
    registering; only then is its session token accepted.
    """
    s = svc(request)
    if s.codespaces is None or s.channel is None:
        raise _err(503, "computer_unconfigured", "No codespace computer is configured on this server.")
    header = request.headers.get("authorization") or ""
    github_token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not github_token:
        raise _err(401, "missing_token", "Registration requires the codespace's GitHub token.")
    api = s.codespaces.api
    try:
        me = await _github_get(s.http, github_token, "/user")
        login = str((me or {}).get("login") or "")
        if not login:
            raise _err(401, "bad_token", "The supplied token is not a valid GitHub token.")
        if api.owner and login.lower() != api.owner.lower():
            raise _err(403, "wrong_owner", f"This token belongs to {login}, not to the configured owner.")
        cs = await _github_get(s.http, github_token, f"/user/codespaces/{body.name}")
        owner = str(((cs or {}).get("owner") or {}).get("login") or "")
        if not cs or not cs.get("name"):
            raise _err(404, "codespace_unknown", "That codespace is not visible to the supplied token.")
        if owner and owner.lower() != login.lower():
            raise _err(403, "foreign_codespace", "That codespace belongs to a different account.")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _err(502, "github_unreachable", f"Could not verify the token with GitHub: {type(exc).__name__}") from exc
    await s.channel.accept_registration(body.name, body.token)
    s.channel.touch(body.name, version=body.version, port=body.port)
    log.info("codespace agent registered: %s (%s)", body.name, body.version)
    return {"ok": True, "name": body.name, "owner": login}


@router.get("/api/codespaces/pull")
async def codespaces_pull(request: Request, name: str = "", version: str = "", port: int = 0):
    """Long-poll: the agent asks for the next unit of work."""
    s = svc(request)
    if s.channel is None:
        raise _err(503, "computer_unconfigured", "No codespace channel is configured on this server.")
    if not name:
        raise _err(400, "missing_name", "The agent must identify its codespace.")
    header = request.headers.get("authorization") or ""
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    expected = await s.channel.token_for(name)
    if not expected or token != expected:
        raise _err(401, "unknown_agent", "This codespace agent is not registered.")
    # A poll proves the machine is alive (presence is updated inside ``pull``); it does NOT prove a
    # task needs it. Counting polls as activity kept the idle clock at zero for ever, so an idle
    # codespace would never stop and its hours were wasted.
    return await s.channel.pull(name, version=version, port=port)


@router.post("/api/codespaces/result/{job_id}")
async def codespaces_result(request: Request, job_id: str):
    """The agent posts the outcome of one job."""
    s = svc(request)
    if s.channel is None:
        raise _err(503, "computer_unconfigured", "No codespace channel is configured on this server.")
    body = await request.json()
    # The caller of the job (``CodespaceComputer.call``) already made the machine busy and touches the
    # idle clock when it finishes; a result arriving late must not extend the session on its own.
    accepted = s.channel.submit_result(job_id, body if isinstance(body, dict) else {})
    return {"ok": True, "accepted": accepted}


async def _github_get(http: httpx.AsyncClient, token: str, path: str) -> Dict[str, Any]:
    resp = await http.get(f"https://api.github.com{path}", timeout=30.0,
                          headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                   "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "Blackthorn-Codespaces"})
    if resp.status_code != 200:
        raise _err(resp.status_code if resp.status_code in (401, 403, 404) else 502, "github_error",
                   f"GitHub {path} returned HTTP {resp.status_code}")
    try:
        return resp.json()
    except ValueError:
        return {}


# ----------------------------------------------------------------- computer session
@router.get("/api/computer/session")
async def computer_session(request: Request):
    """Computer session state for the operator: power state, live link, workspace and events."""
    s = svc(request)
    out: Dict[str, Any] = {"provider": "GitHub Codespaces"}
    computer = s.codespaces
    if computer is None:
        out["session"] = {"state": "OFF", "error": "unconfigured"}
        return out
    try:
        out["session"] = await computer.lifecycle_snapshot()
    except Exception as exc:  # noqa: BLE001
        out["session"] = {"error": type(exc).__name__}
    out["link"] = s.channel.presence(out["session"].get("name") or "") if s.channel is not None else {}
    try:
        raw = await s.store.get_setting("codespace_events")
        out["lifecycle_events"] = (json.loads(raw) if raw else [])[-10:]
    except Exception:  # noqa: BLE001
        out["lifecycle_events"] = []
    return out


@router.post("/api/computer/checkpoint")
async def computer_checkpoint(request: Request):
    """Durability barrier: the codespace writes its checkpoint to the persistent disk."""
    try:
        result = await svc(request).computer.call("/computer/checkpoint", {}, timeout=230.0,
                                                  idempotent=False)
    except ToolError as exc:
        raise _err(502, "checkpoint_failed",
                   "Checkpoint failed: " + (str(getattr(exc, "message", "") or exc) or "the computer did not confirm it")) from exc
    if not result.get("ok"):
        raise _err(502, "checkpoint_failed", str(result.get("error") or "Checkpoint not published")[:160])
    return result
