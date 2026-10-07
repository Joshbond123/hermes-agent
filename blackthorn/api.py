"""HTTP API of the Blackthorn app."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import prompts
from .agent import AgentRun, Deps
from .route import RouteError, RouteResolver
from .runs import Run, RunManager
from .store import ChatStore, NotFound
from .tools import ToolError, ToolRegistry

log = logging.getLogger("blackthorn.api")
router = APIRouter()

MAX_MESSAGE_CHARS = 20_000
SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive", "X-Accel-Buffering": "no"}


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
    gpu: Any
    static_dir: Path
    _prompt_cache: tuple = field(default=(0.0, ""))
    _memory_cache: tuple = field(default=(0.0, ()))

    def deps(self) -> Deps:
        return Deps(settings=self.settings, store=self.store, resolver=self.resolver, registry=self.registry,
                    http=self.http, tavily=self.tavily, computer=self.computer, activity=self.gpu)

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
    return {"ok": True, "app": "blackthorn"}


@router.get("/api/version")
async def version(request: Request) -> Dict[str, Any]:
    from .version import build_info
    return build_info(svc(request).static_dir)


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
    # A closed connection only stops *this* iterator; the run itself keeps going until it finishes or is cancelled.
    yield b"retry: 3000\n\n"
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
    try:  # fail fast and honestly when the model cannot answer; the draft stays in the composer
        route = await s.resolver.get()
    except RouteError as exc:
        raise _err(409, exc.code, exc.message)
    user_prompt, memories = await asyncio.gather(s.user_prompt(), s.memories())
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

    run = s.runs.create(begin["session_id"], begin["assistant_uid"])
    run.emit("run.start", run_id=run.id, session_id=begin["session_id"], assistant_id=begin["assistant_uid"],
             user_id=begin.get("user_uid"), model=route.model, session=begin["session"], created=begin["created"])
    agent = AgentRun(run, s.deps(), system=system, history=begin["history"], user_message=model_text, model=route.model)
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


class AutoOffBody(BaseModel):
    minutes: int


def _gpu_error(exc: Exception) -> HTTPException:
    log.warning("gpu operation failed: %s", exc)
    return _err(502, "gpu_control_failed", f"GPU control failed: {type(exc).__name__}: {str(exc)[:160]}")


@router.get("/api/kaggle-gpu/status")
async def gpu_status(request: Request, refresh: bool = False):
    try:
        return await svc(request).gpu.status(refresh=refresh)
    except Exception as exc:
        raise _gpu_error(exc)


@router.post("/api/kaggle-gpu/turn-on")
async def gpu_on(request: Request):
    s = svc(request)
    try:
        out = await s.gpu.turn_on()
    except Exception as exc:
        raise _gpu_error(exc)
    s.resolver.invalidate()
    return out


@router.post("/api/kaggle-gpu/turn-off")
async def gpu_off(body: OffBody, request: Request):
    s = svc(request)
    if not body.confirm:
        raise _err(422, "confirm_required", "Turning the GPU off stops the model session. Send {\"confirm\": true} to proceed.")
    live = s.runs.active_count()
    try:
        out = await s.gpu.turn_off()
    except Exception as exc:
        raise _gpu_error(exc)
    s.resolver.invalidate()
    out["interrupted_runs"] = live
    return out


@router.get("/api/kaggle-gpu/activity")
async def gpu_activity(request: Request):
    try:
        return await svc(request).gpu.activity()
    except Exception as exc:
        raise _gpu_error(exc)


@router.post("/api/kaggle-gpu/auto-off")
async def gpu_auto_off(body: AutoOffBody, request: Request):
    try:
        return await svc(request).gpu.set_auto_off(body.minutes)
    except ValueError as exc:
        raise _err(422, "invalid", str(exc))
    except Exception as exc:
        raise _gpu_error(exc)


@router.get("/api/kaggle-gpu/logs")
async def gpu_logs(request: Request, limit: int = Query(120, ge=1, le=300)):
    return await svc(request).gpu.logs(limit)
