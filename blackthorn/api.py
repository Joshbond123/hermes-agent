"""HTTP API of the Blackthorn product (mounted by ``hermes_cli.web_server``).

Chat
  POST /api/studio/agent/stream             start a turn and stream it (SSE)
  GET  /api/studio/turns/{id}/stream?after  re-attach to a running/finished turn (SSE)
  POST /api/studio/turns/{id}/cancel        stop generation for real
History
  GET    /api/studio/sessions               list (pinned first, then recent activity)
  GET    /api/studio/sessions/{id}          one chat (+ the turn that is still running, if any)
  PATCH  /api/studio/sessions/{id}          rename / pin / archive (persisted in D1)
  DELETE /api/studio/sessions/{id}
  GET    /api/studio/sessions/{id}/messages
Settings / GPU / build
  GET|PUT /api/system-prompt
  GET     /api/kaggle-gpu/status            instant (served from the supervisor's snapshot)
  POST    /api/kaggle-gpu/{turn-on,turn-off,restart,verify,auto-off}, GET .../logs
  GET     /api/blackthorn/version
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from blackthorn import attachments as attachments_mod
from blackthorn import config, sessions, version
from blackthorn import http as bt_http
from blackthorn.agent.builtin_tools import Deps, SearchKeys, build_registry
from blackthorn.agent.engine import Engine, EngineConfig, Turn, TurnRequest
from blackthorn.agent.kaggle_computer import KaggleComputer
from blackthorn.d1 import D1Error, aget_meta, aset_meta
from blackthorn.gpu import controller
from blackthorn.store import SYSTEM_PROMPT_KEY, D1Store

log = logging.getLogger("blackthorn.api")
router = APIRouter(tags=["blackthorn"])

MAX_MESSAGE_CHARS = 32_000
MAX_SYSTEM_PROMPT_CHARS = 8_000
_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        computer = KaggleComputer(controller.route, on_tunnel_failure=controller.report_tunnel_failure)
        registry = build_registry(Deps(computer=computer, search_keys=SearchKeys(), http_client=bt_http.client))
        _engine = Engine(
            route_provider=controller.route,
            registry=registry,
            store=D1Store(),
            client_factory=bt_http.client,
            on_activity=lambda busy: controller.task_started() if busy else controller.task_finished(),
            on_tunnel_failure=controller.report_tunnel_failure,
            config=EngineConfig(),
        )
    return _engine


def set_engine(engine: Optional[Engine]) -> None:  # used by tests
    global _engine
    _engine = engine


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #
class AttachmentRef(BaseModel):
    path: str = Field(min_length=1, max_length=1024)
    name: Optional[str] = Field(default=None, max_length=255)


class StreamBody(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(default="", max_length=MAX_MESSAGE_CHARS)
    regenerate: bool = False
    thinking: bool = False
    attachments: List[AttachmentRef] = Field(default_factory=list, max_length=attachments_mod.MAX_ATTACHMENTS)
    max_steps: Optional[int] = Field(default=None, ge=1, le=24)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)


def _check_session_id(value: Optional[str]) -> Optional[str]:
    if value is None or value == "":
        return None
    if not _SESSION_ID.match(value):
        raise HTTPException(status_code=400, detail="invalid session id")
    return value


async def _sse(turn: Turn, after: int) -> AsyncIterator[bytes]:
    # a padded first frame defeats proxies that hold back small responses
    yield b"retry: 2000\n: " + b" " * 2048 + b"\n\n"
    async for ev in turn.subscribe(after):
        yield b": ping\n\n" if ev is None else ev.sse()


def _stream_response(turn: Turn, after: int) -> StreamingResponse:
    return StreamingResponse(_sse(turn, after), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/api/studio/agent/stream")
async def agent_stream(body: StreamBody, request: Request):
    session_id = _check_session_id(body.session_id)
    message = body.message.strip()
    if body.regenerate:
        if not session_id:
            raise HTTPException(status_code=400, detail="regenerate needs a session_id")
    elif not message:
        raise HTTPException(status_code=400, detail="message is empty")
    resolved: List[Dict[str, Any]] = []
    if body.attachments:
        resolved = await asyncio.to_thread(
            attachments_mod.load, request, [a.model_dump() for a in body.attachments])
    controller.mark_activity()
    turn = await get_engine().start(TurnRequest(
        session_id=session_id, message=message, regenerate=body.regenerate, thinking=body.thinking,
        attachments=resolved, max_steps=body.max_steps, temperature=body.temperature))
    return _stream_response(turn, 0)


@router.get("/api/studio/turns/{turn_id}/stream")
async def reattach(turn_id: str, after: int = Query(0, ge=0), last_event_id: Optional[str] = Header(None)):
    turn = get_engine().get(turn_id)
    if turn is None:
        raise HTTPException(status_code=404, detail="turn not found (it finished long ago — reload the chat)")
    if last_event_id and last_event_id.isdigit():
        after = max(after, int(last_event_id))
    return _stream_response(turn, after)


@router.post("/api/studio/turns/{turn_id}/cancel")
async def cancel_turn(turn_id: str):
    cancelled = await get_engine().cancel(turn_id)
    return {"ok": True, "cancelled": cancelled}


# --------------------------------------------------------------------------- #
# history
# --------------------------------------------------------------------------- #
class SessionPatch(BaseModel):
    title: Optional[str] = Field(default=None, max_length=400)
    pinned: Optional[bool] = None
    archived: Optional[bool] = None


@router.get("/api/studio/sessions")
async def list_sessions(archived: bool = False, q: str = Query("", max_length=200), limit: int = Query(100, ge=1, le=300)):
    try:
        return {"sessions": await sessions.list_sessions(archived=archived, query=q, limit=limit)}
    except D1Error as exc:
        log.warning("list_sessions: %s", exc)
        raise HTTPException(status_code=503, detail="History is temporarily unavailable (database unreachable).")


@router.get("/api/studio/sessions/{session_id}")
async def get_session(session_id: str):
    _check_session_id(session_id)
    session = await sessions.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="chat not found")
    turn = get_engine().active_turn(session_id)
    active = {"turn_id": turn.id, "assistant_message_id": turn.assistant_message_id, "last_seq": len(turn.events)} if turn else None
    return {"session": session, "active_turn": active}


@router.patch("/api/studio/sessions/{session_id}")
async def patch_session(session_id: str, body: SessionPatch):
    _check_session_id(session_id)
    if body.title is None and body.pinned is None and body.archived is None:
        raise HTTPException(status_code=400, detail="nothing to change")
    try:
        return {"session": await sessions.update_session(
            session_id, title=body.title, pinned=body.pinned, archived=body.archived)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except sessions.SessionNotFound:
        raise HTTPException(status_code=404, detail="chat not found")


@router.delete("/api/studio/sessions/{session_id}")
async def delete_session(session_id: str):
    _check_session_id(session_id)
    turn = get_engine().active_turn(session_id)
    if turn is not None:
        await get_engine().cancel(turn.id)
        if turn.task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(turn.task), timeout=6.0)
            except Exception:  # noqa: BLE001
                pass
    if not await sessions.delete_session(session_id):
        raise HTTPException(status_code=404, detail="chat not found")
    return {"ok": True, "deleted": session_id}


@router.get("/api/studio/sessions/{session_id}/messages")
async def session_messages(session_id: str, limit: int = Query(500, ge=1, le=2000)):
    _check_session_id(session_id)
    if await sessions.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="chat not found")
    return {"messages": await sessions.list_messages(session_id, limit=limit)}


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
class PromptBody(BaseModel):
    prompt: str = Field(default="", max_length=MAX_SYSTEM_PROMPT_CHARS)


@router.get("/api/system-prompt")
async def get_system_prompt():
    from blackthorn.agent.prompts import DEFAULT_PERSONA

    value = await aget_meta(SYSTEM_PROMPT_KEY, "")
    return {"prompt": value, "default": DEFAULT_PERSONA, "max_chars": MAX_SYSTEM_PROMPT_CHARS}


@router.put("/api/system-prompt")
async def put_system_prompt(body: PromptBody):
    await aset_meta(SYSTEM_PROMPT_KEY, body.prompt.strip())
    return {"ok": True, "prompt": body.prompt.strip()}


@router.get("/api/studio/status")
async def studio_status():
    snap = controller.snapshot()
    return {"model": snap.get("model") or config.model_alias(), "status": snap.get("state"),
            "ready": snap.get("state") == "ready"}


@router.get("/api/blackthorn/version")
async def build_version():
    return version.info()


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #
class AutoOff(BaseModel):
    minutes: int


@router.get("/api/kaggle-gpu/status")
async def gpu_status(fresh: bool = False):
    if fresh or controller.snapshot().get("state") == "unknown":
        await controller.refresh(force=bool(fresh))
    return controller.snapshot()


@router.post("/api/kaggle-gpu/turn-on")
async def gpu_turn_on():
    return await controller.turn_on()


@router.post("/api/kaggle-gpu/turn-off")
async def gpu_turn_off():
    return await controller.turn_off(reason="user")


@router.post("/api/kaggle-gpu/restart")
async def gpu_restart():
    return await controller.restart()


@router.post("/api/kaggle-gpu/verify")
async def gpu_verify():
    return {"result": await controller.verify_inference(), "state": controller.snapshot().get("state")}


@router.post("/api/kaggle-gpu/auto-off")
async def gpu_auto_off(body: AutoOff):
    try:
        minutes = await controller.set_auto_off_minutes(body.minutes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    await controller.refresh(force=True)
    return {"ok": True, "auto_off_minutes": minutes, "gpu": controller.snapshot()}


@router.get("/api/kaggle-gpu/logs")
async def gpu_logs(limit: int = Query(120, ge=1, le=400)):
    return await controller.logs(limit)
