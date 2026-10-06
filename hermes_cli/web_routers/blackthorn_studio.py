"""Blackthorn Studio — fast, D1-native streaming chat surface.

Why this exists
---------------
The TUI/PTY chat path is turn-based: the prompt goes into a pseudo-terminal,
the agent decides about tools, and only the *finished* assistant text reaches
the browser.  Because Qwen3.8-27B-Uncensored reasons inside
``<think>`` tags, a trivial question cost ~60 s of silence before anything
became visible, even though the model itself answers at ~150 tok/s.

This router talks to the same Kaggle GPU (same weights, same quant, same
reasoning budget — nothing about the model is reduced) but streams the
OpenAI-compatible SSE response straight through to the browser the moment
each token is produced, splitting ``<think>`` reasoning from the final answer
so the UI can show live deliberation exactly like ChatGPT does.

Sessions and messages are persisted ONLY in Cloudflare D1 (``sessions`` +
``messages`` tables), which is also where the long-term memory snippets come
from, so the "D1-only memory & database" requirement still holds.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
import logging
log = logging.getLogger("blackthorn.studio")
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from hermes_cli.web_routers._common import log

router = APIRouter(tags=["blackthorn-studio"])

STUDIO_SOURCE = "blackthorn-studio"
MAX_HISTORY_MESSAGES = 20
MAX_ATTACHMENT_CHARS = 60_000
MAX_ATTACHMENTS = 8
THINK_OPEN = "<think"
THINK_CLOSE = "</think"
TAG_KEEP = 8  # chars held back so a tag split across SSE chunks is not emitted

SYSTEM_PROMPT = (
    "You are Qwen3.8-27B-Uncensored, running on dual Kaggle T4 GPUs and served "
    "through the Blackthorn Hermes Agent dashboard. You are a precise, helpful "
    "assistant.\n"
    "Reasoning rules: think inside <think>...</think> when the task genuinely needs "
    "it, then ALWAYS close the block with </think> and write the final answer "
    "OUTSIDE it — never leave the answer inside the think block. Keep deliberation "
    "short for simple factual or arithmetic questions.\n"
    "Attached files are authoritative source material: when the user asks about an attached "
    "file, read it and quote the exact value you find there, never a guess.\n"
    "Formatting: markdown for structure; fenced code blocks with a language tag."
)


# --------------------------------------------------------------------------- #
# Fast I/O: pooled HTTP clients (a fresh TLS handshake per call used to cost
# several seconds on Render — both to Cloudflare's D1 API and to the tunnel).
# --------------------------------------------------------------------------- #
_MAX_HISTORY = 20

_d1_client: Optional[httpx.AsyncClient] = None
_upstream_client: Optional[httpx.AsyncClient] = None


def _d1_config() -> tuple[str, str]:
    import cloudflare_d1_client as d1  # local import: reads env at import time

    return d1.D1_API_URL, d1.CLOUDFLARE_API_TOKEN


async def _d1q(sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """One D1 statement over a kept-alive TLS connection."""
    global _d1_client
    if _d1_client is None or _d1_client.is_closed:
        _d1_client = httpx.AsyncClient(
            timeout=httpx.Timeout(25.0),
            limits=httpx.Limits(max_keepalive_connections=12, keepalive_expiry=300.0),
        )
    url, token = _d1_config()
    payload: Dict[str, Any] = {"sql": sql}
    if params is not None:
        payload["params"] = list(params)
    resp = await _d1_client.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"Cloudflare D1 error: {data.get('errors')}")
    results = data.get("result") or []
    if results and isinstance(results[0], dict):
        return results[0].get("results") or []
    return []


def _d1q_sync(sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """Blocking D1 statement for thread-pool / non-async call sites."""
    import urllib.request

    url, token = _d1_config()
    payload: Dict[str, Any] = {"sql": sql}
    if params is not None:
        payload["params"] = list(params)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not data.get("success"):
        raise RuntimeError(f"Cloudflare D1 error: {data.get('errors')}")
    results = data.get("result") or []
    if results and isinstance(results[0], dict):
        return results[0].get("results") or []
    return []


def mark_activity_safe(reason: str = "") -> None:
    """Tell the GPU engine that real work is happening (auto-off timer reset)."""
    try:
        import cloudflare_d1_client as d1_mod

        d1_mod.mark_activity(reason)
    except Exception:
        pass


def invalidate_memory_cache() -> None:
    global _memory_cache
    _memory_cache = None


def invalidate_route_cache(_url: str = "") -> None:
    """Drop the memoised GPU endpoint.

    Called whenever the Kaggle tunnel changes, so a chat can never be sent to a
    tunnel that a kernel restart already killed (which surfaced as HTTP 404).
    """
    global _route_cache
    _route_cache = None
    try:
        from cloudflare_d1_client import _STATUS_CACHE_TS  # noqa: F401  (touch to keep in sync)
    except Exception:
        pass


def register_tunnel_hooks() -> None:
    try:
        import cloudflare_d1_client as _d1

        _d1.add_tunnel_change_hook(invalidate_route_cache)
        _d1.add_tunnel_change_hook(lambda _url: invalidate_memory_cache())
    except Exception as exc:  # pragma: no cover
        log.debug("studio: could not register tunnel hooks: %s", exc)


register_tunnel_hooks()


async def _no_history() -> List[Dict[str, Any]]:
    """A brand-new session has no rows — skip the D1 round-trip entirely."""
    return []


async def _tunnel_client() -> httpx.AsyncClient:
    """Persistent client so the tunnel's TLS connection is reused between turns."""
    global _upstream_client
    if _upstream_client is None or _upstream_client.is_closed:
        _upstream_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=900.0, write=60.0, pool=10.0),
            limits=httpx.Limits(max_keepalive_connections=12, keepalive_expiry=600.0),
        )
    return _upstream_client


def _rows(sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """Synchronous D1 helper (kept for the blocking code paths)."""
    return _d1q_sync(sql, params)


_keepalive_task: Optional[asyncio.Task] = None
_KEEPALIVE_SECONDS = 25.0


async def _tunnel_keepalive() -> None:
    """Ping the tunnel on the pooled client so its TCP/TLS session (and the DNS
    entry) stay warm. Without this, every turn paid a cold handshake to
    Cloudflare — ~3-4 s of the user-visible wait before the first token."""
    while True:
        try:
            await asyncio.sleep(_KEEPALIVE_SECONDS)
            rows = await _d1q("SELECT tunnel_url FROM kaggle_gpu_state WHERE id='primary';")
            url = ((rows[0].get("tunnel_url") if rows else "") or "").rstrip("/")
            if not url:
                continue
            client = await _tunnel_client()
            await client.get(f"{url}/health", timeout=8.0)
        except asyncio.CancelledError:
            raise
        except Exception:
            continue


_route_cache: Optional[tuple[float, Dict[str, Any]]] = None
_memory_cache: Optional[tuple[float, List[str]]] = None
_ROUTE_TTL = 15.0
_MEMORY_TTL = 60.0


async def _cached_route() -> Dict[str, Any]:
    """GPU route from D1, memoised — it only changes when the GPU is toggled."""
    global _route_cache
    now = time.time()
    if _route_cache and now - _route_cache[0] < _ROUTE_TTL:
        return _route_cache[1]
    rows = await _d1q("SELECT status, tunnel_url, api_key, model FROM kaggle_gpu_state WHERE id='primary';")
    route = _route_from_row(rows[0] if rows else {})
    if route["url"]:
        _route_cache = (now, route)
    return route


async def _cached_memories() -> List[str]:
    global _memory_cache
    now = time.time()
    if _memory_cache and now - _memory_cache[0] < _MEMORY_TTL:
        return _memory_cache[1]
    rows = await _d1q(
        "SELECT content FROM hermes_memories WHERE target = 'memory' ORDER BY importance DESC LIMIT 12;"
    )
    mem = [str(r.get("content") or "") for r in rows]
    _memory_cache = (now, mem)
    return mem


async def warmup() -> None:
    """Pre-open the D1 + tunnel connections so the first user turn is not cold."""
    _ensure_keepalive()
    try:
        route = await _cached_route()
        await _cached_memories()
        url = route["url"]
        if url:
            client = await _tunnel_client()
            await client.get(f"{url}/health", timeout=8.0)
    except Exception as exc:
        log.debug("studio warmup note: %s", exc)


def _ensure_keepalive() -> None:
    global _keepalive_task
    if _keepalive_task is None or _keepalive_task.done():
        try:
            _keepalive_task = asyncio.get_running_loop().create_task(_tunnel_keepalive())
        except RuntimeError:  # no running loop (sync import) — first request will start it
            _keepalive_task = None


def _now() -> float:
    return time.time()


def _gpu_route() -> Dict[str, Any]:
    """Resolve the live GPU endpoint straight from D1 (no health probe here).

    The permanent GPU daemon keeps this row fresh, so reading it is a single
    fast query instead of a tunnel round-trip on every chat send.
    """
    rows = _rows(
        "SELECT status, tunnel_url, api_key, model FROM kaggle_gpu_state WHERE id='primary';"
    )
    row = rows[0] if rows else {}
    url = (row.get("tunnel_url") or "").rstrip("/")
    status = (row.get("status") or "").upper()
    if not url:
        try:
            live = _d1().get_kaggle_gpu_status()
            url = (live.get("tunnel_url") or "").rstrip("/")
            row = {**row, "api_key": live.get("api_key") or row.get("api_key"),
                   "model": live.get("model") or row.get("model"),
                   "status": live.get("status") or status}
            status = (row.get("status") or "").upper()
        except Exception as exc:
            log.warning("studio: live GPU status lookup failed: %s", exc)
    # Never hand a stale tunnel to the agent when the GPU is explicitly stopped.
    offline = status in {
        "GPU_STOPPED_SAVING_QUOTA", "STOPPED", "OFF", "OFFLINE", "IDLE_OFF",
        "ERROR", "FAILED", "DEAD",
    }
    if offline:
        url = ""
    return {
        "url": url,
        "api_key": row.get("api_key") or "sk-qwen38-27b-uncensored-7f8a9d2e4b1c6f3a",
        "model": row.get("model") or "Qwen3.8-27B-Uncensored",
        "status": status or "UNKNOWN",
    }


def _memory_lines(limit: int = 12) -> List[str]:
    try:
        entries = _d1().d1_get_memory_entries("memory") or []
    except Exception:
        return []
    return [str(e).strip() for e in entries if str(e).strip()][:limit]


def _read_attachment(request: Request, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolve a file-browser path to readable text using the managed-file policy."""
    path = str(item.get("path") or "").strip()
    if not path:
        return None
    try:
        from pathlib import Path as _Path

        from hermes_cli.web_routers.files import _is_sensitive_path, _resolve_managed_path

        try:
            _policy, target, display_path = _resolve_managed_path(path, request)
        except HTTPException:
            # Relative paths ("uploads/x.txt") are resolved against the managed root.
            if _Path(path).is_absolute():
                raise
            _root_policy, root, _root_display = _resolve_managed_path("", request)
            _policy, target, display_path = _resolve_managed_path(str(_Path(root) / path), request)
        if not target.is_file() or _is_sensitive_path(target):
            return None
        size = target.stat().st_size
        if size > 8 * 1024 * 1024:
            return {"path": display_path, "name": target.name, "note": "file too large to inline"}
        blob = target.read_bytes()
        try:
            text = blob.decode("utf-8")
            binary = False
        except UnicodeDecodeError:
            text = blob.decode("latin-1", errors="replace")
            binary = True
        clipped = text[:MAX_ATTACHMENT_CHARS]
        return {
            "path": display_path,
            "name": target.name,
            "bytes": size,
            "binary": binary,
            "truncated": len(text) > len(clipped),
            "text": clipped,
        }
    except HTTPException:
        return None
    except Exception as exc:
        log.warning("studio: attachment read failed for %s: %s", path, exc)
        return None


def _route_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    status = (row.get("status") or "UNKNOWN").upper()
    url = (row.get("tunnel_url") or "").rstrip("/")
    offline = status in {
        "GPU_STOPPED_SAVING_QUOTA", "STOPPED", "OFF", "OFFLINE", "IDLE_OFF",
        "ERROR", "FAILED", "DEAD",
    }
    if offline:
        url = ""
    return {
        "url": url,
        "api_key": row.get("api_key") or "sk-qwen38-27b-uncensored-7f8a9d2e4b1c6f3a",
        "model": row.get("model") or "Qwen3.8-27B-Uncensored",
        "status": status,
    }


def _compose_system(memories: List[str], attachments: List[Dict[str, Any]]) -> str:
    custom = ""
    try:
        rows = _rows(
            "SELECT value FROM state_meta WHERE key = ? LIMIT 1;",
            ["blackthorn_system_prompt"],
        )
        if rows:
            custom = str(rows[0].get("value") or "").strip()
    except Exception:
        custom = ""
    parts = [custom] if custom else [SYSTEM_PROMPT]
    if custom:
        # Keep core formatting rules even when user overrides persona
        parts.append(
            "Formatting: markdown for structure; fenced code blocks with a language tag. "
            "Only call tools when the user request genuinely requires external data or execution; "
            "never for greetings, thanks, or capability questions."
        )
    memories = [m.strip() for m in memories if str(m).strip()][:12]
    if memories:
        parts.append("Long-term memory (Cloudflare D1):\n" + "\n".join(f"- {m}" for m in memories))
    for att in attachments:
        if att.get("text"):
            note = " (truncated)" if att.get("truncated") else ""
            parts.append(
                f"Attached file `{att['name']}` at {att['path']}{note}:\n```\n{att['text']}\n```"
            )
        elif att.get("note"):
            parts.append(f"Attached file `{att.get('name')}` at {att.get('path')}: {att['note']}.")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class StudioSessionCreate(BaseModel):
    title: Optional[str] = None


class Attachment(BaseModel):
    path: str
    name: Optional[str] = None


class StudioChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    attachments: List[Attachment] = []
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# --------------------------------------------------------------------------- #
# Session + history endpoints
# --------------------------------------------------------------------------- #
@router.get("/api/studio/status")
async def studio_status(request: Request):
    """Fast status for the chat UI — never block on dead tunnels."""
    _ensure_keepalive()
    rows = await _d1q("SELECT status, tunnel_url, api_key, model FROM kaggle_gpu_state WHERE id='primary';")
    route = _route_from_row(rows[0] if rows else {})
    probe: Dict[str, Any] = {"reachable": None, "latency_ms": None}
    status_up = (route.get("status") or "").upper() in (
        "ON", "READY", "RUNNING", "LIVE", "ONLINE", "CONNECTED", "GPU_READY", "ACTIVE"
    )
    # Only probe when GPU is marked ON — avoids 1.5s hang on stale tunnels
    if status_up and route.get("url"):
        try:
            t0 = time.perf_counter()
            client = await _tunnel_client()
            resp = await client.get(f"{route['url']}/health", timeout=0.8)
            probe = {
                "reachable": resp.status_code == 200,
                "latency_ms": round((time.perf_counter() - t0) * 1000),
            }
        except Exception as exc:
            probe = {"reachable": False, "latency_ms": None, "error": str(exc)[:120]}
    elif not status_up:
        probe = {"reachable": False, "latency_ms": None, "skipped": True}
    try:
        mem = await _d1q("SELECT COUNT(*) AS n FROM hermes_memories WHERE target = 'memory';")
        memories = int(mem[0]["n"]) if mem else 0
    except Exception:
        memories = 0
    return {
        "model": route["model"],
        "status": route["status"],
        "endpoint": route["url"] if status_up else None,
        "probe": probe,
        "memories": memories,
    }



_sessions_schema_ready = False

async def _ensure_sessions_archived_col() -> None:
    global _sessions_schema_ready
    if _sessions_schema_ready:
        return
    try:
        await _d1q("ALTER TABLE sessions ADD COLUMN archived INTEGER NOT NULL DEFAULT 0;")
    except Exception:
        pass
    _sessions_schema_ready = True

@router.get("/api/studio/sessions")
async def list_sessions(limit: int = 60):
    """Always return 200 with a list — never 500 the chat history sidebar."""
    out: List[Dict[str, Any]] = []
    try:
        await _ensure_sessions_archived_col()
        rows = await _d1q(
            """
            SELECT id, display_name, started_at, message_count, model
              FROM sessions
             WHERE source = ?
               AND COALESCE(archived, 0) = 0
             ORDER BY started_at DESC
             LIMIT ?;
            """,
            [STUDIO_SOURCE, int(limit)],
        )
        for r in rows or []:
            title = (r.get("display_name") or r.get("title") or "").strip()
            out.append(
                {
                    "id": r["id"],
                    "title": title or "New chat",
                    "preview": title[:140],
                    "started_at": r.get("started_at"),
                    "last_at": r.get("started_at"),
                    "message_count": r.get("message_count") or 0,
                    "model": r.get("model") or "",
                }
            )
    except Exception as exc:
        log.warning("list_sessions failed: %s", exc)
    return {"sessions": out}


@router.post("/api/studio/sessions")
async def create_session(payload: StudioSessionCreate):
    sid = f"studio-{uuid.uuid4().hex[:12]}"
    route = _gpu_route()
    _rows(
        "INSERT INTO sessions (id, source, created_source, session_key, display_name, model, "
        "started_at, message_count, tool_call_count) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0);",
        [sid, STUDIO_SOURCE, STUDIO_SOURCE, sid, (payload.title or "").strip(), route["model"], _now()],
    )
    return {"id": sid, "title": (payload.title or "").strip() or "New chat"}


@router.delete("/api/studio/sessions/{session_id}")
async def delete_session(session_id: str):
    """Permanently remove a session and all of its messages from D1."""
    sid = (session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id required")
    try:
        await _d1q("DELETE FROM messages WHERE session_id = ?;", [sid])
    except Exception:
        _rows("DELETE FROM messages WHERE session_id = ?;", [sid])
    try:
        await _d1q("DELETE FROM sessions WHERE id = ?;", [sid])
    except Exception:
        _rows("DELETE FROM sessions WHERE id = ?;", [sid])
    return {"ok": True, "deleted": sid}




@router.post("/api/studio/sessions/{session_id}/archive")
async def archive_session(session_id: str):
    """Mark a studio session archived so it leaves the active history list."""
    try:
        await _d1q("ALTER TABLE sessions ADD COLUMN archived INTEGER NOT NULL DEFAULT 0;")
    except Exception:
        pass
    await _d1q(
        "UPDATE sessions SET archived = 1 WHERE id = ? AND source = ?;",
        [session_id, STUDIO_SOURCE],
    )
    return {"ok": True, "archived": True}


@router.get("/api/studio/sessions/archived")
async def list_archived_sessions(limit: int = 60):
    try:
        rows = await _d1q(
            """
            SELECT id, display_name, started_at, message_count, model
              FROM sessions
             WHERE source = ? AND COALESCE(archived, 0) = 1
             ORDER BY started_at DESC
             LIMIT ?;
            """,
            [STUDIO_SOURCE, int(limit)],
        )
    except Exception:
        rows = []
    out = []
    for r in rows or []:
        title = (r.get("display_name") or "").strip()
        out.append({
            "id": r["id"],
            "title": title or "Archived chat",
            "preview": title[:140],
            "started_at": r.get("started_at"),
            "message_count": r.get("message_count") or 0,
            "model": r.get("model") or "",
            "archived": True,
        })
    return {"sessions": out}


@router.get("/api/studio/sessions/{session_id}/messages")
async def session_messages(session_id: str, limit: int = 200):
    try:
        rows = _rows(
            """
            SELECT role, content, timestamp, token_count, finish_reason
              FROM messages
             WHERE session_id = ? AND role IN ('user','assistant')
             ORDER BY timestamp ASC
             LIMIT ?;
            """,
            [session_id, int(limit)],
        )
    except Exception:
        rows = []
    return {
        "messages": [
            {
                "role": r.get("role"),
                "content": r.get("content") or "",
                "reasoning": "",
                "timestamp": r.get("timestamp"),
                "token_count": r.get("token_count"),
                "finish_reason": r.get("finish_reason"),
            }
            for r in (rows or [])
        ]
    }


# --------------------------------------------------------------------------- #
# Streaming chat
# --------------------------------------------------------------------------- #
def _sse(event: Dict[str, Any]) -> bytes:
    return f"data: {json.dumps(event)}\n\n".encode("utf-8")


class _ThinkSplitter:
    """Incrementally route model output into reasoning vs. answer channels."""

    def __init__(self) -> None:
        self.buf = ""
        self.in_think = False

    def feed(self, text: str) -> List[tuple[str, str]]:
        self.buf += text
        events: List[tuple[str, str]] = []
        while True:
            if self.in_think:
                idx = self.buf.find(THINK_CLOSE)
                if idx == -1:
                    if len(self.buf) > TAG_KEEP:
                        events.append(("reasoning", self.buf[:-TAG_KEEP]))
                        self.buf = self.buf[-TAG_KEEP:]
                    break
                events.append(("reasoning", self.buf[:idx]))
                self.buf = self.buf[idx + len(THINK_CLOSE):]
                if self.buf.startswith(">"):
                    self.buf = self.buf[1:]
                self.in_think = False
            else:
                idx = self.buf.find(THINK_OPEN)
                if idx == -1:
                    if len(self.buf) > TAG_KEEP:
                        events.append(("content", self.buf[:-TAG_KEEP]))
                        self.buf = self.buf[-TAG_KEEP:]
                    break
                events.append(("content", self.buf[:idx]))
                self.buf = self.buf[idx + len(THINK_OPEN):]
                if self.buf.startswith(">"):
                    self.buf = self.buf[1:]
                self.in_think = True
        return [(kind, chunk) for kind, chunk in events if chunk]

    def flush(self) -> List[tuple[str, str]]:
        kind = "reasoning" if self.in_think else "content"
        rest, self.buf = self.buf, ""
        return [(kind, rest)] if rest else []


@router.post("/api/studio/chat/stream")
async def studio_chat_stream(payload: StudioChatRequest, request: Request):
    """Stream a turn to the browser.

    Hot-path order matters: everything that can run in parallel does, and the
    two D1 writes (user row + session counters) are fired off without blocking
    the model call — on Render each un-pooled D1 round-trip cost ~2 s, which is
    exactly the delay users saw before the first token.
    """
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty")

    _ensure_keepalive()
    mark_activity_safe("chat-turn")
    t_start = time.perf_counter()
    session_id = (payload.session_id or "").strip() or f"studio-{uuid.uuid4().hex[:12]}"

    # ── parallel pre-work: one round-trip for everything we must know ──
    is_new_session = not (payload.session_id or "").strip()
    route_task = _cached_route()
    memory_task = _cached_memories()
    history_task = (
        _no_history()
        if is_new_session
        else _d1q(
            "SELECT role, content FROM messages WHERE session_id = ? AND role IN ('user','assistant') "
            "AND content IS NOT NULL AND content != '' ORDER BY timestamp DESC LIMIT ?;",
            [session_id, MAX_HISTORY_MESSAGES],
        )
    )
    attach_task = asyncio.gather(
        *[asyncio.to_thread(_read_attachment, request, {"path": a.path, "name": a.name})
          for a in payload.attachments[:MAX_ATTACHMENTS]]
    )

    route, history, memory_rows, attachments = await asyncio.gather(
        route_task, history_task, memory_task, attach_task
    )
    attachments = [a for a in attachments if a]
    if not route["url"]:
        raise HTTPException(
            status_code=503,
            detail="Kaggle GPU is OFF. Turn it on with the GPU button in the header, then retry.",
        )

    history.reverse()
    system_prompt = _compose_system(memory_rows, attachments)

    user_note = ""
    if attachments:
        names = ", ".join(a.get("name") or a.get("path") or "file" for a in attachments)
        user_note = f"\n\n[Attached: {names}]"
    stored_user_text = message + user_note

    upstream_messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    upstream_messages += [{"role": m["role"], "content": m.get("content") or ""} for m in history]
    upstream_messages.append({"role": "user", "content": message})

    body: Dict[str, Any] = {"model": route["model"], "messages": upstream_messages, "stream": True}
    if payload.temperature is not None:
        body["temperature"] = payload.temperature
    if payload.max_tokens:
        body["max_tokens"] = payload.max_tokens

    headers = {"Authorization": f"Bearer {route['api_key']}", "Content-Type": "application/json"}
    url = f"{route['url']}/v1/chat/completions"
    prework_ms = int((time.perf_counter() - t_start) * 1000)

    async def _persist_turn(answer: str, reasoning: str, out_tokens: Optional[int], finish_reason: str) -> None:
        """Background writes: they must never delay a token on the wire."""
        try:
            await _d1q(
                "INSERT INTO sessions (id, source, created_source, session_key, display_name, model, "
                "started_at, message_count, tool_call_count) "
                "SELECT ?, ?, ?, ?, '', ?, ?, 0, 0 WHERE NOT EXISTS (SELECT 1 FROM sessions WHERE id = ?);",
                [session_id, STUDIO_SOURCE, STUDIO_SOURCE, session_id, route["model"], _now(), session_id],
            )
            await _d1q(
                "INSERT INTO messages (session_id, role, content, timestamp, active, _compressed_summary) "
                "VALUES (?, 'user', ?, ?, 1, 0);",
                [session_id, stored_user_text, _now()],
            )
            await _d1q(
                "UPDATE sessions SET message_count = COALESCE(message_count, 0) + 1, "
                "display_name = CASE WHEN COALESCE(display_name, '') = '' THEN ? ELSE display_name END "
                "WHERE id = ?;",
                [message.replace("\n", " ")[:60], session_id],
            )
            if answer or reasoning:
                await _d1q(
                    "INSERT INTO messages (session_id, role, content, reasoning, timestamp, token_count, "
                    "finish_reason, active, _compressed_summary) VALUES (?, 'assistant', ?, ?, ?, ?, ?, 1, 0);",
                    [session_id, answer, reasoning, _now(), out_tokens or 0, finish_reason],
                )
                await _d1q(
                    "UPDATE sessions SET message_count = COALESCE(message_count, 0) + 1 WHERE id = ?;",
                    [session_id],
                )
        except Exception as exc:  # pragma: no cover
            log.warning("studio: background persistence failed: %s", exc)

    async def event_stream():
        t0 = time.perf_counter()
        first_token_ms: Optional[int] = None
        first_content_ms: Optional[int] = None
        upstream_ms: Optional[int] = None
        answer_parts: List[str] = []
        reasoning_parts: List[str] = []
        usage: Dict[str, Any] = {}
        finish_reason = "stop"
        splitter = _ThinkSplitter()

        yield _sse(
            {
                "type": "meta",
                "session_id": session_id,
                "model": route["model"],
                "endpoint": route["url"],
                "attachments": [a.get("name") or a.get("path") for a in attachments],
                "prework_ms": prework_ms,
                "started_at": t0,
            }
        )

        def emit(kind: str, text: str):
            nonlocal first_token_ms, first_content_ms
            if not text:
                return None
            now = time.perf_counter()
            if first_token_ms is None:
                first_token_ms = int((now - t0) * 1000) + prework_ms
            if kind == "content":
                answer_parts.append(text)
                if first_content_ms is None:
                    first_content_ms = int((now - t0) * 1000) + prework_ms
            else:
                reasoning_parts.append(text)
            # Wire contract: reasoning deltas and answer deltas ("delta").
            return _sse({"type": "reasoning" if kind == "reasoning" else "delta", "delta": text})

        try:
            client = await _tunnel_client()
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                upstream_ms = prework_ms + int((time.perf_counter() - t0) * 1000)
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode("utf-8", "replace")[:400]
                    yield _sse(
                        {"type": "error", "message": f"GPU endpoint returned HTTP {resp.status_code}: {detail}"}
                    )
                    return
                async for raw_line in resp.aiter_lines():
                    if not raw_line or not raw_line.startswith("data:"):
                        continue
                    data = raw_line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj.get("usage"), dict):
                        usage = obj["usage"]
                    choice = (obj.get("choices") or [{}])[0]
                    if choice.get("finish_reason"):
                        finish_reason = str(choice["finish_reason"])
                    delta = choice.get("delta") or {}
                    reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if reasoning_delta:
                        chunk = emit("reasoning", reasoning_delta)
                        if chunk:
                            yield chunk
                    piece = delta.get("content") or ""
                    if piece:
                        for kind, text in splitter.feed(piece):
                            chunk = emit(kind, text)
                            if chunk:
                                yield chunk
        except asyncio.CancelledError:
            finish_reason = "cancelled"
            for kind, text in splitter.flush():
                chunk = emit(kind, text)
                if chunk:  # keep partial text for the client that is still there
                    pass
            asyncio.create_task(
                _persist_turn("".join(answer_parts).strip(), "".join(reasoning_parts).strip(), None, finish_reason)
            )
            raise
        except Exception as exc:
            log.warning("studio: upstream stream failed: %s", exc)
            for kind, text in splitter.flush():
                chunk = emit(kind, text)
                if chunk:
                    yield chunk
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            asyncio.create_task(
                _persist_turn(
                    "".join(answer_parts).strip(), "".join(reasoning_parts).strip(), None, "error"
                )
            )
            return
        else:
            for kind, text in splitter.flush():
                chunk = emit(kind, text)
                if chunk:
                    yield chunk

        answer = "".join(answer_parts).strip()
        reasoning = "".join(reasoning_parts).strip()
        duration_ms = prework_ms + int((time.perf_counter() - t0) * 1000)
        out_tokens = usage.get("completion_tokens")
        if not out_tokens:
            out_tokens = max(1, round((len(answer) + len(reasoning)) / 4))
        tok_per_s = round(out_tokens / max(duration_ms / 1000.0, 0.001), 1)

        if not answer and reasoning:
            answer = reasoning
            yield _sse({"type": "answer_fallback", "content": reasoning})

        yield _sse(
            {
                "type": "done",
                "session_id": session_id,
                "first_token_ms": first_token_ms,
                "first_content_ms": first_content_ms,
                "upstream_ms": upstream_ms,
                "prework_ms": prework_ms,
                "duration_ms": duration_ms,
                "output_tokens": out_tokens,
                "tokens_per_second": tok_per_s,
                "finish_reason": finish_reason,
                "usage": usage,
            }
        )
        asyncio.create_task(_persist_turn(answer, reasoning, out_tokens, finish_reason))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
