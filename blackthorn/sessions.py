"""Conversation storage (Cloudflare D1): sessions + messages.

Everything the UI shows about history — titles, pinned/archived flags, ordering,
message bodies and tool activity — is read from and written to these D1 rows.  No
state lives only in the browser.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from blackthorn import config
from blackthorn.d1 import D1Result, d1

TITLE_MAX = 120
AUTO_TITLE_MAX = 60
#: Marker stored in ``messages.finish_reason`` while a turn is still generating.
RUNNING = "running"


class SessionNotFound(LookupError):
    pass


class SessionConflict(RuntimeError):
    """The id exists but belongs to a different application (not a Blackthorn chat)."""


def now() -> float:
    return time.time()


def new_session_id() -> str:
    return f"studio-{uuid.uuid4().hex[:12]}"


def auto_title(text: str) -> str:
    one_line = re.sub(r"\s+", " ", (text or "").strip())
    if len(one_line) <= AUTO_TITLE_MAX:
        return one_line or "New chat"
    cut = one_line[:AUTO_TITLE_MAX].rsplit(" ", 1)[0] or one_line[:AUTO_TITLE_MAX]
    return cut.rstrip(" ,.;:-") + "…"


def clean_title(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", (title or "").strip())
    if not cleaned:
        raise ValueError("Title must not be empty")
    return cleaned[:TITLE_MAX]


def _row_to_session(row: Dict[str, Any]) -> Dict[str, Any]:
    title = (row.get("title") or row.get("display_name") or "").strip()
    return {
        "id": row["id"],
        "title": title or "New chat",
        "pinned": bool(row.get("pinned")),
        "archived": bool(row.get("archived")),
        "created_at": row.get("started_at"),
        "updated_at": row.get("updated_at") or row.get("last_activity_at") or row.get("started_at"),
        "message_count": int(row.get("message_count") or 0),
        "model": row.get("model") or "",
    }


_SESSION_COLUMNS = (
    "id, title, display_name, pinned, archived, started_at, last_activity_at, "
    "COALESCE(last_activity_at, started_at) AS updated_at, message_count, model"
)


# --------------------------------------------------------------------------- #
# sessions
# --------------------------------------------------------------------------- #
async def list_sessions(*, archived: bool = False, limit: int = 100, query: str = "") -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit), 300))
    where = ["source = ?", "COALESCE(archived, 0) = ?", "COALESCE(hidden, 0) = 0"]
    params: List[Any] = [config.session_source(), 1 if archived else 0]
    q = (query or "").strip()
    if q:
        like = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        where.append(
            "(COALESCE(title, display_name, '') LIKE ? ESCAPE '\\' OR EXISTS ("
            "SELECT 1 FROM messages m WHERE m.session_id = sessions.id "
            "AND m.role IN ('user','assistant') AND m.content LIKE ? ESCAPE '\\'))"
        )
        params += [like, like]
    sql = (
        f"SELECT {_SESSION_COLUMNS} FROM sessions WHERE {' AND '.join(where)} "
        "ORDER BY COALESCE(pinned, 0) DESC, COALESCE(last_activity_at, started_at) DESC LIMIT ?"
    )
    params.append(limit)
    rows = await d1.aquery(sql, params)
    return [_row_to_session(r) for r in rows]


async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    rows = await d1.aquery(
        f"SELECT {_SESSION_COLUMNS} FROM sessions WHERE id = ? AND source = ? LIMIT 1",
        [session_id, config.session_source()],
    )
    return _row_to_session(rows[0]) if rows else None


async def update_session(
    session_id: str,
    *,
    title: Optional[str] = None,
    pinned: Optional[bool] = None,
    archived: Optional[bool] = None,
) -> Dict[str, Any]:
    """Persist a rename / pin / archive change and return the stored row."""
    sets: List[str] = []
    params: List[Any] = []
    if title is not None:
        cleaned = clean_title(title)
        sets += ["title = ?", "display_name = ?", "title_source = 'user'"]
        params += [cleaned, cleaned]
    if archived is not None:
        sets.append("archived = ?")
        params.append(1 if archived else 0)
        if archived:
            sets.append("pinned = 0")  # an archived chat is never pinned
    if pinned is not None and not (archived is True):
        sets.append("pinned = ?")
        params.append(1 if pinned else 0)
    if not sets:
        existing = await get_session(session_id)
        if not existing:
            raise SessionNotFound(session_id)
        return existing
    results = await d1.abatch(
        [
            (f"UPDATE sessions SET {', '.join(sets)} WHERE id = ? AND source = ?",
             params + [session_id, config.session_source()]),
            (f"SELECT {_SESSION_COLUMNS} FROM sessions WHERE id = ? AND source = ? LIMIT 1",
             [session_id, config.session_source()]),
        ]
    )
    rows = results[1].rows
    if not rows:
        raise SessionNotFound(session_id)
    return _row_to_session(rows[0])


async def delete_session(session_id: str) -> bool:
    source = config.session_source()
    results = await d1.abatch(
        [
            ("DELETE FROM messages WHERE session_id = ? AND EXISTS "
             "(SELECT 1 FROM sessions WHERE id = ? AND source = ?)", [session_id, session_id, source]),
            ("DELETE FROM sessions WHERE id = ? AND source = ?", [session_id, source]),
        ]
    )
    return results[1].changes > 0


# --------------------------------------------------------------------------- #
# messages
# --------------------------------------------------------------------------- #
def _loads(text: Any) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _row_to_message(row: Dict[str, Any]) -> Dict[str, Any]:
    meta = _loads(row.get("display_metadata")) or {}
    return {
        "id": row["id"],
        "role": row["role"],
        "content": row.get("content") or "",
        "created_at": row.get("timestamp"),
        "finish_reason": row.get("finish_reason"),
        "parts": meta.get("parts") if isinstance(meta, dict) else None,
        "usage": meta.get("usage") if isinstance(meta, dict) else None,
        "duration_ms": meta.get("duration_ms") if isinstance(meta, dict) else None,
        "thinking_ms": meta.get("thinking_ms") if isinstance(meta, dict) else None,
        "error": meta.get("error") if isinstance(meta, dict) else None,
        "attachments": meta.get("attachments") if isinstance(meta, dict) else None,
    }


async def list_messages(session_id: str, *, limit: int = 500) -> List[Dict[str, Any]]:
    rows = await d1.aquery(
        "SELECT id, role, content, timestamp, finish_reason, display_metadata FROM messages "
        "WHERE session_id = ? AND role IN ('user','assistant') AND COALESCE(active, 1) = 1 "
        "AND EXISTS (SELECT 1 FROM sessions WHERE id = ? AND source = ?) "
        "ORDER BY timestamp ASC, id ASC LIMIT ?",
        [session_id, session_id, config.session_source(), max(1, min(int(limit), 2000))],
    )
    return [_row_to_message(r) for r in rows]


def history_statement(session_id: str, before_ts: float, limit: int = 40):
    """SELECT for the model context (excludes the turn being created)."""
    return (
        "SELECT role, content FROM messages WHERE session_id = ? AND role IN ('user','assistant') "
        "AND COALESCE(active, 1) = 1 AND content IS NOT NULL AND content != '' AND timestamp < ? "
        "ORDER BY timestamp DESC, id DESC LIMIT ?",
        [session_id, before_ts, int(limit)],
    )


def _msg_insert() -> str:
    """Insert a message only if the session exists *and* belongs to Blackthorn."""
    return (
        "INSERT INTO messages (session_id, role, content, timestamp, active, _compressed_summary, "
        "finish_reason, message_uid, display_metadata) "
        "SELECT ?, ?, ?, ?, 1, 0, ?, ?, ? "
        "WHERE EXISTS (SELECT 1 FROM sessions WHERE id = ? AND source = ?)"
    )


async def begin_turn(
    session_id: str,
    user_text: str,
    *,
    model: str,
    started_at: float,
    create_user_message: bool = True,
    user_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One D1 round trip: ensure the session, store the user message, open the assistant row.

    Returns ``{"user_message_id", "assistant_message_id", "created"}``.
    Raises :class:`SessionConflict` when ``session_id`` belongs to another application.
    """
    source = config.session_source()
    title = auto_title(user_text)
    statements: List[Any] = [
        (
            "INSERT INTO sessions (id, source, created_source, session_key, display_name, title, title_source, "
            "model, started_at, last_activity_at, message_count, tool_call_count) "
            "SELECT ?, ?, ?, ?, ?, ?, 'auto', ?, ?, ?, 0, 0 "
            "WHERE NOT EXISTS (SELECT 1 FROM sessions WHERE id = ?)",
            [session_id, source, source, session_id, title, title, model, started_at, started_at, session_id],
        )
    ]
    if create_user_message:
        meta = json.dumps(user_meta, ensure_ascii=False) if user_meta else None
        statements.append(
            (_msg_insert(),
             [session_id, "user", user_text, started_at, None, uuid.uuid4().hex, meta, session_id, source])
        )
    statements.append(
        (_msg_insert(),
         [session_id, "assistant", "", started_at + 0.001, RUNNING, uuid.uuid4().hex, None, session_id, source])
    )
    results: List[D1Result] = await d1.abatch(statements)
    created = results[0].changes > 0
    idx = 1
    user_id: Optional[int] = None
    if create_user_message:
        if results[idx].changes == 0:
            raise SessionConflict(session_id)
        user_id = results[idx].last_row_id
        idx += 1
    if results[idx].changes == 0:
        raise SessionConflict(session_id)
    return {"user_message_id": user_id, "assistant_message_id": results[idx].last_row_id, "created": created}


async def save_assistant(
    message_id: int,
    session_id: str,
    *,
    content: str,
    finish_reason: str,
    metadata: Dict[str, Any],
    token_count: Optional[int] = None,
) -> None:
    """Write (or checkpoint) the assistant message and refresh the session counters."""
    payload = json.dumps(metadata, ensure_ascii=False)
    if len(payload) > 180_000:  # keep rows far below D1's per-row limit
        metadata = {**metadata, "parts": [p for p in (metadata.get("parts") or []) if p.get("t") == "text"]}
        payload = json.dumps(metadata, ensure_ascii=False)[:180_000]
    ts = now()
    await d1.abatch(
        [
            (
                "UPDATE messages SET content = ?, finish_reason = ?, display_kind = 'turn', "
                "display_metadata = ?, token_count = ? WHERE id = ? AND session_id = ?",
                [content, finish_reason, payload, token_count, message_id, session_id],
            ),
            (
                "UPDATE sessions SET last_activity_at = ?, message_count = ("
                "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role IN ('user','assistant') "
                "AND content != '') WHERE id = ?",
                [ts, session_id, session_id],
            ),
        ]
    )


async def last_user_message(session_id: str) -> Optional[Dict[str, Any]]:
    rows = await d1.aquery(
        "SELECT id, content FROM messages WHERE session_id = ? AND role = 'user' "
        "ORDER BY timestamp DESC, id DESC LIMIT 1",
        [session_id],
    )
    return rows[0] if rows else None


async def drop_after(session_id: str, message_id: int) -> int:
    """Delete every message newer than ``message_id`` (used by regenerate)."""
    res = await d1.aexec(
        "DELETE FROM messages WHERE session_id = ? AND id > ? AND EXISTS "
        "(SELECT 1 FROM sessions WHERE id = ? AND source = ?)",
        [session_id, message_id, session_id, config.session_source()],
    )
    return res.changes


async def recover_interrupted() -> int:
    """After a restart no turn is running any more: close the rows that say otherwise."""
    res = await d1.aexec(
        "UPDATE messages SET finish_reason = 'interrupted' WHERE finish_reason = ?", [RUNNING]
    )
    return res.changes
