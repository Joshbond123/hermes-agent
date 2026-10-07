"""Conversation storage on the real ``sessions`` / ``messages`` tables (Cloudflare D1 in production).

Design notes
------------
* A turn is persisted *before* the model is called: one atomic batch reads the history, creates the
  session if needed, inserts the user message and inserts the assistant message as ``streaming``.
  Progress is then written incrementally and finalised exactly once, so a refresh, a crash or a
  cancel can never lose the user's message or the answer produced so far.
* Message order is the autoincrement ``id`` (never a timestamp).
* The assistant message keeps its full structure (ordered text + tool parts) as JSON in
  ``messages.display_metadata`` (``display_kind = 'bt.v1'``); ``messages.content`` always holds the
  plain answer text so other readers of the table keep working.
* ``messages.finish_reason`` carries the status: streaming | stop | length | cancelled | error | interrupted.
* pin / archive / rename / delete are real columns on ``sessions`` (pinned, archived, title) — not UI state.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .sql import Result, SqlError, Statement, dumps

KIND = "bt.v1"
STREAMING = "streaming"
TERMINAL = ("stop", "length", "cancelled", "error", "interrupted")

SQLITE_SCHEMA: Tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY, source TEXT, created_source TEXT, session_key TEXT, display_name TEXT,
        title TEXT, title_source TEXT, model TEXT, started_at REAL, ended_at REAL, end_reason TEXT,
        last_activity_at REAL, message_count INTEGER DEFAULT 0, tool_call_count INTEGER DEFAULT 0,
        pinned INTEGER DEFAULT 0, archived INTEGER DEFAULT 0, hidden INTEGER DEFAULT 0)""",
    """CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, tool_call_id TEXT,
        tool_calls TEXT, tool_name TEXT, timestamp REAL, token_count INTEGER, finish_reason TEXT, reasoning TEXT,
        active INTEGER DEFAULT 1, _compressed_summary INTEGER DEFAULT 0, message_uid TEXT,
        display_kind TEXT, display_metadata TEXT)""",
    "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)",
    "CREATE TABLE IF NOT EXISTS state_meta (key TEXT, value TEXT)",
    """CREATE TABLE IF NOT EXISTS hermes_memories (
        id TEXT PRIMARY KEY, target TEXT, content TEXT, memory_type TEXT, importance REAL, user_id TEXT,
        session_id TEXT, profile TEXT, created_at REAL, updated_at REAL)""",
    """CREATE TABLE IF NOT EXISTS kaggle_gpu_state (
        id TEXT PRIMARY KEY, status TEXT, tunnel_url TEXT, api_key TEXT, model TEXT, gpu_info TEXT,
        updated_at REAL, detail TEXT)""",
)

# Columns the store relies on; added with ALTER TABLE when an older database lacks them.
REQUIRED_SESSION_COLS = {"title": "TEXT", "title_source": "TEXT", "last_activity_at": "REAL",
                         "pinned": "INTEGER DEFAULT 0", "archived": "INTEGER DEFAULT 0", "hidden": "INTEGER DEFAULT 0"}
REQUIRED_MESSAGE_COLS = {"message_uid": "TEXT", "display_kind": "TEXT", "display_metadata": "TEXT",
                         "finish_reason": "TEXT", "token_count": "INTEGER"}


class NotFound(LookupError):
    pass


def new_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def clean_title(text: str, limit: int = 60) -> str:
    line = " ".join((text or "").strip().split())
    if len(line) <= limit:
        return line
    return line[: limit - 1].rstrip() + "…"


def _flag(value: Any) -> bool:
    return bool(int(value or 0))


def _session_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "title": (row.get("title") or "").strip() or "New chat",
        "pinned": _flag(row.get("pinned")),
        "archived": _flag(row.get("archived")),
        "created_at": row.get("started_at"),
        "updated_at": row.get("updated_at") or row.get("started_at"),
        "message_count": int(row.get("message_count") or 0),
    }


def _parse_meta(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (TypeError, ValueError):
        return {}


def message_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    meta = _parse_meta(row.get("display_metadata")) if row.get("display_kind") == KIND else {}
    content = row.get("content") or ""
    parts = meta.get("parts")
    if not isinstance(parts, list) or not parts:
        parts = [{"type": "text", "text": content}] if content else []
    status = row.get("finish_reason") or ("stop" if row.get("role") == "assistant" else "")
    out: Dict[str, Any] = {
        "id": row.get("message_uid") or f"m{row['id']}",
        "seq": row["id"],
        "role": row["role"],
        "content": content,
        "parts": parts,
        "status": status if row.get("role") == "assistant" else "",
        "created_at": row.get("timestamp"),
    }
    info = meta.get("meta")
    if isinstance(info, dict):
        out["meta"] = info
    if meta.get("attachments"):
        out["attachments"] = meta["attachments"]
    return out


class ChatStore:
    def __init__(self, executor: Any, *, source: str = "blackthorn-studio", backend: str = "d1"):
        self.db = executor
        self.source = source
        self.backend = backend
        self._ready = False

    # ------------------------------------------------------------------ schema
    async def ensure_schema(self) -> None:
        if self._ready:
            return
        if self.backend == "sqlite":
            await self.db.batch([(sql, ()) for sql in SQLITE_SCHEMA])
        else:
            existing = await self.db.batch([("PRAGMA table_info(sessions)", ()), ("PRAGMA table_info(messages)", ())])
            have_sessions = {r["name"] for r in existing[0].rows}
            have_messages = {r["name"] for r in existing[1].rows}
            stmts: List[Statement] = []
            if not have_sessions or not have_messages:
                stmts.extend((sql, ()) for sql in SQLITE_SCHEMA[:3])
            else:
                stmts.extend((f"ALTER TABLE sessions ADD COLUMN {c} {t}", ()) for c, t in REQUIRED_SESSION_COLS.items()
                             if c not in have_sessions)
                stmts.extend((f"ALTER TABLE messages ADD COLUMN {c} {t}", ()) for c, t in REQUIRED_MESSAGE_COLS.items()
                             if c not in have_messages)
            for stmt in stmts:  # one at a time: a duplicate-column race must not poison the rest
                try:
                    await self.db.query(*stmt)
                except SqlError as exc:
                    if "duplicate column" not in str(exc).lower():
                        raise
        self._ready = True

    # ------------------------------------------------------------------ sessions
    async def list_sessions(self, *, archived: bool = False, q: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        await self.ensure_schema()
        params: List[Any] = [self.source, 1 if archived else 0]
        search = ""
        q = (q or "").strip()
        if q:
            like = f"%{q.replace('%', '').replace('_', ' ')}%"
            search = (" AND (COALESCE(NULLIF(s.title,''), s.display_name, '') LIKE ? OR EXISTS ("
                      "SELECT 1 FROM messages m WHERE m.session_id = s.id AND m.content LIKE ?))")
            params += [like, like]
        params.append(int(max(1, min(limit, 300))))
        res = await self.db.query(
            "SELECT s.id, COALESCE(NULLIF(s.title,''), NULLIF(s.display_name,''), '') AS title, "
            "COALESCE(s.pinned,0) AS pinned, COALESCE(s.archived,0) AS archived, s.started_at, "
            "COALESCE(s.last_activity_at, s.started_at) AS updated_at, COALESCE(s.message_count,0) AS message_count "
            "FROM sessions s WHERE s.source = ? AND COALESCE(s.hidden,0) = 0 AND COALESCE(s.archived,0) = ? "
            "AND COALESCE(s.message_count,0) > 0" + search +
            " ORDER BY COALESCE(s.pinned,0) DESC, COALESCE(s.last_activity_at, s.started_at) DESC LIMIT ?",
            params,
        )
        return [_session_dict(r) for r in res.rows]

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        await self.ensure_schema()
        res = await self.db.query(
            "SELECT id, COALESCE(NULLIF(title,''), NULLIF(display_name,''), '') AS title, COALESCE(pinned,0) AS pinned, "
            "COALESCE(archived,0) AS archived, started_at, COALESCE(last_activity_at, started_at) AS updated_at, "
            "COALESCE(message_count,0) AS message_count FROM sessions WHERE id = ? AND source = ?",
            [session_id, self.source])
        if not res.rows:
            raise NotFound(session_id)
        return _session_dict(res.rows[0])

    async def update_session(self, session_id: str, *, title: Optional[str] = None, pinned: Optional[bool] = None,
                             archived: Optional[bool] = None) -> Dict[str, Any]:
        await self.ensure_schema()
        sets: List[str] = []
        params: List[Any] = []
        if title is not None:
            clean = clean_title(title, 120)
            if not clean:
                raise ValueError("title must not be empty")
            sets += ["title = ?", "display_name = ?", "title_source = 'user'"]
            params += [clean, clean]
        if pinned is not None:
            sets.append("pinned = ?")
            params.append(1 if pinned else 0)
        if archived is not None:
            sets.append("archived = ?")
            params.append(1 if archived else 0)
            if archived:
                sets.append("pinned = 0")  # an archived chat leaves the pinned list
        if not sets:
            return await self.get_session(session_id)
        res = await self.db.query(f"UPDATE sessions SET {', '.join(sets)} WHERE id = ? AND source = ?",
                                  params + [session_id, self.source])
        if res.changes == 0:
            raise NotFound(session_id)
        return await self.get_session(session_id)

    async def delete_session(self, session_id: str) -> None:
        await self.ensure_schema()
        out = await self.db.batch([
            ("DELETE FROM messages WHERE session_id = ? AND EXISTS (SELECT 1 FROM sessions WHERE id = ? AND source = ?)",
             [session_id, session_id, self.source]),
            ("DELETE FROM sessions WHERE id = ? AND source = ?", [session_id, self.source]),
        ])
        if out[1].changes == 0:
            raise NotFound(session_id)

    # ------------------------------------------------------------------ messages
    async def get_messages(self, session_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        await self.ensure_schema()
        res = await self.db.query(
            "SELECT m.id, m.role, m.content, m.timestamp, m.finish_reason, m.message_uid, m.display_kind, "
            "m.display_metadata FROM messages m WHERE m.session_id = ? AND m.role IN ('user','assistant') "
            "AND COALESCE(m.active,1) = 1 ORDER BY m.id ASC LIMIT ?", [session_id, int(limit)])
        return [message_dict(r) for r in res.rows]

    async def begin_turn(self, session_id: Optional[str], user_text: str, *, model: str,
                         attachments: Optional[List[Dict[str, Any]]] = None, history_limit: int = 24,
                         ) -> Dict[str, Any]:
        """Atomically persist the user message + a ``streaming`` assistant placeholder; return prior history."""
        await self.ensure_schema()
        created = not session_id
        sid = session_id or new_id("studio-")
        now = time.time()
        user_uid, asst_uid = new_id("m"), new_id("m")
        title = clean_title(user_text)
        meta_user = dumps({"v": 1, "attachments": attachments}) if attachments else None
        guard = "EXISTS (SELECT 1 FROM sessions WHERE id = ? AND source = ?)"
        stmts: List[Statement] = [
            # 0: history strictly BEFORE this turn (statements in a batch run in order)
            ("SELECT role, content FROM messages WHERE session_id = ? AND role IN ('user','assistant') "
             "AND COALESCE(active,1) = 1 AND COALESCE(finish_reason,'') <> 'streaming' AND COALESCE(content,'') <> '' "
             "ORDER BY id DESC LIMIT ?", [sid, int(history_limit)]),
            # 1: create the session when it does not exist yet
            ("INSERT INTO sessions (id, source, created_source, session_key, display_name, title, title_source, model, "
             "started_at, last_activity_at, message_count, tool_call_count, pinned, archived) "
             "SELECT ?, ?, ?, ?, ?, ?, 'auto', ?, ?, ?, 0, 0, 0, 0 "
             "WHERE NOT EXISTS (SELECT 1 FROM sessions WHERE id = ?)",
             [sid, self.source, self.source, sid, title, title, model, now, now, sid]),
            # 2: user message (only if the session exists and belongs to this product)
            (f"INSERT INTO messages (session_id, role, content, timestamp, active, _compressed_summary, message_uid, "
             f"display_kind, display_metadata) SELECT ?, 'user', ?, ?, 1, 0, ?, ?, ? WHERE {guard}",
             [sid, user_text, now, user_uid, KIND if meta_user else None, meta_user, sid, self.source]),
            # 3: assistant placeholder, finalised later
            (f"INSERT INTO messages (session_id, role, content, timestamp, active, _compressed_summary, message_uid, "
             f"finish_reason, display_kind, display_metadata) SELECT ?, 'assistant', '', ?, 1, 0, ?, 'streaming', ?, ? "
             f"WHERE {guard}", [sid, now + 0.001, asst_uid, KIND, dumps({"v": 1, "parts": []}), sid, self.source]),
            # 4: counters / title / un-archive on new activity
            ("UPDATE sessions SET message_count = COALESCE(message_count,0) + 2, last_activity_at = ?, archived = 0, "
             "title = CASE WHEN COALESCE(title,'') = '' THEN ? ELSE title END, "
             "display_name = CASE WHEN COALESCE(display_name,'') = '' THEN ? ELSE display_name END "
             "WHERE id = ? AND source = ?", [now, title, title, sid, self.source]),
            # 5: final session row for the client
            ("SELECT id, COALESCE(NULLIF(title,''), NULLIF(display_name,''), '') AS title, COALESCE(pinned,0) AS pinned, "
             "COALESCE(archived,0) AS archived, started_at, COALESCE(last_activity_at, started_at) AS updated_at, "
             "COALESCE(message_count,0) AS message_count FROM sessions WHERE id = ? AND source = ?", [sid, self.source]),
        ]
        out = await self.db.batch(stmts)
        if out[2].changes == 0 or not out[5].rows:
            raise NotFound(sid)
        history = [{"role": r["role"], "content": r["content"]} for r in reversed(out[0].rows)]
        return {"session": _session_dict(out[5].rows[0]), "session_id": sid, "created": created,
                "user_uid": user_uid, "assistant_uid": asst_uid, "history": history}

    async def begin_regeneration(self, session_id: str, *, model: str, history_limit: int = 24) -> Dict[str, Any]:
        """Replace the last assistant answer: drop it and add a fresh ``streaming`` placeholder."""
        await self.ensure_schema()
        sess = await self.get_session(session_id)  # raises NotFound
        rows = await self.db.query(
            "SELECT id, role, content FROM messages WHERE session_id = ? AND role IN ('user','assistant') "
            "AND COALESCE(active,1) = 1 ORDER BY id DESC LIMIT ?", [session_id, int(history_limit) + 2])
        ordered = list(rows.rows)  # newest first
        removed = 0
        if ordered and ordered[0]["role"] == "assistant":
            removed = 1
            ordered = ordered[1:]
        if not ordered or ordered[0]["role"] != "user":
            raise ValueError("nothing to regenerate: the conversation has no user message")
        user_text = ordered[0]["content"] or ""
        history = [{"role": r["role"], "content": r["content"]} for r in reversed(ordered[1:])
                   if (r["content"] or "").strip()]
        now = time.time()
        asst_uid = new_id("m")
        stmts: List[Statement] = []
        if removed:
            stmts.append(("DELETE FROM messages WHERE id = ?", [rows.rows[0]["id"]]))
        stmts.append(("INSERT INTO messages (session_id, role, content, timestamp, active, _compressed_summary, message_uid, "
                      "finish_reason, display_kind, display_metadata) VALUES (?, 'assistant', '', ?, 1, 0, ?, 'streaming', ?, ?)",
                      [session_id, now, asst_uid, KIND, dumps({"v": 1, "parts": []})]))
        stmts.append(("UPDATE sessions SET last_activity_at = ?, archived = 0, message_count = COALESCE(message_count,0) + ? "
                      "WHERE id = ? AND source = ?", [now, 0 if removed else 1, session_id, self.source]))
        await self.db.batch(stmts)
        sess["archived"] = False
        return {"session": sess, "session_id": session_id, "created": False, "user_uid": None,
                "assistant_uid": asst_uid, "history": history, "user_text": user_text}

    async def update_assistant(self, uid: str, *, text: str, parts: List[Dict[str, Any]], status: str,
                               meta: Optional[Dict[str, Any]] = None, tokens: Optional[int] = None) -> None:
        payload = {"v": 1, "parts": parts}
        if meta:
            payload["meta"] = meta
        await self.db.query(
            "UPDATE messages SET content = ?, display_kind = ?, display_metadata = ?, finish_reason = ?, "
            "token_count = COALESCE(?, token_count) WHERE message_uid = ?",
            [text, KIND, dumps(payload), status, tokens, uid])

    async def finish_session_activity(self, session_id: str, *, tool_calls: int = 0) -> None:
        await self.db.query(
            "UPDATE sessions SET last_activity_at = ?, tool_call_count = COALESCE(tool_call_count,0) + ? WHERE id = ?",
            [time.time(), int(tool_calls), session_id])

    async def sweep_interrupted(self, session_id: Optional[str] = None) -> int:
        """Rows still marked ``streaming`` with no live run were cut off (restart / crash): mark them honestly."""
        await self.ensure_schema()
        if session_id:
            res = await self.db.query(
                "UPDATE messages SET finish_reason = 'interrupted' WHERE finish_reason = 'streaming' AND session_id = ?",
                [session_id])
        else:
            res = await self.db.query("UPDATE messages SET finish_reason = 'interrupted' WHERE finish_reason = 'streaming'")
        return res.changes

    # ------------------------------------------------------------------ settings & memory
    async def get_setting(self, key: str) -> str:
        res = await self.db.query("SELECT value FROM state_meta WHERE key = ? LIMIT 1", [key])
        return str(res.rows[0].get("value") or "") if res.rows else ""

    async def set_setting(self, key: str, value: str) -> None:
        await self.db.batch([("DELETE FROM state_meta WHERE key = ?", [key]),
                             ("INSERT INTO state_meta (key, value) VALUES (?, ?)", [key, value])])

    async def list_memories(self, limit: int = 8) -> List[str]:
        res = await self.db.query(
            "SELECT content FROM hermes_memories WHERE target = 'memory' ORDER BY importance DESC, updated_at DESC LIMIT ?",
            [int(limit)])
        return [str(r.get("content") or "").strip() for r in res.rows if (r.get("content") or "").strip()]

    async def add_memory(self, note: str) -> str:
        mid, now = new_id("mem-"), time.time()
        await self.db.query(
            "INSERT INTO hermes_memories (id, target, content, memory_type, importance, user_id, session_id, profile, "
            "created_at, updated_at) VALUES (?, 'memory', ?, 'factual', 0.9, 'blackthorn', '', 'default', ?, ?)",
            [mid, note, now, now])
        return mid

    async def gpu_row(self) -> Dict[str, Any]:
        res = await self.db.query(
            "SELECT status, tunnel_url, api_key, model, gpu_info, updated_at FROM kaggle_gpu_state WHERE id = 'primary' LIMIT 1")
        return dict(res.rows[0]) if res.rows else {}
