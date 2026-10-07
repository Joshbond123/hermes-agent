"""D1-backed implementation of the engine's persistence interface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from blackthorn import sessions
from blackthorn.agent.engine import Context
from blackthorn.d1 import d1

SYSTEM_PROMPT_KEY = "blackthorn_system_prompt"


class D1Store:
    async def begin_turn(self, session_id: str, user_text: str, *, model: str, started_at: float,
                         create_user_message: bool, user_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return await sessions.begin_turn(
            session_id, user_text, model=model, started_at=started_at,
            create_user_message=create_user_message, user_meta=user_meta)

    async def load_context(self, session_id: str, *, before_ts: Optional[float],
                           before_id: Optional[int]) -> Context:
        """History + custom system prompt + memories in ONE round trip."""
        if before_id is not None:
            history_stmt = (
                "SELECT role, content FROM messages WHERE session_id = ? AND role IN ('user','assistant') "
                "AND COALESCE(active, 1) = 1 AND content IS NOT NULL AND content != '' AND id < ? "
                "ORDER BY timestamp DESC, id DESC LIMIT 60",
                [session_id, before_id],
            )
        else:
            sql, params = sessions.history_statement(session_id, before_ts or 1e18, 60)
            history_stmt = (sql, params)
        res = await d1.abatch([
            history_stmt,
            ("SELECT value FROM state_meta WHERE key = ? LIMIT 1", [SYSTEM_PROMPT_KEY]),
            ("SELECT content FROM hermes_memories WHERE target = 'memory' "
             "ORDER BY importance DESC, updated_at DESC LIMIT 12", None),
        ])
        custom = str(res[1].rows[0].get("value") or "") if res[1].rows else ""
        memories = [str(r.get("content") or "") for r in res[2].rows]
        return Context(history=res[0].rows, custom_prompt=custom, memories=memories)

    async def save_assistant(self, message_id: int, session_id: str, *, content: str, finish_reason: str,
                             metadata: Dict[str, Any], token_count: Optional[int]) -> None:
        await sessions.save_assistant(message_id, session_id, content=content, finish_reason=finish_reason,
                                      metadata=metadata, token_count=token_count)

    async def last_user_message(self, session_id: str) -> Optional[Dict[str, Any]]:
        return await sessions.last_user_message(session_id)

    async def drop_after(self, session_id: str, message_id: int) -> int:
        return await sessions.drop_after(session_id, message_id)
