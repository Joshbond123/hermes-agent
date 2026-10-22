"""Durable memory (Cloudflare D1 ``hermes_memories``)."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import ToolContext, ToolResult, ToolSpec, redact


async def remember(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    note = " ".join(args["note"].split())
    await ctx.store.add_memory(note)
    return ToolResult(ok=True, content=f"Saved to memory: {note}", summary="saved", data={})


async def session_search(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = str(args.get("query") or "").strip()
    if len(query) < 2:
        return ToolResult.failure("Provide at least two characters to search past conversations.")
    try:
        rows = await ctx.store.search_session_messages(query, session_id=str(args.get("session_id") or ""),
                                                       limit=int(args.get("limit") or 5))
    except Exception:
        return ToolResult.failure("Past conversation search is unavailable; do not guess its results.")
    if not rows:
        return ToolResult(ok=True, content="No matching saved Blackthorn conversation messages.", summary="no matches")
    lines = []
    for row in rows:
        # A prior assistant's prose is not evidence that a command actually ran.
        kind = "user request" if row["role"] == "user" else "prior assistant text (unverified claim)"
        content = redact(" ".join(str(row.get("content") or "").split()))[:650]
        lines.append(f"session={row['session_id']} message={row['id']} ({kind}, status={row.get('finish_reason') or 'unknown'}): {content}")
    return ToolResult(ok=True, content="Saved conversation matches (quote only what these records show):\n" +
                      "\n".join(lines), summary=f"{len(lines)} saved message(s)")


SCHEMA = {"type": "object", "properties": {"note": {"type": "string", "minLength": 3, "maxLength": 500}}, "required": ["note"]}


def specs() -> List[ToolSpec]:
    return [ToolSpec("remember",
                     "Save a lasting fact/preference about the user. Only when asked to remember or when a stable "
                     "preference is stated.",
                     SCHEMA, remember, timeout=30.0, kind="memory", describe=lambda a: str(a.get("note") or "")),
            ToolSpec("session_search",
                     "Search actual saved Blackthorn messages across past conversations when the user asks what "
                     "happened earlier. Results from assistant messages are claims, not proof of a tool action; "
                     "verify consequential results before reporting them. This is literal search, not semantic search.",
                     {"type": "object", "properties": {
                         "query": {"type": "string", "minLength": 2, "maxLength": 120},
                         "session_id": {"type": "string", "description": "Optional saved conversation ID to search within."},
                         "limit": {"type": "integer", "minimum": 1, "maximum": 8}},
                      "required": ["query"]}, session_search, timeout=30.0, kind="read",
                     describe=lambda a: "search saved conversations: " + str(a.get("query") or ""))]
