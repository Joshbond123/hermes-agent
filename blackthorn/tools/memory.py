"""Durable memory (Cloudflare D1 ``hermes_memories``)."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import ToolContext, ToolResult, ToolSpec


async def remember(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    note = " ".join(args["note"].split())
    await ctx.store.add_memory(note)
    return ToolResult(ok=True, content=f"Saved to memory: {note}", summary="saved", data={})


SCHEMA = {"type": "object", "properties": {"note": {"type": "string", "minLength": 3, "maxLength": 500}}, "required": ["note"]}


def specs() -> List[ToolSpec]:
    return [ToolSpec("remember",
                     "Save a lasting fact/preference about the user. Only when asked to remember or when a stable "
                     "preference is stated.",
                     SCHEMA, remember, timeout=30.0, kind="memory", describe=lambda a: str(a.get("note") or ""))]
