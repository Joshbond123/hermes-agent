from __future__ import annotations

from typing import Any, Dict

from . import computer, memory, security, task, web
from .base import ToolContext, ToolError, ToolRegistry, ToolResult, ToolSpec, clip, loads_args, redact, validate_args
from .task import TaskBoard


def default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for spec in [*web.specs(), *computer.specs(), *memory.specs(), *security.specs(), *task.specs()]:
        reg.register(spec)
    # Hermes-style progressive disclosure: uncommon native tools remain discoverable and
    # callable without paying for all of their schemas on every ordinary computer task.
    # The security catalog's tool_search is a DIFFERENT tool (it does not execute these).
    optional = {"browse", "remember", "session_search", "skill_search", "skill_load",
                "skill_read", "tool_search", "sample_triage", "cyber_run", "cyber_preflight"}

    async def optional_tool(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = args["name"]
        if name not in optional:
            return ToolResult.failure("That tool is not available through optional_tool. "
                                      "Use a directly listed tool or the named optional tools.")
        return await reg.run(name, args["arguments"], ctx)

    reg.register(ToolSpec(
        "optional_tool",
        "Call an optional built-in when its full schema is not listed. Options: "
        "browse({url, timeout_ms?, screenshot?}) opens a JavaScript page; "
        "remember({note}) saves memory; session_search({query}) finds earlier chats; "
        "skill_search({query}), skill_load({name}), skill_read({name,path}) explore skills; "
        "tool_search({query}) searches the unified security catalog (897 skills + 183 tools); "
        "cyber_run({tool_id,args?}) installs-once and executes one catalogued security tool; "
        "cyber_preflight() reports installed binaries; "
        "sample_triage({sample_id}) statically inspects a sample. "
        "Invalid arguments receive the specific tool's validation error; nothing is installed by discovery.",
        {"type": "object", "properties": {
            "name": {"type": "string", "enum": sorted(optional)},
            "arguments": {"type": "object"}}, "required": ["name", "arguments"]},
        optional_tool, timeout=370.0, describe=lambda a: str(a.get("name") or "")))
    return reg


__all__ = ["TaskBoard", "ToolContext", "ToolError", "ToolRegistry", "ToolResult", "ToolSpec", "clip",
           "default_registry", "loads_args", "redact", "validate_args"]
