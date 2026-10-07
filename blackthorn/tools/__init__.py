from __future__ import annotations

from . import computer, memory, web
from .base import ToolContext, ToolError, ToolRegistry, ToolResult, ToolSpec, clip, loads_args, redact, validate_args


def default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for spec in [*web.specs(), *computer.specs(), *memory.specs()]:
        reg.register(spec)
    return reg


__all__ = ["ToolContext", "ToolError", "ToolRegistry", "ToolResult", "ToolSpec", "clip", "default_registry",
           "loads_args", "redact", "validate_args"]
