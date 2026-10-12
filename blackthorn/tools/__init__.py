from __future__ import annotations

from . import computer, memory, security, task, web
from .base import ToolContext, ToolError, ToolRegistry, ToolResult, ToolSpec, clip, loads_args, redact, validate_args
from .task import TaskBoard


def default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for spec in [*web.specs(), *computer.specs(), *memory.specs(), *security.specs(), *task.specs()]:
        reg.register(spec)
    return reg


__all__ = ["TaskBoard", "ToolContext", "ToolError", "ToolRegistry", "ToolResult", "ToolSpec", "clip",
           "default_registry", "loads_args", "redact", "validate_args"]
