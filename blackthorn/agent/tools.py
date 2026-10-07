"""Tool registry: JSON-schema'd tools the *model* chooses from via native tool calling.

Nothing here decides whether a tool should be used — that is the model's job.  This
module only guarantees that, once the model asks for a tool, the call is well formed
(schema validation + safe coercion), permitted, bounded in time and output size, and
that every failure comes back to the model as plain text it can act on.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple


class ToolError(Exception):
    """A failure the model should hear about (returned as the tool result)."""


@dataclass
class ToolResult:
    ok: bool
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    session_id: str
    cancel: asyncio.Event


Handler = Callable[[Dict[str, Any], ToolContext], Awaitable[ToolResult]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Handler
    label: str
    timeout_s: float = 60.0
    max_output_chars: int = 8000

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]


# --------------------------------------------------------------------------- #
# argument parsing + validation
# --------------------------------------------------------------------------- #
def parse_arguments(raw: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Turn the model's ``arguments`` payload into a dict, or explain why not."""
    if isinstance(raw, dict):
        return raw, None
    if raw is None:
        return {}, None
    text = str(raw).strip()
    if not text:
        return {}, None
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"arguments are not valid JSON ({exc.msg} at position {exc.pos})"
    if isinstance(value, str):  # double-encoded
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None, "arguments must be a JSON object"
    if not isinstance(value, dict):
        return None, "arguments must be a JSON object"
    return value, None


_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no"}


def _coerce(value: Any, spec: Dict[str, Any], name: str, errors: List[str]) -> Any:
    kind = spec.get("type")
    if kind == "string":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
        if not isinstance(value, str):
            errors.append(f"'{name}' must be a string")
            return value
        if "maxLength" in spec and len(value) > spec["maxLength"]:
            errors.append(f"'{name}' is too long (max {spec['maxLength']} characters)")
        if "minLength" in spec and len(value) < spec["minLength"]:
            errors.append(f"'{name}' must not be empty")
    elif kind in ("integer", "number"):
        if isinstance(value, bool):
            errors.append(f"'{name}' must be a number")
            return value
        if isinstance(value, str):
            try:
                value = int(value) if kind == "integer" else float(value)
            except ValueError:
                errors.append(f"'{name}' must be a {kind}")
                return value
        if isinstance(value, float) and kind == "integer":
            if value != int(value):
                errors.append(f"'{name}' must be an integer")
                return value
            value = int(value)
        if not isinstance(value, (int, float)):
            errors.append(f"'{name}' must be a {kind}")
            return value
        if "minimum" in spec and value < spec["minimum"]:
            errors.append(f"'{name}' must be >= {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            errors.append(f"'{name}' must be <= {spec['maximum']}")
    elif kind == "boolean":
        if isinstance(value, str) and value.lower() in _TRUE | _FALSE:
            value = value.lower() in _TRUE
        if not isinstance(value, bool):
            errors.append(f"'{name}' must be true or false")
    elif kind == "array":
        if not isinstance(value, list):
            errors.append(f"'{name}' must be an array")
    elif kind == "object":
        if not isinstance(value, dict):
            errors.append(f"'{name}' must be an object")
    if "enum" in spec and value not in spec["enum"]:
        errors.append(f"'{name}' must be one of {spec['enum']}")
    return value


def validate_arguments(schema: Dict[str, Any], args: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Validate/coerce ``args`` against a (subset of) JSON schema. Returns (clean, errors)."""
    props: Dict[str, Any] = schema.get("properties") or {}
    errors: List[str] = []
    clean: Dict[str, Any] = {}
    for key in args:
        if key not in props:
            errors.append(f"unknown argument '{key}' (allowed: {', '.join(props) or 'none'})")
    for key in schema.get("required") or []:
        if key not in args or args[key] is None:
            errors.append(f"missing required argument '{key}'")
    for key, spec in props.items():
        if key in args and args[key] is not None:
            clean[key] = _coerce(args[key], spec, key, errors)
        elif "default" in spec:
            clean[key] = spec["default"]
    return clean, errors


def clip(text: str, limit: int) -> Tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = text[: max(0, limit - 160)]
    return head + f"\n… [truncated {len(text) - len(head)} characters]", True


def canonical_key(name: str, args: Dict[str, Any]) -> str:
    """Stable identity of a call (used to detect the model repeating itself)."""
    return name + "\x00" + json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


async def run_tool(tool: Tool, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Execute with a timeout; never raises (failures become ``ToolResult(ok=False)``)."""
    try:
        result = await asyncio.wait_for(tool.handler(args, ctx), timeout=tool.timeout_s)
    except asyncio.TimeoutError:
        return ToolResult(False, f"error: the tool '{tool.name}' timed out after {int(tool.timeout_s)}s")
    except ToolError as exc:
        return ToolResult(False, f"error: {exc}")
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - tool bugs must not kill the turn
        return ToolResult(False, f"error: {type(exc).__name__}: {exc}")
    text, truncated = clip(result.text, tool.max_output_chars)
    meta = dict(result.meta)
    if truncated:
        meta["truncated"] = True
    return ToolResult(result.ok, text, meta)
