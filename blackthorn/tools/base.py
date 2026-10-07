"""Tool registry: JSON-schema validation, permissions, timeouts, redaction and result budgets."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- redaction
_REDACTIONS: Tuple[Tuple[re.Pattern, str], ...] = tuple((re.compile(p, re.I | re.S), r) for p, r in (
    (r"(authorization:\s*bearer\s+)[A-Za-z0-9._~+/=-]{8,}", r"\1***"),
    (r"\b(bearer)\s+[A-Za-z0-9._~+/=-]{16,}", r"\1 ***"),
    (r"\b(api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|passwd|password|pwd)\b(\s*[:=]\s*)([\"']?)[^\s\"']{6,}",
     r"\1\2\3***"),
    (r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}", "***"),
    (r"\bbt-[A-Za-z0-9_-]{24,}", "***"),
    (r"\bcfut_[A-Za-z0-9]{16,}", "***"),
    (r"\bKGAT_[A-Za-z0-9]{16,}", "***"),
    (r"\btvly-[A-Za-z0-9_-]{16,}", "***"),
    (r"\bgh[pousr]_[A-Za-z0-9]{20,}", "***"),
    (r"\bgithub_pat_[A-Za-z0-9_]{20,}", "***"),
    (r"\bAKIA[0-9A-Z]{16}\b", "***"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", "[private key removed]"),
))


def redact(text: str) -> str:
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


def clip(text: str, limit: int, mode: str = "head") -> Tuple[str, bool]:
    """Bound ``text`` to ``limit`` characters. mode: head | tail | ends (keeps both ends, tail-heavy)."""
    if limit <= 0 or len(text) <= limit:
        return text, False
    omitted = len(text) - limit
    marker = f"\n[… {omitted} characters omitted …]\n"
    if mode == "tail":
        return marker.lstrip("\n") + text[-limit:], True
    if mode == "ends":
        head = int(limit * 0.35)
        return text[:head] + marker + text[-(limit - head):], True
    return text[:limit] + marker.rstrip("\n"), True


# --------------------------------------------------------------------------- results
@dataclass
class ToolResult:
    ok: bool
    content: str                      # what the model reads
    summary: str = ""                 # one redacted line for the UI
    error: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    truncated: bool = False

    @staticmethod
    def failure(message: str, *, hint: str = "", data: Optional[Dict[str, Any]] = None) -> "ToolResult":
        body = f"ERROR: {message}" + (f"\n{hint}" if hint else "")
        return ToolResult(ok=False, content=body, summary=redact(message)[:200], error=message, data=data or {})


class ToolError(RuntimeError):
    def __init__(self, kind: str, message: str, hint: str = ""):
        super().__init__(message)
        self.kind, self.message, self.hint = kind, message, hint


# --------------------------------------------------------------------------- validation
def _coerce(value: Any, spec: Dict[str, Any], name: str) -> Any:
    kind = spec.get("type")
    if kind == "string":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
        if not isinstance(value, str):
            raise ValueError(f"'{name}' must be a string")
        if "minLength" in spec and len(value.strip()) < spec["minLength"]:
            raise ValueError(f"'{name}' must not be empty")
        if "maxLength" in spec and len(value) > spec["maxLength"]:
            raise ValueError(f"'{name}' is too long ({len(value)} > {spec['maxLength']} characters)")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"'{name}' must be one of {spec['enum']}")
        return value
    if kind in ("integer", "number"):
        if isinstance(value, bool):
            raise ValueError(f"'{name}' must be a number")
        if isinstance(value, str):
            try:
                value = float(value.strip())
            except ValueError:
                raise ValueError(f"'{name}' must be a number") from None
        if not isinstance(value, (int, float)):
            raise ValueError(f"'{name}' must be a number")
        if kind == "integer":
            if int(value) != value:
                raise ValueError(f"'{name}' must be a whole number")
            value = int(value)
        if "minimum" in spec:
            value = max(spec["minimum"], value)
        if "maximum" in spec:
            value = min(spec["maximum"], value)
        return value
    if kind == "boolean":
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        if not isinstance(value, bool):
            raise ValueError(f"'{name}' must be true or false")
        return value
    return value


def validate_args(schema: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Validate/coerce ``args`` against the (small) JSON-schema subset the tools use. Raises ValueError."""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError("arguments must be a JSON object")
    props: Dict[str, Any] = schema.get("properties", {})
    out: Dict[str, Any] = {}
    for key in schema.get("required", []):
        if key not in args or args[key] is None:
            raise ValueError(f"missing required argument '{key}'")
    for key, spec in props.items():
        if key in args and args[key] is not None:
            out[key] = _coerce(args[key], spec, key)
        elif "default" in spec:
            out[key] = spec["default"]
    return out


_WIRE_DROP = {"minLength", "maxLength", "minimum", "maximum", "default"}


def _slim(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _slim(v) for k, v in node.items() if k not in _WIRE_DROP}
    return node


# --------------------------------------------------------------------------- registry
@dataclass
class ToolContext:
    settings: Any
    store: Any
    computer: Any
    http: Any
    tavily: Any
    session_id: str = ""
    run_id: str = ""


Handler = Callable[[Dict[str, Any], ToolContext], Awaitable[ToolResult]]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Handler
    timeout: float = 90.0
    kind: str = "read"                 # read | write | exec | net | memory
    describe: Callable[[Dict[str, Any]], str] = lambda a: ""

    def schema(self) -> Dict[str, Any]:
        """Wire schema sent to the model. Validation-only keys stay server-side to save context tokens."""
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": _slim(self.parameters)}}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def names(self) -> List[str]:
        return list(self._tools)

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def describe(self, name: str, args: Dict[str, Any]) -> str:
        spec = self._tools.get(name)
        if not spec:
            return ""
        try:
            return redact(spec.describe(args) or "")[:200]
        except Exception:
            return ""

    async def run(self, name: str, raw_args: Any, ctx: ToolContext) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult.failure(f"unknown tool '{name}'", hint=f"Available tools: {', '.join(self._tools)}.")
        try:
            args = validate_args(spec.parameters, raw_args)
        except ValueError as exc:
            return ToolResult.failure(f"invalid arguments for {name}: {exc}",
                                      hint="Fix the arguments and call the tool again, or answer without it.")
        try:
            return await asyncio.wait_for(spec.handler(args, ctx), timeout=spec.timeout)
        except asyncio.TimeoutError:
            return ToolResult.failure(f"{name} timed out after {int(spec.timeout)}s")
        except ToolError as exc:
            return ToolResult.failure(exc.message, hint=exc.hint, data={"kind": exc.kind})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a tool bug must never kill the run
            return ToolResult.failure(f"{name} failed: {type(exc).__name__}: {str(exc)[:200]}")


def loads_args(raw: str) -> Dict[str, Any]:
    """Parse the JSON string a model streamed as tool arguments. Raises ValueError with a model-readable message."""
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"arguments are not valid JSON ({exc.msg} at char {exc.pos})") from None
    if not isinstance(value, dict):
        raise ValueError("arguments must be a JSON object")
    return value
