"""Streaming client for the OpenAI-compatible chat endpoint on the GPU gateway.

Yields normalised events and raises typed errors; contains no policy about tools.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

from blackthorn.route import DEAD_HTTP_STATUS, Route, TunnelDown


class ModelError(RuntimeError):
    """The model endpoint failed in a way that is not a dead tunnel."""

    def __init__(self, message: str, *, status: int = 0, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass
class ContentDelta:
    text: str


@dataclass
class ReasoningDelta:
    """Private reasoning tokens.  Only their *existence* is surfaced, never the text."""

    chars: int


@dataclass
class ToolCallDelta:
    index: int
    id: Optional[str]
    name: Optional[str]
    arguments: str


@dataclass
class Finish:
    reason: str
    usage: Optional[Dict[str, Any]] = None


ModelEvent = Union[ContentDelta, ReasoningDelta, ToolCallDelta, Finish]


@dataclass
class AssembledToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ToolCallAssembler:
    """Collects streamed tool-call fragments (they arrive split by ``index``)."""

    _calls: Dict[int, Dict[str, str]] = field(default_factory=dict)

    def add(self, delta: ToolCallDelta) -> None:
        slot = self._calls.setdefault(delta.index, {"id": "", "name": "", "arguments": ""})
        if delta.id:
            slot["id"] = delta.id
        if delta.name:
            slot["name"] = delta.name
        if delta.arguments:
            slot["arguments"] += delta.arguments

    def __bool__(self) -> bool:
        return any(c["name"] for c in self._calls.values())

    def calls(self) -> List[AssembledToolCall]:
        out: List[AssembledToolCall] = []
        for index in sorted(self._calls):
            slot = self._calls[index]
            if not slot["name"]:
                continue
            out.append(
                AssembledToolCall(slot["id"] or f"call_{uuid.uuid4().hex[:10]}", slot["name"], slot["arguments"])
            )
        return out


def _arguments_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return ""


def parse_chunk(obj: Dict[str, Any]) -> List[ModelEvent]:
    """Normalise one decoded SSE JSON object."""
    events: List[ModelEvent] = []
    err = obj.get("error")
    if err:
        message = err.get("message") if isinstance(err, dict) else str(err)
        raise ModelError(f"model error: {message}")
    choices = obj.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    delta = choice.get("delta") or {}
    reasoning = delta.get("reasoning") or delta.get("reasoning_content")
    if reasoning:
        events.append(ReasoningDelta(len(str(reasoning))))
    content = delta.get("content")
    if content:
        events.append(ContentDelta(str(content)))
    for tc in delta.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        events.append(
            ToolCallDelta(
                index=int(tc.get("index") or 0),
                id=tc.get("id"),
                name=fn.get("name"),
                arguments=_arguments_text(fn.get("arguments")),
            )
        )
    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
    reason = choice.get("finish_reason")
    if reason or usage:
        events.append(Finish(str(reason or ""), usage))
    return events


async def stream_chat(
    client: httpx.AsyncClient,
    route: Route,
    body: Dict[str, Any],
    *,
    idle_timeout: float = 180.0,
    max_malformed: int = 25,
) -> AsyncIterator[ModelEvent]:
    """POST ``/v1/chat/completions`` with ``stream=true`` and yield events as they arrive."""
    url = route.url + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {route.api_key}", "Accept": "text/event-stream"}
    timeout = httpx.Timeout(connect=10.0, read=idle_timeout, write=60.0, pool=10.0)
    malformed = 0
    try:
        async with client.stream("POST", url, json={**body, "stream": True}, headers=headers, timeout=timeout) as resp:
            if resp.status_code != 200:
                text = (await resp.aread()).decode("utf-8", "replace")
                ctype = (resp.headers.get("content-type") or "").lower()
                if resp.status_code in DEAD_HTTP_STATUS and (
                    "html" in ctype or resp.status_code in (502, 503, 521, 522, 523, 524, 530)
                ):
                    raise TunnelDown(
                        f"the GPU tunnel is down (HTTP {resp.status_code})", http_status=resp.status_code
                    )
                detail = text.strip()[:300]
                if "<html" in detail.lower():
                    detail = "non-JSON error page"
                raise ModelError(
                    f"the model endpoint returned HTTP {resp.status_code}: {detail}",
                    status=resp.status_code,
                    retryable=resp.status_code in (429, 500),
                )
            async for raw in resp.aiter_lines():
                line = raw.strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    malformed += 1
                    if malformed > max_malformed:
                        raise ModelError("the model stream was corrupted (too many malformed events)")
                    continue
                if not isinstance(obj, dict):
                    malformed += 1
                    continue
                for event in parse_chunk(obj):
                    yield event
    except httpx.ReadTimeout as exc:
        raise ModelError(
            f"the model produced no output for {int(idle_timeout)} seconds", retryable=True
        ) from exc
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise TunnelDown(f"could not connect to the GPU ({type(exc).__name__})") from exc
    except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
        raise ModelError(f"the connection to the GPU was lost mid-answer ({type(exc).__name__})", retryable=True) from exc
