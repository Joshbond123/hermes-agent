"""Streaming client for the OpenAI-compatible endpoint exposed by the Kaggle gateway."""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .route import Route

DEAD_TUNNEL_CODES = {404, 410, 502, 503, 504, 521, 522, 523, 524, 530}


class LLMError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False, status: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.status = status


class ContextOverflow(LLMError):
    pass


def _error_text(body: bytes) -> str:
    try:
        data = json.loads(body.decode("utf-8", "replace"))
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            return str(err.get("message") or err)[:400]
        if err:
            return str(err)[:400]
        if isinstance(data, dict) and data.get("detail"):
            return str(data["detail"])[:400]
    except ValueError:
        pass
    text = body.decode("utf-8", "replace")
    if "<html" in text.lower():
        return "the tunnel returned an HTML error page"
    return text[:300]


def _is_overflow(message: str) -> bool:
    m = message.lower()
    return ("context" in m and any(w in m for w in ("exceed", "size", "length", "too long", "window"))) or "n_ctx" in m or (
        "prompt" in m and "too long" in m)


INROK_HOST_SUFFIX = ".share.inrok.in"


async def route_alive(client: httpx.AsyncClient, route: Route) -> bool:
    """Cheap liveness probe of the GPU route, used only to label a silent stream (dead vs slow)."""
    try:
        resp = await client.get(f"{route.url}/health", headers={**tunnel_headers(route.url), "Authorization": f"Bearer {route.api_key}"},
                                timeout=httpx.Timeout(8.0, connect=5.0))
        return resp.status_code == 200
    except Exception:  # noqa: BLE001 - a failed probe is simply 'not alive'
        return False


def tunnel_headers(url: str) -> Dict[str, str]:
    """Extra headers for a public tunnel URL.

    Inrok shows a browser interstitial on shared sites unless the client sends
    ``skip_zrok_interstitial``. The header is only added for Inrok hosts, so other
    upstreams never see it.
    """
    from urllib.parse import urlsplit
    try:
        host = (urlsplit(str(url or "")).hostname or "").lower()
    except ValueError:
        host = ""
    return {"skip_zrok_interstitial": "true"} if host.endswith(INROK_HOST_SUFFIX) else {}


def new_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=20.0),
        # keepalive_expiry MUST stay below the tunnel edge's idle close (~100s): reusing a socket the
        # edge already dropped writes into a half-open connection and the request hangs silently.
        limits=httpx.Limits(max_keepalive_connections=8, keepalive_expiry=15.0),
        headers={"User-Agent": "Blackthorn/5"},
    )


async def stream_chat(route: Route, messages: List[Dict[str, Any]], *, client: httpx.AsyncClient,
                      tools: Optional[List[Dict[str, Any]]] = None, idle_timeout: float = 150.0,
                      first_byte_timeout: float = 45.0,
                      connect_timeout: float = 20.0, temperature: Optional[float] = None,
                      extra: Optional[Dict[str, Any]] = None) -> AsyncIterator[Dict[str, Any]]:
    """Yield normalised events: content | reasoning | tool | finish | usage.

    * ``content``   {"v": text}                       visible answer text
    * ``reasoning`` {"v": text}                       private thinking (callers must not forward it)
    * ``tool``      {"index", "id", "name", "args"}   incremental native tool-call fragment
    * ``finish``    {"v": reason}
    * ``usage``     {"prompt": n, "completion": n}
    """
    body: Dict[str, Any] = {"model": route.model, "messages": messages, "stream": True,
                            "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if temperature is not None:
        body["temperature"] = temperature
    if extra:
        body.update(extra)
    timeout = httpx.Timeout(connect=connect_timeout, read=idle_timeout, write=30.0, pool=15.0)
    # Two separate clocks. The first-byte watchdog covers request -> response headers -> first
    # SSE line (prefill included). A stream that never starts is abandoned after
    # ``first_byte_timeout`` and the connection is closed, not left to the idle timeout.
    # ``idle_timeout`` only governs gaps between later lines of an already-started answer.
    stack = AsyncExitStack()
    try:
        cm = client.stream("POST", f"{route.url}/v1/chat/completions", json=body, timeout=timeout,
                           headers={**tunnel_headers(route.url), "Authorization": f"Bearer {route.api_key}",
                                    "Accept": "text/event-stream"})
        try:
            resp = await asyncio.wait_for(stack.enter_async_context(cm), timeout=first_byte_timeout)
        except asyncio.TimeoutError as exc:
            alive = await route_alive(client, route)
            if alive:
                raise LLMError("gpu_stalled", f"The model sent no response within {int(first_byte_timeout)}s (GPU is online); the request was closed and will be retried.",
                               retryable=True) from exc
            raise LLMError("gpu_unreachable", f"No response from the GPU within {int(first_byte_timeout)}s and its route is not answering; retrying on a fresh connection.",
                           retryable=True) from exc
        if resp.status_code != 200:
            detail = _error_text(await resp.aread())
            if resp.status_code == 400 and _is_overflow(detail):
                raise ContextOverflow("context_overflow", detail, status=400)
            if resp.status_code in (401, 403):
                raise LLMError("gpu_auth", "The GPU gateway rejected the access key. Turn the GPU off and on again to refresh it.",
                               status=resp.status_code)
            if resp.status_code in DEAD_TUNNEL_CODES:
                raise LLMError("gpu_unreachable", f"The GPU tunnel is unreachable (HTTP {resp.status_code}).",
                               retryable=True, status=resp.status_code)
            raise LLMError("gpu_http", f"The GPU endpoint returned HTTP {resp.status_code}: {detail}",
                           retryable=resp.status_code >= 500, status=resp.status_code)
        lines = resp.aiter_lines().__aiter__()
        first_line = True
        while True:
            budget = first_byte_timeout if first_line else idle_timeout
            try:
                raw = await asyncio.wait_for(lines.__anext__(), timeout=budget)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError as exc:
                if first_line:
                    raise LLMError("gpu_stalled", f"The model sent no output within {int(first_byte_timeout)}s; the request was closed and will be retried.",
                                   retryable=True) from exc
                raise LLMError("gpu_stalled", f"The model produced no output for {int(idle_timeout)}s and the request was abandoned.",
                               retryable=True) from exc
            first_line = False
            if not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get("error"):
                message = _error_text(json.dumps({"error": obj["error"]}).encode())
                if _is_overflow(message):
                    raise ContextOverflow("context_overflow", message)
                raise LLMError("gpu_stream", message, retryable=False)
            usage = obj.get("usage") if isinstance(obj, dict) else None
            if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                yield {"t": "usage", "prompt": int(usage.get("prompt_tokens") or 0),
                       "completion": int(usage.get("completion_tokens") or 0)}
            choices = obj.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if reasoning:
                yield {"t": "reasoning", "v": reasoning}
            content = delta.get("content") or ""
            if content:
                yield {"t": "content", "v": content}
            for tc in delta.get("tool_calls") or []:
                fn = tc.get("function") or {}
                yield {"t": "tool", "index": int(tc.get("index", 0) or 0), "id": tc.get("id"),
                       "name": fn.get("name"), "args": fn.get("arguments") or ""}
            if choice.get("finish_reason"):
                yield {"t": "finish", "v": str(choice["finish_reason"])}
    except httpx.ConnectTimeout as exc:
        raise LLMError("gpu_unreachable", "Timed out connecting to the GPU tunnel.", retryable=True) from exc
    except httpx.ConnectError as exc:
        raise LLMError("gpu_unreachable", "Could not connect to the GPU tunnel (it may have stopped).", retryable=True) from exc
    except httpx.ReadTimeout as exc:
        raise LLMError("gpu_stalled", f"The model produced no output for {int(idle_timeout)}s and the request was abandoned.", retryable=True) from exc
    except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
        raise LLMError("gpu_dropped", "The connection to the GPU dropped mid-response.", retryable=True) from exc
    finally:
        # Always close the upstream connection: a stalled or abandoned stream must not keep
        # a dead socket in the pool for the next request.
        await stack.aclose()
