"""Streaming client for the OpenAI-compatible endpoint exposed by the Kaggle gateway."""

from __future__ import annotations

import asyncio
import json
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


async def _anext(iterator):
    return await iterator.__anext__()


async def _route_alive(client: httpx.AsyncClient, route: Route) -> bool:
    try:
        async with client.stream("GET", f"{route.url}/health",
                                 headers={"Authorization": f"Bearer {route.api_key}"},
                                 timeout=httpx.Timeout(8.0, connect=5.0)) as resp:
            return resp.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _is_overflow(message: str) -> bool:
    m = message.lower()
    return ("context" in m and any(w in m for w in ("exceed", "size", "length", "too long", "window"))) or "n_ctx" in m or (
        "prompt" in m and "too long" in m)


def new_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=20.0),
        # keepalive_expiry MUST stay below the tunnel worker's idle-close (~100s):
        # reusing a socket the edge already dropped writes into a half-open connection
        # and the request hangs silently until the read timeout (4-minute responses).
        limits=httpx.Limits(max_keepalive_connections=8, keepalive_expiry=15.0),
        headers={"User-Agent": "Blackthorn/5"},
    )


async def stream_chat(route: Route, messages: List[Dict[str, Any]], *, client: httpx.AsyncClient,
                      tools: Optional[List[Dict[str, Any]]] = None, idle_timeout: float = 150.0,
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
    import time as _time
    t_req = _time.monotonic()

    def _log(msg: str) -> None:
        print(f"[llm] {msg}", flush=True)

    try:
        async with client.stream("POST", f"{route.url}/v1/chat/completions", json=body, timeout=timeout,
                                 headers={"Authorization": f"Bearer {route.api_key}", "Accept": "text/event-stream"}) as resp:
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
            _log(f"connected in {_time.monotonic() - t_req:.1f}s")
            t_first = _time.monotonic()
            lines = resp.aiter_lines()
            # First-byte watchdog: a pooled socket the edge already closed is half-open and
            # stays silent forever. Detect it in ~30s and fail over to a fresh connection.
            # A live-but-slow server (health endpoint answers) keeps its full budget.
            pending = asyncio.ensure_future(_anext(lines))
            done, _ = await asyncio.wait({pending}, timeout=30.0)
            if not done:
                healthy = await _route_alive(client, route)
                _log(f"no first line after 30s (route {'alive' if healthy else 'DEAD'})")
                if not healthy:
                    pending.cancel()
                    raise LLMError("gpu_dropped", "No response reached the app for 30s and the GPU route is not answering; retrying on a fresh connection.",
                                   retryable=True)
                done, _ = await asyncio.wait({pending}, timeout=60.0)  # slow prefill gets a real chance
                if not done:
                    pending.cancel()
                    raise LLMError("gpu_stalled", "The GPU is online but produced no output for 90s; the request was abandoned.")
            first = pending.result()
            raw = first
            while raw is not None:
                if raw.startswith("data:"):
                    data = raw[5:].strip()
                    if data and data != "[DONE]":
                        try:
                            obj = json.loads(data)
                        except ValueError:
                            obj = None
                        if isinstance(obj, dict) and obj.get("error"):
                            message = _error_text(json.dumps({"error": obj["error"]}).encode())
                            if _is_overflow(message):
                                raise ContextOverflow("context_overflow", message)
                            raise LLMError("gpu_stream", message, retryable=False)
                        if isinstance(obj, dict):
                            usage = obj.get("usage")
                            if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                                yield {"t": "usage", "prompt": int(usage.get("prompt_tokens") or 0),
                                       "completion": int(usage.get("completion_tokens") or 0)}
                            choices = obj.get("choices") or []
                            if choices:
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
                    elif data == "[DONE]":
                        break
                try:
                    raw = await _anext(lines)
                except StopAsyncIteration:
                    break
    except httpx.ConnectTimeout as exc:
        raise LLMError("gpu_unreachable", "Timed out connecting to the GPU tunnel.", retryable=True) from exc
    except httpx.ConnectError as exc:
        raise LLMError("gpu_unreachable", "Could not connect to the GPU tunnel (it may have stopped).", retryable=True) from exc
    except httpx.ReadTimeout as exc:
        waited = _time.monotonic() - t_req
        if waited < min(idle_timeout, 25.0):
            # nothing at all came back within seconds: almost certainly a dead pooled
            # socket / half-open connection - retryable, the caller re-runs on a fresh one.
            raise LLMError("gpu_dropped", f"Connection to the GPU was half-open (silent for {waited:.0f}s); retrying on a fresh connection.",
                           retryable=True) from exc
        raise LLMError("gpu_stalled", f"The model produced no output for {int(idle_timeout)}s and the request was abandoned.") from exc
    except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
        raise LLMError("gpu_dropped", "The connection to the GPU dropped mid-response.", retryable=True) from exc
    except httpx.HTTPError as exc:
        raise LLMError("gpu_dropped", f"The connection to the GPU failed ({type(exc).__name__}).", retryable=True) from exc
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001 - any transport-layer surprise is a recoverable drop
        raise LLMError("gpu_dropped", f"The connection to the GPU failed ({type(exc).__name__}: {str(exc)[:80]}).", retryable=True) from exc
    except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
        raise LLMError("gpu_dropped", "The connection to the GPU dropped mid-response.", retryable=False) from exc
