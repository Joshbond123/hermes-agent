"""Streaming client for the OpenAI-compatible endpoint that serves the agent's model.

NVIDIA's hosted endpoint and any other OpenAI-compatible provider work through the same code path;
provider-specific request fields are added only for the providers that accept them.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from contextlib import AsyncExitStack
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .route import Route

DEAD_TUNNEL_CODES = {404, 410, 502, 503, 504, 521, 522, 523, 524, 530}


class LLMError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False, status: Optional[int] = None,
                 retry_after: Optional[float] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.status = status
        # Seconds the provider itself asked us to wait (``Retry-After``), when it said so.
        self.retry_after = retry_after


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


async def server_context(client: httpx.AsyncClient, route: Route, timeout: float = 2.5) -> int:
    """The context window the runtime *actually* serves, or 0 when it cannot be read.

    Raising a number in a config file does not give the engine more capacity, and guessing the
    other way (assuming more than the engine has) turns every long step into a ContextOverflow
    retry. llama.cpp reports the real window on /props; a gateway that does not implement it is
    simply unknown, and the caller keeps its configured budget.
    """
    if (getattr(route, "provider", "") or "") not in ("llamacpp", ""):
        # Hosted providers publish no runtime settings endpoint; the configured budget is used.
        return 0
    headers = {**tunnel_headers(route.url), "Authorization": f"Bearer {route.api_key}"}
    for path in ("/props", "/v1/props"):
        try:
            resp = await client.get(f"{route.url}{path}", headers=headers, timeout=timeout)
            if resp.status_code != 200:
                continue
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        for key in ("default_generation_settings", "generation_settings"):
            settings = data.get(key) or {}
            if isinstance(settings, dict):
                value = settings.get("n_ctx") or settings.get("context_length")
                if isinstance(value, int) and value > 0:
                    return value
        for key in ("n_ctx", "context_length", "max_context_length", "n_ctx_train"):
            value = data.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return 0


async def route_alive(client: httpx.AsyncClient, route: Route) -> bool:
    """Cheap liveness probe of the model route, used only to label a silent stream (dead vs slow)."""
    path = "/health" if (getattr(route, "provider", "") == "llamacpp") else "/v1/models"
    try:
        resp = await client.get(f"{route.url}{path}", headers={**tunnel_headers(route.url), "Authorization": f"Bearer {route.api_key}"},
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



# ==================================================================================================
# One model endpoint, one request at a time
# --------------------------------------------------------------------------------------------------
# A hosted endpoint answers 429 when it is asked for more than it will serve. The agent can start
# several runs (two tabs, a retry, a resumed task) and each of them used to fire its own request, so
# the service saw concurrent demand from a single account and throttled all of it. Every request now
# passes a per-endpoint gate: at most ``BLACKTHORN_MODEL_CONCURRENCY`` in flight (default 1), a small
# minimum gap between starts, and — after a 429 — a shared cooldown that every caller waits out, so a
# throttle is respected once instead of being hit again by the next run.
# ==================================================================================================
RATE_LIMIT_MIN_WAIT = 1.0
RATE_LIMIT_MAX_WAIT = 120.0


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, "") or default))
    except ValueError:
        return default


class EndpointGate:
    """Bounded concurrency, a minimum interval and a shared cooldown for one endpoint."""

    def __init__(self, concurrency: int = 1, min_interval: float = 0.0):
        self._slots = asyncio.Semaphore(concurrency)
        self._min_interval = max(0.0, min_interval)
        self._next_start = 0.0
        self._cooldown_until = 0.0
        self._lock = asyncio.Lock()

    def cooldown(self, seconds: float) -> None:
        until = time.monotonic() + max(0.0, seconds)
        if until > self._cooldown_until:
            self._cooldown_until = until

    def cooldown_remaining(self) -> float:
        return max(0.0, self._cooldown_until - time.monotonic())

    async def enter(self) -> None:
        await self._slots.acquire()
        # The cooldown is checked *inside* the slot so a caller that queued during a throttle does
        # not slip through the moment the slot frees.
        while True:
            wait = self.cooldown_remaining()
            if wait <= 0:
                break
            await asyncio.sleep(min(wait, 5.0))
        async with self._lock:
            gap = self._next_start - time.monotonic()
            if gap > 0:
                await asyncio.sleep(gap)
            self._next_start = time.monotonic() + self._min_interval

    def leave(self) -> None:
        self._slots.release()


_GATES: Dict[str, EndpointGate] = {}


def gate_for(url: str) -> EndpointGate:
    key = (url or "").rstrip("/")
    gate = _GATES.get(key)
    if gate is None:
        gate = EndpointGate(concurrency=_int_env("BLACKTHORN_MODEL_CONCURRENCY", 1),
                            min_interval=float(os.environ.get("BLACKTHORN_MODEL_MIN_INTERVAL_MS", "0") or 0) / 1000.0)
        _GATES[key] = gate
    return gate


def retry_after_seconds(resp: httpx.Response, detail: str = "") -> Optional[float]:
    """The provider's own instruction, from ``Retry-After`` (seconds or HTTP date) or ``X-RateLimit-Reset``."""
    raw = (resp.headers.get("retry-after") or "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime
                when = parsedate_to_datetime(raw)
                return max(0.0, when.timestamp() - time.time())
            except (TypeError, ValueError):
                pass
    for name in ("x-ratelimit-reset-requests", "x-ratelimit-reset"):
        value = (resp.headers.get(name) or "").strip().rstrip("s")
        if value:
            try:
                return max(0.0, float(value))
            except ValueError:
                pass
    # A JSON body sometimes carries the wait ("retry_after": 20).
    body = detail.lower()
    for key in ('"retry_after"', '"retry-after"'):
        at = body.find(key)
        if at >= 0:
            digits = ""
            for ch in body[at + len(key):]:
                if ch.isdigit() or ch == ".":
                    digits += ch
                elif digits:
                    break
            if digits:
                try:
                    return float(digits)
                except ValueError:
                    pass
    return None


def backoff_delay(attempt: int, retry_after: Optional[float] = None) -> float:
    """Exponential backoff with jitter, never shorter than what the provider asked for."""
    base = min(RATE_LIMIT_MAX_WAIT, 2.0 * (2 ** max(0, attempt - 1)))
    jitter = random.uniform(0, base / 3)
    delay = base + jitter
    if retry_after is not None:
        delay = max(delay, min(RATE_LIMIT_MAX_WAIT, max(RATE_LIMIT_MIN_WAIT, retry_after)))
    return min(RATE_LIMIT_MAX_WAIT, delay)


async def stream_chat(route: Route, messages: List[Dict[str, Any]], *, client: httpx.AsyncClient,
                      tools: Optional[List[Dict[str, Any]]] = None, idle_timeout: float = 150.0,
                      first_byte_timeout: float = 45.0,
                      connect_timeout: float = 20.0, temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      extra: Optional[Dict[str, Any]] = None) -> AsyncIterator[Dict[str, Any]]:
    """Yield normalised events: content | reasoning | tool | finish | usage.

    * ``content``   {"v": text}                       visible answer text
    * ``reasoning`` {"v": text}                       private thinking (callers must not forward it)
    * ``tool``      {"index", "id", "name", "args"}   incremental native tool-call fragment
    * ``finish``    {"v": reason}
    * ``usage``     {"prompt": n, "completion": n}
    """
    provider = getattr(route, "provider", "nvidia") or "nvidia"
    body: Dict[str, Any] = {"model": route.model, "messages": messages, "stream": True,
                            "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    body["max_tokens"] = int(max_tokens or 8192)
    if provider == "llamacpp":
        # llama.cpp keeps a model resident between requests; other providers reject the field.
        body.setdefault("keep_alive", "30m")
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
    gate = gate_for(route.url)
    await gate.enter()
    try:
        cm = client.stream("POST", f"{route.url}/v1/chat/completions", json=body, timeout=timeout,
                           headers={**tunnel_headers(route.url), "Authorization": f"Bearer {route.api_key}",
                                    "Accept": "text/event-stream"})
        try:
            resp = await asyncio.wait_for(stack.enter_async_context(cm), timeout=first_byte_timeout)
        except asyncio.TimeoutError as exc:
            alive = await route_alive(client, route)
            if alive:
                raise LLMError("model_stalled", f"The model sent no response within {int(first_byte_timeout)}s (the endpoint is online); the request was closed and will be retried.",
                               retryable=True) from exc
            raise LLMError("model_unreachable", f"No response from the model provider within {int(first_byte_timeout)}s; retrying on a fresh connection.",
                           retryable=True) from exc
        if resp.status_code != 200:
            detail = _error_text(await resp.aread())
            if resp.status_code == 400 and _is_overflow(detail):
                raise ContextOverflow("context_overflow", detail, status=400)
            if resp.status_code in (401, 403):
                # No model output has been accepted on a non-200 response. The caller may
                # refresh the D1 route and retry this model step (never its earlier tools).
                raise LLMError("model_auth", "The model provider did not accept this request's access key.",
                               status=resp.status_code)
            if resp.status_code == 429:
                wait = retry_after_seconds(resp, detail)
                gate_for(route.url).cooldown(wait if wait is not None else 5.0)
                # The provider is busy, not broken: this is retryable and the caller must keep the
                # task alive across it. The wait it asked for is carried with the error.
                raise LLMError("model_rate_limited",
                               "The model provider is throttling requests right now.",
                               retryable=True, status=429, retry_after=wait)
            if resp.status_code in DEAD_TUNNEL_CODES:
                raise LLMError("model_unreachable", f"The model endpoint is unreachable (HTTP {resp.status_code}).",
                               retryable=True, status=resp.status_code)
            raise LLMError("model_http", f"The model endpoint returned HTTP {resp.status_code}: {detail}",
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
                    raise LLMError("model_stalled", f"The model sent no output within {int(first_byte_timeout)}s; the request was closed and will be retried.",
                                   retryable=True) from exc
                raise LLMError("model_stalled", f"The model produced no output for {int(idle_timeout)}s and the request was abandoned.",
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
                raise LLMError("model_stream", message, retryable=False)
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
        raise LLMError("model_unreachable", "Timed out connecting to the model endpoint.", retryable=True) from exc
    except httpx.ConnectError as exc:
        raise LLMError("model_unreachable", "Could not connect to the model endpoint.", retryable=True) from exc
    except httpx.ReadTimeout as exc:
        raise LLMError("model_stalled", f"The model produced no output for {int(idle_timeout)}s and the request was abandoned.", retryable=True) from exc
    except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
        raise LLMError("model_dropped", "The connection to the model dropped mid-response.", retryable=True) from exc
    finally:
        # Always close the upstream connection: a stalled or abandoned stream must not keep
        # a dead socket in the pool for the next request.
        await stack.aclose()
        gate.leave()
