"""Tools that act on the remote Kaggle computer through the gateway /computer/* API."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..route import RouteError
from .base import ToolContext, ToolError, ToolResult, ToolSpec, clip

log = logging.getLogger("blackthorn.tools.computer")

# Permanent tunnel death (do not keep retrying the same stale URL forever without refresh).
DEAD = {404, 410, 521, 522, 523, 530}
# Recoverable tunnel/gateway failures — refresh endpoint and retry with backoff.
TRANSIENT = {401, 502, 503, 504}

# Updated on every successful computer call — GPU auto-off consults this.
_LAST_COMPUTER_OK: float = 0.0


def computer_recently_active(window_s: float = 300.0) -> bool:
    """True if a computer tool succeeded within ``window_s`` seconds."""
    return _LAST_COMPUTER_OK > 0 and (time.time() - _LAST_COMPUTER_OK) < window_s


class ComputerClient:
    """HTTP client for the agent's personal computer (Kaggle tunnel).

    A dedicated computer host (id='computer' in kaggle_gpu_state) is preferred; when it is
    offline the model host may expose /computer/* as a fallback.
    """

    def __init__(self, http: httpx.AsyncClient, resolver: Any) -> None:
        self.http = http
        self.resolver = resolver
        self._cached_url: Optional[str] = None
        self._cached_key: Optional[str] = None
        self._cached_at: float = 0.0

    def invalidate(self) -> None:
        self._cached_url = None
        self._cached_key = None
        self._cached_at = 0.0

    async def _endpoint(self) -> Tuple[str, str]:
        now = time.time()
        if self._cached_url and self._cached_key and (now - self._cached_at) < 30.0:
            return self._cached_url, self._cached_key
        # Prefer dedicated computer route when resolver supports it.
        route = None
        err: Optional[Exception] = None
        for getter in ("get_computer", "computer", "get"):
            fn = getattr(self.resolver, getter, None)
            if fn is None:
                continue
            try:
                route = await fn() if asyncio.iscoroutinefunction(fn) else fn()
                if route:
                    break
            except Exception as exc:
                err = exc
                continue
        if route is None:
            raise RouteError("offline", f"No computer route available: {err}", retryable=True)
        url = str(getattr(route, "base_url", None) or getattr(route, "url", "") or "").rstrip("/")
        key = str(getattr(route, "api_key", None) or getattr(route, "key", "") or "")
        if not url:
            raise RouteError("offline", "Computer route has no URL.", retryable=True)
        self._cached_url, self._cached_key, self._cached_at = url, key, now
        return url, key

    async def post(self, path: str, body: Optional[Dict[str, Any]] = None, *, timeout: float = 60.0) -> Dict[str, Any]:
        timeout = max(float(timeout), 60.0)
        http_timeout = timeout + 60.0
        attempts = 8
        last_err: Optional[ToolError] = None
        for attempt in range(1, attempts + 1):
            try:
                url, api_key = await self._endpoint()
            except RouteError as exc:
                raise ToolError("computer_offline", str(exc), retryable=True) from exc
            headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
            try:
                resp = await self.http.request(
                    "POST",
                    f"{url}{path}",
                    headers=headers,
                    json=body or {},
                    timeout=httpx.Timeout(http_timeout, connect=20.0),
                )
            except httpx.TimeoutException:
                last_err = ToolError(
                    "timeout",
                    f"The remote computer did not answer within {int(timeout)}s.",
                    retryable=True,
                )
                self.invalidate()
                await asyncio.sleep(min(0.5 * (2 ** (attempt - 1)), 12.0))
                continue
            except httpx.HTTPError as exc:
                last_err = ToolError("network", f"Computer request failed: {exc}", retryable=True)
                self.invalidate()
                await asyncio.sleep(min(0.5 * (2 ** (attempt - 1)), 12.0))
                continue

            ctype = (resp.headers.get("content-type") or "").lower()
            if resp.status_code in TRANSIENT:
                self.invalidate()
                last_err = ToolError(
                    "offline",
                    f"The remote computer tunnel is down (HTTP {resp.status_code}); retrying.",
                    retryable=True,
                )
                if attempt >= attempts:
                    raise last_err
                await asyncio.sleep(min(0.5 * (2 ** (attempt - 1)), 12.0))
                continue
            if resp.status_code in DEAD or "text/html" in ctype:
                self.invalidate()
                detail = ""
                try:
                    detail = str((resp.json() or {}).get("detail") or "")[:180]
                except Exception:
                    detail = (resp.text or "")[:120]
                msg = detail or f"The remote computer tunnel is down (HTTP {resp.status_code})."
                last_err = ToolError("offline", msg, retryable=True)
                if attempt >= attempts:
                    raise last_err
                await asyncio.sleep(min(0.5 * (2 ** (attempt - 1)), 12.0))
                continue
            if resp.status_code >= 400:
                raise ToolError(
                    "computer_error",
                    f"Computer returned HTTP {resp.status_code}: {(resp.text or '')[:300]}",
                    retryable=resp.status_code >= 500,
                )
            try:
                global _LAST_COMPUTER_OK
                data = resp.json()
                _LAST_COMPUTER_OK = time.time()
                return data
            except ValueError as exc:
                raise ToolError(
                    "computer_error",
                    f"Computer returned non-JSON: {(resp.text or '')[:200]}",
                    retryable=False,
                ) from exc
        assert last_err is not None
        raise last_err

    async def exec(self, command: str, timeout: float = 120.0, cwd: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"command": command, "timeout": int(timeout)}
        if cwd:
            body["cwd"] = cwd
        return await self.post("/computer/exec", body, timeout=timeout)

    async def info(self) -> Dict[str, Any]:
        return await self.post("/computer/info", {}, timeout=60.0)


async def _client(ctx: ToolContext) -> ComputerClient:
    client = getattr(ctx, "computer", None)
    if client is None:
        raise ToolError("computer_offline", "Computer client is not configured.", retryable=False)
    return client


async def run_command(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    cmd = str(args.get("command") or "").strip()
    if not cmd:
        raise ToolError("bad_args", "command is required")
    timeout = float(args.get("timeout") or 120)
    cwd = args.get("cwd")
    client = await _client(ctx)
    data = await client.exec(cmd, timeout=timeout, cwd=str(cwd) if cwd else None)
    code = int(data.get("exit_code") or data.get("code") or 0)
    out = str(data.get("stdout") or data.get("output") or "")
    err = str(data.get("stderr") or "")
    text = out if not err else (out + ("\n" if out else "") + err)
    text = clip(text, ctx.settings.tool_result_chars, "ends")
    ok = bool(data.get("ok", True)) and code == 0
    return ToolResult(ok, text or f"(exit {code})", {"exit_code": code})


async def computer_info(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    client = await _client(ctx)
    data = await client.info()
    keys = ("hostname", "cwd", "workspace", "disk_free_gb", "disk_total_gb", "python", "platform")
    summary = {k: data.get(k) for k in keys if k in data}
    return ToolResult(True, str(summary or data)[:2000], data if isinstance(data, dict) else {})


def tool_specs() -> List[ToolSpec]:
    return [
        ToolSpec(
            "run_command",
            "Run a shell command on the remote Kaggle computer workspace.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "number"},
                    "cwd": {"type": "string"},
                },
                "required": ["command"],
            },
            run_command,
            timeout=300.0,
            describe=lambda a: str(a.get("command") or "")[:80],
        ),
        ToolSpec(
            "computer_info",
            "Return hostname, workspace path, and disk usage for the remote computer.",
            {"type": "object", "properties": {}},
            computer_info,
            timeout=90.0,
            describe=lambda a: "",
        ),
    ]
