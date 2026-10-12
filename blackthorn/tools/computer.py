"""Tools that act on the remote Kaggle computer through the /computer/* API."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..route import Route, RouteError
from .base import ToolContext, ToolError, ToolResult, ToolSpec, clip

log = logging.getLogger("blackthorn.tools.computer")

# Permanent tunnel death codes (refresh URL still helps once).
DEAD = {404, 410, 521, 522, 523, 530}
# Recoverable edge/gateway failures during long agent tasks.
TRANSIENT = {401, 502, 503, 504}

_LAST_COMPUTER_OK: float = 0.0


def computer_recently_active(window_s: float = 300.0) -> bool:
    return _LAST_COMPUTER_OK > 0 and (time.time() - _LAST_COMPUTER_OK) < window_s


class ComputerClient:
    """Call the agent computer with dual-channel failover.

    Preferred order:
      1) dedicated computer tunnel (D1 id=computer)
      2) model host /computer/* fallback (D1 id=primary)

    On HTTP 502/503/504 the client invalidates caches, switches channel, and
    retries with exponential backoff so mid-task tunnel blips do not fail the agent.
    """

    def __init__(self, http: httpx.AsyncClient, resolver: Any) -> None:
        self.http = http
        self.resolver = resolver
        self._cache: Dict[str, Tuple[float, str, str]] = {}  # role -> (ts, url, key)

    def invalidate(self) -> None:
        self._cache.clear()
        inv = getattr(self.resolver, "invalidate", None)
        if callable(inv):
            try:
                inv()
            except Exception:
                pass
        inv_c = getattr(self.resolver, "invalidate_computer", None)
        if callable(inv_c):
            try:
                inv_c()
            except Exception:
                pass

    async def _route_for(self, role: str, *, force: bool = False) -> Tuple[str, str]:
        now = time.time()
        if not force and role in self._cache and (now - self._cache[role][0]) < 20.0:
            _, url, key = self._cache[role]
            return url, key
        route: Optional[Route] = None
        if role == "computer":
            getter = getattr(self.resolver, "get_computer", None)
            if getter is not None:
                route = await getter(force=force) if asyncio.iscoroutinefunction(getter) else getter()
        if route is None:
            route = await self.resolver.get(force=force)
        url = str(getattr(route, "base_url", "") or "").rstrip("/")
        key = str(getattr(route, "api_key", "") or "")
        if not url:
            raise RouteError("offline", f"No {role} tunnel URL published.", retryable=True)
        self._cache[role] = (now, url, key)
        return url, key

    async def _channels(self) -> List[str]:
        """Ordered channel list: dedicated computer first when available."""
        channels = ["computer", "model"]
        # If get_computer just falls back to model, still try once then model.
        return channels

    async def post(
        self,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        *,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        timeout = max(float(timeout), 60.0)
        http_timeout = timeout + 90.0
        attempts = 10
        last_err: Optional[ToolError] = None
        channels = await self._channels()
        for attempt in range(1, attempts + 1):
            role = channels[(attempt - 1) % len(channels)]
            force = attempt > 1
            try:
                url, api_key = await self._route_for(role, force=force)
            except RouteError as exc:
                last_err = ToolError("computer_offline", str(exc), retryable=True)
                await asyncio.sleep(min(0.4 * attempt, 8.0))
                continue
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                
            }
            try:
                resp = await self.http.request(
                    "POST",
                    f"{url}{path}",
                    headers=headers,
                    json=body or {},
                    timeout=httpx.Timeout(http_timeout, connect=25.0),
                )
            except httpx.TimeoutException:
                last_err = ToolError(
                    "timeout",
                    f"The remote computer did not answer within {int(timeout)}s.",
                    retryable=True,
                )
                self._cache.pop(role, None)
                await asyncio.sleep(min(0.5 * (2 ** (attempt - 1)), 15.0))
                continue
            except httpx.HTTPError as exc:
                last_err = ToolError("network", f"Computer request failed: {exc}", retryable=True)
                self._cache.pop(role, None)
                await asyncio.sleep(min(0.5 * (2 ** (attempt - 1)), 15.0))
                continue

            ctype = (resp.headers.get("content-type") or "").lower()
            code = resp.status_code
            if code in TRANSIENT or code in DEAD or "text/html" in ctype:
                self._cache.pop(role, None)
                # Force resolver refresh next time
                if hasattr(self.resolver, "invalidate"):
                    try:
                        self.resolver.invalidate()
                    except Exception:
                        pass
                detail = ""
                try:
                    detail = str((resp.json() or {}).get("detail") or "")[:180]
                except Exception:
                    detail = (resp.text or "")[:120]
                last_err = ToolError(
                    "offline",
                    detail
                    or f"The remote computer tunnel is down (HTTP {code}) via {role}; retrying.",
                    retryable=True,
                )
                log.warning(
                    "computer %s failed HTTP %s on %s (attempt %s/%s)",
                    path, code, role, attempt, attempts,
                )
                if attempt >= attempts:
                    raise last_err
                await asyncio.sleep(min(0.6 * (2 ** (attempt - 1)), 15.0))
                continue
            if code >= 400:
                raise ToolError(
                    "computer_error",
                    f"Computer returned HTTP {code}: {(resp.text or '')[:300]}",
                    retryable=code >= 500,
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


async def run_command(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    cmd = str(args.get("command") or "").strip()
    if not cmd:
        raise ToolError("bad_args", "command is required")
    timeout = float(args.get("timeout") or 120)
    cwd = args.get("cwd")
    client = getattr(ctx, "computer", None)
    if client is None:
        raise ToolError("computer_offline", "Computer client is not configured.", retryable=False)
    data = await client.exec(cmd, timeout=timeout, cwd=str(cwd) if cwd else None)
    code = int(data.get("exit_code") or data.get("code") or 0)
    out = str(data.get("stdout") or data.get("output") or "")
    err = str(data.get("stderr") or "")
    text = out if not err else (out + ("\n" if out else "") + err)
    text = clip(text, ctx.settings.tool_result_chars, "ends")
    ok = bool(data.get("ok", True)) and code == 0
    return ToolResult(ok, text or f"(exit {code})", {"exit_code": code})


async def computer_info(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    client = getattr(ctx, "computer", None)
    if client is None:
        raise ToolError("computer_offline", "Computer client is not configured.", retryable=False)
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
