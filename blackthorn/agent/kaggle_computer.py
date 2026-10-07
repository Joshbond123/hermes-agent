"""Client for the Kaggle "computer" API exposed by the GPU gateway.

All shell / file / fetch tools run on the Kaggle machine, never on Render (Render is
the control plane only).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, Optional

import httpx

from blackthorn import http as bt_http
from blackthorn.agent.tools import ToolError
from blackthorn.route import DEAD_HTTP_STATUS, Route, RouteProvider, TunnelDown

WORKSPACE_ROOT = "/kaggle/working/blackthorn_workspace"


def normalize_path(path: Any) -> str:
    """Workspace-relative path for the gateway.

    The gateway joins whatever it receives onto the workspace root after stripping a
    leading ``/``.  Models routinely pass the absolute workspace path they were told
    about, which used to create ``…/blackthorn_workspace/kaggle/working/…`` copies.
    Accept absolute paths *inside* the workspace, reject anything that escapes it.
    """
    raw = str(path if path is not None else ".").strip().replace("\\", "/")
    if not raw:
        raw = "."
    if raw == WORKSPACE_ROOT or raw.startswith(WORKSPACE_ROOT + "/"):
        raw = raw[len(WORKSPACE_ROOT):].lstrip("/") or "."
    elif raw.startswith("/"):
        raise ToolError(
            f"absolute path '{raw}' is outside the workspace; use a path relative to {WORKSPACE_ROOT} "
            "(or run a terminal command to reach other locations)"
        )
    parts = []
    for seg in raw.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if not parts:
                raise ToolError("path escapes the workspace")
            parts.pop()
        else:
            parts.append(seg)
    return "/".join(parts) or "."


_HTML_ERROR = re.compile(r"(?i)cloudflare|tunnel|error 10\d\d|<html")


class KaggleComputer:
    def __init__(self, route_provider: RouteProvider, *, on_tunnel_failure: Optional[Any] = None) -> None:
        self._route = route_provider
        self._on_failure = on_tunnel_failure

    async def _base(self) -> Route:
        route = await self._route()
        if route is None or not route.url:
            raise ToolError("the Kaggle computer is offline — start the GPU first")
        return route

    async def call(
        self, path: str, payload: Optional[Dict[str, Any]] = None, *, method: str = "POST", timeout: float = 90.0
    ) -> Dict[str, Any]:
        route = await self._base()
        headers = {"Authorization": f"Bearer {route.api_key}"}
        try:
            client = bt_http.client()
            if method == "GET":
                resp = await client.get(route.url + path, headers=headers, timeout=timeout)
            else:
                resp = await client.post(route.url + path, json=payload or {}, headers=headers, timeout=timeout)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError) as exc:
            await self._failed(f"{type(exc).__name__}")
            raise TunnelDown(f"the Kaggle computer is unreachable ({type(exc).__name__})") from exc
        except httpx.TimeoutException as exc:
            raise ToolError(f"the Kaggle computer did not answer within {int(timeout)}s") from exc
        if resp.status_code == 401:
            raise ToolError("the Kaggle computer rejected the credentials (HTTP 401)")
        ctype = (resp.headers.get("content-type") or "").lower()
        if resp.status_code in DEAD_HTTP_STATUS or ("text/html" in ctype and _HTML_ERROR.search(resp.text[:2000])):
            await self._failed(f"http {resp.status_code}")
            raise TunnelDown(
                f"the GPU tunnel is down (HTTP {resp.status_code}); the session may have stopped",
                http_status=resp.status_code,
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise ToolError(f"unexpected response from the Kaggle computer (HTTP {resp.status_code})") from exc
        if resp.status_code >= 400:
            detail = data.get("detail") if isinstance(data, dict) else data
            raise ToolError(f"the Kaggle computer returned HTTP {resp.status_code}: {detail}")
        if not isinstance(data, dict):
            raise ToolError("unexpected response shape from the Kaggle computer")
        return data

    async def _failed(self, why: str) -> None:
        if self._on_failure is not None:
            try:
                res = self._on_failure(why)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:  # noqa: BLE001
                pass
