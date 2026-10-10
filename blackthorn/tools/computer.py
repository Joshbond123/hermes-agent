"""Tools that act on the remote Kaggle computer through the gateway's /computer/* API."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import httpx

from ..route import RouteError, RouteResolver
from .base import ToolContext, ToolError, ToolResult, ToolSpec, clip

DEAD = {404, 410, 502, 503, 504, 521, 522, 523, 524, 530}

# Commands that could destroy the host or the deployment are refused outright (kept from the original agent).
DENY_PATTERNS = (
    r"rm\s+-rf\s+/(\s|$)", r"rm\s+-rf\s+/\*", r"\bmkfs", r"\bdd\s+if=", r"\bshutdown\b", r"\breboot\b",
    r":\(\)\s*\{", r">\s*/dev/sd", r"chmod\s+-R\s+777\s+/(\s|$)", r"\bmv\s+/(\s|$)", r"kill\s+-9\s+1\b", r"\bkillall5\b",
    r"(sudo\s+)?rm\s+(-[a-zA-Z]*\s+)*\/(\s|$)",  # wiping the root filesystem, with or without sudo
)


class ComputerClient:
    """Talks to the agent's personal computer. A dedicated computer host (id='computer' in
    kaggle_gpu_state) is preferred; when none is enrolled the model host answers instead."""

    def __init__(self, resolver: RouteResolver, http: httpx.AsyncClient, store: Any = None):
        self._resolver = resolver
        self._http = http
        self._store = store

    async def _endpoint(self) -> tuple:
        if self._store is not None:
            try:
                row = await self._store.computer_row()
            except Exception:  # noqa: BLE001 - fall back to the model host
                row = {}
            url = str(row.get("tunnel_url") or "").rstrip("/")
            if url.endswith("/v1"):
                url = url[:-3]
            key = str(row.get("api_key") or "")
            status = str(row.get("status") or "").upper()
            offline = {
                "", "OFF", "OFFLINE", "GPU_STOPPED_SAVING_QUOTA", "STOPPED", "STOPPING_KAGGLE_GPU",
                "ERROR", "FAILED", "BOOT_FAILED", "TUNNEL_ERROR", "GPU_UNAVAILABLE",
            }
            # Only use the dedicated computer host when it has a live tunnel.
            if url and key and status not in offline:
                return url, key
        route = await self._resolver.get()
        return route.url, route.api_key

    async def call(self, path: str, payload: Dict[str, Any] | None = None, *, method: str = "POST",
                   timeout: float = 90.0) -> Dict[str, Any]:
        """Call the remote computer through the relay. Retries transient auth/timeout/502 blips.

        The GPU relay is in-process on Render: a deploy or a missed long-poll can drop a single
        request even while the notebook is healthy. Retrying with a refreshed key is the durable fix.
        """
        import asyncio
        # Never let a model-chosen short timeout kill a healthy but busy notebook.
        timeout = max(45.0, float(timeout))
        last_err: Optional[Exception] = None
        attempts = 4
        for attempt in range(1, attempts + 1):
            try:
                base_url, api_key = await self._endpoint()
            except RouteError as exc:
                raise ToolError("offline", "The remote computer is offline.", hint=exc.message) from exc
            try:
                resp = await self._http.request(
                    method, f"{base_url}{path}", json=payload if method == "POST" else None, timeout=timeout,
                    headers={"Authorization": f"Bearer {api_key}"})
            except httpx.ConnectError as exc:
                self._resolver.invalidate()
                last_err = ToolError("offline", "Could not reach the remote computer (the tunnel may have stopped).",
                                     hint="Open the GPU panel and check that the GPU is Ready.")
                if attempt >= attempts:
                    raise last_err from exc
                await asyncio.sleep(min(4.0, 0.5 * attempt))
                continue
            except httpx.TimeoutException as exc:
                last_err = ToolError("timeout", f"The remote computer did not answer within {int(timeout)}s.")
                if attempt >= attempts:
                    raise last_err from exc
                self._resolver.invalidate()
                await asyncio.sleep(min(4.0, 0.6 * attempt))
                continue
            except httpx.HTTPError as exc:
                raise ToolError("network", f"Network error talking to the remote computer: {type(exc).__name__}") from exc

            ctype = (resp.headers.get("content-type") or "").lower()
            if resp.status_code in DEAD or "text/html" in ctype:
                self._resolver.invalidate()
                last_err = ToolError("offline", f"The remote computer tunnel is down (HTTP {resp.status_code}).",
                                     hint="Open the GPU panel and turn the GPU on again if it is not Ready.")
                if attempt >= attempts:
                    raise last_err
                await asyncio.sleep(min(4.0, 0.5 * attempt))
                continue
            if resp.status_code in (401, 403):
                # Key rotation / multi-worker race: refresh D1 key and retry.
                self._resolver.invalidate()
                last_err = ToolError("auth", "The remote computer rejected the access key.",
                                     hint="Turn the GPU off and on to refresh it.")
                if attempt >= attempts:
                    raise last_err
                await asyncio.sleep(0.4 * attempt)
                continue
            try:
                data = resp.json()
            except ValueError:
                raise ToolError("protocol", f"The remote computer returned a non-JSON response (HTTP {resp.status_code}).") from None
            if resp.status_code >= 400:
                # 502/503 from the relay are transient — retry.
                if resp.status_code in (502, 503, 504) and attempt < attempts:
                    self._resolver.invalidate()
                    await asyncio.sleep(min(4.0, 0.6 * attempt))
                    continue
                raise ToolError("rejected", str(data.get("detail") or data.get("error") or f"HTTP {resp.status_code}")[:300])
            if not isinstance(data, dict):
                raise ToolError("protocol", "Unexpected response shape from the remote computer.")
            return data
        if last_err:
            raise last_err
        raise ToolError("offline", "The remote computer is offline.")


def _need(data: Dict[str, Any]) -> Dict[str, Any]:
    if data.get("ok") is False:
        raise ToolError("failed", str(data.get("error") or data.get("output") or "operation failed")[:400])
    return data


async def run_command(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    cmd = args["command"].strip()
    for pat in DENY_PATTERNS:
        if re.search(pat, cmd, re.I):
            return ToolResult.failure("refused: the command matches a destructive pattern and was not run",
                                      hint="Choose a safer command or explain to the user why it cannot be run.",
                                      data={"kind": "denied"})
    # Floor command timeout: short model-chosen values (e.g. 20s) were the main source of
    # "did not answer within 20s" while the notebook was merely busy on the shared relay.
    timeout = max(60, int(args.get("timeout_seconds") or 90))
    data = await ctx.computer.call("/computer/exec", {"command": cmd, "timeout_seconds": timeout, "cwd": args.get("cwd") or "."},
                                   timeout=float(timeout + 60))
    out = str(data.get("output") or "")
    code = data.get("exit_code", -1)
    body, cut = clip(out, ctx.settings.tool_result_chars, "ends")
    ok = bool(data.get("ok", True)) and code == 0
    head = f"exit code {code}" + ("" if ok else " (command failed)")
    ui, _ = clip(out.strip(), ctx.settings.ui_result_chars, "ends")
    return ToolResult(ok=True, content=f"{head}\n{body}", summary=head, truncated=cut,
                      data={"exit_code": code, "output": ui, "failed": not ok})


async def list_files(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    data = _need(await ctx.computer.call("/computer/list_files", {"path": args.get("path") or "."}))
    lines = str(data.get("listing") or "(empty)").splitlines()
    shown = lines[:80]
    text = "\n".join(shown) + (f"\n[… {len(lines) - 80} more entries …]" if len(lines) > 80 else "")
    return ToolResult(ok=True, content=f"{data.get('path')}\n{text}", summary=f"{len(lines)} entries", truncated=len(lines) > 80,
                      data={"output": "\n".join(lines[:30])})


async def read_file(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    data = _need(await ctx.computer.call("/computer/read_file", {"path": args["path"]}))
    text = str(data.get("content") or "")
    offset, limit = int(args.get("offset") or 0), int(args.get("max_chars") or ctx.settings.tool_result_chars)
    piece = text[offset: offset + limit]
    more = len(text) - (offset + len(piece))
    note = f"\n[… {more} more characters; call again with offset={offset + len(piece)} …]" if more > 0 else ""
    return ToolResult(ok=True, content=f"{data.get('path')} (characters {offset}-{offset + len(piece)} of {len(text)})\n{piece}{note}",
                      summary=f"{len(text)} characters", truncated=more > 0, data={"chars": len(text)})


async def write_file(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    content = args["content"]
    data = _need(await ctx.computer.call("/computer/write_file", {"path": args["path"], "content": content}))
    return ToolResult(ok=True, content=f"Wrote {data.get('bytes', len(content.encode()))} bytes to {data.get('path')}",
                      summary=f"{data.get('bytes', len(content.encode()))} bytes written", data={"bytes": data.get("bytes")})


async def fetch_url(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    url = args["url"].strip()
    if not re.match(r"^https?://", url, re.I):
        return ToolResult.failure("url must start with http:// or https://")
    data = _need(await ctx.computer.call("/computer/fetch_url", {"url": url}, timeout=60.0))
    text = str(data.get("text") or "")
    offset, limit = int(args.get("offset") or 0), int(args.get("max_chars") or ctx.settings.tool_result_chars)
    piece = text[offset: offset + limit]
    more = len(text) - (offset + len(piece))
    note = f"\n[… {more} more characters; call again with offset={offset + len(piece)} to continue …]" if more > 0 else ""
    return ToolResult(ok=True, content=f"{url} (characters {offset}-{offset + len(piece)} of {len(text)})\n{piece}{note}",
                      summary=f"{len(text)} characters of text", truncated=more > 0, data={"chars": len(text)})


async def computer_info(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    data = await ctx.computer.call("/computer/info", method="GET", timeout=60.0)
    keep = {k: data.get(k) for k in ("hostname", "platform", "python", "gpu", "cuda_available", "disk_free_gb",
                                      "disk_total_gb", "workspace") if k in data}
    lines = [f"{k}: {str(v).replace(chr(10), ' + ')}" for k, v in keep.items()]
    return ToolResult(ok=True, content="\n".join(lines), summary=str(keep.get("gpu") or "online").replace("\n", " + ")[:120], data=keep)


SCHEMAS = {
    "run_command": {"type": "object", "properties": {
        "command": {"type": "string", "minLength": 1, "maxLength": 6000, "description": "bash command"},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 3600, "default": 60}}, "required": ["command"]},
    "list_files": {"type": "object", "properties": {"path": {"type": "string", "default": ".", "description": "directory"}}},
    "read_file": {"type": "object", "properties": {
        "path": {"type": "string", "minLength": 1}, "offset": {"type": "integer", "minimum": 0, "default": 0},
        "max_chars": {"type": "integer", "minimum": 200, "maximum": 6000}}, "required": ["path"]},
    "write_file": {"type": "object", "properties": {
        "path": {"type": "string", "minLength": 1, "maxLength": 300},
        "content": {"type": "string", "maxLength": 120000}}, "required": ["path", "content"]},
    "fetch_url": {"type": "object", "properties": {
        "url": {"type": "string", "minLength": 8, "maxLength": 2000}, "offset": {"type": "integer", "minimum": 0, "default": 0},
        "max_chars": {"type": "integer", "minimum": 200, "maximum": 6000}}, "required": ["url"]},
    "computer_info": {"type": "object", "properties": {}},
}


def specs() -> list[ToolSpec]:
    return [
        ToolSpec("run_command", "Run a bash command on the remote Linux computer (Python, GPU). Time-limited.",
                 SCHEMAS["run_command"], run_command, timeout=3660.0, kind="exec", describe=lambda a: f"$ {a.get('command', '')}"),
        ToolSpec("list_files", "List a remote workspace directory.", SCHEMAS["list_files"], list_files,
                 timeout=45.0, describe=lambda a: str(a.get("path") or ".")),
        ToolSpec("read_file", "Read a remote text file (offset to page).", SCHEMAS["read_file"], read_file,
                 timeout=45.0, describe=lambda a: str(a.get("path") or "")),
        ToolSpec("write_file", "Write a remote text file.", SCHEMAS["write_file"], write_file,
                 timeout=60.0, kind="write", describe=lambda a: f"{a.get('path', '')} ({len(str(a.get('content') or ''))} characters)"),
        ToolSpec("fetch_url", "Open a web page as text (offset to page).", SCHEMAS["fetch_url"], fetch_url,
                 timeout=75.0, kind="net", describe=lambda a: str(a.get("url") or "")),
        ToolSpec("computer_info", "Remote computer GPU, disk, OS.", SCHEMAS["computer_info"], computer_info,
                 timeout=90.0, describe=lambda a: ""),
    ]
