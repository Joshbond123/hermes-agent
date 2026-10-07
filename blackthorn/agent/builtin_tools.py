"""The tools Blackthorn offers the model (schemas, permissions, limits).

Descriptions say *when a tool is appropriate*; whether to call one is always the
model's decision (native tool calling).  Every tool is bounded: argument validation
(see :mod:`blackthorn.agent.tools`), a timeout, an output cap, and — for the shell —
a deny-list of host-destroying commands.
"""

from __future__ import annotations

import ipaddress
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from blackthorn import d1 as d1mod
from blackthorn.agent.kaggle_computer import WORKSPACE_ROOT, KaggleComputer, normalize_path
from blackthorn.agent.tools import Tool, ToolContext, ToolError, ToolRegistry, ToolResult

# Commands that would destroy the machine or take the model server down with it.
DENY_PATTERNS = [
    (r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rR][a-zA-Z]*\s+(-[a-zA-Z]+\s+)*(/|~|\$HOME|/\*)(\s|$)", "recursive delete of / or ~"),
    (r"\bmkfs(\.\w+)?\b", "filesystem formatting"),
    (r"\bdd\s+[^|;]*of=/dev/", "raw device write"),
    (r">\s*/dev/(sd|nvme|vd)[a-z0-9]*", "raw device write"),
    (r"\b(shutdown|reboot|poweroff|halt)\b", "power control"),
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "fork bomb"),
    (r"\bchmod\s+-R\s+[0-7]{3,4}\s+/(\s|$)", "recursive chmod of /"),
    (r"\bkill\s+-9\s+(-1|1)(\s|$)", "killing init / all processes"),
    (r"\bkillall5\b", "killing all processes"),
    (r"\b(pkill|killall)\b[^|;]*\b(ollama|llama|cloudflared|gateway|uvicorn)\b", "stopping the model server"),
    (r"\bkill\b[^|;]*\$\(\s*pgrep[^)]*\b(ollama|llama|cloudflared|uvicorn)\b", "stopping the model server"),
    (r"\bsudo\s+rm\b", "privileged delete"),
]
_DENY = [(re.compile(p, re.I), why) for p, why in DENY_PATTERNS]


def check_command(command: str) -> None:
    for pattern, why in _DENY:
        if pattern.search(command):
            raise ToolError(f"refused: the command matches a blocked pattern ({why})")


def check_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ToolError("url must start with http:// or https://")
    host = parsed.hostname.lower()
    if host in ("localhost", "metadata.google.internal") or host.endswith(".local") or host.endswith(".internal"):
        raise ToolError("refused: local and internal addresses are not allowed")
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ToolError("refused: private network addresses are not allowed")
    except ValueError:
        pass  # a hostname
    return url.strip()


# --------------------------------------------------------------------------- #
# web search keys (kept in D1 / env — never in the repository)
# --------------------------------------------------------------------------- #
class SearchKeys:
    """Round-robin over the configured Tavily keys; a key that is rejected is skipped."""

    def __init__(self, loader: Optional[Callable[[], Awaitable[List[str]]]] = None) -> None:
        self._loader = loader or self._default_loader
        self._keys: List[str] = []
        self._loaded_at = 0.0
        self._cursor = 0

    @staticmethod
    async def _default_loader() -> List[str]:
        import os

        keys: List[str] = []
        raw = ""
        try:
            raw = await d1mod.aget_meta("tavily_api_keys")
        except Exception:  # noqa: BLE001 - D1 hiccup must not break search entirely
            raw = ""
        if raw:
            try:
                data = json.loads(raw)
                keys = [k for k in (data.get("keys") if isinstance(data, dict) else data) or [] if isinstance(k, str)]
            except ValueError:
                keys = []
        for name in ("TAVILY_API_KEYS", "TAVILY_API_KEY"):
            keys += [k.strip() for k in os.environ.get(name, "").split(",") if k.strip()]
        seen: List[str] = []
        for key in keys:
            if key not in seen:
                seen.append(key)
        return seen

    async def ordered(self) -> List[str]:
        if not self._keys or time.monotonic() - self._loaded_at > 300:
            self._keys = await self._loader()
            self._loaded_at = time.monotonic()
        if not self._keys:
            return []
        start = self._cursor % len(self._keys)
        self._cursor += 1
        return self._keys[start:] + self._keys[:start]


# --------------------------------------------------------------------------- #
@dataclass
class Deps:
    computer: KaggleComputer
    search_keys: SearchKeys
    http_client: Callable[[], httpx.AsyncClient]


def _obj(properties: Dict[str, Any], required: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


def build_registry(deps: Deps) -> ToolRegistry:
    reg = ToolRegistry()

    # ---- web_search -------------------------------------------------------
    async def web_search(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = args["query"].strip()
        keys = await deps.search_keys.ordered()
        if not keys:
            raise ToolError("web search is not configured (no search API key available)")
        last_status = 0
        for key in keys:
            resp = await deps.http_client().post(
                "https://api.tavily.com/search",
                json={"query": query, "max_results": int(args.get("max_results") or 5), "include_answer": False},
                headers={"Authorization": f"Bearer {key}"},
                timeout=25.0,
            )
            last_status = resp.status_code
            if resp.status_code in (401, 403, 429, 432, 433):
                continue  # this key is exhausted/invalid — try the next one
            if resp.status_code >= 400:
                raise ToolError(f"search provider returned HTTP {resp.status_code}")
            data = resp.json()
            results = data.get("results") or []
            if not results:
                return ToolResult(True, f'No web results for "{query}".')
            lines = [f'Web results for "{query}":']
            for i, item in enumerate(results, 1):
                title = (item.get("title") or "Untitled").strip()
                url = (item.get("url") or "").strip()
                snippet = re.sub(r"\s+", " ", (item.get("content") or "")).strip()[:500]
                lines.append(f"{i}. {title}\n   {url}\n   {snippet}")
            return ToolResult(True, "\n".join(lines), {"results": len(results)})
        raise ToolError(f"all search keys were rejected (last HTTP {last_status})")

    reg.register(Tool(
        name="web_search",
        label="Web search",
        description=(
            "Search the live web for current or external information you cannot reliably know: recent "
            "events and news, prices, product or library versions, documentation, facts that change. "
            "Returns titles, URLs and snippets. Not needed for greetings, opinions, or things you can "
            "answer or compute yourself."
        ),
        parameters=_obj({
            "query": {"type": "string", "minLength": 2, "maxLength": 400, "description": "What to search for."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5},
        }, ["query"]),
        handler=web_search,
        timeout_s=90.0,
    ))

    # ---- fetch_url --------------------------------------------------------
    async def fetch_url(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        url = check_url(args["url"])
        data = await deps.computer.call("/computer/fetch_url", {"url": url}, timeout=70.0)
        if not data.get("ok"):
            raise ToolError(str(data.get("error") or "fetch failed"))
        return ToolResult(True, f"Page text of {url}:\n{data.get('text') or '(empty page)'}")

    reg.register(Tool(
        name="fetch_url",
        label="Fetch page",
        description=(
            "Download one web page (http/https) and return its readable text. Use it when the user gives "
            "a URL, or after a search when you need the full content of a specific result."
        ),
        parameters=_obj({"url": {"type": "string", "minLength": 8, "maxLength": 2000}}, ["url"]),
        handler=fetch_url,
        timeout_s=90.0,
    ))

    # ---- terminal ---------------------------------------------------------
    async def terminal(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = args["command"].strip()
        check_command(command)
        timeout = int(args.get("timeout_seconds") or 60)
        cwd = normalize_path(args.get("cwd") or ".")
        data = await deps.computer.call(
            "/computer/exec",
            {"command": command, "timeout_seconds": timeout, "cwd": cwd},
            timeout=float(timeout) + 30.0,
        )
        code = data.get("exit_code", -1)
        output = str(data.get("output") or "(no output)")
        ok = bool(data.get("ok")) and code == 0
        return ToolResult(ok, f"[exit {code}]\n{output}", {"exit_code": code})

    reg.register(Tool(
        name="terminal",
        label="Terminal",
        description=(
            "Run a shell command on the Kaggle computer (Linux, 2x Tesla T4 GPUs, internet, Python) in the "
            "persistent workspace and get the exit code and combined output. Use it to run code, install "
            "packages, inspect the machine or GPUs, or do any command-line task. Not for questions you "
            "can answer directly."
        ),
        parameters=_obj({
            "command": {"type": "string", "minLength": 1, "maxLength": 8000, "description": "Shell command."},
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300, "default": 60},
            "cwd": {"type": "string", "default": ".", "description": "Directory relative to the workspace."},
        }, ["command"]),
        handler=terminal,
        timeout_s=340.0,
        max_output_chars=9000,
    ))

    # ---- files ------------------------------------------------------------
    async def list_files(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = normalize_path(args.get("path") or ".")
        data = await deps.computer.call("/computer/list_files", {"path": path}, timeout=30.0)
        if not data.get("ok"):
            raise ToolError(str(data.get("error") or "could not list directory"))
        return ToolResult(True, f"{data.get('path')}\n{data.get('listing') or '(empty)'}")

    reg.register(Tool(
        name="list_files",
        label="List files",
        description=f"List a directory in the workspace ({WORKSPACE_ROOT}). Paths are relative to it.",
        parameters=_obj({"path": {"type": "string", "default": "."}}),
        handler=list_files,
        timeout_s=45.0,
    ))

    async def read_file(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = normalize_path(args["path"])
        data = await deps.computer.call("/computer/read_file", {"path": path}, timeout=30.0)
        if not data.get("ok"):
            raise ToolError(str(data.get("error") or "could not read file"))
        return ToolResult(True, f"{data.get('path')}\n{data.get('content') or ''}")

    reg.register(Tool(
        name="read_file",
        label="Read file",
        description="Read a UTF-8 text file from the workspace (up to about 100,000 characters).",
        parameters=_obj({"path": {"type": "string", "minLength": 1, "maxLength": 500}}, ["path"]),
        handler=read_file,
        timeout_s=45.0,
        max_output_chars=12000,
    ))

    async def write_file(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = normalize_path(args["path"])
        if path == ".":
            raise ToolError("path must name a file")
        data = await deps.computer.call(
            "/computer/write_file", {"path": path, "content": args["content"]}, timeout=45.0
        )
        if not data.get("ok"):
            raise ToolError(str(data.get("error") or "could not write file"))
        return ToolResult(True, f"Wrote {data.get('bytes', len(args['content']))} bytes to {data.get('path')}")

    reg.register(Tool(
        name="write_file",
        label="Write file",
        description=(
            "Create or overwrite a text file in the workspace; parent directories are created. "
            "Paths are relative to the workspace root."
        ),
        parameters=_obj({
            "path": {"type": "string", "minLength": 1, "maxLength": 500},
            "content": {"type": "string", "maxLength": 400000},
        }, ["path", "content"]),
        handler=write_file,
        timeout_s=60.0,
    ))

    # ---- computer_info ----------------------------------------------------
    async def computer_info(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        data = await deps.computer.call("/computer/info", method="GET", timeout=25.0)
        return ToolResult(True, json.dumps(data, indent=2, ensure_ascii=False))

    reg.register(Tool(
        name="computer_info",
        label="Computer info",
        description=(
            "Report the Kaggle computer's hostname, GPU model(s), free disk space and workspace path. "
            "Use when asked about the machine, its GPUs or its resources."
        ),
        parameters=_obj({}),
        handler=computer_info,
        timeout_s=40.0,
    ))

    # ---- remember ---------------------------------------------------------
    async def remember(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        note = args["note"].strip()
        ts = time.time()
        await d1mod.d1.aexec(
            "INSERT INTO hermes_memories (id, target, content, memory_type, importance, user_id, session_id, "
            "profile, created_at, updated_at) VALUES (?, 'memory', ?, 'factual', 0.9, 'blackthorn', ?, 'default', ?, ?)",
            [uuid.uuid4().hex[:24], note, ctx.session_id, ts, ts],
        )
        return ToolResult(True, f"Remembered: {note}")

    reg.register(Tool(
        name="remember",
        label="Remember",
        description=(
            "Save a short durable fact or preference to long-term memory. Use only when the user asks you "
            "to remember something or states a lasting preference."
        ),
        parameters=_obj({"note": {"type": "string", "minLength": 3, "maxLength": 600}}, ["note"]),
        handler=remember,
        timeout_s=30.0,
    ))

    return reg
