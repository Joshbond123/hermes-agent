"""Blackthorn agent runtime — a real streaming agent loop with visible activity.

Unlike the TUI/PTY path (turn-based, silent for a minute) this module runs the
agent loop *in the request*, streams every step to the browser over SSE, and
never exposes the model's private deliberation.  The user sees:

  • status steps           ("Reading your request…", "Thinking", "Writing the answer")
  • tool calls             (name + arguments, marked running → ok/error)
  • tool output            (live stdout while a command runs, capped)
  • the final answer       (streamed progressively)

The model drives tools through a small, strictly-parsed protocol:

    <tool_call>{"name": "run_command", "arguments": {"command": "ls"}}</tool_call>

Everything runs on Blackthorn's own backend: commands execute in the managed
workspace root with a timeout + output cap, files are read/written through the
same policy the file browser uses, and memories persist to Cloudflare D1.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from hermes_cli.web_routers import blackthorn_studio as studio
from hermes_cli.web_routers._common import log

router = APIRouter(tags=["blackthorn-agent"])

MAX_STEPS = 24
TOOL_OUTPUT_CHARS = 6000
COMMAND_TIMEOUT = 60.0
FETCH_TIMEOUT = 20.0
TAG_KEEP = 12  # chars held back so a delimiter split across chunks is never leaked

TOOL_SYSTEM_PROMPT = (
    "You are Blackthorn, powered by Qwen3.8-27B-Uncensored. "
    "Your EXCLUSIVE workspace is the Kaggle Computer (/kaggle/working/blackthorn_workspace). "
    "ALL terminal, file, package, build, browser, and code execution happens on Kaggle Computer — never on Render. "
    "The dual T4 GPUs are available for CUDA/ML workloads on that same Kaggle machine. Cloudflare D1 holds durable memory.\n"
    "\n"
    "TOOL SELECTION — CRITICAL:\n"
    "Tools are AVAILABLE but must NOT be used automatically. Most messages need NO tool.\n"
    "Answer directly in markdown whenever the request does not genuinely require external data or execution.\n"
    "\n"
    "NEVER call any tool for:\n"
    "- greetings: hi, hello, hey, good morning, good evening\n"
    "- courtesy: thanks, thank you, ok, okay, bye, goodbye\n"
    "- casual chat or small talk\n"
    "- explaining your own capabilities (what can you do?)\n"
    "- rewriting, summarizing, or answering from known general knowledge\n"
    "- normal coding questions that do not require running code or reading files\n"
    "\n"
    "ONLY call a tool when the user's request clearly requires it:\n"
    "- web_search / tavily_search / fetch_url / browser: current events, latest news, live web research, or explicit browse/search\n"
    "- terminal / run_command: user asks to run, install, diagnose, or execute something on the computer\n"
    "- read_file / write_file / list_files: user asks about workspace files or uploads\n"
    "- computer_info: user asks about the machine/GPU/workspace\n"
    "- remember: user explicitly asks to remember a fact\n"
    "\n"
    "TOOL PROTOCOL (only when a tool is required):\n"
    "Emit EXACTLY one block and STOP (no other text after it):\n"
    '<tool_call>{"name": "TOOL_NAME", "arguments": {...}}</tool_call>\n'
    "Available tools:\n"
    "- web_search(query: str) — live web search via Tavily\n"
    "- tavily_search(query: str) — alias of web_search\n"
    "- terminal(command: str, timeout_seconds: int=60) — run a shell command\n"
    "- run_command(command: str, timeout_seconds: int=60) — alias of terminal\n"
    "- read_file(path: str) — read a workspace file\n"
    "- write_file(path: str, content: str) — write a workspace file\n"
    '- list_files(path: str=".") — list a directory\n'
    "- fetch_url(url: str) — fetch a page as plain text\n"
    "- browser(url: str) — open a URL from Kaggle Computer\n"
    "- remember(note: str) — store a durable fact in D1\n"
    "- computer_info() — hostname, GPU, disk, workspace path\n"
    "\n"
    "RULES:\n"
    "1. Never invent a tool result — the system runs the tool and returns real output.\n"
    "2. At most one tool per reply.\n"
    "3. After a tool result, answer in markdown or call another tool if still needed.\n"
    "4. Keep <think>...</think> brief and private; put the user-visible answer outside it.\n"
    "5. Prefer answering with zero tools. Do not search the web for greetings or capability questions."
)

NO_TOOL_SYSTEM_ADDON = (
    "\n\nThis turn requires NO tools. Answer the user directly in markdown. "
    "Do not emit <tool_call>, do not search the web, do not run commands, do not touch files."
)


# Commands that could destroy the host or the deployment — refused outright.
_DENY_PATTERNS = (
    r"rm\s+-rf\s+/(\s|$)", r"rm\s+-rf\s+/\*", r"mkfs", r"\bdd\s+if=", r"\bshutdown\b",
    r"\breboot\b", r":\(\)\s*\{", r">\s*/dev/sd", r"chmod\s+-R\s+777\s+/",
    r"mv\s+/(\s|$)", r"kill\s+-9\s+1\b", r"killall5", r"\bsudo\s+rm\b",
)


# --------------------------------------------------------------------------- #
# Wire protocol
# --------------------------------------------------------------------------- #
class AgentChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    attachments: List[studio.Attachment] = []
    max_steps: Optional[int] = None
    temperature: Optional[float] = None


def _sse(event: Dict[str, Any]) -> bytes:
    return f"data: {json.dumps(event)}\n\n".encode("utf-8")


# --------------------------------------------------------------------------- #
# Stream splitter: separates private thinking, tool calls, and public answer
# --------------------------------------------------------------------------- #
class AgentStreamSplitter:
    """Incrementally classify model output into think / tool_call / answer.

    The answer is released with a small hold-back window so a delimiter that is
    split across SSE chunks can never leak into the visible transcript.
    """

    OPEN_THINK = "<think"
    CLOSE_THINK = "</think"
    OPEN_TOOL = "<tool_call"
    CLOSE_TOOL = "</tool_call"

    def __init__(self) -> None:
        self.buf = ""
        self.mode = "answer"  # answer | think | tool
        self.answer_parts: List[str] = []
        self.think_parts: List[str] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.think_started_at: Optional[float] = None
        self.thinking_ever = False

    # -- internal ----------------------------------------------------------
    def _consume_tag_prefix(self, tag: str) -> bool:
        if self.buf.startswith(tag):
            self.buf = self.buf[len(tag):]
            if self.buf.startswith(">"):
                self.buf = self.buf[1:]
            return True
        return False

    # -- public ------------------------------------------------------------
    def feed(self, text: str) -> List[Dict[str, Any]]:
        """Returns ordered events: {"kind":"answer"|"think_start"|"think_end"|"tool"}"""
        self.buf += text
        events: List[Dict[str, Any]] = []
        while True:
            if self.mode == "answer":
                think_at = self.buf.find(self.OPEN_THINK)
                tool_at = self.buf.find(self.OPEN_TOOL)
                candidates = [i for i in (think_at, tool_at) if i != -1]
                if not candidates:
                    if len(self.buf) > TAG_KEEP:
                        chunk, self.buf = self.buf[:-TAG_KEEP], self.buf[-TAG_KEEP:]
                        self.answer_parts.append(chunk)
                        events.append({"kind": "answer", "text": chunk})
                    break
                idx = min(candidates)
                if idx > 0:
                    chunk, self.buf = self.buf[:idx], self.buf[idx:]
                    self.answer_parts.append(chunk)
                    events.append({"kind": "answer", "text": chunk})
                if self.buf.startswith(self.OPEN_TOOL):
                    if self._consume_tag_prefix(self.OPEN_TOOL):
                        self.mode = "tool"
                        self.think_started_at = None
                        continue
                if self._consume_tag_prefix(self.OPEN_THINK):
                    self.mode = "think"
                    self.thinking_ever = True
                    self.think_started_at = time.time()
                    events.append({"kind": "think_start"})
                    continue
            elif self.mode == "think":
                close = self.buf.find(self.CLOSE_THINK)
                if close == -1:
                    if len(self.buf) > TAG_KEEP:
                        self.think_parts.append(self.buf[:-TAG_KEEP])
                        self.buf = self.buf[-TAG_KEEP:]
                    break
                self.think_parts.append(self.buf[:close])
                self.buf = self.buf[close + len(self.CLOSE_THINK):]
                if self.buf.startswith(">"):
                    self.buf = self.buf[1:]
                self.mode = "answer"
                events.append({"kind": "think_end", "seconds": self._think_seconds()})
            else:  # tool
                close = self.buf.find(self.CLOSE_TOOL)
                if close == -1:
                    break
                payload = self.buf[:close]
                self.buf = self.buf[close + len(self.CLOSE_TOOL):]
                if self.buf.startswith(">"):
                    self.buf = self.buf[1:]
                self.mode = "answer"
                call = self._parse_tool_payload(payload)
                if call:
                    self.tool_calls.append(call)
                    events.append({"kind": "tool", "call": call})
        return events

    def _think_seconds(self) -> float:
        return round(time.time() - (self.think_started_at or time.time()), 2)

    def _parse_tool_payload(self, payload: str) -> Optional[Dict[str, Any]]:
        raw = payload.strip()
        if raw.startswith(">"):
            raw = raw[1:]
        raw = raw.strip()
        # tolerate ```json fences and stray prose around the object
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
        if fence:
            raw = fence.group(1)
        else:
            brace = re.search(r"\{.*\}", raw, re.S)
            if brace:
                raw = brace.group(0)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.debug("agent: unparsable tool payload: %r", raw[:200])
            return None
        if not isinstance(data, dict):
            return None
        name = str(data.get("name") or data.get("tool") or "").strip()
        args = data.get("arguments") or data.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not name:
            return None
        return {"name": name, "arguments": args if isinstance(args, dict) else {}}

    def flush(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if self.buf:
            tail, self.buf = self.buf, ""
            if self.mode == "think":
                self.think_parts.append(tail)
                events.append({"kind": "think_end", "seconds": self._think_seconds()})
            elif self.mode == "answer":
                self.answer_parts.append(tail)
                events.append({"kind": "answer", "text": tail})
        return events

    @property
    def answer(self) -> str:
        return "".join(self.answer_parts).strip()


# --------------------------------------------------------------------------- #

# Tools — ALL execution happens on Kaggle Computer via the tunnel API.
# Render is control-plane only; it never runs shell/files/browser for the agent.
# --------------------------------------------------------------------------- #
import httpx as _httpx

def _kaggle_tunnel() -> tuple:
    """Return (base_url, api_key) for the live Kaggle Computer+GPU tunnel."""
    try:
        rows = studio._d1q_sync(
            "SELECT status, tunnel_url, api_key FROM kaggle_gpu_state WHERE id='primary' LIMIT 1;"
        )
        if rows:
            url = (rows[0].get("tunnel_url") or "").rstrip("/")
            key = rows[0].get("api_key") or ""
            status = str(rows[0].get("status") or "").upper()
            if url and status not in ("OFF", "GPU_STOPPED_SAVING_QUOTA", "BOOT_FAILED", ""):
                # strip /v1 if present
                if url.endswith("/v1"):
                    url = url[:-3]
                return url, key
    except Exception as exc:
        log.warning("kaggle tunnel lookup: %s", exc)
    return "", ""


def _mark_tunnel_dead(reason: str, http_code: int = 0) -> None:
    """Clear a confirmed-dead Cloudflare tunnel from D1 so status/UI stop reporting ONLINE."""
    try:
        studio._d1q_sync(
            "UPDATE kaggle_gpu_state SET status = ?, tunnel_url = '', updated_at = ? WHERE id = 'primary';",
            ["TUNNEL_ERROR", time.time()],
        )
        log.warning("kaggle tunnel marked dead (HTTP %s): %s", http_code or "?", reason[:160])
    except Exception as exc:
        log.debug("mark tunnel dead note: %s", exc)
    try:
        studio.invalidate_route_cache()
    except Exception:
        pass


def _computer_post(path: str, payload: dict, timeout: float = 90.0) -> dict:
    """POST to Kaggle Computer API with dead-tunnel detection and one retry on fresh URL."""
    base, key = _kaggle_tunnel()
    if not base:
        return {
            "ok": False,
            "error": "Kaggle Computer is offline. Turn ON the GPU/Computer from the header first.",
            "host": "none",
        }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": "BlackthornHermes/1.0",
    }
    dead_codes = {404, 410, 502, 503, 521, 522, 523, 524, 530}

    def _once(url_base: str) -> dict:
        url = f"{url_base}{path}"
        with _httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            code = resp.status_code
            if code == 401:
                return {"ok": False, "error": "Kaggle Computer auth failed", "host": "kaggle-computer"}
            if code in dead_codes:
                body_preview = (resp.text or "")[:120]
                return {
                    "ok": False,
                    "error": f"tunnel_dead:{code}",
                    "http_code": code,
                    "detail": body_preview,
                    "host": "kaggle-computer",
                }
            # Cloudflare error pages are HTML, not JSON
            ctype = (resp.headers.get("content-type") or "").lower()
            if "text/html" in ctype and ("cloudflare" in (resp.text or "").lower() or "530" in (resp.text or "")):
                return {
                    "ok": False,
                    "error": f"tunnel_dead:{code or 530}",
                    "http_code": code or 530,
                    "detail": "Cloudflare tunnel error page",
                    "host": "kaggle-computer",
                }
            try:
                data = resp.json()
            except Exception:
                return {
                    "ok": False,
                    "error": f"non-JSON response HTTP {code}: {(resp.text or '')[:200]}",
                    "host": "kaggle-computer",
                }
            if not isinstance(data, dict):
                return {"ok": False, "error": f"bad response: {resp.text[:200]}", "host": "kaggle-computer"}
            data.setdefault("host", "kaggle-computer")
            return data

    try:
        result = _once(base)
        if result.get("error", "").startswith("tunnel_dead:"):
            code = int(result.get("http_code") or 530)
            _mark_tunnel_dead(result.get("detail") or result["error"], code)
            # One recovery attempt: re-read D1 for a replacement tunnel
            fresh_base, fresh_key = _kaggle_tunnel()
            if fresh_base and fresh_base != base:
                headers["Authorization"] = f"Bearer {fresh_key}"
                result = _once(fresh_base)
                if not result.get("error", "").startswith("tunnel_dead:"):
                    return result
            return {
                "ok": False,
                "error": (
                    f"Kaggle Computer tunnel is down (HTTP {code}). "
                    "The Cloudflare quick tunnel expired or the Kaggle session stopped. "
                    "Turn the GPU ON from the header, wait until Ready, then retry."
                ),
                "host": "kaggle-computer",
                "http_code": code,
            }
        return result
    except Exception as exc:
        msg = str(exc)
        if any(x in msg for x in ("530", "1033", "ConnectError", "ConnectTimeout", "Name or service not known")):
            _mark_tunnel_dead(msg, 530)
            return {
                "ok": False,
                "error": (
                    "Kaggle Computer tunnel unreachable. "
                    "Turn the GPU ON from the header and wait until Ready."
                ),
                "host": "kaggle-computer",
            }
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "host": "kaggle-computer"}


def _workspace_root(request: Request) -> Path:
    """Logical root label only — real FS is on Kaggle Computer."""
    return Path("/kaggle/working/blackthorn_workspace")


def _truncate(text: str, limit: int = TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit - 400]
    return f"{head}\n… [{len(text) - len(head)} more characters truncated]"


def tool_list_files(root: Path, args: Dict[str, Any]) -> str:
    data = _computer_post("/computer/list_files", {"path": str(args.get("path") or ".")})
    if not data.get("ok"):
        return f"error: {data.get('error') or data} [host={data.get('host')}]"
    return f"[kaggle-computer] {data.get('path')}\n{data.get('listing') or '(empty)'}"


def tool_read_file(root: Path, args: Dict[str, Any]) -> str:
    data = _computer_post("/computer/read_file", {"path": str(args.get("path") or "")})
    if not data.get("ok"):
        return f"error: {data.get('error') or data} [host={data.get('host')}]"
    return f"[kaggle-computer] {data.get('path')}\n{_truncate(str(data.get('content') or ''))}"


def tool_write_file(root: Path, args: Dict[str, Any]) -> str:
    path = str(args.get("path") or "").strip()
    if not path:
        return "error: path is required"
    content = args.get("content")
    if content is None:
        return "error: content is required"
    data = _computer_post("/computer/write_file", {"path": path, "content": content})
    if not data.get("ok"):
        return f"error: {data.get('error') or data} [host={data.get('host')}]"
    return f"wrote {data.get('bytes', '?')} bytes to {data.get('path')} [kaggle-computer]"


def tool_run_command(root: Path, args: Dict[str, Any]) -> str:
    cmd = str(args.get("command") or "").strip()
    if not cmd:
        return "error: command is required"
    # Safety: refuse host-destructive patterns (still enforced on Kaggle)
    for pat in _DENY_PATTERNS:
        if re.search(pat, cmd, re.I):
            return f"error: refused dangerous command pattern: {pat}"
    timeout = int(args.get("timeout_seconds") or 60)
    data = _computer_post(
        "/computer/exec",
        {"command": cmd, "timeout_seconds": timeout, "cwd": args.get("cwd") or "."},
        timeout=float(timeout) + 30,
    )
    out = data.get("output") or data.get("error") or ""
    code = data.get("exit_code", -1)
    host = data.get("host", "kaggle-computer")
    return f"[host={host} cwd={data.get('cwd', '')} exit={code}]\n{_truncate(str(out))}"


def tool_fetch_url(root: Path, args: Dict[str, Any]) -> str:
    url = str(args.get("url") or "").strip()
    data = _computer_post("/computer/fetch_url", {"url": url}, timeout=60.0)
    if not data.get("ok"):
        return f"error: {data.get('error') or data} [host={data.get('host')}]"
    return f"[kaggle-computer browser] {url}\n{_truncate(str(data.get('text') or ''))}"


def tool_remember(root: Path, args: Dict[str, Any]) -> str:
    note = str(args.get("note") or "").strip()
    if not note:
        return "error: note is required"
    try:
        studio._d1q_sync(
            "INSERT OR REPLACE INTO hermes_memories "
            "(id, target, content, memory_type, importance, user_id, session_id, profile, created_at, updated_at) "
            "VALUES (?, 'memory', ?, 'factual', 0.9, ?, '', 'default', ?, ?);",
            [uuid.uuid4().hex[:24], note, "blackthorn", time.time(), time.time()],
        )
        studio.invalidate_memory_cache()
        return f"remembered: {note}"
    except Exception as exc:
        return f"error: could not store memory ({exc})"


def tool_computer_info(root: Path, args: Dict[str, Any]) -> str:
    base, key = _kaggle_tunnel()
    if not base:
        return "error: Kaggle Computer offline"
    try:
        with _httpx.Client(timeout=20.0) as client:
            resp = client.get(
                f"{base}/computer/info",
                headers={"Authorization": f"Bearer {key}"},
            )
            data = resp.json()
        return json.dumps(data, indent=2)
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"


def _next_tavily_key() -> str:
    """Rotate Tavily API keys stored in D1 state_meta.tavily_api_keys."""
    raw = _d1_get("tavily_api_keys")
    try:
        data = json.loads(raw) if raw else {"keys": [], "index": 0}
    except Exception:
        data = {"keys": [], "index": 0}
    keys = data.get("keys") or []
    if not keys:
        return os.environ.get("TAVILY_API_KEY", "")
    idx = int(data.get("index") or 0) % len(keys)
    key = keys[idx]
    data["index"] = (idx + 1) % len(keys)
    try:
        _d1_set("tavily_api_keys", json.dumps(data))
    except Exception:
        pass
    return key



def tool_tavily_search(root: Path, args: Dict[str, Any]) -> str:
    import urllib.request
    query = str(args.get("query") or "").strip()
    if not query:
        return "error: query required"
    api_key = _next_tavily_key()
    if not api_key:
        return "error: no Tavily API key configured"
    payload = json.dumps({"api_key": api_key, "query": query, "max_results": 5}).encode()
    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
        results = data.get("results") or []
        if not results:
            return "no results"
        blocks = []
        sources = []
        for i, r in enumerate(results[:5], 1):
            title = (r.get("title") or "Untitled").strip()
            url = (r.get("url") or "").strip()
            snippet = (r.get("content") or "")[:400].strip()
            blocks.append(f"[{i}] {title}\nURL: {url}\n{snippet}")
            if url:
                sources.append(f"- [{title}]({url})")
        out = "Search results:\n\n" + "\n\n".join(blocks)
        if sources:
            out += "\n\nSources:\n" + "\n".join(sources)
        return out
    except Exception as exc:
        return f"error: tavily {exc}"


TOOLS: Dict[str, Callable[[Path, Dict[str, Any]], str]] = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "run_command": tool_run_command,
    "terminal": tool_run_command,
    "fetch_url": tool_fetch_url,
    "browser": tool_fetch_url,
    "remember": tool_remember,
    "tavily_search": tool_tavily_search,
    "web_search": tool_tavily_search,
    "computer_info": tool_computer_info,
}

TOOL_LABELS = {
    "list_files": "Listing files on Kaggle Computer",
    "read_file": "Reading file on Kaggle Computer",
    "write_file": "Writing file on Kaggle Computer",
    "run_command": "Running command on Kaggle Computer",
    "terminal": "Terminal on Kaggle Computer",
    "fetch_url": "Browsing via Kaggle Computer",
    "browser": "Browsing via Kaggle Computer",
    "remember": "Saving to memory",
    "tavily_search": "Web search",
    "web_search": "Web search",
    "computer_info": "Inspecting Kaggle Computer",
}


# OpenAI-compatible tools schema — sent to models that support native function calling
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Live web search via Tavily. Use for latest news, current events, research.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a shell command on Kaggle Computer and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "default": 60},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_info",
            "description": "Show Kaggle Computer hostname, GPU list, disk free, workspace path.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List a directory on Kaggle Computer workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the Kaggle workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a text file on the Kaggle workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a URL as plain text from Kaggle Computer.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Store a durable fact in Cloudflare D1 memory.",
            "parameters": {
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "required": ["note"],
            },
        },
    },
]



def _user_message_needs_tools(text: str, has_attachments: bool = False) -> bool:
    """Backend tool-selection gate: True only when the user request likely needs a tool.

    Separates TOOL AVAILABILITY from TOOL INVOCATION. Simple greetings and
    conversational turns must not trigger web search, terminal, or file tools.
    """
    if has_attachments:
        return True
    t = (text or "").strip().lower()
    if not t:
        return False
    # Pure greetings / courtesy — never tools
    pure = re.sub(r"[^a-z0-9\s]", "", t).strip()
    pure = re.sub(r"\s+", " ", pure)
    greetings = {
        "hi", "hello", "hey", "hiya", "yo", "sup", "howdy",
        "good morning", "good afternoon", "good evening", "good night",
        "thanks", "thank you", "thx", "ty", "ok", "okay", "k", "cool",
        "bye", "goodbye", "see you", "cya", "cheers", "nice", "great",
        "yes", "no", "yep", "nope", "sure",
    }
    if pure in greetings or pure in {g + " there" for g in ("hi", "hello", "hey")}:
        return False
    # Capability / identity questions — answer from prompt, no tools
    capability = (
        "what can you do", "who are you", "what are you", "your capabilities",
        "what do you do", "help me", "how can you help", "introduce yourself",
    )
    if any(c in pure for c in capability) and len(pure) < 80:
        return False
    # Explicit tool triggers
    web_triggers = (
        "search the web", "search online", "google ", "look up", "latest news",
        "current events", "what is the latest", "what happened", "browse ",
        "fetch url", "open this url", "open the url", "web search", "tavily",
        "from the internet", "online research", "research online",
    )
    terminal_triggers = (
        "run this", "run the", "execute ", "install ", "pip install", "apt ",
        "shell ", "terminal", "bash ", "command line", "ls ", "cd ",
        "write a script and run", "run python", "run the code",
    )
    file_triggers = (
        "read the file", "open the file", "list files", "list the files",
        "read file", "write file", "create a file", "save to file",
        "uploaded", "attachment", "summarize the pdf", "read the pdf",
        "workspace", "list directory",
    )
    kaggle_triggers = (
        "kaggle", "gpu status", "computer info", "what gpu", "disk space",
        "hostname", "workspace path",
    )
    if any(x in t for x in web_triggers + terminal_triggers + file_triggers + kaggle_triggers):
        return True
    # Short conversational messages without action verbs → no tools
    words = pure.split()
    if len(words) <= 12 and not any(
        w in pure for w in (
            "search", "browse", "run", "install", "execute", "fetch", "download",
            "write", "create", "delete", "list", "read", "open", "code", "script",
            "news", "latest", "current", "online", "web", "url", "http",
            "research", "investigate", "scan", "assess", "terminal", "gpu",
            "kaggle", "workspace", "file", "command", "python", "shell",
        )
    ):
        return False
    # Very short messages (≤4 words) never need tools unless an explicit trigger matched above
    if len(words) <= 4:
        return False
    # Default: allow tools (model still decides via tool_choice auto)
    return True


def _intent_looks_like_tool_need(text: str) -> bool:
    """True when the model narrated tool intent instead of emitting a tool_call."""
    t = (text or "").lower()
    if not t.strip():
        return False
    markers = (
        "i should search", "i will search", "i'll search", "let me search",
        "i should use", "i will use", "i'll use", "let me use",
        "i need to search", "i need to run", "i need to call",
        "i should call", "i will call", "i'll call",
        "web search", "use the tool", "using the tool",
        "search the web", "look up", "fetch the",
        "run a command", "run the command", "check the computer",
    )
    return any(m in t for m in markers)




def _tool_title(name: str, args: Dict[str, Any]) -> str:
    label = TOOL_LABELS.get(name, name.replace("_", " ").title())
    detail = args.get("command") or args.get("path") or args.get("url") or args.get("note") or ""
    detail = str(detail).strip().replace("\n", " ")
    return f"{label}: {detail[:120]}" if detail else label


def _tool_code(name: str, args: Dict[str, Any]) -> str:
    if name == "run_command":
        return f"$ {args.get('command', '')}"
    if name in ("read_file", "list_files"):
        return f"{name}({args.get('path', '.')})"
    if name == "write_file":
        body = str(args.get("content") or "")
        preview = body if len(body) <= 400 else body[:400] + "…"
        return f"write_file({args.get('path')})\n{preview}"
    return f"{name}({json.dumps(args)[:200]})"


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
@router.post("/api/studio/agent/stream")
async def agent_stream(payload: AgentChatRequest, request: Request):
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty")

    studio._ensure_keepalive()
    t_start = time.perf_counter()
    session_id = (payload.session_id or "").strip() or f"studio-{uuid.uuid4().hex[:12]}"
    root = _workspace_root(request)
    max_steps = max(1, min(int(payload.max_steps or MAX_STEPS), 32))

    is_new_session = not (payload.session_id or "").strip()
    route_task = studio._cached_route()
    memory_task = studio._cached_memories()
    history_task = (
        studio._no_history()
        if is_new_session
        else studio._d1q(
            "SELECT role, content FROM messages WHERE session_id = ? AND role IN ('user','assistant') "
            "AND content IS NOT NULL AND content != '' ORDER BY timestamp DESC LIMIT ?;",
            [session_id, studio.MAX_HISTORY_MESSAGES],
        )
    )
    attach_task = asyncio.gather(
        *[
            asyncio.to_thread(studio._read_attachment, request, {"path": a.path, "name": a.name})
            for a in payload.attachments[: studio.MAX_ATTACHMENTS]
        ]
    )
    route, history, memories, attachments = await asyncio.gather(
        route_task, history_task, memory_task, attach_task
    )
    attachments = [a for a in attachments if a]
    if not route["url"]:
        raise HTTPException(
            status_code=503,
            detail="Kaggle GPU is offline. Turn it on from the header, then send the message again.",
        )

    history.reverse()
    needs_tools = _user_message_needs_tools(message, has_attachments=bool(attachments))
    if not needs_tools:
        max_steps = 1  # single direct answer — no tool loop
    system_prompt = studio._compose_system(memories, attachments) + "\n\n" + TOOL_SYSTEM_PROMPT
    system_prompt += """

AUTONOMY RULES (mandatory):
- Complete the user's task in this single turn using as many tool calls as needed.
- Never ask the user to say "continue" or send another message to finish.
- After each tool result, immediately decide the next tool call or the final answer.
- Prefer action (tools) over narrating plans. Emit real tool calls, not descriptions of them.
- When the task is done, give a clear final answer summarizing results and evidence.
"""
    system_prompt += f"\nWorkspace root: {root}"
    if not needs_tools:
        system_prompt += NO_TOOL_SYSTEM_ADDON
    user_note = ""
    if attachments:
        names = ", ".join(a.get("name") or a.get("path") or "file" for a in attachments)
        user_note = f"\n\n[Attached: {names}]"

    base_messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    base_messages += [{"role": m["role"], "content": m.get("content") or ""} for m in history]
    base_messages.append({"role": "user", "content": message})

    prework_ms = int((time.perf_counter() - t_start) * 1000)
    url = f"{route['url']}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {route['api_key']}", "Content-Type": "application/json"}

    def _trace_json(steps: List[Dict[str, Any]]) -> str:
        """Compact activity trace persisted with the answer so history keeps it."""
        slim = [
            {
                "id": st.get("id"), "kind": st.get("kind"), "title": str(st.get("title") or "")[:200],
                "status": st.get("status"), "duration_ms": st.get("duration_ms"),
                "code": (str(st.get("code"))[:400] if st.get("code") else None),
            }
            for st in steps
        ]
        try:
            return json.dumps(slim)[:8000]
        except Exception:
            return ""

    async def persist(answer: str, out_tokens: Optional[int], finish_reason: str,
                      steps: List[Dict[str, Any]]) -> None:
        try:
            await studio._d1q(
                "INSERT INTO sessions (id, source, created_source, session_key, display_name, model, "
                "started_at, message_count, tool_call_count) "
                "SELECT ?, ?, ?, ?, '', ?, ?, 0, 0 WHERE NOT EXISTS (SELECT 1 FROM sessions WHERE id = ?);",
                [session_id, studio.STUDIO_SOURCE, studio.STUDIO_SOURCE, session_id,
                 route["model"], studio._now(), session_id],
            )
            await studio._d1q(
                "INSERT INTO messages (session_id, role, content, timestamp, active, _compressed_summary) "
                "VALUES (?, 'user', ?, ?, 1, 0);",
                [session_id, message + user_note, studio._now()],
            )
            await studio._d1q(
                "UPDATE sessions SET message_count = COALESCE(message_count, 0) + 1, "
                "display_name = CASE WHEN COALESCE(display_name, '') = '' THEN ? ELSE display_name END "
                "WHERE id = ?;",
                [message.replace("\n", " ")[:60], session_id],
            )
            if answer:
                await studio._d1q(
                    "INSERT INTO messages (session_id, role, content, reasoning, timestamp, token_count, "
                    "finish_reason, active, _compressed_summary) VALUES (?, 'assistant', ?, ?, ?, ?, ?, 1, 0);",
                    [session_id, answer, _trace_json(steps), studio._now(), out_tokens or 0, finish_reason],
                )
                await studio._d1q(
                    "UPDATE sessions SET message_count = COALESCE(message_count, 0) + 1 WHERE id = ?;",
                    [session_id],
                )
        except Exception as exc:
            log.warning("agent: persistence failed: %s", exc)

    async def event_stream():
        nonlocal route, url
        import cloudflare_d1_client as d1_mod

        studio.mark_activity_safe("agent-turn")
        d1_mod.task_started("agent-turn")
        steps: List[Dict[str, Any]] = []
        step_seq = 0
        cancelled = False
        stop_all = False

        async def client_gone() -> bool:
            """True once the browser aborted the request (Stop button / tab close)."""
            try:
                return bool(await request.is_disconnected())
            except Exception:
                return False

        def activity(kind: str, title: str, *, status: str = "running", detail: str = "",
                     code: str = "", step_id: Optional[str] = None,
                     duration_ms: Optional[int] = None) -> bytes:
            nonlocal step_seq
            if step_id is None:
                step_seq += 1
                step_id = f"a{step_seq}"
            steps.append({
                "id": step_id, "kind": kind, "title": title, "status": status,
                "detail": detail, "code": code, "duration_ms": duration_ms,
            })
            return _sse({
                "type": "activity", "id": step_id, "kind": kind, "title": title,
                "status": status, "detail": detail, "code": code, "duration_ms": duration_ms,
            })

        yield _sse({
            "type": "meta", "session_id": session_id, "model": route["model"],
            "endpoint": route["url"], "mode": "agent", "prework_ms": prework_ms,
            "attachments": [a.get("name") or a.get("path") for a in attachments],
        })
        # Do not emit mock status titles ("Reading your request", etc.).
        # Only real tool / plan / thinking events are streamed below.

        messages = list(base_messages)
        answer_text = ""
        out_tokens = 0
        finish_reason = "stop"
        t_start_model = time.perf_counter()
        first_token_ms: Optional[int] = None

        try:
            client = await studio._tunnel_client()
            retried_route = False
            route_retry_signal = False
            step_budget = max_steps
            for step_no in range(1, step_budget + 1):
                try:
                    # Refresh idle clock only — do NOT nest task_started (would leak counters)
                    studio.mark_activity_safe("agent-step")
                except Exception:
                    pass
                if stop_all or await client_gone():
                    cancelled = True
                    finish_reason = finish_reason if finish_reason == "cancelled" else "cancelled"
                    break
                if route_retry_signal:
                    # the tunnel moved mid-run: re-issue this same step against the
                    # refreshed endpoint without eating into the step budget
                    route_retry_signal = False
                    step_budget += 1
                    continue
                splitter = AgentStreamSplitter()
                think_step_id: Optional[str] = None
                body = {
                    "model": route["model"],
                    "messages": messages,
                    "stream": True,
                    # Prefer visible tool calls over hidden reasoning when the backend supports it
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                # TOOL GATE: only expose tools when the user request actually needs them.
                # Separates availability from invocation — greetings never get a tool schema.
                if needs_tools:
                    body["tools"] = OPENAI_TOOLS
                    body["tool_choice"] = "auto"
                if payload.temperature is not None:
                    body["temperature"] = payload.temperature

                # Some llama.cpp/ollama builds reject the OpenAI tools field — fall back once
                if body.get("tools") and not getattr(splitter, "_tools_stripped", False):
                    pass  # first attempt keeps tools
                async with client.stream("POST", url, json=body, headers=headers) as resp:
                    if resp.status_code != 200:
                        detail = (await resp.aread()).decode("utf-8", "replace")[:300]
                        # Backend does not support tools — strip and retry this step
                        if resp.status_code in (400, 422) and body.get("tools") and not getattr(splitter, "_tools_stripped", False):
                            body.pop("tools", None)
                            body.pop("tool_choice", None)
                            splitter._tools_stripped = True
                            log.info("agent: backend rejected tools — retrying without function-calling schema")
                            step_budget += 1
                            continue
                        # The tunnel may have been replaced by a kernel restart while
                        # Render still held the old route: refresh it and retry once.
                        # 530 = Cloudflare Tunnel error 1033 (cloudflared down / stale quick tunnel)
                        if not retried_route and resp.status_code in (404, 410, 502, 503, 521, 522, 523, 524, 530):
                            retried_route = True
                            studio.invalidate_route_cache()
                            yield _sse({"type": "activity", "id": "route-refresh", "kind": "status",
                                        "title": "GPU tunnel unreachable — probing for a live endpoint",
                                        "status": "warn",
                                        "detail": f"HTTP {resp.status_code} from previous tunnel"})
                            # Force a live status refresh (may clear a dead tunnel from D1)
                            try:
                                import cloudflare_d1_client as _d1
                                await asyncio.to_thread(_d1.get_kaggle_gpu_status, True)
                            except Exception as _pe:
                                log.debug("agent: forced status refresh note: %s", _pe)
                            fresh = await studio._cached_route()
                            fresh_url = (fresh.get("url") or "").rstrip("/")
                            old_url = (route.get("url") or "").rstrip("/")
                            if fresh_url and fresh_url != old_url:
                                log.info("agent: tunnel moved %s → %s — retrying",
                                         old_url[:40], fresh_url[:40])
                                route = fresh
                                url = f"{route['url']}/v1/chat/completions"
                                messages = [{"role": "system", "content": system_prompt}] + base_messages[1:]
                                route_retry_signal = True
                                continue
                            # Same dead URL or no URL — do not dump Cloudflare HTML at the user
                            clean = (
                                "The Kaggle GPU tunnel is down (Cloudflare HTTP "
                                f"{resp.status_code}). The GPU session may have stopped or the "
                                "quick tunnel expired. Turn the GPU ON from the header, wait until "
                                "status is Ready, then retry."
                            )
                            yield _sse({"type": "error", "message": clean})
                            finish_reason = "error"
                            break
                        # Non-retriable upstream error — strip HTML bodies
                        clean_detail = detail
                        if "<" in detail and "html" in detail.lower():
                            clean_detail = (
                                f"GPU endpoint returned HTTP {resp.status_code} "
                                "(non-JSON / tunnel error page). The endpoint is not healthy."
                            )
                        else:
                            clean_detail = f"The GPU endpoint returned HTTP {resp.status_code}: {detail[:240]}"
                        yield _sse({"type": "error", "message": clean_detail})
                        finish_reason = "error"
                        break

                    chunk_no = 0
                    async for raw_line in resp.aiter_lines():
                        chunk_no += 1
                        if chunk_no % 8 == 0 and await client_gone():
                            cancelled = True
                            finish_reason = "cancelled"
                            stop_all = True
                            break
                        if not raw_line or not raw_line.startswith("data:"):
                            continue
                        data = raw_line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        usage = obj.get("usage")
                        if isinstance(usage, dict) and usage.get("completion_tokens"):
                            out_tokens += int(usage["completion_tokens"])
                        choice = (obj.get("choices") or [{}])[0]
                        if choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])
                        delta = choice.get("delta") or {}
                        # Native OpenAI-style tool_calls (function calling)
                        native_tcs = delta.get("tool_calls") or []
                        if native_tcs:
                            if not hasattr(splitter, "_pending_tools"):
                                splitter._pending_tools = []
                            if not hasattr(splitter, "_native_tc_buf"):
                                splitter._native_tc_buf = {}
                            for tc in native_tcs:
                                idx = str(tc.get("index", 0))
                                buf = splitter._native_tc_buf.setdefault(idx, {"name": "", "arguments": ""})
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    buf["name"] = fn["name"]
                                if fn.get("arguments"):
                                    buf["arguments"] += fn["arguments"]
                        # Qwen reasoning models put private CoT in reasoning_content —
                        # never show it, but still scan it for <tool_call> blocks so tools run.
                        piece = delta.get("content") or ""
                        reasoning_piece = (
                            delta.get("reasoning_content")
                            or delta.get("reasoning")
                            or ""
                        )
                        if reasoning_piece:
                            for event in splitter.feed(reasoning_piece):
                                if event["kind"] == "tool":
                                    call = event["call"]
                                    yield activity(
                                        "plan",
                                        f"Decided to use `{call['name']}`",
                                        status="ok",
                                        code=_tool_code(call["name"], call["arguments"]),
                                    )
                                    if not hasattr(splitter, "_pending_tools"):
                                        splitter._pending_tools = []
                                    splitter._pending_tools.append(call)
                        if not piece and not native_tcs:
                            continue
                        for event in splitter.feed(piece):
                            if event["kind"] == "answer":
                                if first_token_ms is None:
                                    first_token_ms = int((time.perf_counter() - t_start_model) * 1000) + prework_ms
                                yield _sse({"type": "delta", "delta": event["text"]})
                            elif event["kind"] == "think_start":
                                # Keep reasoning private — do not stream thinking activity to the UI
                                think_step_id = None
                            elif event["kind"] == "think_end":
                                think_step_id = None
                            elif event["kind"] == "tool":
                                call = event["call"]
                                yield activity("plan", f"Decided to use `{call['name']}`",
                                               status="ok", code=_tool_code(call["name"], call["arguments"]))
                for event in splitter.flush():
                    if event["kind"] == "answer":
                        yield _sse({"type": "delta", "delta": event["text"]})
                    elif event["kind"] == "think_end" and think_step_id:
                        yield activity("thinking", "Reasoning", status="ok",
                                       step_id=think_step_id,
                                       duration_ms=int(event.get("seconds", 0) * 1000))

                answer_text += splitter.answer
                calls = list(splitter.tool_calls)
                # tool calls detected inside private reasoning channel or native API
                calls.extend(getattr(splitter, "_pending_tools", []) or [])
                splitter._pending_tools = []
                # Finalize native OpenAI tool_calls buffers accumulated across stream chunks
                native_buf = getattr(splitter, "_native_tc_buf", {}) or {}
                for _idx, buf in native_buf.items():
                    name = (buf.get("name") or "").strip()
                    if not name:
                        continue
                    args_raw = buf.get("arguments") or "{}"
                    try:
                        args = json.loads(args_raw) if args_raw.strip() else {}
                    except json.JSONDecodeError:
                        args = {"_raw": args_raw}
                    if not isinstance(args, dict):
                        args = {"value": args}
                    calls.append({"name": name, "arguments": args})
                splitter._native_tc_buf = {}
                # de-dupe by name+args
                seen = set()
                uniq = []
                for c in calls:
                    key = (c.get("name"), json.dumps(c.get("arguments") or {}, sort_keys=True))
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append(c)
                calls = uniq

                if not calls:
                    # Model narrated tool intent without emitting a real tool_call — re-prompt
                    # Only when this turn was allowed to use tools (gate already decided).
                    if needs_tools and answer_text.strip() and _intent_looks_like_tool_need(answer_text) and step_no < max_steps:
                        yield activity(
                            "plan",
                            "Model described a tool but did not call it — requesting a proper tool_call",
                            status="warn",
                            detail=answer_text[:200],
                        )
                        messages.append({"role": "assistant", "content": answer_text})
                        messages.append({
                            "role": "user",
                            "content": (
                                "You described using a tool but did not emit a tool_call block. "
                                "Emit EXACTLY one <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call> "
                                "now and nothing else. For web research use name web_search."
                            ),
                        })
                        answer_text = ""
                        continue
                    if answer_text.strip():
                        break
                    # model produced nothing usable — nudge it once, then give up
                    if step_no == max_steps:
                        break
                    messages.append({"role": "assistant", "content": ""})
                    messages.append({"role": "user",
                                     "content": "Your previous reply was empty. Answer the user's request now, or emit a <tool_call> if you need a tool."})
                    continue

                for call in calls[:1]:  # one tool per step keeps the transcript readable
                    name = call["name"]
                    args = call["arguments"]
                    tool_id = f"t{step_no}"
                    yield activity("tool", _tool_title(name, args), status="running",
                                   code=_tool_code(name, args), step_id=tool_id)
                    tool_started = time.perf_counter()
                    runner = TOOLS.get(name)
                    if runner is None:
                        result = f"error: unknown tool '{name}'. Available: {', '.join(TOOLS)}"
                        status = "error"
                    else:
                        try:
                            result = await asyncio.to_thread(runner, root, args)
                            status = "error" if result.startswith(("error:", "refused:")) else "ok"
                        except Exception as exc:
                            result = f"error: {type(exc).__name__}: {exc}"
                            status = "error"
                    duration_ms = int((time.perf_counter() - tool_started) * 1000)
                    yield activity("tool", _tool_title(name, args), status=status,
                                   detail=_truncate(result, 1200), code=_tool_code(name, args),
                                   step_id=tool_id, duration_ms=duration_ms)
                    messages.append({"role": "assistant",
                                     "content": f'<tool_call>{{"name": "{name}", "arguments": {json.dumps(args)}}}</tool_call>'})
                    messages.append({"role": "user",
                                     "content": f"Tool `{name}` finished. Real result:\n\n```\n{_truncate(result)}\n```\n\n"
                                                "Continue autonomously: call another tool now if needed, or give the FINAL complete answer. Do not ask the user to continue."})
                answer_text = ""  # intermediate text is not the final answer
                if finish_reason == "tool_calls":
                    finish_reason = "stop"  # continue autonomous loop
        except asyncio.CancelledError:
            cancelled = True
            finish_reason = "cancelled"
            raise
        except Exception as exc:
            log.warning("agent: stream failed: %s", exc)
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            finish_reason = "error"
        finally:
            d1_mod.task_finished("agent-turn")

        # Nothing below yields from a finally block: a client abort raises
        # GeneratorExit, and yielding after that is a RuntimeError.
        answer = (answer_text or "").strip()
        duration_ms = prework_ms + int((time.perf_counter() - t_start_model) * 1000)
        # If the model stopped on tool_calls without a final answer, surface a clear
        # completion note so history is not empty "thought-only" rows.
        if not answer and steps and finish_reason in ("tool_calls", "length", "stop"):
            tool_titles = [s.get("title") or s.get("kind") for s in steps if s.get("kind") == "tool"]
            if tool_titles:
                answer = (
                    "Completed tool steps in this turn:\n"
                    + "\n".join(f"- {t}" for t in tool_titles[:20])
                    + ("\n\n(Partial run — step budget reached before a final narrative answer.)"
                       if finish_reason == "tool_calls" else "")
                )
                # stream the fallback so the live UI is not blank
                yield _sse({"type": "delta", "delta": answer})
        # Always emit done so the client closes cleanly and persists activity.
        yield _sse({
            "type": "done", "session_id": session_id, "finish_reason": finish_reason or "stop",
            "first_token_ms": first_token_ms, "duration_ms": duration_ms,
            "output_tokens": out_tokens or None,
            "cancelled": cancelled,
            "steps": steps,
        })
        asyncio.create_task(persist(answer, out_tokens or None, finish_reason or "stop", steps))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


# ---- System prompt + Tavily rotation (Blackthorn product) ----

class SystemPromptBody(BaseModel):
    prompt: str = ""


def _d1_get(key: str) -> str:
    try:
        rows = studio._d1q_sync(
            "SELECT value FROM state_meta WHERE key = ? LIMIT 1;", [key]
        )
        if rows:
            return str(rows[0].get("value") or "")
    except Exception:
        pass
    return ""


def _d1_set(key: str, value: str) -> None:
    try:
        studio._d1q_sync("DELETE FROM state_meta WHERE key = ?;", [key])
        studio._d1q_sync(
            "INSERT INTO state_meta (key, value) VALUES (?, ?);", [key, value]
        )
    except Exception as exc:
        log(f"d1 set {key} failed: {exc}")


@router.get("/api/system-prompt")
async def get_system_prompt():
    prompt = _d1_get("blackthorn_system_prompt")
    return {"prompt": prompt}


@router.put("/api/system-prompt")
async def put_system_prompt(body: SystemPromptBody):
    _d1_set("blackthorn_system_prompt", body.prompt or "")
    return {"ok": True, "prompt": body.prompt or ""}


