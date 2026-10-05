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

MAX_STEPS = 6
TOOL_OUTPUT_CHARS = 6000
COMMAND_TIMEOUT = 60.0
FETCH_TIMEOUT = 20.0
TAG_KEEP = 12  # chars held back so a delimiter split across chunks is never leaked

TOOL_SYSTEM_PROMPT = (
    "You are Blackthorn, powered by Qwen3.8-27B-Uncensored on dual Kaggle T4 GPUs, with a shell, "
    "a file workspace, and Cloudflare D1 memory.\n"
    "TOOLS — when a task needs real work (running something, reading or writing files, "
    "fetching a page, remembering a fact), call a tool by emitting EXACTLY this block and "
    "then stop:\n"
    '<tool_call>{"name": "TOOL_NAME", "arguments": {...}}</tool_call>\n'
    "Available tools:\n"
    "- run_command(command: str, timeout_seconds: int=60) — run a shell command in the "
    "workspace root and see its output\n"
    "- read_file(path: str) — read a text file\n"
    "- write_file(path: str, content: str) — create or overwrite a file\n"
    "- list_files(path: str=\".\") — list a directory\n"
    "- fetch_url(url: str) — fetch a web page as text\n"
    "- tavily_search(query: str) — web search via Tavily (rotating API keys)\n- remember(note: str) — store a durable fact in long-term memory\n"
    "RULES: never invent a tool result — the system runs the tool and sends you the real "
    "output, then you continue. Use at most one tool per reply. Keep <think> blocks brief "
    "and private; the user never sees them, so put the answer itself in normal text. "
    "When no tool is needed, answer directly in markdown."
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
# Tools (bounded, real, workspace-scoped)
# --------------------------------------------------------------------------- #
def _workspace_root(request: Request) -> Path:
    try:
        from hermes_cli.web_routers.files import _resolve_managed_path

        _policy, root, _display = _resolve_managed_path("", request)
        return Path(root)
    except Exception:
        return Path(os.environ.get("HERMES_HOME") or os.getcwd())


def _safe_join(root: Path, raw: str) -> Path:
    candidate = Path(raw or ".")
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if root.resolve() not in resolved.parents and resolved != root.resolve():
        raise ValueError(f"path escapes the workspace root: {raw}")
    return resolved


def _truncate(text: str, limit: int = TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit - 400]
    return f"{head}\n… [{len(text) - len(head)} more characters truncated]"


def tool_list_files(root: Path, args: Dict[str, Any]) -> str:
    try:
        target = _safe_join(root, str(args.get("path") or "."))
    except ValueError as exc:
        return f"error: {exc}"
    if not target.is_dir():
        return f"error: {target} is not a directory"
    rows = []
    for entry in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:200]:
        try:
            size = "" if entry.is_dir() else f"  {entry.stat().st_size:,} B"
        except OSError:
            size = ""
        rows.append(f"{'DIR ' if entry.is_dir() else 'FILE'} {entry.name}{size}")
    return "\n".join(rows) or "(empty directory)"


def tool_read_file(root: Path, args: Dict[str, Any]) -> str:
    try:
        target = _safe_join(root, str(args.get("path") or ""))
    except ValueError as exc:
        return f"error: {exc}"
    if not target.is_file():
        return f"error: {args.get('path')} is not a file"
    if target.stat().st_size > 4 * 1024 * 1024:
        return "error: file is larger than 4 MB"
    data = target.read_bytes()
    try:
        return _truncate(data.decode("utf-8"))
    except UnicodeDecodeError:
        return f"error: {target.name} is binary ({len(data)} bytes)"


def tool_write_file(root: Path, args: Dict[str, Any]) -> str:
    rel = str(args.get("path") or "").strip()
    if not rel:
        return "error: path is required"
    try:
        target = _safe_join(root, rel)
    except ValueError as exc:
        return f"error: {exc}"
    target.parent.mkdir(parents=True, exist_ok=True)
    content = str(args.get("content") or "")
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} characters to {target.relative_to(root) if root in target.parents else target}"


def tool_run_command(root: Path, args: Dict[str, Any]) -> str:
    command = str(args.get("command") or "").strip()
    if not command:
        return "error: command is required"
    lowered = command.lower()
    for pattern in _DENY_PATTERNS:
        if re.search(pattern, lowered):
            return f"refused: command matches a destructive pattern ({pattern})"
    try:
        timeout = min(float(args.get("timeout_seconds") or COMMAND_TIMEOUT), 120.0)
    except (TypeError, ValueError):
        timeout = COMMAND_TIMEOUT
    try:
        proc = subprocess.run(
            command, shell=True, cwd=str(root), capture_output=True, text=True,
            timeout=timeout, env={**os.environ, "GIT_PAGER": "cat", "PAGER": "cat"},
        )
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout:.0f}s"
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"
    out = (proc.stdout or "") + (proc.stderr or "")
    body = _truncate(out.strip() or "(no output)")
    return f"exit_code={proc.returncode}\n{body}"


def tool_fetch_url(root: Path, args: Dict[str, Any]) -> str:
    import urllib.request

    url = str(args.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return "error: url must start with http(s)://"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BlackthornAgent/1.0"})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
            raw = resp.read(1_500_000)
            charset = resp.headers.get_content_charset() or "utf-8"
        text = raw.decode(charset, errors="replace")
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _truncate(text)


def tool_remember(root: Path, args: Dict[str, Any]) -> str:
    note = str(args.get("note") or "").strip()
    if not note:
        return "error: note is required"
    try:
        store = studio._d1q_sync(
            "INSERT OR REPLACE INTO hermes_memories "
            "(id, target, content, memory_type, importance, user_id, session_id, profile, created_at, updated_at) "
            "VALUES (?, 'memory', ?, 'factual', 0.9, ?, '', 'default', ?, ?);",
            [uuid.uuid4().hex[:24], note, "blackthorn", time.time(), time.time()],
        )
        del store
        studio.invalidate_memory_cache()
        return f"remembered: {note}"
    except Exception as exc:
        return f"error: could not store memory ({exc})"


TOOLS: Dict[str, Callable[[Path, Dict[str, Any]], str]] = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "run_command": tool_run_command,
    "fetch_url": tool_fetch_url,
    "remember": tool_remember,
}

TOOL_LABELS = {
    "list_files": "Listing files",
    "read_file": "Reading a file",
    "write_file": "Writing a file",
    "run_command": "Running a command",
    "fetch_url": "Fetching a web page",
    "remember": "Saving to memory",
}


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
    max_steps = max(1, min(int(payload.max_steps or MAX_STEPS), 8))

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
    system_prompt = studio._compose_system(memories, attachments) + "\n\n" + TOOL_SYSTEM_PROMPT
    system_prompt += f"\nWorkspace root: {root}"
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
        yield activity("status", "Reading your request", status="ok",
                       detail=f'"{message[:160]}"' + (f" + {len(attachments)} attachment(s)" if attachments else ""))

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
                body = {"model": route["model"], "messages": messages, "stream": True}
                if payload.temperature is not None:
                    body["temperature"] = payload.temperature

                async with client.stream("POST", url, json=body, headers=headers) as resp:
                    if resp.status_code != 200:
                        detail = (await resp.aread()).decode("utf-8", "replace")[:300]
                        # The tunnel may have been replaced by a kernel restart while
                        # Render still held the old route: refresh it and retry once.
                        if not retried_route and resp.status_code in (404, 410, 502, 503):
                            retried_route = True
                            studio.invalidate_route_cache()
                            fresh = await studio._cached_route()
                            if fresh.get("url") and fresh.get("url") != route.get("url"):
                                log.info("agent: tunnel moved %s → %s — retrying",
                                         (route.get("url") or "")[:40], fresh["url"][:40])
                                route = fresh
                                url = f"{route['url']}/v1/chat/completions"
                                messages = [{"role": "system", "content": system_prompt}] + base_messages[1:]
                                route_retry_signal = True
                                # continue (not break!) so the `for` loop re-runs this
                                # step against the refreshed tunnel
                                continue
                            yield _sse({"type": "activity", "id": "route-refresh", "kind": "status",
                                        "title": "GPU endpoint moved — reconnecting", "status": "warn",
                                        "detail": detail[:200]})
                            continue
                        yield _sse({"type": "error",
                                    "message": f"The GPU endpoint returned HTTP {resp.status_code}: {detail}"})
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
                        # reasoning_content (if the model exposes it) stays private
                        piece = delta.get("content") or ""
                        if not piece:
                            continue
                        for event in splitter.feed(piece):
                            if event["kind"] == "answer":
                                if first_token_ms is None:
                                    first_token_ms = int((time.perf_counter() - t_start_model) * 1000) + prework_ms
                                yield _sse({"type": "delta", "delta": event["text"]})
                            elif event["kind"] == "think_start":
                                chunk = activity("thinking", "Thinking", status="running",
                                                 detail="Reasoning privately about the next step")
                                think_step_id = json.loads(chunk[6:].decode("utf-8"))["id"]
                                yield chunk
                            elif event["kind"] == "think_end":
                                if think_step_id:
                                    yield activity("thinking", "Thought for a moment", status="ok",
                                                   step_id=think_step_id,
                                                   duration_ms=int(event.get("seconds", 0) * 1000))
                                    think_step_id = None
                            elif event["kind"] == "tool":
                                call = event["call"]
                                yield activity("plan", f"Decided to use `{call['name']}`",
                                               status="ok", code=_tool_code(call["name"], call["arguments"]))
                for event in splitter.flush():
                    if event["kind"] == "answer":
                        yield _sse({"type": "delta", "delta": event["text"]})
                    elif event["kind"] == "think_end" and think_step_id:
                        yield activity("thinking", "Thought for a moment", status="ok",
                                       step_id=think_step_id,
                                       duration_ms=int(event.get("seconds", 0) * 1000))

                answer_text += splitter.answer
                calls = splitter.tool_calls

                if not calls:
                    if answer_text.strip():
                        break
                    # model produced nothing usable — nudge it once, then give up
                    if step_no == max_steps:
                        break
                    messages.append({"role": "assistant", "content": ""})
                    messages.append({"role": "user",
                                     "content": "Your previous reply was empty. Answer the user's request now."})
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
                                                "Continue and give the user the final answer, or call another tool if needed."})
                answer_text = ""  # intermediate text is not the final answer
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
        if answer or finish_reason in ("stop", "cancelled"):
            yield _sse({
                "type": "done", "session_id": session_id, "finish_reason": finish_reason,
                "first_token_ms": first_token_ms, "duration_ms": duration_ms,
                "output_tokens": out_tokens or None,
                "cancelled": cancelled,
                "steps": steps,
            })
        asyncio.create_task(persist(answer, out_tokens or None, finish_reason, steps))

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
        rows = studio._d1_sql(
            "SELECT value FROM state_meta WHERE key = ? LIMIT 1;", [key]
        )
        if rows:
            return str(rows[0].get("value") or "")
    except Exception:
        pass
    return ""


def _d1_set(key: str, value: str) -> None:
    try:
        studio._d1_sql("DELETE FROM state_meta WHERE key = ?;", [key])
        studio._d1_sql(
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
        lines = [f"- {r.get('title','')}: {r.get('url','')}\n  {r.get('content','')[:300]}" for r in results[:5]]
        return "\n".join(lines) if lines else "no results"
    except Exception as exc:
        return f"error: tavily {exc}"


TOOLS["tavily_search"] = tool_tavily_search
TOOL_LABELS["tavily_search"] = "Searching the web"
