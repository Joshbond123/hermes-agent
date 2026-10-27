"""The agent loop.

One streamed OpenAI-compatible chat completion per step, with the model choosing tools itself through
native ``tool_calls`` (no keyword gate, no text protocol). Each step's text is streamed to the client
immediately; tool results go back as proper ``role: "tool"`` messages. The loop is bounded (steps, tool calls,
duplicate calls, malformed arguments, wall-clock) and always ends with a real answer or an honest error.
Private reasoning (``reasoning`` deltas and ``<think>`` blocks) is never forwarded or stored.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

from . import prompts
from .llm import ContextOverflow, LLMError, stream_chat
from .route import RouteError, RouteResolver
from .runs import Run
from .tools import ToolContext, ToolRegistry, ToolResult, loads_args, redact

log = logging.getLogger("blackthorn.agent")

FINALIZE_NOTE = ("Tool use is finished for this turn. Using only the information gathered above, write the final answer "
                 "now. If something could not be done or found, say so plainly.")

LENGTH_CONTINUE_NOTE = ("Your reply was cut off by the model's output token limit. Continue from exactly "
                        "where it stopped, without repeating anything already said. If you were about to call "
                        "tools, call them now and keep working until every item is complete. Keep updates brief.")

CONTINUE_NOTE = ("Your previous reply was cut off by a network error before it finished. Continue exactly from the "
                 "end of that reply: do not repeat any text that was already written, just finish the thought.")

# Server-generated notes. They are appended as user messages because that is what the wire protocol
# needs, but they are NOT user intent: the last *real* user message always outranks them for history
# protection and for the model's sense of what was asked (upstream issue #14665).
GATE_NOTE = ("[server] Some items on your task list are still open. Continue with the next one now, using tools. "
             "Do not repeat a step already marked completed. If you really cannot continue, say exactly what is "
             "blocking you instead of reporting success.")
RESUME_NOTE = ("[server] The earliest part of this turn was compacted to fit the model's context window. "
               "Keep working from the tool results above until the user's request is genuinely complete, then "
               "report only what the recorded results show.")
NOTE_PREFIXES = (FINALIZE_NOTE[:40], LENGTH_CONTINUE_NOTE[:40], CONTINUE_NOTE[:40], GATE_NOTE[:40], RESUME_NOTE[:40],
                 prompts.COMPACT_HEADER[:40])
GATE_STEPS = 6                      # extra steps a run may spend on the completion gate


class ThinkFilter:
    """Split streamed content into visible text and ``<think>…</think>`` text without ever swallowing normal text.

    Only a trailing fragment that could still become a tag is held back, and only until the next chunk decides it.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.buf = ""
        self.inside = False

    @staticmethod
    def _partial(buf: str, tag: str) -> int:
        for n in range(min(len(tag) - 1, len(buf)), 0, -1):
            if tag.startswith(buf[-n:]):
                return n
        return 0

    def feed(self, chunk: str) -> Tuple[str, str]:
        self.buf += chunk
        visible: List[str] = []
        hidden: List[str] = []
        while True:
            tag = self.CLOSE if self.inside else self.OPEN
            i = self.buf.find(tag)
            if i >= 0:
                (hidden if self.inside else visible).append(self.buf[:i])
                self.buf = self.buf[i + len(tag):]
                self.inside = not self.inside
                continue
            keep = self._partial(self.buf, tag)
            cut = len(self.buf) - keep
            (hidden if self.inside else visible).append(self.buf[:cut])
            self.buf = self.buf[cut:]
            break
        return "".join(visible), "".join(hidden)

    def flush(self) -> Tuple[str, str]:
        rest, self.buf = self.buf, ""
        return ("", rest) if self.inside else (rest, "")


@dataclass
class Deps:
    settings: Any
    store: Any
    resolver: RouteResolver
    registry: ToolRegistry
    http: httpx.AsyncClient
    tavily: Any
    computer: Any
    activity: Any = None   # run_started() / run_touched() / run_finished(): keeps the GPU's inactivity clock honest


@dataclass
class StepResult:
    text: str = ""
    calls: List[Dict[str, str]] = field(default_factory=list)
    finish: Optional[str] = None


class RunFailure(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


class AgentRun:
    def __init__(self, run: Run, deps: Deps, *, system: str, history: List[Dict[str, str]], user_message: str,
                 model: str = "", flush_interval: float = 1.5, turns: Optional[List[Dict[str, Any]]] = None,
                 board: Any = None, previous_attempt: Optional[Dict[str, Any]] = None, resume_hint: str = ""):
        self.run, self.deps = run, deps
        self.system, self.history, self.user_message = system, history, user_message
        # Rich history: every stored turn with its finish status and the ledger of what its tools really
        # did. Falls back to the plain role/content history for callers that do not pass it.
        self.turns: List[Dict[str, Any]] = list(turns) if turns is not None else [
            {"role": m.get("role"), "content": m.get("content") or "", "status": "", "ledger": []} for m in (history or [])]
        self.board = board                                  # persistent task list (TaskBoard)
        self.previous_attempt = previous_attempt            # ledger of a discarded regenerate attempt
        self.resume_hint = resume_hint
        self.model = model
        self.parts: List[Dict[str, Any]] = []
        self.ledger: List[Dict[str, Any]] = []              # what ran in THIS turn, in completion order
        self._epoch = 0                                     # bumped by a successful write
        # A re-run after a real change is legitimate (edit, then re-run the tests), but the same call
        # must not be repeatable for ever: the lifetime cap stops a loop that keeps re-running one
        # command and calling each attempt "new work".
        self._repeat_total_cap = max(4, int(deps.settings.max_repeat_calls) * 4)
        self._nudges = 0
        self._state_cache = ""
        self._messages: List[Dict[str, Any]] = []
        self.tool_calls_total = 0
        self.strikes = 0
        self.malformed = 0
        self.seen: Dict[Tuple[str, str, int], int] = {}
        self.seen_total: Dict[Tuple[str, str], int] = {}   # lifetime cap per (tool, args), all epochs
        self.steps = 0
        self.usage = {"prompt": 0, "completion": 0}
        self.finish_reason: Optional[str] = None
        self.first_token_ms: Optional[int] = None
        self.timings_ms = {"lease": 0, "route": 0, "model": 0, "tools": 0, "final_save": 0}
        self._t0 = time.monotonic()
        self._flush_interval = flush_interval
        self._last_flush = 0.0
        self._flush_task: Optional[asyncio.Task] = None
        self._produced = False
        self._history_len = 0
        self._last_step_text = ""
        self._recoveries = 0
        s = deps.settings
        self._budget = prompts.prompt_budget(s.context_tokens, s.completion_reserve_tokens)
        self._schemas = deps.registry.schemas()
        # Remote malware tools are optional, not 2 extra schemas on every normal
        # conversation. Keeping them off unrelated turns saves prefill and prevents
        # history compaction from evicting the user's original task instruction.
        import re
        context = self.user_message + " " + (board.goal if board is not None and board.is_open() else "")
        if self.user_message.strip().lower() in ("continue", "yes", "go on", "do it"):
            context += " " + " ".join(str(t.get("content") or "")[:500] for t in self.turns[-4:] if t.get("role") == "user")
        self._remote_analysis = bool(re.search(r"malware|ransomware|sample|detonat|sandbox|virus|malicious", context, re.I))
        # An explicitly requested public upload/report has a known short workflow:
        # omit unrelated catalog, browser and shell schemas for this turn. That
        # saves ~1k prefill tokens on the slow partially-offloaded 27B GPU, while
        # keeping the general research/computer toolbox on broader tasks.
        specific_remote = self._remote_analysis and bool(re.search(
            r"(submit|upload|report|lookup|public remote analysis).{0,120}(malware|analyzer)|"
            r"(malware|analyzer).{0,120}(submit|upload|report|lookup)", context, re.I))
        if specific_remote:
            allowed = {"malware_analyzer_submit", "malware_analyzer_report", "sample_triage",
                       "list_files", "read_file", "write_file", "web_search", "todo_list"}
            self._active_schemas = [schema for schema in self._schemas if schema["function"]["name"] in allowed]
        elif self._remote_analysis:
            self._active_schemas = self._schemas
        else:
            # Qwen's template expands every function description into the prompt on EVERY
            # model step. Keep the nine core research/computer/task tools available, and load
            # specialised schemas only when the current task (or its recent context) needs them.
            # Full schemas remain in the registry: no tool or capability is uninstalled.
            # On the live dual-T4 Qwen, model prefill dominates a computer turn.
            # Offer the five computer primitives first; disclose unrelated schemas
            # only when the task calls for them (or a prior ledger used them).
            core = {"run_command", "list_files", "read_file", "write_file", "job_status"}
            context_lower = (context + " " + " ".join(str(t.get("content") or "")[:300]
                            for t in self.turns[-4:] if t.get("role") == "user")).lower()
            def mentions(*words: str) -> bool:
                return any(word in context_lower for word in words)
            if mentions("search", "research", "latest", "current", "news", "source", "web", "internet"):
                core.add("web_search")
            if mentions("url", "http", "fetch", "page"):
                core.add("fetch_url")
            if mentions("computer info", "computer status", "system specs", "hardware"):
                core.add("computer_info")
            if mentions("plan", "multi-step", "long task", "todo", "project"):
                core.add("todo_list")
            if mentions("optional tool", "catalog", "skill", "browser", "browse", "sample"):
                core.add("optional_tool")
            if mentions("remember", "memory", "recall", "previous", "earlier", "history", "session", "past chat"):
                core.update(("remember", "session_search"))
            if mentions("skill", "playbook", "workflow", "training"):
                core.update(("skill_search", "skill_load", "skill_read"))
            if mentions("catalog", "security tool", "pentest", "nmap", "install", "scan", "tool search"):
                core.add("tool_search")
            if mentions("browse", "browser", "web page", "website", "javascript", "screenshot", "login", "web app"):
                core.add("browse")
            if mentions("triage", "suspicious file", "sample"):
                core.add("sample_triage")
            # A continued task can still call any specialised tool that actually ran earlier.
            core.update(entry.get("n") for turn in self.turns[-8:]
                        for entry in (turn.get("ledger") or []) if isinstance(entry, dict))
            self._active_schemas = [schema for schema in self._schemas
                                    if schema["function"]["name"] in core]
        self._schema_cost = prompts.schema_tokens(self._active_schemas)

    # ------------------------------------------------------------------ helpers
    def text(self) -> str:
        return "\n\n".join(p["text"] for p in self.parts if p["type"] == "text" and p["text"])

    def _elapsed_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    def _check(self) -> None:
        if self.run.cancel_event.is_set():
            raise asyncio.CancelledError()
        if time.monotonic() - self._t0 > self.deps.settings.run_timeout_s:
            raise RunFailure("run_timeout", f"The run exceeded its {int(self.deps.settings.run_timeout_s)}s limit and was stopped.")

    def _emit_text(self, text: str) -> None:
        if not text:
            return
        if self.first_token_ms is None:
            self.first_token_ms = self._elapsed_ms()
        if self.parts and self.parts[-1]["type"] == "text":
            self.parts[-1]["text"] += text
        else:
            self.parts.append({"type": "text", "text": text})
        self._produced = True
        self.run.emit("text.delta", text=text)
        self._maybe_flush()

    def _meta(self, status: str, error: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"model": self.model, "duration_ms": self._elapsed_ms(), "first_token_ms": self.first_token_ms,
                                "steps": self.steps, "tool_calls": self.tool_calls_total, "finish_reason": self.finish_reason,
                                "usage": dict(self.usage), "status": status, "timings_ms": dict(self.timings_ms)}
        if error:
            meta["error"] = error
        return meta

    async def _persist(self, status: str, error: Optional[Dict[str, Any]] = None) -> None:
        await self.deps.store.update_assistant(
            self.run.assistant_uid, text=self.text(), parts=self.parts, status=status, meta=self._meta(status, error),
            tokens=self.usage["completion"] or None, ledger=list(self.ledger))

    async def _flush_now(self) -> None:
        """Force a save (used after each tool batch so a restart resumes from real results)."""
        try:
            if self._flush_task and not self._flush_task.done():
                await asyncio.shield(self._flush_task)
            else:
                await self._safe_flush()
        except Exception as exc:  # noqa: BLE001
            log.warning("tool-batch save failed: %s", exc)

    # ------------------------------------------------------------------ memory / task state
    def _prior_steps(self, limit: int = 8) -> List[Dict[str, Any]]:
        """The most recent ledger entries from earlier turns of this session.

        Kept in the state layer so "what is already done" survives even when the turn that did it has
        been compacted out of the history tail.
        """
        out: List[Dict[str, Any]] = []
        for turn in reversed(self.turns):
            for entry in reversed(turn.get("ledger") or []):
                if not isinstance(entry, dict):
                    continue
                out.append(entry)
                if len(out) >= limit:
                    return list(reversed(out))
        return list(reversed(out))

    def _state_block(self) -> str:
        """Ephemeral machine-generated state layer, injected per API call (Hermes does the same)."""
        todos = self.board.read() if self.board is not None else []
        return prompts.task_state_block(goal=self.user_message,
                                        task_goal=self.board.goal if self.board is not None and self.board.is_open() else "",
                                        todos=todos, ledger=self.ledger,
                                        prior_steps=self._prior_steps(),
                                        previous_attempt=self.previous_attempt, resume_hint=self.resume_hint)

    def _system_content(self) -> str:
        block = self._state_block()
        return f"{self.system}\n\n{block}" if block else self.system

    def _sync_state(self) -> None:
        """Refresh the injected state in place; unchanged state leaves the prompt prefix untouched."""
        if not self._messages:
            return
        content = self._system_content()
        if content != self._state_cache:
            self._state_cache = content
            self._messages[0] = {"role": "system", "content": content}

    def _force_compact(self) -> None:
        """The server rejected a prompt our estimator thought fitted: shrink it for real.

        Two passes — whole tool groups / whole turns at half the current estimate, then (because the
        estimator is the thing that was wrong) literally drop the oldest half of the unprotected
        messages. The system prompt and the user's real request always survive.
        """
        cpt = self.deps.settings.chars_per_token
        msgs = self._messages
        if len(msgs) < 3:
            return
        protect = sorted(self._protected_indices())
        prompts.shrink_tool_messages(msgs, int(prompts.estimate_messages(msgs, cpt) * 0.5), cpt, protect=protect)
        removable = [i for i in range(1, len(msgs)) if i not in set(self._protected_indices())]
        if removable:
            for i in sorted(removable[:max(1, len(removable) // 2)], reverse=True):
                msgs.pop(i)
        prompts._sanitize_pairs(msgs)

    def _protected_indices(self) -> List[int]:
        """Indices compaction must never drop: the system prompt and the user's real request(s)."""
        out = {0}
        # Preserve the session-opening request, not just the newest turn. Hermes keeps a
        # protected head; otherwise a long-running follow-up can forget the original goal.
        opening = next((prompts.one_line(t.get("content") or "", 500)
                        for t in self.turns if t.get("role") == "user" and str(t.get("content") or "").strip()), "")
        if opening:
            out.update(i for i, msg in enumerate(self._messages[1:], 1)
                       if msg.get("role") == "user" and msg.get("content") == opening)
        for i in range(len(self._messages) - 1, 0, -1):
            msg = self._messages[i]
            if msg.get("role") != "user":
                continue
            content = (msg.get("content") or "").strip()
            if any(content.startswith(p) for p in NOTE_PREFIXES if p):
                continue                                    # a server note is expendable
            out.add(i)
            break
        return sorted(out)

    def _emit_unfinished(self, status: str) -> None:
        """Tell the user (and the next turn) when the plan did not finish. Never silent."""
        if self.board is None or status not in ("stop", "length"):
            return
        pending = self.board.pending()
        if not pending:
            return
        listing = "; ".join(f"{i['id']}: {prompts.one_line(i.get('content') or '', 60)}" for i in pending[:4])
        self.run.emit("notice", level="warn",
                      text=f"This response ended with {len(pending)} unfinished task(s): {listing}. "
                           "Ask me to continue and I will resume from here.")

    def _continuation_note(self) -> Optional[str]:
        """Ask the model to keep going while its own task list still has open items."""
        s = self.deps.settings
        if self.board is None or self._nudges >= int(getattr(s, "max_completion_nudges", 3)):
            return None
        if self.tool_calls_total >= s.max_tool_calls or self.strikes >= 3:
            return None
        pending = self.board.pending()
        if not pending:
            return None
        self._nudges += 1
        listing = "\n".join(f"- {i['id']}: {prompts.one_line(i.get('content') or '', 90)} [{i.get('status')}]"
                            for i in pending[:8])
        return f"{GATE_NOTE}\nStill open:\n{listing}\nUpdate the list with todo_list as each one finishes."

    def _finalize_note(self) -> str:
        pending = self.board.pending() if self.board is not None else []
        if not pending:
            return FINALIZE_NOTE
        listing = "\n".join(f"- {i['id']}: {prompts.one_line(i.get('content') or '', 80)}" for i in pending[:8])
        return (FINALIZE_NOTE + f"\nThese items are still open: \n{listing}\n"
                "Say plainly which of them are not done and why. Do not imply they were completed.")

    def _maybe_flush(self) -> None:
        now = time.monotonic()
        if now - self._last_flush < self._flush_interval or (self._flush_task and not self._flush_task.done()):
            return
        self._last_flush = now
        self._flush_task = asyncio.get_running_loop().create_task(self._safe_flush())

    async def _safe_flush(self) -> None:
        try:
            await self._persist("streaming")
        except Exception as exc:  # progress saves are best-effort; the final save is what must succeed
            log.warning("progress save failed: %s", exc)

    # ------------------------------------------------------------------ entry point
    def _activity(self, hook: str, *args: Any) -> None:
        fn = getattr(self.deps.activity, hook, None) if self.deps.activity is not None else None
        if fn is not None:
            try:
                fn(*args)
            except Exception:  # bookkeeping must never affect a run
                pass

    async def _lease_heartbeat(self) -> None:
        """Durable, expiring run lease: shutdown checks other Render workers too."""
        import json
        ticks = 0
        while True:
            await asyncio.sleep(30)
            ticks += 1
            try:
                await self.deps.store.set_setting("bt_run:" + self.run.id, json.dumps({
                    "session_id": self.run.session_id, "expires": time.time() + 120}))
            except Exception as exc:
                log.warning("run lease refresh failed: %s", type(exc).__name__)
            # Provider TTL is an IDLE timer, not a durable disk. While this real
            # agent task is running (including slow model prefill), touch its VM
            # every two minutes; never do this when idle/off to evade the tier.
            if ticks % 4 == 0 and hasattr(self.deps.computer, "keepalive"):
                try:
                    await self.deps.computer.keepalive()
                except Exception as exc:
                    log.warning("active computer keepalive failed: %s", type(exc).__name__)

    async def execute(self) -> None:
        status, error = "stop", None
        self._activity("run_started")
        lease = asyncio.create_task(self._lease_heartbeat())
        # Immediate UI feedback: do not wait for GPU connect / prefill before the user sees activity.
        self.run.emit("thinking", state="start")
        self._early_thinking = True
        try:
            import json
            # Before any tool executes, D1 must hold a cross-worker shutdown lease.
            lease_started = time.monotonic()
            await self.deps.store.set_setting("bt_run:" + self.run.id, json.dumps({
                "session_id": self.run.session_id, "expires": time.time() + 120}))
            self.timings_ms["lease"] += int((time.monotonic() - lease_started) * 1000)
            await self._loop()
            status = "length" if self.finish_reason == "length" else "stop"
        except asyncio.CancelledError:
            status = "cancelled" if self.run.cancel_event.is_set() else "interrupted"
        except RunFailure as exc:
            status, error = "error", {"code": exc.code, "message": exc.message, "retryable": exc.retryable}
        except RouteError as exc:
            status, error = "error", {"code": exc.code, "message": exc.message, "retryable": exc.code == "gpu_booting"}
        except LLMError as exc:
            status, error = "error", {"code": exc.code, "message": exc.message, "retryable": exc.retryable}
        except Exception as exc:  # noqa: BLE001 - a run must always end cleanly
            log.exception("agent run crashed")
            status, error = "error", {"code": "internal", "message": f"{type(exc).__name__}: {str(exc)[:240]}", "retryable": True}
        finally:
            lease.cancel()
            try:
                await lease
            except asyncio.CancelledError:
                pass
            try:
                if hasattr(self.deps.store, "delete_setting"):
                    await self.deps.store.delete_setting("bt_run:" + self.run.id)
                else:
                    await self.deps.store.set_setting("bt_run:" + self.run.id, "")
            except Exception:
                pass   # an expiring lease is safer than a premature shutdown
            for part in self.parts:
                if part["type"] == "tool" and part.get("status") == "running":
                    part["status"] = "cancelled" if status in ("cancelled", "interrupted") else "error"
            if error:
                self.run.emit("error", **error)
            if status == "length":
                self.run.emit("notice", level="warn", text="The answer was cut off by the model's context/token limit.")
            # Honest close-out: an unfinished plan is reported as unfinished, not as success.
            self._emit_unfinished(status)
            saved = True
            final_started = time.monotonic()
            try:
                if self._flush_task and not self._flush_task.done():
                    await asyncio.shield(self._flush_task)
                await asyncio.shield(self._persist(status, error))
                await asyncio.shield(self.deps.store.finish_session_activity(self.run.session_id, tool_calls=self.tool_calls_total))
            except Exception as exc:  # noqa: BLE001
                saved = False
                log.error("final save failed: %s", exc)
                self.run.emit("notice", level="error", text="This response could not be saved to history: " + str(exc)[:160])
            self.timings_ms["final_save"] += int((time.monotonic() - final_started) * 1000)
            self._activity("run_finished")
            self.run.emit("run.end", status=status, finish_reason=self.finish_reason, duration_ms=self._elapsed_ms(),
                          first_token_ms=self.first_token_ms, message_id=self.run.assistant_uid, session_id=self.run.session_id,
                          steps=self.steps, tool_calls=self.tool_calls_total, usage=dict(self.usage), saved=saved,
                          timings_ms=dict(self.timings_ms))
            self.run.finish(status)

    # ------------------------------------------------------------------ the loop
    def _tools_for_step(self, step: int):
        """Omit tool schemas on tiny chit-chat to cut prefill latency.

        Only the very first turn of a brand-new conversation can lose its tools: on a follow-up such as
        "continue", "yes" or "do it" the model must still be able to act on the work in progress.
        """
        if not self._active_schemas:
            return None
        if self._history_len or (self.board is not None and self.board.is_open()):
            return self._active_schemas
        text = (self.user_message or "").strip().lower()
        if step <= 1 and len(text) < 48:
            keys = ("run", "file", "search", "code", "install", "browse", "skill", "computer",
                    "write", "read", "exec", "git", "http", "scan", "tool", "ls", "mkdir", "pip", "npm",
                    "remember", "previous", "history", "earlier", "task", "continue", "recall")
            if not any(k in text for k in keys):
                return None
        return self._active_schemas

    async def _loop(self) -> None:
        s = self.deps.settings
        cpt = s.chars_per_token
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self._system_content()}]
        self._state_cache = messages[0]["content"]
        used = prompts.estimate_tokens(messages[0]["content"], cpt) + prompts.estimate_tokens(self.user_message, cpt) + 24
        # Head (the session's opening request) + a budgeted tail of whole turns + a server-built summary
        # of whatever does not fit — Hermes' context layout, instead of a silent newest-first truncation.
        history, dropped, _summary = prompts.compile_history(self.turns, self._budget - self._schema_cost - used, cpt)
        messages += history
        self._history_len = len(history)
        if dropped:
            self.run.emit("internal.compaction", count=dropped, stage="history")
        messages.append({"role": "user", "content": self.user_message})
        self._messages = messages

        force_final = False
        empty_retries = 0
        length_continues = 0
        while True:
            self._check()
            self.steps += 1
            if not force_final and (self.steps > s.max_steps or self.tool_calls_total >= s.max_tool_calls
                                    or self.strikes >= 3 or self.malformed >= s.max_malformed_calls):
                force_final = True
                reason = ("repeated identical tool calls" if self.strikes >= 3 else
                          "invalid tool arguments" if self.malformed >= s.max_malformed_calls else "the tool-call limit")
                self.run.emit("notice", level="warn", text=f"Stopped using tools ({reason}); writing the answer from what was gathered.")
                self._messages.append({"role": "user", "content": self._finalize_note()})
            result = await self._model_step(self._messages, None if force_final else self._tools_for_step(self.steps))
            if result.calls and not force_final:
                await self._run_tools(result, self._messages)
                continue
            if not result.text.strip() and empty_retries < 1:
                # The model produced nothing visible (an intermittent quirk: reasoning, then stop). Nothing was shown or
                # saved for this step, so asking once more is invisible apart from the extra wait.
                empty_retries += 1
                self.run.emit("notice", level="info", text="The model returned an empty answer; asking once more.")
                continue
            if result.finish == "length" and length_continues < 12:
                # Silent seamless continuation — never show a token-ceiling notice to the user.
                length_continues += 1
                if result.text.strip():
                    self._messages.append({"role": "assistant", "content": result.text})
                self._messages.append({"role": "user", "content": LENGTH_CONTINUE_NOTE})
                continue
            # Completion gate: a stop with open tasks is not a finished task. Give the model a bounded
            # number of chances to continue instead of letting it declare victory over a half-done plan.
            note = self._continuation_note()
            if note is not None and self.steps < s.max_steps + GATE_STEPS:
                open_count = len(self.board.pending()) if self.board is not None else 0
                self.run.emit("notice", level="info",
                              text=f"{open_count} task(s) from the list are still open — continuing instead of stopping.")
                if result.text.strip():
                    self._messages.append({"role": "assistant", "content": result.text})
                self._messages.append({"role": "user", "content": note})
                self.steps += 1
                continue
            self.finish_reason = result.finish or "stop"
            self._last_step_text = result.text
            break

        if not self._last_step_text.strip():
            if any(p["type"] == "tool" for p in self.parts):
                raise RunFailure("no_answer", "The model ran its tools but did not write an answer. Retry to ask again.", True)
            raise RunFailure("empty_response", "The model returned an empty response (it produced only reasoning). Retry to ask again.", True)

    # ------------------------------------------------------------------ one model call (with recovery)
    async def _model_step(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> StepResult:
        s = self.deps.settings
        attempt = 0
        auth_refreshes = 0
        while True:
            attempt += 1
            self._check()
            budget = self._budget - (self._schema_cost if tools else 0)
            self._sync_state()
            before = len(messages)
            if not prompts.shrink_tool_messages(messages, budget, s.chars_per_token,
                                                protect=self._protected_indices()):
                raise RunFailure(
                    "context_full",
                    "This single message is too large for the model window even after compaction. "
                    "Shorten the message or clear attachments and try again.",
                    True,
                )
            if len(messages) < before:
                self.run.emit("internal.compaction", stage="mid_task")
                # Mid-task resume after compaction (Hermes-style): the model must not answer from a hole.
                if messages[-1].get("role") == "tool" and (
                    prompts.estimate_messages(messages, s.chars_per_token)
                    + prompts.estimate_tokens(RESUME_NOTE, s.chars_per_token) + 3 <= budget
                ):
                    messages.append({"role": "user", "content": RESUME_NOTE})
            route_started = time.monotonic()
            route = await self.deps.resolver.get()
            self.timings_ms["route"] += int((time.monotonic() - route_started) * 1000)
            self.model = route.model
            self._produced = False

            def _text_len() -> int:
                return sum(len(part["text"]) for part in self.parts if part["type"] == "text")

            before = _text_len()
            model_started = time.monotonic()
            try:
                return await self._consume(route, messages, tools)
            except ContextOverflow:
                # The server's window is smaller than our estimate, so the server wins: compact for real
                # (whole tool groups first, then whole turns) and keep the user's request. Nothing is
                # silently deleted, and a summary of what was dropped replaces it wherever possible —
                # Hermes' contract: compact and continue, never restart the session.
                if attempt >= 3:
                    raise
                if attempt >= 2:
                    for m in messages:
                        if m.get("role") == "tool" and len(m["content"]) > 400:
                            m["content"] = m["content"][:400] + "\n[… clipped to fit the context window …]"
                self._force_compact()
                self.run.emit("internal.compaction", stage="overflow_retry")
                if messages[-1].get("role") == "tool":
                    messages.append({"role": "user", "content": RESUME_NOTE})
                continue
            except LLMError as exc:
                if exc.code == "gpu_auth":
                    # A notebook may rotate its per-boot key while a long task is
                    # running. Retry only the rejected model step after an uncached D1
                    # read; the completed tool ledger is NOT replayed. An unchanged
                    # route/key (or a second rejection after refresh) is a real auth
                    # problem and must remain visible instead of looping indefinitely.
                    self.deps.resolver.invalidate()
                    try:
                        fresh = await self.deps.resolver.get(force=True)
                    except Exception:
                        raise exc
                    if auth_refreshes >= 1:
                        raise LLMError("gpu_auth", "The GPU gateway still rejects the refreshed route. "
                                       "Check the GPU connection and retry once it is healthy.",
                                       status=exc.status) from exc
                    if fresh.api_key == route.api_key and fresh.url == route.url:
                        # An edge can reject a single stream request even when the
                        # current key is valid. Only retry unchanged credentials
                        # when a cheap authenticated endpoint actually accepts them.
                        try:
                            check = await self.deps.http.get(
                                f"{fresh.url}/v1/models",
                                headers={"Authorization": f"Bearer {fresh.api_key}"}, timeout=5.0)
                            if check.status_code != 200:
                                raise exc
                        except (OSError, httpx.HTTPError):
                            raise exc
                    auth_refreshes += 1
                    self.run.emit("notice", level="info", text="GPU connection refreshed — resuming this model step…")
                    continue
                # Some engines truncate the conversation *before* applying Qwen3.8's template.
                # When their SSE error says the latest user query vanished, perform ONE bounded
                # compaction and repeat the real request near the tail. Do not lose the ledger or
                # retry indefinitely; the normal context_full error remains actionable.
                if (exc.code in ("gpu_stream", "gpu_http") and
                        "no user query found in messages" in exc.message.lower() and attempt == 1):
                    self._force_compact()
                    messages.append({"role": "user", "content":
                                     RESUME_NOTE + "\nReal user request (restated): " + self.user_message[:1400]})
                    self.run.emit("internal.compaction", stage="qwen_query_retry")
                    continue
                # Transport faults (edge timeout 524, dropped tunnel, stalled stream) are not
                # the model failing: recover instead of killing a long run mid-task. A dead GPU
                # still fails fast (retryable=False on auth/protocol errors).
                transient = (exc.retryable or
                             exc.code in ("gpu_unreachable", "gpu_dropped", "gpu_stalled") or
                             (exc.status is not None and exc.status in (502, 503, 521, 522, 523, 524)))
                # More patience on edge/proxy blips; auth/protocol still fail fast (retryable=False).
                # Bounded and visible (attempt N of limit); the first-byte watchdog already
                # fails over half-open sockets quickly, so each attempt is cheap.
                limit = 6 if (exc.status in (502, 503, 521, 522, 523, 524) or
                              exc.code in ("gpu_unreachable", "gpu_dropped", "gpu_stalled")) else 3
                if not transient or attempt >= limit:
                    raise
                self.deps.resolver.invalidate()
                partial = "".join(part["text"] for part in self.parts if part["type"] == "text")[before:]
                # Every recovery is visible with a bounded attempt count: a silent wait of minutes
                # looked like a hang, and an unbounded "retrying" loop is what this replaces.
                if partial.strip():
                    if self._recoveries >= 5:
                        raise
                    self._recoveries += 1
                    self.run.emit("notice", level="info",
                                  text=f"Connection dropped mid-answer; continuing from the cut point (attempt {attempt + 1} of {limit}).")
                    messages.append({"role": "assistant", "content": partial})
                    messages.append({"role": "user", "content": CONTINUE_NOTE})
                else:
                    self.run.emit("notice", level="info",
                                  text=f"Reconnecting to the GPU — retrying (attempt {attempt + 1} of {limit})…")
                await asyncio.sleep(min(6.0, 0.6 * (2 ** (attempt - 1))))
                continue
            finally:
                self.timings_ms["model"] += int((time.monotonic() - model_started) * 1000)

    async def _consume(self, route: Any, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> StepResult:
        s = self.deps.settings
        flt = ThinkFilter()
        thinking_since: Optional[float] = None
        acc: Dict[int, Dict[str, str]] = {}
        text_chunks: List[str] = []
        finish: Optional[str] = None

        def start_thinking() -> None:
            nonlocal thinking_since
            if thinking_since is None:
                thinking_since = time.monotonic()
                if getattr(self, "_early_thinking", False):
                    self._early_thinking = False
                else:
                    self.run.emit("thinking", state="start")

        def end_thinking() -> None:
            nonlocal thinking_since
            if thinking_since is not None:
                self.run.emit("thinking", state="end", ms=int((time.monotonic() - thinking_since) * 1000))
                thinking_since = None

        def visible(text: str) -> None:
            end_thinking()
            text_chunks.append(text)
            self._emit_text(text)

        extra = None if s.thinking else {"chat_template_kwargs": {"enable_thinking": False}}
        async for ev in stream_chat(route, messages, client=self.deps.http, tools=tools, idle_timeout=s.llm_idle_timeout_s,
                                    first_byte_timeout=s.llm_first_byte_timeout_s,
                                    connect_timeout=s.llm_connect_timeout_s, extra=extra):
            self._check()
            kind = ev["t"]
            if kind == "reasoning":
                start_thinking()  # the *content* of reasoning is discarded; only the fact that it is happening is shown
            elif kind == "content":
                vis, hid = flt.feed(ev["v"])
                if hid:
                    start_thinking()
                if vis:
                    visible(vis)
            elif kind == "tool":
                end_thinking()
                slot = acc.setdefault(ev["index"], {"id": "", "name": "", "args": ""})
                if ev.get("id"):
                    slot["id"] = ev["id"]
                if ev.get("name"):
                    slot["name"] += ev["name"]
                slot["args"] += ev.get("args") or ""
                self._produced = True
                if self.first_token_ms is None:
                    self.first_token_ms = self._elapsed_ms()
            elif kind == "usage":
                self.usage["prompt"] = max(self.usage["prompt"], ev["prompt"])
                self.usage["completion"] += ev["completion"]
            elif kind == "finish":
                finish = ev["v"]
        vis, hid = flt.flush()
        if vis:
            visible(vis)
        end_thinking()

        calls: List[Dict[str, str]] = []
        for index in sorted(acc):
            slot = acc[index]
            if not slot["name"].strip():
                continue
            calls.append({"id": slot["id"] or f"call_{self.run.id[-6:]}_{self.steps}_{index}", "name": slot["name"].strip(),
                          "args": slot["args"]})
        return StepResult(text="".join(text_chunks), calls=calls, finish=finish)

    # ------------------------------------------------------------------ tools
    async def _run_tools(self, step: StepResult, messages: List[Dict[str, Any]]) -> None:
        s = self.deps.settings
        wire_calls = []
        for c in step.calls:
            try:
                json.loads(c["args"] or "{}")
                arg_str = c["args"] or "{}"
            except ValueError:
                arg_str = "{}"
            wire_calls.append({"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": arg_str}})
        messages.append({"role": "assistant", "content": step.text or "", "tool_calls": wire_calls})

        ctx = ToolContext(settings=s, store=self.deps.store, computer=self.deps.computer, http=self.deps.http,
                          tavily=self.deps.tavily, session_id=self.run.session_id, run_id=self.run.id,
                          board=self.board, user_message=self.user_message)
        # Every call the model asked for is executed — none are ever dropped or skipped.
        # max_calls_per_step bounds *concurrency* (protects the GPU computer from a stampede),
        # not how much work a step may do. Each call reports its result the moment it finishes
        # so the UI shows live progress instead of one late batch.
        sem = asyncio.Semaphore(max(1, int(s.max_calls_per_step)))
        seen_lock = asyncio.Lock()

        async def _execute(call: Dict[str, str]) -> ToolResult:
            name = call["name"]
            parse_error: Optional[str] = None
            args: Dict[str, Any] = {}
            try:
                args = loads_args(call["args"])
            except ValueError as exc:
                parse_error = str(exc)
            if parse_error:
                return ToolResult.failure(f"invalid arguments for {name}: {parse_error}", hint="Send valid JSON arguments.")
            if self.deps.registry.get(name) is None:
                return await self.deps.registry.run(name, args, ctx)
            # Duplicate suppression is keyed on (tool, args, epoch): the epoch advances whenever a
            # write/exec succeeds, so a legitimate re-run after a real change (edit a file, then run the
            # tests again) is allowed, while a stuck loop on an unchanged world is still stopped.
            key = (name, json.dumps(args, sort_keys=True, default=str), self._epoch)
            async with seen_lock:
                count = self.seen.get(key, 0)
                total = self.seen_total.get(key[:2], 0)
                if count >= s.max_repeat_calls or total >= self._repeat_total_cap:
                    return ToolResult.failure(
                        f"duplicate call suppressed: {name} was already run with these exact arguments "
                        f"{max(count, total)} times",
                        hint="Use the earlier result in the conversation to answer, or try different arguments.")
                self.seen[key] = count + 1
                self.seen_total[key[:2]] = total + 1
            async with sem:
                return await self.deps.registry.run(name, args, ctx)

        pending: List[tuple] = []
        for call in step.calls:
            self._check()
            name, call_id = call["name"], call["id"]
            try:
                preview_args = loads_args(call["args"])
            except ValueError:
                preview_args = {}
            summary = self.deps.registry.describe(name, preview_args) if preview_args or not call["args"] else ""
            part: Dict[str, Any] = {"type": "tool", "id": call_id, "name": name, "status": "running", "args": summary,
                                    "summary": "", "step": self.steps, "started_at": time.time()}
            self.parts.append(part)
            self._activity("run_touched", "tool")
            self.run.emit("tool.start", id=call_id, name=name, summary=summary, step=self.steps)
            pending.append((call, part, time.monotonic()))

        contents: Dict[str, str] = {}
        finished: Dict[int, tuple] = {}
        emit_next = 0

        def _publish(index: int) -> None:
            """Emit tool.end events in the model's call order, as soon as each prefix is done."""
            nonlocal emit_next
            while emit_next in finished:
                part, result, duration = finished[emit_next]
                call_id, name = part["id"], part["name"]
                ui_out = redact(str(result.data.get("output") or ""))[: s.ui_result_chars] if result.data.get("output") else ""
                part.update(status="ok" if result.ok else "error", summary=redact(result.summary)[:240], duration_ms=duration)
                if not result.ok:
                    part["error"] = redact(result.error)[:300]
                if ui_out:
                    part["output"] = ui_out
                if result.data.get("sources"):
                    part["sources"] = result.data["sources"][:6]
                if result.data.get("answer"):
                    part["answer"] = str(result.data["answer"])[:500]
                if result.data.get("exit_code") is not None:
                    part["exit_code"] = result.data["exit_code"]
                self.run.emit("tool.end", id=call_id, name=name, status=part["status"], summary=part["summary"],
                              duration_ms=duration, error=part.get("error"), output=ui_out or None,
                              sources=part.get("sources"), answer=part.get("answer"),
                              exit_code=part.get("exit_code"), truncated=result.truncated)
                contents[call_id] = result.content
                self._maybe_flush()
                emit_next += 1

        async def _run_one(index: int, call: Dict[str, str], part: Dict[str, Any], started: float) -> None:
            try:
                result = await _execute(call)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # a tool bug must never kill the run
                result = ToolResult.failure(f"{type(exc).__name__}: {exc}")
            # The execution ledger: what ran, with which arguments, and what came back. It is the only
            # evidence of work, so it is written before the model ever sees the result and is persisted
            # with the message — the model's later claims are checked against it, not trusted.
            try:
                preview = json.dumps(loads_args(call["args"]), ensure_ascii=False, sort_keys=True, default=str)
            except ValueError:
                preview = str(call.get("args") or "")
            self.ledger.append({
                "n": part["name"], "a": redact(preview)[:400], "ok": bool(result.ok),
                "s": "ok" if result.ok else "error",
                "e": redact(str(result.error or ""))[:300],
                "o": redact(str((result.data or {}).get("output") or ""))[:200] if result.ok else "",
            })
            # Only a write advances the epoch. A shell command that fails the same way five times is the
            # classic stuck loop, so exec does not refill the repeat budget on its own; re-running a
            # command after a *file change* is verification, and that stays allowed.
            spec = self.deps.registry.get(call["name"])
            if result.ok and spec is not None and spec.kind == "write":
                self._epoch += 1      # the world changed: re-running the same call is now meaningful
            err = str(result.error or "")
            if err.startswith("invalid arguments") or err.startswith("unknown tool"):
                self.malformed += 1
            elif err.startswith("duplicate call"):
                self.strikes += 1
            else:
                self.tool_calls_total += 1
            finished[index] = (part, result, max(1, int((time.monotonic() - started) * 1000)))
            _publish(index)

        tool_started = time.monotonic()
        try:
            await asyncio.gather(*(_run_one(i, c, p, t) for i, (c, p, t) in enumerate(pending)))
        finally:
            self.timings_ms["tools"] += int((time.monotonic() - tool_started) * 1000)
        # tool messages are appended in the model's call order
        for call, _part, _t in pending:
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": contents.get(call["id"], "ERROR: tool produced no result")})
        # Flush now: the ledger + parts must be durable before the model's next step, so a restart or a
        # crash resumes from results that really happened instead of an empty turn.
        await self._flush_now()
