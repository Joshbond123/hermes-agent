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
                 model: str = "", flush_interval: float = 1.5):
        self.run, self.deps = run, deps
        self.system, self.history, self.user_message = system, history, user_message
        self.model = model
        self.parts: List[Dict[str, Any]] = []
        self.tool_calls_total = 0
        self.strikes = 0
        self.malformed = 0
        self.seen: Dict[Tuple[str, str], int] = {}
        self.steps = 0
        self.usage = {"prompt": 0, "completion": 0}
        self.finish_reason: Optional[str] = None
        self.first_token_ms: Optional[int] = None
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
        self._schema_cost = prompts.schema_tokens(self._schemas)

    # ------------------------------------------------------------------ helpers
    def text(self) -> str:
        return "\n\n".join(p["text"].strip() for p in self.parts if p["type"] == "text" and p["text"].strip())

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
                                "usage": dict(self.usage), "status": status}
        if error:
            meta["error"] = error
        return meta

    async def _persist(self, status: str, error: Optional[Dict[str, Any]] = None) -> None:
        await self.deps.store.update_assistant(
            self.run.assistant_uid, text=self.text(), parts=self.parts, status=status, meta=self._meta(status, error),
            tokens=self.usage["completion"] or None)

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

    async def execute(self) -> None:
        status, error = "stop", None
        self._activity("run_started")
        try:
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
            for part in self.parts:
                if part["type"] == "tool" and part.get("status") == "running":
                    part["status"] = "cancelled" if status in ("cancelled", "interrupted") else "error"
            if error:
                self.run.emit("error", **error)
            if status == "length":
                self.run.emit("notice", level="warn", text="The answer was cut off by the model's context/token limit.")
            saved = True
            try:
                if self._flush_task and not self._flush_task.done():
                    await asyncio.shield(self._flush_task)
                await asyncio.shield(self._persist(status, error))
                await asyncio.shield(self.deps.store.finish_session_activity(self.run.session_id, tool_calls=self.tool_calls_total))
            except Exception as exc:  # noqa: BLE001
                saved = False
                log.error("final save failed: %s", exc)
                self.run.emit("notice", level="error", text="This response could not be saved to history: " + str(exc)[:160])
            self._activity("run_finished")
            self.run.emit("run.end", status=status, finish_reason=self.finish_reason, duration_ms=self._elapsed_ms(),
                          first_token_ms=self.first_token_ms, message_id=self.run.assistant_uid, session_id=self.run.session_id,
                          steps=self.steps, tool_calls=self.tool_calls_total, usage=dict(self.usage), saved=saved)
            self.run.finish(status)

    # ------------------------------------------------------------------ the loop
    async def _loop(self) -> None:
        s = self.deps.settings
        cpt = s.chars_per_token
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system}]
        used = prompts.estimate_tokens(self.system, cpt) + prompts.estimate_tokens(self.user_message, cpt) + 24
        history = prompts.fit_history(self.history, self._budget - self._schema_cost - used, cpt)
        messages += history
        self._history_len = len(history)
        messages.append({"role": "user", "content": self.user_message})

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
                messages.append({"role": "user", "content": FINALIZE_NOTE})
            result = await self._model_step(messages, None if force_final else self._schemas)
            if result.calls and not force_final:
                await self._run_tools(result, messages)
                continue
            if not result.text.strip() and empty_retries < 1:
                # The model produced nothing visible (an intermittent quirk: reasoning, then stop). Nothing was shown or
                # saved for this step, so asking once more is invisible apart from the extra wait.
                empty_retries += 1
                self.run.emit("notice", level="info", text="The model returned an empty answer; asking once more.")
                continue
            if result.finish == "length" and length_continues < 2:
                # Generation hit the output ceiling mid-answer. Keep the partial and resume: a long
                # task must not die because the reply itself ran out of room.
                length_continues += 1
                self.run.emit("notice", level="info",
                              text="The reply hit the model's token ceiling mid-answer; continuing where it stopped.")
                if result.text.strip():
                    messages.append({"role": "assistant", "content": result.text})
                messages.append({"role": "user", "content": LENGTH_CONTINUE_NOTE})
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
        while True:
            attempt += 1
            self._check()
            budget = self._budget - (self._schema_cost if tools else 0)
            if not prompts.shrink_tool_messages(messages, budget, s.chars_per_token):
                # Last resort: keep system + latest user only and continue (do not force a new chat).
                if len(messages) > 2:
                    messages[:] = [messages[0], messages[-1]]
                    self.run.emit(
                        "notice",
                        level="info",
                        text="Earlier turns were compacted so this conversation can continue in the model window.",
                    )
                    if not prompts.shrink_tool_messages(messages, budget, s.chars_per_token):
                        raise RunFailure(
                            "context_full",
                            "This single message is too large for the model window even after compaction. "
                            "Shorten the message or clear attachments and try again.",
                            True,
                        )
                else:
                    raise RunFailure(
                        "context_full",
                        "This single message is too large for the model window even after compaction. "
                        "Shorten the message or clear attachments and try again.",
                        True,
                    )
            route = await self.deps.resolver.get()
            self.model = route.model
            self._produced = False

            def _text_len() -> int:
                return sum(len(part["text"]) for part in self.parts if part["type"] == "text")

            before = _text_len()
            try:
                return await self._consume(route, messages, tools)
            except ContextOverflow:
                # The server's window is smaller than our estimate. 1st: drop prior history. 2nd: clip tool output hard.
                if attempt >= 3:
                    raise
                if attempt == 1 and self._history_len:
                    del messages[1: 1 + self._history_len]
                    self._history_len = 0
                else:
                    for m in messages:
                        if m.get("role") == "tool" and len(m["content"]) > 400:
                            m["content"] = m["content"][:400] + "\n[… clipped to fit the context window …]"
                self.run.emit("notice", level="info", text="The conversation was too long for the model window; trimmed it and retried.")
                continue
            except LLMError as exc:
                # Transport faults (edge timeout 524, dropped tunnel, stalled stream) are not
                # the model failing: recover instead of killing a long run mid-task. A dead GPU
                # still fails fast (retryable=False on auth/protocol errors).
                transient = (exc.retryable or
                             exc.code in ("gpu_unreachable", "gpu_dropped", "gpu_stalled") or
                             (exc.status is not None and exc.status in (502, 503, 521, 522, 523, 524)))
                limit = 4 if (exc.status == 524 or exc.code == "gpu_dropped") else 2
                if not transient or attempt >= limit:
                    raise
                self.deps.resolver.invalidate()
                partial = "".join(part["text"] for part in self.parts if part["type"] == "text")[before:]
                if partial.strip():
                    # Text already reached the user; keep it and let the model continue from the
                    # cut point instead of failing the run over one dropped packet.
                    if self._recoveries >= 3:
                        raise
                    self._recoveries += 1
                    self.run.emit("notice", level="info",
                                  text="The link to the GPU dropped mid-answer; continuing from where it stopped.")
                    messages.append({"role": "assistant", "content": partial})
                    messages.append({"role": "user", "content": CONTINUE_NOTE})
                else:
                    self.run.emit("notice", level="info", text="GPU connection hiccup — retrying.")
                await asyncio.sleep(min(4.0, 0.8 * attempt))
                continue

    async def _consume(self, route: Any, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> StepResult:
        s = self.deps.settings
        flt = ThinkFilter()
        thinking_since: Optional[float] = None
        acc: Dict[int, Dict[str, str]] = {}
        text_chunks: List[str] = []
        finish: Optional[str] = None
        lead_trim = True

        def start_thinking() -> None:
            nonlocal thinking_since
            if thinking_since is None:
                thinking_since = time.monotonic()
                self.run.emit("thinking", state="start")

        def end_thinking() -> None:
            nonlocal thinking_since
            if thinking_since is not None:
                self.run.emit("thinking", state="end", ms=int((time.monotonic() - thinking_since) * 1000))
                thinking_since = None

        def visible(text: str) -> None:
            nonlocal lead_trim
            if lead_trim:
                text = text.lstrip()
                if not text:
                    return
                lead_trim = False
            end_thinking()
            text_chunks.append(text)
            self._emit_text(text)

        async for ev in stream_chat(route, messages, client=self.deps.http, tools=tools, idle_timeout=s.llm_idle_timeout_s,
                                    connect_timeout=s.llm_connect_timeout_s):
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
                          tavily=self.deps.tavily, session_id=self.run.session_id, run_id=self.run.id)
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
            key = (name, json.dumps(args, sort_keys=True, default=str))
            async with seen_lock:
                count = self.seen.get(key, 0)
                if count >= s.max_repeat_calls:
                    return ToolResult.failure(
                        f"duplicate call suppressed: {name} was already run with these exact arguments {count} times",
                        hint="Use the earlier result in the conversation to answer, or try different arguments.")
                self.seen[key] = count + 1
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
            err = str(result.error or "")
            if err.startswith("invalid arguments") or err.startswith("unknown tool"):
                self.malformed += 1
            elif err.startswith("duplicate call"):
                self.strikes += 1
            else:
                self.tool_calls_total += 1
            finished[index] = (part, result, max(1, int((time.monotonic() - started) * 1000)))
            _publish(index)

        await asyncio.gather(*(_run_one(i, c, p, t) for i, (c, p, t) in enumerate(pending)))
        # tool messages are appended in the model's call order
        for call, _part, _t in pending:
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": contents.get(call["id"], "ERROR: tool produced no result")})
