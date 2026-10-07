"""The turn engine: one streamed assistant message per user message.

Design
------
* A *turn* runs in its own asyncio task, **detached from the HTTP request**.  A browser
  refresh, a sleeping phone or a proxy hiccup therefore never aborts a generation; the
  turn keeps running, checkpoints to D1, and any client can re-attach.
* Every event has a sequence number.  A client that re-attaches with ``after=<last seq>``
  receives exactly the events it missed — nothing is duplicated, nothing is lost.
* **Stop is real**: ``cancel`` cancels the in-flight model stream (closing the upstream
  connection, which makes the server abort generation) and any running tool, then saves
  what was produced.
* Tool use is the model's decision (native ``tools`` + ``tool_choice=auto``).  The engine
  only enforces safety: schema validation, a step budget, repeated-call and failure
  guards, and honest reporting when a budget is hit.
* Reasoning text is never forwarded, stored or logged; only the fact (and duration) of
  thinking is surfaced.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Protocol, Tuple

import httpx

from blackthorn.agent import prompts
from blackthorn.agent.model_client import (
    ContentDelta,
    Finish,
    ModelError,
    ReasoningDelta,
    ToolCallAssembler,
    ToolCallDelta,
    stream_chat,
)
from blackthorn.agent.redact import preview, redact
from blackthorn.agent.stream_filter import ContentFilter, parse_text_tool_calls
from blackthorn.agent.tools import (
    ToolContext,
    ToolRegistry,
    ToolResult,
    canonical_key,
    parse_arguments,
    run_tool,
    validate_arguments,
)
from blackthorn.route import Route, RouteProvider, TunnelDown

log = logging.getLogger("blackthorn.engine")

HEARTBEAT_S = 10.0
FINISHED_TTL_S = 900.0


# --------------------------------------------------------------------------- #
# events + turn
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Event:
    seq: int
    type: str
    data: Dict[str, Any]

    def sse(self) -> bytes:
        body = json.dumps(self.data, ensure_ascii=False, separators=(",", ":"))
        return f"id: {self.seq}\nevent: {self.type}\ndata: {body}\n\n".encode("utf-8")


class Turn:
    def __init__(self, session_id: str) -> None:
        self.id = f"turn-{uuid.uuid4().hex[:12]}"
        self.session_id = session_id
        self.events: List[Event] = []
        self.finished = False
        self.finished_at = 0.0
        self.cancel = asyncio.Event()
        self.task: Optional[asyncio.Task] = None
        self.assistant_message_id: Optional[int] = None
        self._waiter = asyncio.Event()

    def emit(self, type_: str, **data: Any) -> Event:
        ev = Event(len(self.events) + 1, type_, data)
        self.events.append(ev)
        waiter, self._waiter = self._waiter, asyncio.Event()
        waiter.set()
        return ev

    def mark_finished(self) -> None:
        if not self.finished:
            self.finished = True
            self.finished_at = time.monotonic()
        waiter, self._waiter = self._waiter, asyncio.Event()
        waiter.set()

    async def subscribe(self, after: int = 0) -> AsyncIterator[Optional[Event]]:
        """Yield buffered events with ``seq > after`` then live ones.

        ``None`` is yielded as a heartbeat when nothing happened for a while.
        Ends after the ``done`` event.
        """
        idx = max(0, int(after))
        while True:
            waiter = self._waiter  # capture first: an emit after this point sets *this* waiter
            if idx < len(self.events):
                ev = self.events[idx]
                idx += 1
                yield ev
                if ev.type == "done":
                    return
                continue
            if self.finished:
                return
            try:
                await asyncio.wait_for(waiter.wait(), timeout=HEARTBEAT_S)
            except asyncio.TimeoutError:
                yield None


class TurnCancelled(Exception):
    pass


# --------------------------------------------------------------------------- #
# persistence interface (D1 in production, in-memory in tests)
# --------------------------------------------------------------------------- #
@dataclass
class Context:
    history: List[Dict[str, str]]
    custom_prompt: str
    memories: List[str]


class Store(Protocol):
    async def begin_turn(self, session_id: str, user_text: str, *, model: str, started_at: float,
                         create_user_message: bool, user_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]: ...

    async def load_context(self, session_id: str, *, before_ts: Optional[float],
                           before_id: Optional[int]) -> Context: ...

    async def save_assistant(self, message_id: int, session_id: str, *, content: str, finish_reason: str,
                             metadata: Dict[str, Any], token_count: Optional[int]) -> None: ...

    async def last_user_message(self, session_id: str) -> Optional[Dict[str, Any]]: ...

    async def drop_after(self, session_id: str, message_id: int) -> int: ...


# --------------------------------------------------------------------------- #
@dataclass
class TurnRequest:
    session_id: Optional[str] = None
    message: str = ""
    regenerate: bool = False
    thinking: bool = False
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    max_steps: Optional[int] = None
    temperature: Optional[float] = None


@dataclass
class EngineConfig:
    max_steps: int = 16
    max_tool_calls: int = 30
    max_calls_per_step: int = 4
    repeat_limit: int = 2          # identical (tool, args) executions allowed per turn
    strike_limit: int = 3          # consecutive failed/invalid calls before forcing a final answer
    turn_timeout_s: float = 1200.0
    idle_timeout_s: float = 180.0
    history_limit_chars: int = 60_000
    checkpoint_interval_s: float = 6.0
    step_max_tokens: int = 6144
    tool_output_ui_chars: int = 1500


@dataclass
class _State:
    """Mutable record of what the turn has produced so far (survives cancellation)."""

    parts: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    steps: int = 0
    first_token_ms: Optional[int] = None
    thinking_ms: int = 0
    reasoning_started: Optional[float] = None
    dirty: bool = False
    finish_reason: str = "stop"
    error: Optional[Dict[str, Any]] = None
    started: float = field(default_factory=time.monotonic)

    def text(self) -> str:
        return "\n\n".join(p["text"].strip() for p in self.parts if p.get("t") == "text" and p.get("text", "").strip())

    def metadata(self, model: str) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "parts": [dict(p) for p in self.parts],
            "usage": dict(self.usage),
            "duration_ms": int((time.monotonic() - self.started) * 1000),
            "thinking_ms": self.thinking_ms,
            "steps": self.steps,
            "model": model,
        }
        if self.error:
            meta["error"] = self.error
        return meta


@dataclass
class _Step:
    text: str
    calls: List[Tuple[str, str, str]]  # (id, name, raw arguments)
    finish: str


def _friendly_error(exc: BaseException) -> Dict[str, Any]:
    if isinstance(exc, TunnelDown):
        return {
            "code": "gpu_unreachable",
            "message": "The GPU endpoint is unreachable — the Kaggle session may have stopped. "
                       "Check the GPU status, start it again if needed, then retry.",
            "retryable": True,
            "action": "check_gpu",
        }
    if isinstance(exc, ModelError):
        return {"code": "model_error", "message": str(exc), "retryable": bool(exc.retryable), "action": "retry"}
    return {"code": "internal", "message": f"Unexpected error ({type(exc).__name__}). Please retry.",
            "retryable": True, "action": "retry"}


def display_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Arguments as shown (collapsed) in the UI: redacted, long strings clipped."""
    shown: Dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str):
            limit = 1500 if key == "content" else 2000
            clipped = preview(value, limit)
            if len(value) > limit:
                clipped += f"  [{len(value)} characters total]"
            shown[str(key)] = clipped
        elif isinstance(value, (int, float, bool)) or value is None:
            shown[str(key)] = value
        else:
            shown[str(key)] = preview(json.dumps(value, ensure_ascii=False, default=str), 400)
    return shown


class Engine:
    def __init__(
        self,
        *,
        route_provider: RouteProvider,
        registry: ToolRegistry,
        store: Store,
        client_factory: Callable[[], httpx.AsyncClient],
        attachment_loader: Optional[Callable[[List[Dict[str, Any]]], Awaitable[List[Dict[str, Any]]]]] = None,
        on_activity: Optional[Callable[[bool], None]] = None,
        on_tunnel_failure: Optional[Callable[[str], Any]] = None,
        config: Optional[EngineConfig] = None,
        new_session_id: Optional[Callable[[], str]] = None,
    ) -> None:
        self.route_provider = route_provider
        self.registry = registry
        self.store = store
        self.client_factory = client_factory
        self.attachment_loader = attachment_loader
        self.on_activity = on_activity or (lambda busy: None)
        self.on_tunnel_failure = on_tunnel_failure
        self.cfg = config or EngineConfig()
        self.turns: Dict[str, Turn] = {}
        self.by_session: Dict[str, str] = {}
        self._new_session_id = new_session_id or (lambda: f"studio-{uuid.uuid4().hex[:12]}")

    # -- public ------------------------------------------------------------
    def get(self, turn_id: str) -> Optional[Turn]:
        return self.turns.get(turn_id)

    def active_turn(self, session_id: str) -> Optional[Turn]:
        tid = self.by_session.get(session_id)
        turn = self.turns.get(tid) if tid else None
        return turn if turn is not None and not turn.finished else None

    async def start(self, req: TurnRequest) -> Turn:
        session_id = (req.session_id or "").strip() or self._new_session_id()
        previous = self.active_turn(session_id)
        if previous is not None:  # one live turn per conversation
            previous.cancel.set()
            if previous.task is not None:
                with contextlib.suppress(asyncio.TimeoutError, Exception):
                    await asyncio.wait_for(asyncio.shield(previous.task), timeout=8.0)
        turn = Turn(session_id)
        self.turns[turn.id] = turn
        self.by_session[session_id] = turn.id
        turn.task = asyncio.create_task(self._run(turn, req), name=turn.id)
        self._gc()
        return turn

    async def cancel(self, turn_id: str) -> bool:
        turn = self.turns.get(turn_id)
        if turn is None or turn.finished:
            return False
        turn.cancel.set()
        return True

    async def shutdown(self) -> None:
        pending = [t.task for t in self.turns.values() if t.task is not None and not t.task.done()]
        for t in self.turns.values():
            t.cancel.set()
        if pending:
            await asyncio.wait(pending, timeout=6.0)

    def _gc(self) -> None:
        cutoff = time.monotonic() - FINISHED_TTL_S
        for tid in [tid for tid, t in self.turns.items() if t.finished and t.finished_at < cutoff]:
            turn = self.turns.pop(tid)
            if self.by_session.get(turn.session_id) == tid:
                self.by_session.pop(turn.session_id, None)

    # -- internals -----------------------------------------------------------
    async def _cancellable(self, turn: Turn, coro: Awaitable[Any]) -> Any:
        task = asyncio.ensure_future(coro)
        waiter = asyncio.ensure_future(turn.cancel.wait())
        try:
            done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            task.cancel()
            waiter.cancel()
            raise
        if task in done:
            waiter.cancel()
            return task.result()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        raise TurnCancelled()

    def _end_reasoning(self, turn: Turn, st: _State) -> None:
        if st.reasoning_started is not None:
            ms = int((time.monotonic() - st.reasoning_started) * 1000)
            st.thinking_ms += ms
            st.reasoning_started = None
            turn.emit("reasoning", active=False, duration_ms=ms)

    def _emit_text(self, turn: Turn, st: _State, text: str) -> None:
        if not text:
            return
        if st.first_token_ms is None:
            st.first_token_ms = int((time.monotonic() - st.started) * 1000)
        if not st.parts or st.parts[-1].get("t") != "text":
            st.parts.append({"t": "text", "text": ""})
        st.parts[-1]["text"] += text
        st.dirty = True
        turn.emit("delta", text=text)

    async def _model_step(self, turn: Turn, st: _State, route: Route, body: Dict[str, Any]) -> _Step:
        filt = ContentFilter()
        asm = ToolCallAssembler()
        finish = ""
        step_text: List[str] = []

        def emit_visible(chunk: str) -> None:
            if chunk:
                step_text.append(chunk)
                self._emit_text(turn, st, chunk)

        async for ev in stream_chat(self.client_factory(), route, body, idle_timeout=self.cfg.idle_timeout_s):
            if isinstance(ev, ReasoningDelta):
                if st.reasoning_started is None:
                    st.reasoning_started = time.monotonic()
                    turn.emit("reasoning", active=True)
            elif isinstance(ev, ContentDelta):
                self._end_reasoning(turn, st)
                emit_visible(filt.feed(ev.text))
            elif isinstance(ev, ToolCallDelta):
                self._end_reasoning(turn, st)
                asm.add(ev)
            elif isinstance(ev, Finish):
                if ev.reason:
                    finish = ev.reason
                if ev.usage:
                    st.usage["prompt_tokens"] += int(ev.usage.get("prompt_tokens") or 0)
                    st.usage["completion_tokens"] += int(ev.usage.get("completion_tokens") or 0)
        emit_visible(filt.flush())
        self._end_reasoning(turn, st)
        calls: List[Tuple[str, str, str]] = [(c.id, c.name, c.arguments) for c in asm.calls()]
        if not calls and filt.tool_blocks:  # text-form tool call the server did not convert
            for i, c in enumerate(parse_text_tool_calls(filt.tool_blocks)):
                calls.append((f"call_{uuid.uuid4().hex[:10]}", c["name"], c["arguments"]))
        return _Step("".join(step_text), calls, finish or ("tool_calls" if calls else "stop"))

    async def _step_with_retry(self, turn: Turn, st: _State, route_box: List[Route], body: Dict[str, Any]) -> _Step:
        """Run one model step; retry once if it failed before producing any visible output."""
        for attempt in (1, 2):
            first_event = len(turn.events)
            try:
                return await self._model_step(turn, st, route_box[0], body)
            except (TunnelDown, ModelError) as exc:
                produced = any(e.type in ("delta", "tool_start") for e in turn.events[first_event:])
                retryable = isinstance(exc, TunnelDown) or getattr(exc, "retryable", False)
                if attempt == 2 or produced or not retryable:
                    raise
                if isinstance(exc, TunnelDown) and self.on_tunnel_failure is not None:
                    with contextlib.suppress(Exception):
                        res = self.on_tunnel_failure(str(exc))
                        if asyncio.iscoroutine(res):
                            await res
                fresh = await self.route_provider()
                if isinstance(exc, TunnelDown) and (fresh is None or fresh.url == route_box[0].url):
                    raise  # nothing better to retry against
                if fresh is not None:
                    route_box[0] = fresh
                turn.emit("notice", level="warn", text="Connection to the GPU hiccupped — retrying once.")
                await asyncio.sleep(1.0)
        raise AssertionError("unreachable")

    async def _execute_call(
        self, turn: Turn, st: _State, call: Tuple[str, str, str], seen: Counter, ctx: ToolContext
    ) -> Tuple[str, bool, Dict[str, Any]]:
        """Run one requested tool call. Returns (model-visible text, ok, normalized args)."""
        call_id, name, raw_args = call
        tool = self.registry.get(name)
        label = tool.label if tool else name
        args, parse_err = parse_arguments(raw_args)
        shown_args = display_args(args if isinstance(args, dict) else {"arguments": str(raw_args)})
        part: Dict[str, Any] = {"t": "tool", "id": call_id, "name": name, "label": label,
                                "status": "running", "args": shown_args}
        st.parts.append(part)
        st.dirty = True
        turn.emit("tool_start", id=call_id, name=name, label=label, args=shown_args)
        started = time.monotonic()
        result: ToolResult
        clean: Dict[str, Any] = {}
        if tool is None:
            result = ToolResult(False, f"error: there is no tool named '{name}'. Available tools: "
                                       f"{', '.join(self.registry.names())}")
        elif parse_err:
            result = ToolResult(False, f"error: {parse_err}. Send the arguments as a JSON object.")
        else:
            clean, errors = validate_arguments(tool.parameters, args or {})
            if errors:
                result = ToolResult(False, "error: invalid arguments — " + "; ".join(errors))
            else:
                key = canonical_key(name, clean)
                if seen[key] >= self.cfg.repeat_limit:
                    result = ToolResult(
                        False,
                        f"error: this exact call was already made {seen[key]} times in this conversation turn. "
                        "Use the earlier result, change the arguments, or finish with what you have.",
                    )
                else:
                    seen[key] += 1
                    result = await self._cancellable(turn, run_tool(tool, clean, ctx))
        text = redact(result.text)
        ui_text = preview(text, self.cfg.tool_output_ui_chars)
        duration = int((time.monotonic() - started) * 1000)
        part.update({"status": "ok" if result.ok else "error", "duration_ms": duration, "output": ui_text})
        if result.meta.get("exit_code") is not None:
            part["exit_code"] = result.meta["exit_code"]
        st.dirty = True
        turn.emit("tool_result", id=call_id, ok=result.ok, duration_ms=duration, output=ui_text,
                  truncated=bool(result.meta.get("truncated")), exit_code=result.meta.get("exit_code"))
        return text, result.ok, clean if clean else (args if isinstance(args, dict) else {})

    # -- the run ---------------------------------------------------------------
    async def _run(self, turn: Turn, req: TurnRequest) -> None:
        st = _State()
        model_name = ""
        self.on_activity(True)
        try:
            turn.emit("turn", state="accepted", turn_id=turn.id, session_id=turn.session_id)
            try:
                model_name = await self._prepare_and_run(turn, req, st)
            except TurnCancelled:
                st.finish_reason = "cancelled"
            except asyncio.TimeoutError:
                st.finish_reason = "timeout"
                st.error = {"code": "timeout", "message": "The turn exceeded the time limit and was stopped.",
                            "retryable": True, "action": "retry"}
                turn.emit("error", **st.error)
            except asyncio.CancelledError:
                st.finish_reason = "interrupted"
                raise
            except (TunnelDown, ModelError) as exc:
                st.finish_reason = "error"
                st.error = _friendly_error(exc)
                turn.emit("error", **st.error)
            except Exception as exc:  # noqa: BLE001
                log.exception("turn %s failed", turn.id)
                st.finish_reason = "error"
                st.error = _friendly_error(exc)
                turn.emit("error", **st.error)
        finally:
            await asyncio.shield(self._finalize(turn, st, model_name))
            self.on_activity(False)
            turn.mark_finished()

    async def _prepare_and_run(self, turn: Turn, req: TurnRequest, st: _State) -> str:
        cfg = self.cfg
        started_wall = time.time()
        session_id = turn.session_id
        route_task = asyncio.ensure_future(self.route_provider())
        attachments: List[Dict[str, Any]] = []
        if req.attachments and self.attachment_loader is not None:
            attachments = await self.attachment_loader(req.attachments)

        user_text = req.message.strip()
        before_id: Optional[int] = None
        create_user = True
        if req.regenerate:
            last = await self.store.last_user_message(session_id)
            if not last:
                raise ModelError("there is no message to regenerate")
            user_text = str(last.get("content") or "")
            before_id = int(last["id"])
            await self.store.drop_after(session_id, before_id)
            create_user = False
        route = await route_task
        model_name = route.model if route else ""
        user_meta = {"attachments": [a.get("name") or a.get("path") for a in attachments]} if attachments else None
        begin_task = asyncio.ensure_future(self.store.begin_turn(
            session_id, user_text, model=model_name, started_at=started_wall,
            create_user_message=create_user, user_meta=user_meta))
        ctx_task = asyncio.ensure_future(self.store.load_context(
            session_id, before_ts=None if before_id else started_wall, before_id=before_id))
        rows, ctx = await asyncio.gather(begin_task, ctx_task)
        turn.assistant_message_id = rows.get("assistant_message_id")
        turn.emit("turn", state="ready", turn_id=turn.id, session_id=session_id, model=model_name,
                  user_message_id=rows.get("user_message_id"), assistant_message_id=turn.assistant_message_id,
                  created=bool(rows.get("created")))
        if route is None:
            st.finish_reason = "error"
            st.error = {"code": "gpu_offline",
                        "message": "The GPU is off. Start it from the GPU control in the header, then retry.",
                        "retryable": True, "action": "start_gpu"}
            turn.emit("error", **st.error)
            return model_name

        ticker = asyncio.create_task(self._checkpointer(turn, st, model_name))
        try:
            await asyncio.wait_for(
                self._loop(turn, req, st, route, ctx, user_text, attachments), timeout=cfg.turn_timeout_s
            )
        finally:
            ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await ticker
        return model_name

    def _history_messages(self, ctx: Context) -> List[Dict[str, str]]:
        budget = self.cfg.history_limit_chars
        picked: List[Dict[str, str]] = []
        for row in ctx.history:  # newest first
            content = str(row.get("content") or "")
            if budget - len(content) < 0 and picked:
                break
            budget -= len(content)
            picked.append({"role": str(row.get("role")), "content": content})
        picked.reverse()
        return picked

    async def _loop(self, turn: Turn, req: TurnRequest, st: _State, route: Route, ctx: Context,
                    user_text: str, attachments: List[Dict[str, Any]]) -> None:
        cfg = self.cfg
        system = prompts.build_system_prompt(custom=ctx.custom_prompt, memories=ctx.memories)
        model_user = user_text
        extra = prompts.attachment_block(attachments)
        if extra:
            model_user = f"{user_text}\n\n{extra}"
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        messages += self._history_messages(ctx)
        messages.append({"role": "user", "content": model_user})
        tools = self.registry.schemas()
        max_steps = max(1, min(int(req.max_steps or cfg.max_steps), 24))
        route_box = [route]
        seen: Counter = Counter()
        tool_ctx = ToolContext(session_id=turn.session_id, cancel=turn.cancel)
        total_calls = 0
        strikes = 0
        force_final = False
        nudged_empty = False

        while True:
            st.steps += 1
            body: Dict[str, Any] = {
                "model": route_box[0].model,
                "messages": messages,
                "stream_options": {"include_usage": True},
                "max_tokens": cfg.step_max_tokens,
            }
            if not force_final and tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            if not req.thinking:
                body["reasoning_effort"] = "none"
            if req.temperature is not None:
                body["temperature"] = float(req.temperature)

            step = await self._cancellable(turn, self._step_with_retry(turn, st, route_box, body))

            if step.finish == "length":
                st.finish_reason = "length"
            if not step.calls:
                if step.text.strip():
                    return  # a normal final answer
                # the model said nothing at all in this step
                if not nudged_empty:
                    nudged_empty = True
                    messages.append({"role": "assistant", "content": ""})
                    messages.append({"role": "user", "content": "[system] Your last reply was empty. Reply to the user now."})
                    continue
                st.finish_reason = "incomplete"
                self._emit_text(turn, st, ("\n\n" if st.text() else "") +
                                "I could not produce a final answer. Please try again or rephrase the request.")
                return

            # --- the model asked for tools -------------------------------------------------
            if force_final:  # tools were withheld but text-form calls came anyway: stop here
                st.finish_reason = "incomplete"
                if not step.text.strip():
                    self._emit_text(turn, st, ("\n\n" if st.text() else "") +
                                    "I reached my tool budget before I could finish.")
                return
            assistant_calls = []
            results: List[Dict[str, Any]] = []
            for call_id, name, raw in step.calls[: cfg.max_calls_per_step]:
                total_calls += 1
                text, ok, clean = await self._execute_call(turn, st, (call_id, name, raw), seen, tool_ctx)
                strikes = 0 if ok else strikes + 1
                assistant_calls.append({
                    "id": call_id, "type": "function",
                    "function": {"name": name, "arguments": json.dumps(clean, ensure_ascii=False)},
                })
                results.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": text})
            messages.append({"role": "assistant", "content": step.text or "", "tool_calls": assistant_calls})
            messages.extend(results)

            if turn.cancel.is_set():
                raise TurnCancelled()
            exhausted = ""
            if strikes >= cfg.strike_limit:
                exhausted = "too many consecutive failed tool calls"
            elif total_calls >= cfg.max_tool_calls or st.steps >= max_steps:
                exhausted = "the tool-call budget for one reply was reached"
            if exhausted:
                force_final = True
                if st.finish_reason == "stop":
                    st.finish_reason = "incomplete"
                turn.emit("notice", level="warn", text=f"Stopping tool use: {exhausted}.")
                messages.append({"role": "user", "content": (
                    f"[system] Tool use is now disabled ({exhausted}). Do not call tools. Give your final "
                    "answer using what you have, and say clearly anything you could not complete.")})

    async def _checkpointer(self, turn: Turn, st: _State, model: str) -> None:
        while True:
            await asyncio.sleep(self.cfg.checkpoint_interval_s)
            if not st.dirty or turn.assistant_message_id is None:
                continue
            st.dirty = False
            with contextlib.suppress(Exception):
                await self.store.save_assistant(
                    turn.assistant_message_id, turn.session_id, content=st.text(),
                    finish_reason="running", metadata=st.metadata(model), token_count=None)

    async def _finalize(self, turn: Turn, st: _State, model: str) -> None:
        # close any tool part that never produced a result
        for part in st.parts:
            if part.get("t") == "tool" and part.get("status") == "running":
                part["status"] = "cancelled" if st.finish_reason in ("cancelled", "interrupted") else "error"
        self._end_reasoning(turn, st)
        if st.finish_reason == "stop" and st.error:
            st.finish_reason = "error"
        content = st.text()
        if turn.assistant_message_id is not None:
            try:
                await self.store.save_assistant(
                    turn.assistant_message_id, turn.session_id, content=content,
                    finish_reason=st.finish_reason, metadata=st.metadata(model),
                    token_count=st.usage.get("completion_tokens") or None)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not persist turn %s: %s", turn.id, exc)
                turn.emit("notice", level="warn", text="The answer could not be saved to history.")
        turn.mark_finished()  # before `done`: a cancel arriving once `done` is visible must be a no-op
        turn.emit(
            "done", finish_reason=st.finish_reason, usage=dict(st.usage), steps=st.steps,
            duration_ms=int((time.monotonic() - st.started) * 1000), first_token_ms=st.first_token_ms,
            thinking_ms=st.thinking_ms, message_id=turn.assistant_message_id, session_id=turn.session_id,
            error=st.error,
        )
