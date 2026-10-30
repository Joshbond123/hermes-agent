"""System prompt composition and token budgeting for the agent context window."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Keep this short so tool schemas + multi-step tool results still fit.
# There are NO keyword rules here — the model decides about tools itself through native tool calling.
BASE_PROMPT = (
    "You are Blackthorn, a capable AI assistant. Answer from your own knowledge when you can; call a tool when "
    "the answer needs live information or an action (web, code, files, the remote computer).\n"
    "The computer is a PandaStack Linux microVM; /workspace is a temp cache mirrored to the private GitHub "
    "Computer-drive, the durable store. Kaggle Qwen supplies the GPUs.\n"
    "Be honest: say what failed, never claim a run you did not make, never repeat a call that already returned. "
    "An earlier answer is a claim, not proof; check the computer before stating what it contains.\n"
    "Decide yourself what a task needs: tool_search finds an executable tool, skill_search a playbook, then "
    "skill_load/skill_read. Searching and loading cost nothing; guessing costs the task.\n"
    "Malware: sample_triage for statics; the dedicated lab for behaviour (lab_detonate, lab_exec for root-level "
    "work, lab_os for Linux/Windows readiness) then lab_close, because an open lab bills.\n"
    "Reply in Markdown, code fenced with a language tag, concise."
)


def system_prompt(user_prompt: str = "", memories: Sequence[str] = (), now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    parts = [BASE_PROMPT, f"Today is {now.strftime('%A, %d %B %Y')} (UTC)."]
    user_prompt = (user_prompt or "").strip()
    if user_prompt:
        parts.append("Additional instructions from the user:\n" + user_prompt[:1500])
    mem = [m.strip()[:160] for m in memories if m and m.strip()][:6]
    if mem:
        parts.append("Remembered about the user:\n" + "\n".join(f"- {m}" for m in mem))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- token estimates
def estimate_tokens(text: str, chars_per_token: float = 3.2) -> int:
    return int(math.ceil(len(text or "") / chars_per_token)) + 4


def estimate_messages(messages: Sequence[Dict[str, Any]], chars_per_token: float = 3.2) -> int:
    total = 0
    for m in messages:
        total += estimate_tokens(str(m.get("content") or ""), chars_per_token) + 3
        for tc in m.get("tool_calls") or []:
            total += estimate_tokens(json.dumps(tc.get("function", {})), chars_per_token)
    return total


def schema_tokens(schemas: Sequence[Dict[str, Any]]) -> int:
    """The chat template renders tool JSON verbosely: measured ≈ chars / 2.6 on the live model."""
    return int(math.ceil(len(json.dumps(schemas, separators=(",", ":"))) / 2.6)) + 60 if schemas else 0


def prompt_budget(context_tokens: int, completion_reserve: int) -> int:
    return max(512, context_tokens - completion_reserve)


STUB = "[earlier tool output removed to fit the model's context window]"


def fit_history(history: List[Dict[str, str]], budget_tokens: int, chars_per_token: float = 3.2) -> List[Dict[str, str]]:
    """Newest-first fill of the history budget; an over-long single message is clipped, never dropped silently."""
    kept: List[Dict[str, str]] = []
    used = 0
    for msg in reversed(history):
        text = msg.get("content") or ""
        if not text.strip():
            continue
        cost = estimate_tokens(text, chars_per_token) + 3
        if used + cost > budget_tokens:
            room = budget_tokens - used - 8
            if not kept and room > 80:  # keep at least the most recent turn, clipped
                clipped = text[-int(room * chars_per_token):]
                kept.append({"role": msg["role"], "content": "[…earlier part omitted…] " + clipped})
            break
        kept.append({"role": msg["role"], "content": text})
        used += cost
    kept.reverse()
    while kept and kept[0]["role"] != "user":  # a conversation must start with the user
        kept.pop(0)
    return kept


def shrink_tool_messages(messages: List[Dict[str, Any]], budget_tokens: int, chars_per_token: float = 3.2,
                         protect: Sequence[int] = ()) -> bool:
    """Make ``messages`` fit without forcing the user to start a new chat.

    Hermes-compatible strategy (mutates in place), applied cheapest-first and always on
    whole ``tool_calls``/``tool`` groups so a call is never separated from its results:

      1. Stub the oldest tool outputs (never the newest group's).
      2. Clip the newest tool output.
      3. Drop whole tool groups, oldest first (never the newest group, never a protected message).
      4. Drop whole non-tool turns, oldest first (system + protected messages are never dropped).
      5. Clip long assistant messages.
      6. Last resort: system + an explicit compaction notice + the protected tail.

    ``protect`` holds indices that must survive (the system prompt, the pinned opening request and
    the current real user message). Step 3/4 also re-pairs orphaned tool results
    (``_sanitize_tool_pairs`` in Hermes) so the request stays valid for the server.

    Returns True when the result fits, or when only system + the protected message remain
    (always keep the current turn runnable).
    """
    if not messages:
        return False
    # Pin actual message objects, not integer positions: every deletion shifts indices. The
    # original implementation pinned index N and then removed old turns before N, eventually
    # deleting the current user request and sending system+tools to Qwen's Jinja template.
    pinned = {id(messages[0])}
    pinned.update(id(messages[i]) for i in protect if isinstance(i, int) and 0 <= i < len(messages))
    def real_user(m: Dict[str, Any]) -> bool:
        text = str(m.get("content") or "").strip()
        return m.get("role") == "user" and not (text.startswith("<tool_response>") and text.endswith("</tool_response>"))
    users = [m for m in messages if real_user(m)]
    if not users:
        return False  # never send a request Qwen's actual template cannot render
    if not any(m.get("role") == "user" and id(m) in pinned for m in messages):
        pinned.add(id(users[-1]))
    _sanitize_pairs(messages)

    def total() -> int:
        return estimate_messages(messages, chars_per_token)

    def fits() -> bool:
        # Compaction can remove the assistant between two user turns. Repair alternation
        # *before* the fit check; otherwise Qwen receives an invalid wire history.
        alternate = _alternate(messages)
        if estimate_messages(alternate, chars_per_token) > budget_tokens:
            return False
        messages[:] = alternate
        return True

    if fits():
        return True

    groups = _tool_groups(messages)
    in_progress = {id(messages[i]) for i in groups[-1][1:]} if groups else set()
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    for i in tool_idx[:-1]:
        if id(messages[i]) not in in_progress:
            messages[i]["content"] = STUB
    if fits():
        return True

    if tool_idx:
        i = tool_idx[-1]
        content = str(messages[i].get("content") or "")
        while total() > budget_tokens and len(content) > 400:
            content = content[: int(len(content) * 0.7)] + "\n[… clipped to fit the context window …]"
            messages[i]["content"] = content
        if fits():
            return True

    # Old complete tool groups are expendable; keep the latest batch while feasible.
    while total() > budget_tokens:
        groups = _tool_groups(messages)
        if len(groups) < 2 or any(id(messages[i]) in pinned for i in groups[0]):
            break
        for i in sorted(groups[0], reverse=True):
            messages.pop(i)
        if fits():
            return True

    # Never delete a protected request, even after all earlier turns shift left.
    while total() > budget_tokens and len(messages) > 2:
        drop_at = next((i for i in range(1, len(messages))
                        if id(messages[i]) not in pinned and messages[i].get("role") in ("user", "assistant")
                        and not messages[i].get("tool_calls")), None)
        if drop_at is None:
            break
        messages.pop(drop_at)
        if fits():
            return True

    for m in messages:
        if m.get("role") == "assistant" and isinstance(m.get("content"), str) and len(m["content"]) > 800:
            while total() > budget_tokens and len(m["content"]) > 400:
                m["content"] = m["content"][: int(len(m["content"]) * 0.6)] + "\n[… clipped …]"
            if fits():
                return True

    # Last resort: keep both the opening request and the current real request where
    # feasible.
    protected = [m for m in messages[1:] if id(m) in pinned]
    minimal = _alternate([messages[0], *protected])
    if estimate_messages(minimal, chars_per_token) <= budget_tokens:
        messages[:] = minimal
        return True

    # The user's own request (or an attachment inside it) is larger than the whole window.
    # Refusing to run is what produced "This single message is too large for the model window
    # even after compaction" and left the user with nothing to do but start over. A single
    # message cannot be dropped — it *is* the task — so keep its head and tail, mark the gap
    # explicitly, and let the run proceed: the model can always ask for a specific part, and
    # the elided text is still in the stored turn for a later, smaller request.
    # Room is whatever the window actually has left after the system prompt - a fixed floor here
    # would silently declare "fits" while the request was still far too big.
    room = budget_tokens - estimate_messages([messages[0]], chars_per_token) - 32
    budget_chars = max(600, int(room * chars_per_token))
    oversized = sorted(((m for m in protected if len(str(m.get("content") or "")) > budget_chars)),
                       key=lambda m: -len(str(m.get("content") or "")))
    for m in oversized:
        text = str(m.get("content") or "")
        note = (ELISION_NOTE % (len(text), len(text) - budget_chars + 240))
        keep = max(240, budget_chars - len(note) - 40)
        head = int(keep * 0.7)
        tail = keep - head
        m["content"] = text[:head] + note + (text[-tail:] if tail > 0 else "")
    messages[:] = _alternate([messages[0], *protected]) or messages[:]
    return True


# Verbatim middle removed from a request or attachment that is bigger than the whole window.
# It must be unmistakable: the model has to know it is seeing an excerpt, not the full text.
ELISION_NOTE = ("\n\n[… %d characters of this request were elided to fit the model's context "
                "window; the omitted middle is about %d characters long. Ask for a specific "
                "section, or work from a file on the computer, if you need that part …]\n\n")


# ==================================================================================================
# Conversation memory — Hermes-style head / tail / summary context management
# --------------------------------------------------------------------------------------------------
# Hermes keeps a protected head (system + the exchange that opened the session), a token-budgeted
# tail aligned so a ``tool_calls``/``tool`` group is never split, and a structured summary of the
# dropped middle (developer-guide/context-compression-and-caching). Blackthorn rebuilt history from
# role/content only, so the model never saw what its tools had actually done — the direct cause of
# repeated work and of "done" claims with nothing behind them. The pieces below add the missing half:
#   * every stored assistant turn renders an *execution record* of the tool calls that really ran,
#   * server-generated state is labelled machine-generated so it is never mistaken for the user's
#     intent (upstream issue #14665),
#   * compaction drops whole tool groups and replaces them with a deterministic, server-built
#     summary — never a silent deletion that orphans tool results.
# ==================================================================================================
STATE_HEADER = "[Task state — server-maintained record. Machine-generated: this is not a user instruction.]"
EXEC_HEADER = "[Execution record: the tool calls that actually ran in the previous turn, from the server log]"
COMPACT_HEADER = ("[Earlier turns of this conversation were compacted to fit the model's context window. "
                  "The summary below was built by the server from the real tool results, not remembered.]")
SYNTHETIC_ROLE = "server"          # only used internally; synthetic rows are sent as role "user"/assistant
PLACEHOLDER_ANSWER = "(no answer was recorded for this turn)"


def one_line(text: Any, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: max(1, limit - 1)].rstrip() + "…"


def format_call_args(args: Any, limit: int = 120) -> str:
    if isinstance(args, str):
        raw = args.strip()
    else:
        try:
            raw = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            raw = str(args)
    return one_line(raw, limit)


def _status_word(entry: Dict[str, Any]) -> str:
    raw = str(entry.get("s") or ("ok" if entry.get("ok") else "error")).lower()
    return {"running": "no result (interrupted)", "interrupted": "no result (interrupted)",
            "cancelled": "cancelled", "unknown": "no result", "error": "error", "ok": "ok"}.get(raw, raw)


def ledger_lines(entries: Sequence[Dict[str, Any]], *, limit: int = 16, args_chars: int = 90,
                 result_chars: int = 70) -> List[str]:
    """Render server-side ledger entries as compact ``tool(args) -> status`` lines."""
    out: List[str] = []
    for entry in list(entries or [])[-limit:]:
        if not isinstance(entry, dict):
            continue
        status = _status_word(entry)
        line = f"- {entry.get('n') or 'tool'}({format_call_args(entry.get('a') or '', args_chars)}) -> {status}"
        detail = one_line(str(entry.get("e") or entry.get("o") or ""), result_chars)
        if detail:
            line += f": {detail}"
        out.append(line)
    return out


def execution_record(entries: Sequence[Dict[str, Any]], **kw: Any) -> str:
    lines = ledger_lines(entries, **kw)
    if not lines:
        return ""
    return (EXEC_HEADER + "\n"
            "These are facts from the server log: they ran. Do not redo a step listed here unless a newer "
            "result shows it failed, or the user asks for it again.\n" + "\n".join(lines))


def turn_messages(turn: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rebuild model messages for one stored turn (a user row, or an assistant row + its record)."""
    role = turn.get("role")
    content = str(turn.get("content") or "")
    if role != "assistant":
        if not content.strip():
            return []
        return [{"role": "user", "content": content}]

    status = str(turn.get("status") or "")
    blocks: List[str] = []
    if content.strip():
        blocks.append(content.strip())
    record = execution_record(turn.get("ledger") or [])
    if record:
        blocks.append(record)
    if status in ("interrupted", "error", "cancelled"):
        blocks.append(f"[The previous turn ended unfinished (status: {status}). It did not complete the task. "
                      "Continue from the work recorded above; do not start over and do not claim it finished.]")
    if not blocks:
        blocks.append(PLACEHOLDER_ANSWER if status != "streaming" else "")
    return [{"role": "assistant", "content": "\n\n".join(b for b in blocks if b)}]


def _clip_group(group: List[Dict[str, Any]], budget_tokens: int, chars_per_token: float) -> List[Dict[str, Any]]:
    """Clip the newest turn group so that at least something of it survives the budget."""
    room = max(120, int(budget_tokens * 0.9))
    out: List[Dict[str, Any]] = []
    for msg in group:
        text = str(msg.get("content") or "")
        limit = int(room * chars_per_token)
        if len(text) > limit:
            msg = dict(msg)
            msg["content"] = "[…earlier part omitted…] " + text[-limit:]
        out.append(msg)
        room -= estimate_tokens(str(msg.get("content") or ""), chars_per_token)
    return out


def _alternate(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Hermes enforces strict alternation: never two user or two assistant messages in a row."""
    out: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role in ("user", "assistant") and out and out[-1].get("role") == role:
            filler = {"role": "assistant", "content": PLACEHOLDER_ANSWER} if role == "user" else {"role": "user", "content": "(continued)"}
            out.append(filler)
        out.append(msg)
    return out


def _tool_groups(messages: List[Dict[str, Any]]) -> List[List[int]]:
    """Indices of each ``assistant(tool_calls)`` + following ``tool`` result group."""
    groups: List[List[int]] = []
    i, n = 1, len(messages)          # index 0 is the system prompt and is never part of a group
    while i < n:
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            group = [i]
            j = i + 1
            while j < n and messages[j].get("role") == "tool":
                group.append(j)
                j += 1
            groups.append(group)
            i = j
        else:
            i += 1
    return groups


def _sanitize_pairs(messages: List[Dict[str, Any]]) -> None:
    """Keep *one* immediate tool result per call, inserting a stub only if absent.

    Previously the scan looked only at results already emitted, so it inserted a stub for
    EVERY call before visiting the real results. Four parallel calls became eight results,
    doubling prefill and sometimes pushing a valid prompt over the context limit.
    """
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "tool":  # orphan (or a duplicate after this call's group)
            i += 1
            continue
        out.append(msg)
        i += 1
        calls = msg.get("tool_calls") or []
        if not calls:
            continue
        found: Dict[str, Dict[str, Any]] = {}
        while i < len(messages) and messages[i].get("role") == "tool":
            result = messages[i]
            cid = result.get("tool_call_id")
            if cid and cid not in found:
                found[cid] = result
            i += 1
        for call in calls:
            cid = (call or {}).get("id")
            if cid:
                out.append(found.get(cid) or {"role": "tool", "tool_call_id": cid, "content": STUB})
    messages[:] = out


def compact_summary(turns: Sequence[Dict[str, Any]], *, requests: int = 6, done: int = 14, failed: int = 8) -> str:
    """Deterministic structured summary of dropped turns (Hermes' deterministic fallback summary).

    Built only from stored rows and the server's own execution ledger, so it can never invent a step.
    """
    asked: List[str] = []
    completed: List[str] = []
    broken: List[str] = []
    quoted: List[str] = []
    for turn in turns:
        role = turn.get("role")
        content = one_line(turn.get("content") or "", 150)
        if role == "user":
            if content:
                asked.append(content)
                quoted.append(f"User: {content}")
            continue
        entries = [e for e in (turn.get("ledger") or []) if isinstance(e, dict)]
        for entry in entries:
            line = f"{entry.get('n') or 'tool'}({format_call_args(entry.get('a') or '', 60)})"
            if entry.get("ok"):
                completed.append(line)
            else:
                broken.append(f"{line} -> {_status_word(entry)}: {one_line(str(entry.get('e') or ''), 70)}")
        if content:
            quoted.append(f"Blackthorn: {content}")
    lines = [COMPACT_HEADER]
    if asked:
        # The opening request is the session's goal: it is kept even when the list is trimmed.
        kept: List[str] = []
        for a in asked[:2] + asked[-(max(1, requests - 2)):]:
            if a not in kept:
                kept.append(a)
        lines.append("Earlier requests in this conversation (oldest first):")
        lines += [f"- {a}" for a in kept]
    if completed:
        lines.append("Steps that completed (server-verified tool results):")
        lines += [f"- {c}" for c in completed[-done:]]
    if broken:
        lines.append("Steps that failed or never returned a result:")
        lines += [f"- {b}" for b in broken[-failed:]]
    if not completed and not broken:
        lines.append("No tool ran in the dropped turns. Verbatim excerpts (newest last):")
        lines += [f"- {q}" for q in quoted[-8:]]
    return "\n".join(lines)


def compile_history(turns: Sequence[Dict[str, Any]], budget_tokens: int, chars_per_token: float = 3.2,
                    ) -> Tuple[List[Dict[str, Any]], int, Optional[str]]:
    """Rebuild the model-visible history from stored turns.

    Head: the session's first real request (pinned, so the original instruction survives compaction).
    Tail: whole turn groups, newest-first, until the budget is spent.
    Middle: replaced by :func:`compact_summary` when anything is dropped.

    Returns ``(messages, dropped_turn_count, summary_or_None)``.
    """
    budget = max(200, int(budget_tokens))
    groups: List[Tuple[int, List[Dict[str, Any]]]] = []
    for turn_index, turn in enumerate(turns):
        group = turn_messages(turn)
        if group:
            groups.append((turn_index, group))
    if not groups:
        return [], 0, None
    flat = [m for _i, g in groups for m in g]
    if estimate_messages(flat, chars_per_token) <= budget:
        return _alternate(flat), 0, None

    # ---- tail: newest whole turn groups first
    tail: List[Dict[str, Any]] = []
    used = 0
    start = len(groups)
    for idx in range(len(groups) - 1, -1, -1):
        group = groups[idx][1]
        cost = estimate_messages(group, chars_per_token)
        if tail and used + cost > budget:
            break
        if not tail and cost > budget:
            group = _clip_group(group, budget, chars_per_token)
            cost = estimate_messages(group, chars_per_token)
        tail = group + tail
        used += cost
        start = idx
        if used >= budget:
            break

    # ---- head: pin the first real user request of the session
    head: List[Dict[str, Any]] = []
    head_turn_index = next((i for i, (_ti, g) in enumerate(groups) if g[0].get("role") == "user"), None)
    if head_turn_index is not None and head_turn_index < start:
        first = dict(groups[head_turn_index][1][0])
        first["content"] = one_line(str(first.get("content") or ""), 500)
        head = [first]
        used += estimate_messages(head, chars_per_token)

    # The tail keeps ``len(groups) - start`` turn groups; everything older is summarised.
    dropped_turns = list(turns)[:groups[start][0]] if start < len(groups) else []

    summary = compact_summary(dropped_turns) if dropped_turns else None
    messages = head
    if summary:
        messages = messages + [{"role": "user", "content": summary}]
    messages = messages + tail
    return _alternate(messages), len(dropped_turns), summary


def task_state_block(*, goal: str = "", task_goal: str = "", todos: Sequence[Dict[str, Any]] = (),
                     ledger: Sequence[Dict[str, Any]] = (),
                     prior_steps: Sequence[Dict[str, Any]] = (), previous_attempt: Optional[Dict[str, Any]] = None,
                     resume_hint: str = "") -> str:
    """Ephemeral, machine-generated state layer (Hermes injects these per API call, never persists them).

    It rides in the system message so it can never be mistaken for the newest user instruction
    (upstream issue #14665) and so the prompt prefix stays stable while the state is unchanged.
    """
    blocks: List[str] = []
    if goal.strip():
        blocks.append(f"Current user request: {one_line(goal, 260)}")
    if task_goal.strip() and task_goal.strip() != goal.strip():
        blocks.append("Original active task (saved with the plan; newer user instructions take precedence): "
                      + one_line(task_goal, 1400))
    if todos:
        marks = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]", "cancelled": "[~]"}
        lines = []
        for item in list(todos)[:12]:
            status = str(item.get("status") or "pending").lower()
            lines.append(f"  {marks.get(status, '[ ]')} {item.get('id') or '?'}: {one_line(item.get('content') or '', 80)}")
        blocks.append("Tasks:\n" + "\n".join(lines))
    lines = ledger_lines(ledger, limit=10, args_chars=70, result_chars=50)
    if lines:
        blocks.append("Done this turn (tool results):\n" + "\n".join(lines))
    prior = ledger_lines(prior_steps, limit=8, args_chars=70, result_chars=50)
    if prior:
        blocks.append("Already done earlier in this session (do not redo):\n" + "\n".join(prior))
    if previous_attempt:
        entries = previous_attempt.get("ledger") or []
        plines = ledger_lines(entries, limit=10, args_chars=70, result_chars=50)
        if plines:
            blocks.append("The discarded previous answer had already run (do not repeat):\n" + "\n".join(plines))
    if resume_hint.strip():
        blocks.append(resume_hint.strip())
    if not blocks:
        return ""
    footer = ("Only a step listed here actually happened: never claim a command, edit or check was done unless "
              "it appears in this record. If a step failed, fix it or say plainly that it failed."
              if (todos or ledger or prior_steps or previous_attempt) else
              "Only state what tool results show.")
    return STATE_HEADER + "\n" + "\n".join(blocks) + "\n" + footer


# --------------------------------------------------------------------------- attachments
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    base = (name or "file").replace("\\", "/").split("/")[-1].strip() or "file"
    base = _SAFE.sub("_", base)[:80].strip("._") or "file"
    return base


def user_message_with_attachments(text: str, attachments: Sequence[Dict[str, Any]]) -> str:
    """Model-facing user message: the text plus a short note + preview per attachment."""
    if not attachments:
        return text
    blocks: List[str] = []
    for a in attachments:
        where = a.get("workspace_path")
        head = f"{a['name']} ({a['chars']} characters" + (f", saved at {where}" if where else ", not saved to the workspace") + ")"
        preview = a.get("preview") or ""
        blocks.append(f"Attached file: {head}\n```\n{preview}\n```" + ("\n[preview truncated]" if a.get("truncated") else ""))
    return text + "\n\n" + "\n\n".join(blocks)
