"""System prompt composition and token budgeting for a small (4096-token) context window."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Deliberately short: the live model has a 4096-token window and the tool schemas already cost ~800 of it.
# There are NO keyword rules here — the model decides about tools itself through native tool calling.
BASE_PROMPT = (
    "You are Blackthorn, a capable AI assistant.\n"
    "Answer directly from your own knowledge whenever you can. Call a tool only when the answer needs live "
    "information or an action you cannot do from memory (current events, web pages, running code, files on the "
    "remote computer). Greetings, chit-chat and general knowledge never need a tool.\n"
    "The remote computer is a Linux machine (2×T4 GPU) with workspace /kaggle/working/blackthorn_workspace.\n"
    "Be honest: if a tool fails, say what failed; never claim you ran something you did not run, and never repeat "
    "a tool call that already returned a result.\n"
    "Reply in Markdown; put code in fenced blocks with a language tag. Keep answers concise."
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


def shrink_tool_messages(messages: List[Dict[str, Any]], budget_tokens: int, chars_per_token: float = 3.2) -> bool:
    """Make ``messages`` fit by replacing the OLDEST tool outputs with a stub, then clipping the newest.

    Returns True when the result fits. Mutates ``messages`` in place.
    """
    def total() -> int:
        return estimate_messages(messages, chars_per_token)

    if total() <= budget_tokens:
        return True
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    for i in tool_idx[:-1]:
        if messages[i]["content"] != STUB:
            messages[i]["content"] = STUB
            if total() <= budget_tokens:
                return True
    if tool_idx:
        last = messages[tool_idx[-1]]
        while total() > budget_tokens and len(last["content"]) > 400:
            last["content"] = last["content"][: int(len(last["content"]) * 0.7)] + "\n[… clipped to fit the context window …]"
    return total() <= budget_tokens


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
