"""System prompt assembly.

The prompt describes the environment and the rules of honest tool use.  It does **not**
contain trigger lists or per-intent routing: the model receives the tool schemas through
the API's native ``tools`` field and decides for itself when a tool is warranted.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

DEFAULT_PERSONA = (
    "You are Blackthorn, a capable, direct and honest AI assistant. You run on the "
    "Qwen3.8-27B-Uncensored model on dual Tesla T4 GPUs hosted on Kaggle."
)

OPERATING_RULES = (
    "Operating rules:\n"
    "- Answer directly whenever you can. Use a tool only when it is needed to obtain information you do "
    "not have or to act on the machine; otherwise just answer.\n"
    "- Never state that you searched, ran, saved, installed or checked something unless a tool result in "
    "this conversation confirms it. If a tool fails, say so plainly and either try a different approach "
    "or report what could not be done.\n"
    "- Do not repeat a tool call that already returned the information you need.\n"
    "- Files and commands live in the Kaggle workspace; use paths relative to it.\n"
    "- Format answers in Markdown. Put code in fenced blocks with a language tag. Keep answers focused."
)


def build_system_prompt(
    *,
    custom: str = "",
    memories: Optional[List[str]] = None,
    now: Optional[_dt.datetime] = None,
) -> str:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    persona = (custom or "").strip() or DEFAULT_PERSONA
    parts = [persona, OPERATING_RULES, f"Current date and time (UTC): {now.strftime('%Y-%m-%d %H:%M')}."]
    mem = [m.strip() for m in (memories or []) if m and m.strip()][:12]
    if mem:
        parts.append("Long-term memory about the user:\n" + "\n".join(f"- {m}" for m in mem))
    return "\n\n".join(parts)


def attachment_block(attachments: List[Dict[str, Any]]) -> str:
    """Model-visible text for attached files (appended to the user's message)."""
    blocks: List[str] = []
    for att in attachments:
        name = att.get("name") or att.get("path") or "file"
        if att.get("text"):
            note = " (truncated)" if att.get("truncated") else ""
            blocks.append(f"Attached file `{name}`{note}:\n```\n{att['text']}\n```")
        elif att.get("note"):
            blocks.append(f"Attached file `{name}`: {att['note']}.")
    return "\n\n".join(blocks)
