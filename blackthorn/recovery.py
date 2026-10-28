"""Failure classification and retry policy for tool execution.

The two production defects this module exists to fix:

* ``Provider VM expired while idle; the next computer tool call will create a
  replacement and restore the GitHub workspace.`` — an *infrastructure* failure
  (the sandbox is gone), not a tool-logic failure and not a duplicate call.
* ``Stopped using tools (repeated identical tool calls); writing the answer
  from what was gathered.`` — the agent aborting a long task because a
  recoverable infrastructure failure was counted as a duplicate tool call.

The old loop counted *every* execution of the same ``(tool, args)`` pair, so a
recoverable provider failure burned the duplicate budget, produced three
"strikes", and forced a final answer. The fix is to separate three questions:

1. Did the call produce a *logical* outcome (the tool ran and returned
   something) or did the *environment* fail before the tool logic ran?
2. Is the environmental failure retryable, and with what backoff?
3. Has the model genuinely stalled (same successful call, unchanged world),
   or is it re-issuing a call that never actually completed?

Only case 3 may ever end a run. Everything else is retried with bounded
backoff and, when that fails, escalates to a *changed strategy* instruction
instead of an identical retry.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ToolError.kind / ToolResult.data["kind"] values that mean "the environment
# failed", never "the tool rejected the request".
INFRA_KINDS = frozenset({
    "offline", "timeout", "network", "transport", "storage", "auth",
    "quota", "provider", "session_expired", "vm_expired", "rate_limited",
    "computer_offline", "vpn_unavailable", "remote_unavailable",
    "infra_unrecovered", "provider_quota", "handover",
})

# Kinds that can never be healed by repeating the same call.
PERMANENT_KINDS = frozenset({
    "invalid", "not_found", "unsupported", "binary", "bad_request",
    "protocol", "rejected", "tool_error", "vpn_unavailable",
})

# Sentence fragments that identify a provider/session failure even when a
# lower layer reported it as a generic error (relay HTML pages, 502s, and the
# PandaStack expiry path all surface this way).
_INFRA_HINTS = (
    "vm expired", "expired while idle", "sandbox no longer exists",
    "computer is missing", "computer is offline", "computer could not be reached",
    "computer state could not be", "connection refused", "tunnel is down",
    "temporarily unavailable", "timed out", "timeout", "bad gateway",
    "service unavailable", "gateway timeout", "network error",
    "could not be woken", "recovery failed", "quota", "rate limit",
    "too many requests", "denied access", "401", "403", "429", "502", "503", "504",
)


def signature(name: str, args: Any) -> str:
    """Stable, secret-free fingerprint of one tool call."""
    try:
        blob = json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        blob = str(args)
    return hashlib.sha256(f"{name}\x00{blob}".encode("utf-8", "replace")).hexdigest()[:32]


def looks_like_infra(message: str) -> bool:
    text = (message or "").lower()
    return any(hint in text for hint in _INFRA_HINTS)


def classify(result: Any) -> str:
    """Return ``ok`` | ``invalid`` | ``duplicate`` | ``infra`` | ``logic``."""
    if getattr(result, "ok", False):
        return "ok"
    data = getattr(result, "data", None) or {}
    kind = str(data.get("kind") or "").lower()
    error = str(getattr(result, "error", "") or "")
    if error.startswith("duplicate call"):
        return "duplicate"
    if error.startswith("invalid arguments") or error.startswith("unknown tool"):
        return "invalid"
    if kind in INFRA_KINDS:
        return "infra"
    if kind in PERMANENT_KINDS:
        return "logic"
    if looks_like_infra(error):
        return "infra"
    return "logic"


@dataclass
class RetryPolicy:
    """Bounded retries, exponential backoff with jitter, and a hard deadline.

    ``max_attempts`` counts *total* attempts including the first one.
    """

    max_attempts: int = 3
    base_delay_s: float = 1.5
    max_delay_s: float = 12.0
    jitter: float = 0.25
    total_budget_s: float = 150.0
    started_at: float = field(default_factory=time.monotonic)

    def exhausted_deadline(self) -> bool:
        return (time.monotonic() - self.started_at) >= self.total_budget_s

    def delay(self, attempt: int) -> float:
        """Delay before attempt number ``attempt`` (1-based)."""
        raw = min(self.max_delay_s, self.base_delay_s * (2 ** max(0, attempt - 1)))
        spread = raw * max(0.0, self.jitter)
        return max(0.0, raw + random.uniform(-spread, spread))

    def should_retry(self, attempt: int, verdict: str) -> bool:
        if verdict != "infra":
            return False
        if attempt >= self.max_attempts:
            return False
        return not self.exhausted_deadline()

    async def sleep_for(self, attempt: int) -> None:
        delay = self.delay(attempt)
        remaining = self.total_budget_s - (time.monotonic() - self.started_at)
        await asyncio.sleep(max(0.0, min(delay, remaining)))


@dataclass
class CallLedger:
    """Per-run accounting for one tool signature.

    ``completed`` counts calls that actually reached the tool logic (success or
    a genuine logical error). ``infra_failures`` counts environment failures,
    which are retried and never count as the model repeating itself.
    """

    completed: int = 0
    infra_failures: int = 0
    strikes: int = 0
    epoch: Optional[int] = None
    last_error: str = ""
    strategy_changed: bool = False

    def note_infra(self, message: str) -> None:
        self.infra_failures += 1
        self.last_error = (message or "")[:300]

    def note_completed(self) -> None:
        self.completed += 1

    def as_dict(self) -> Dict[str, Any]:
        return {"completed": self.completed, "infra_failures": self.infra_failures,
                "strikes": self.strikes}


def infra_directive(name: str, attempts: int, error: str, *, recovered: bool, occurrence: int = 1) -> str:
    """Instruction appended to a tool result after an unrecovered infrastructure failure.

    It is deliberately *different* on every occurrence (attempt count, verbatim
    error, explicit next actions) so the model never receives an identical
    message twice and cannot mistake it for a duplicate-call suppression.
    """
    head = (f"The computer/provider was unavailable for `{name}` (failure #{occurrence}, attempt {attempts} "
            f"of {attempts}): {error[:400]}"
            if not recovered else
            f"`{name}` succeeded on attempt {attempts} after the computer recovered "
            f"(failure #{occurrence}): {error[:300]}")
    return (
        f"{head}\n"
        "This is an infrastructure failure, not a bad request and not a duplicate call. "
        "The sandbox is recreated and the GitHub workspace is restored automatically by the runtime. "
        "Next steps, in order: (1) if the result above already contains usable output, continue from it; "
        "(2) re-issue this exact call once — it is not counted as a repeat; "
        "(3) if it fails again, switch approach: use a different tool (write_file/read_file instead of a "
        "shell heredoc, or a background job), or narrow the operation; "
        "(4) report the blocker plainly only after those attempts fail."
    )


def strategy_change_note(name: str, args_preview: str, attempts: int) -> str:
    """Server note injected when the model repeats a call that already completed."""
    return (
        f"[server] `{name}` with these arguments has now run {attempts} time(s) and returned a result that is "
        "already in this conversation. Repeating it unchanged cannot produce new information. "
        "Change strategy now: use the existing result, vary the arguments (different path, pattern, host or "
        "flag), pick a different tool, or proceed to the next step of the task. "
        f"Arguments were: {args_preview[:200]}. Do not call it again unchanged."
    )
