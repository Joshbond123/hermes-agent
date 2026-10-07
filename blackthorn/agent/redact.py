"""Scrub credential-looking strings before anything is shown to the UI or stored.

Tool arguments and outputs are displayed (collapsed) in the chat and persisted with the
message, so they must never carry secrets.  This is a defence-in-depth filter; tools
should also avoid returning secrets in the first place.
"""

from __future__ import annotations

import re
from typing import Any

_PATTERNS = [
    re.compile(r"(?i)\b(bearer|token|authorization)\b(\s*[:=]\s*|\s+)([A-Za-z0-9._~+/=-]{16,})"),
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bKGAT_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bcfut_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\btvly-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\brnd_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"(?i)\b(api[_-]?key|secret|password|passwd)\b(\s*[:=]\s*)(['\"]?)([^\s'\"]{8,})\3"),
]


def redact(text: Any) -> str:
    """Return ``text`` with credential-shaped substrings replaced by ``[REDACTED]``."""
    if text is None:
        return ""
    out = str(text)
    for pattern in _PATTERNS:
        if pattern.groups >= 3 and "bearer" in pattern.pattern:
            out = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", out)
        elif pattern.groups >= 4:
            out = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}[REDACTED]{m.group(3)}", out)
        else:
            out = pattern.sub("[REDACTED]", out)
    return out


def preview(text: Any, limit: int = 1200) -> str:
    """Redacted, length-limited text for display."""
    cleaned = redact(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"
