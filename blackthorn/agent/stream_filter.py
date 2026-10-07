"""Incremental filter between raw model text and the visible answer.

Guarantees that the user never sees

* ``<think>…</think>`` reasoning (some servers leave it inline in ``content``), or
* ``<tool_call>…</tool_call>`` blocks (the text form of a native tool call when the
  server did not convert it),

even when a tag is split across stream chunks.  Text before a tag is released
immediately; only a possible *partial* tag at the very end is held back.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"
_ALL_OPENERS = (_THINK_OPEN, _TOOL_OPEN, _THINK_CLOSE)  # stray </think> in normal text is dropped


def _partial_suffix_len(buf: str, markers: tuple) -> int:
    """Length of the longest suffix of ``buf`` that is a proper prefix of any marker."""
    best = 0
    lowered = buf.lower()
    for marker in markers:
        for k in range(min(len(marker) - 1, len(lowered)), 0, -1):
            if lowered.endswith(marker[:k]):
                best = max(best, k)
                break
    return best


class ContentFilter:
    NORMAL, THINK, TOOL = "normal", "think", "tool"

    def __init__(self) -> None:
        self.mode = self.NORMAL
        self.buf = ""
        self.tool_blocks: List[str] = []
        self._tool_acc = ""
        self.saw_inline_think = False

    # -- helpers -----------------------------------------------------------
    def _find(self, hay: str, needle: str) -> int:
        return hay.lower().find(needle)

    def feed(self, text: str) -> str:
        """Add model text; return the part that is safe to show right now."""
        self.buf += text
        out: List[str] = []
        while True:
            if self.mode == self.NORMAL:
                hits = [(self._find(self.buf, m), m) for m in _ALL_OPENERS]
                hits = [(i, m) for i, m in hits if i != -1]
                if not hits:
                    keep = _partial_suffix_len(self.buf, _ALL_OPENERS)
                    if keep:
                        out.append(self.buf[:-keep])
                        self.buf = self.buf[-keep:]
                    else:
                        out.append(self.buf)
                        self.buf = ""
                    break
                idx, marker = min(hits)
                out.append(self.buf[:idx])
                self.buf = self.buf[idx + len(marker):]
                if marker == _THINK_OPEN:
                    self.mode = self.THINK
                    self.saw_inline_think = True
                elif marker == _TOOL_OPEN:
                    self.mode = self.TOOL
                    self._tool_acc = ""
                # stray close tag: just dropped
            elif self.mode == self.THINK:
                idx = self._find(self.buf, _THINK_CLOSE)
                if idx == -1:
                    keep = _partial_suffix_len(self.buf, (_THINK_CLOSE,))
                    self.buf = self.buf[-keep:] if keep else ""
                    break
                self.buf = self.buf[idx + len(_THINK_CLOSE):]
                self.mode = self.NORMAL
            else:  # TOOL
                idx = self._find(self.buf, _TOOL_CLOSE)
                if idx == -1:
                    keep = _partial_suffix_len(self.buf, (_TOOL_CLOSE,))
                    if keep:
                        self._tool_acc += self.buf[:-keep]
                        self.buf = self.buf[-keep:]
                    else:
                        self._tool_acc += self.buf
                        self.buf = ""
                    break
                self._tool_acc += self.buf[:idx]
                self.tool_blocks.append(self._tool_acc)
                self._tool_acc = ""
                self.buf = self.buf[idx + len(_TOOL_CLOSE):]
                self.mode = self.NORMAL
        return "".join(out)

    def flush(self) -> str:
        """End of stream: release whatever is still held back."""
        out = ""
        if self.mode == self.NORMAL:
            out = self.buf
        elif self.mode == self.TOOL:
            self.tool_blocks.append(self._tool_acc + self.buf)
        # an unterminated <think> block is reasoning: discard
        self.buf = ""
        self._tool_acc = ""
        self.mode = self.NORMAL
        return out


def parse_text_tool_calls(blocks: List[str]) -> List[Dict[str, Any]]:
    """Parse ``{"name": …, "arguments": …}`` JSON found inside ``<tool_call>`` blocks."""
    calls: List[Dict[str, Any]] = []
    for block in blocks:
        raw = block.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.S)
        if fence:
            raw = fence.group(1)
        else:
            brace = re.search(r"\{.*\}", raw, re.S)
            if brace:
                raw = brace.group(0)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        name = str(data.get("name") or data.get("tool") or "").strip()
        args: Optional[Any] = data.get("arguments", data.get("args", {}))
        if not name:
            continue
        calls.append({"name": name, "arguments": args if isinstance(args, str) else json.dumps(args or {})})
    return calls
