"""Task list for multi-step work — Hermes' ``todo_list`` tool, persisted per session.

Upstream keeps the list on the in-memory ``TodoStore`` (one per ``AIAgent``) and re-injects it after
every context compression (``tools/todo_tool.py``: ``write``/``merge``/``read``/``snapshot``, statuses
pending | in_progress | completed | cancelled, ``TODO_INJECTION_HEADER``). Blackthorn runs one process
serving many sessions and restarts often, so the same semantics are kept but the list is stored in
D1 ``state_meta`` (``bt_todo:<session_id>``) — the same place Hermes keeps session goals — and restored
when a turn starts, so a restart or a compaction cannot lose the plan.

The list is the agent's *declared* plan. What actually happened is tracked separately by the agent's
execution ledger; the two are compared before a run is allowed to finish.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from .base import ToolContext, ToolResult, ToolSpec, redact

VALID_STATUSES = ("pending", "in_progress", "completed", "cancelled")
ACTIVE_STATUSES = ("pending", "in_progress")
STATUS_MARKS = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]", "cancelled": "[~]"}
MAX_ITEMS = 256
MAX_CONTENT_CHARS = 400


def _cap(text: str) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= MAX_CONTENT_CHARS else text[: MAX_CONTENT_CHARS - 1].rstrip() + "…"


def _normalize(items: Sequence[Any], *, existing: Sequence[Dict[str, Any]] = ()) -> List[Dict[str, str]]:
    """Validate/normalise a whole list, assigning ids when the model omits them and dropping junk."""
    out: List[Dict[str, str]] = []
    for raw in items or []:
        item = _normalize_item(raw, len(out) + 1, existing=list(existing) + out)
        if item is not None:
            out.append(item)
    return out[:MAX_ITEMS]


def _normalize_item(raw: Any, position: int, *, existing: Sequence[Dict[str, Any]] = (),
                    require_content: bool = True) -> Optional[Dict[str, str]]:
    """Validate one item. ``require_content=False`` allows a status-only update (id + status)."""
    if not isinstance(raw, dict):
        return None
    content = _cap(raw.get("content") or raw.get("task") or raw.get("title") or "")
    if not content and require_content:
        return None
    taken = {str(i.get("id")) for i in existing}
    item_id = str(raw.get("id") or "").strip()
    if not item_id:
        item_id = str(position)
    n = 0
    base, candidate = item_id, item_id
    while candidate in taken and str(raw.get("id") or "").strip() == "":
        n += 1
        candidate = f"{base}.{n}"
    status = str(raw.get("status") or ("pending" if require_content else "")).strip().lower()
    if status not in VALID_STATUSES:
        status = "pending"
    item: Dict[str, str] = {"id": candidate, "content": content, "status": status}
    parent = str(raw.get("parent") or "").strip()
    if parent and parent != candidate:
        item["parent"] = parent
    return item


class TaskBoard:
    """One session's task list: in-memory during a run, persisted after every write."""

    def __init__(self, store: Any = None, session_id: str = "", items: Optional[Sequence[Dict[str, Any]]] = None,
                 goal: str = "") -> None:
        self.store = store
        self.session_id = session_id or ""
        self.goal = goal[:2000]
        self.items: List[Dict[str, str]] = _normalize(items or [])
        self._revision = 0

    @classmethod
    async def load(cls, store: Any, session_id: str) -> "TaskBoard":
        state = await store.load_task_state(session_id)
        return cls(store, session_id, state["todos"], state["goal"])

    # ------------------------------------------------------------------ reads
    def read(self) -> List[Dict[str, str]]:
        return [dict(i) for i in self.items]

    def pending(self) -> List[Dict[str, str]]:
        return [dict(i) for i in self.items if i.get("status") in ACTIVE_STATUSES]

    def counts(self) -> Dict[str, int]:
        out = {s: 0 for s in VALID_STATUSES}
        for item in self.items:
            out[item.get("status", "pending")] = out.get(item.get("status", "pending"), 0) + 1
        return out

    def is_open(self) -> bool:
        return bool(self.pending())

    def format(self) -> str:
        if not self.items:
            return ("No tasks recorded yet. For a request with two or more steps, call todo_list first with the "
                    "steps in order, then update each item's status as its tool result comes back.")
        lines = [f"  {STATUS_MARKS.get(i.get('status', 'pending'), '[ ]')} {i['id']}: {i['content']}" for i in self.items]
        counts = self.counts()
        return ("Current task list (server-persisted):\n" + "\n".join(lines) +
                f"\n{counts.get('completed', 0)} completed, {counts.get('in_progress', 0)} in progress, "
                f"{counts.get('pending', 0)} pending. Mark an item completed only after its tool returned successfully.")

    # ------------------------------------------------------------------ writes
    def write(self, todos: Sequence[Any], merge: bool = False) -> List[Dict[str, str]]:
        """Replace (default) or merge the list; a rejected write leaves the list untouched.

        Merge applies only the fields present, so ``{"id": "2", "status": "completed"}`` flips a status
        without the model having to resend the whole list (and without wiping the text it omits).
        """
        if merge and self.items:
            by_id: Dict[str, Dict[str, str]] = {i["id"]: i for i in self.items}
            for raw in todos or []:
                if not isinstance(raw, dict):
                    continue
                item_id = str(raw.get("id") or "").strip()
                cur = by_id.get(item_id)
                if cur is None:                      # new item: needs content
                    new = _normalize_item(raw, len(self.items) + 1, existing=self.items)
                    if new is None:
                        continue
                    by_id[new["id"]] = new
                    self.items.append(new)
                    continue
                content = raw.get("content")
                if isinstance(content, str) and content.strip():
                    cur["content"] = _cap(content)
                status = str(raw.get("status") or "").strip().lower()
                if status in VALID_STATUSES:
                    cur["status"] = status
                if "parent" in raw:
                    parent = str(raw.get("parent") or "").strip()
                    if parent and parent != cur["id"]:
                        cur["parent"] = parent
                    else:
                        cur.pop("parent", None)
                by_id[cur["id"]] = cur
            self.items = [by_id[i["id"]] for i in self.items]
        else:
            self.items = _normalize(todos)
        self.items = self.items[:MAX_ITEMS]
        self._revision += 1
        return self.read()

    async def save(self) -> None:
        if self.store is None or not self.session_id:
            return
        # A failed write must reach the tool result; claiming "Saved" here would
        # manufacture a memory the next worker cannot actually recover.
        await self.store.save_task_state(self.session_id, self.read(), self.goal)


async def todo_list(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    board: Optional[TaskBoard] = getattr(ctx, "board", None)
    if board is None:
        return ToolResult.failure("the task list is not available in this run")
    todos = args.get("todos")
    if todos is None:
        return ToolResult(ok=True, content=board.format(), summary=f"{len(board.items)} task(s)",
                          data={"todos": board.read()})
    if not isinstance(todos, list):
        return ToolResult.failure("invalid arguments for todo_list: 'todos' must be a list of objects",
                                  hint='Send [{"id": "1", "content": "...", "status": "completed"}].')
    prior, prior_goal = board.read(), board.goal
    if not board.items and todos:
        board.goal = " ".join(str(getattr(ctx, "user_message", "") or "").split())[:2000]
    items = board.write(todos, merge=bool(args.get("merge", True)))
    try:
        await board.save()
    except Exception:
        board.items, board.goal = prior, prior_goal
        return ToolResult.failure("Task list was NOT saved. Persistent storage is unavailable; retry later.")
    return ToolResult(ok=True, content=board.format() + "\nSaved. Do the next pending item now.",
                      summary=f"{len(items)} task(s)", data={"todos": items})


SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "todos": {
            "type": "array",
            "description": "Full or partial list. Omit to read. Each item: {id, content, status}.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Stable id; reuse it to update a status."},
                    "content": {"type": "string", "minLength": 1, "maxLength": 400},
                    "status": {"type": "string", "enum": list(VALID_STATUSES)},
                },
                "required": ["content"],
            },
        },
        "merge": {"type": "boolean", "default": True,
                  "description": "true (default) updates only the items listed; false replaces the whole list."},
    },
}


def specs() -> List[ToolSpec]:
    def describe(args: Dict[str, Any]) -> str:
        todos = args.get("todos")
        if not isinstance(todos, list):
            return "read task list"
        bits = []
        for item in todos[:4]:
            if isinstance(item, dict):
                bits.append(f"{item.get('id') or '?'}:{item.get('status') or ''}")
        return "tasks " + ", ".join(bits)

    return [ToolSpec(
        "todo_list",
        "Your persistent task list for multi-step requests (3+ steps): record the steps, mark each one "
        "pending/in_progress/completed as you go, and read it back after a long run. Omit 'todos' to read "
        "the current list. Mark completed only after the tool for that step succeeded.",
        SCHEMA, todo_list, timeout=30.0, kind="state", describe=describe)]
