"""Resolve uploaded files (managed-file policy of the dashboard) into text for the model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("blackthorn.attachments")

MAX_ATTACHMENTS = 8
MAX_ATTACHMENT_CHARS = 60_000
MAX_INLINE_BYTES = 8 * 1024 * 1024


def _read_one(request: Any, item: Dict[str, Any]) -> Dict[str, Any] | None:
    from fastapi import HTTPException

    from hermes_cli.web_routers.files import _is_sensitive_path, _resolve_managed_path

    path = str(item.get("path") or "").strip()
    if not path:
        return None
    try:
        try:
            _policy, target, display = _resolve_managed_path(path, request)
        except HTTPException:
            if Path(path).is_absolute():
                raise
            _p, root, _d = _resolve_managed_path("", request)
            _policy, target, display = _resolve_managed_path(str(Path(root) / path), request)
        if not target.is_file() or _is_sensitive_path(target):
            return None
        size = target.stat().st_size
        if size > MAX_INLINE_BYTES:
            return {"path": display, "name": target.name, "note": "file too large to inline"}
        blob = target.read_bytes()
        try:
            text, binary = blob.decode("utf-8"), False
        except UnicodeDecodeError:
            text, binary = blob.decode("latin-1", errors="replace"), True
        clipped = text[:MAX_ATTACHMENT_CHARS]
        return {"path": display, "name": item.get("name") or target.name, "bytes": size, "binary": binary,
                "truncated": len(text) > len(clipped), "text": clipped}
    except HTTPException:
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("attachment read failed for %s: %s", path, exc)
        return None


def load(request: Any, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Blocking; call from a worker thread."""
    out: List[Dict[str, Any]] = []
    for item in items[:MAX_ATTACHMENTS]:
        loaded = _read_one(request, item)
        if loaded:
            out.append(loaded)
    return out
