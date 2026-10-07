"""Which build is running? (exposed at /api/blackthorn/version so it can always be proven)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

STARTED_AT = time.time()


def _dist_dir() -> Path:
    env = os.environ.get("HERMES_WEB_DIST")
    return Path(env) if env else Path(__file__).resolve().parent.parent / "hermes_cli" / "web_dist"


def info() -> Dict[str, Any]:
    build: Dict[str, Any] = {}
    try:
        build = json.loads((_dist_dir() / "build-info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        build = {}
    return {
        "commit": os.environ.get("RENDER_GIT_COMMIT") or build.get("commit") or "unknown",
        "branch": os.environ.get("RENDER_GIT_BRANCH") or build.get("branch") or "unknown",
        "service": os.environ.get("RENDER_SERVICE_NAME") or "local",
        "web_build": build,
        "web_dist_present": (_dist_dir() / "index.html").is_file(),
        "started_at": STARTED_AT,
        "uptime_s": int(time.time() - STARTED_AT),
    }
