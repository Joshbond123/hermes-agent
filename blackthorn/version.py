"""Which code is actually running? Commit + an integrity manifest of the files that matter.

``MANIFEST.json`` is written at commit time (``python -m blackthorn.version --write``) and verified by the test
suite. At runtime ``/api/version`` re-hashes the shipped files, so any out-of-band modification of the deployed tree
(for example an overlay extracted over the checkout) shows up immediately as ``drift``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from . import __version__

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = Path(__file__).resolve().parent / "MANIFEST.json"
STARTED_AT = time.time()

TRACKED = (
    "blackthorn_start.py", "cloudflare_d1_client.py", "kaggle_cyber_ornith/cyber_ornith_server.py",
)
TRACKED_DIRS = ("blackthorn",)
SKIP_PARTS = {"__pycache__", "ui", "node_modules", "tests"}
SKIP_FILES = {"MANIFEST.json"}


def _iter_files(root: Path):
    for rel in TRACKED:
        if (root / rel).is_file():
            yield rel
    for d in TRACKED_DIRS:
        for path in sorted((root / d).rglob("*")):
            if not path.is_file() or path.name in SKIP_FILES or path.suffix == ".pyc":
                continue
            parts = path.relative_to(root / d).parts
            if any(part in SKIP_PARTS for part in parts[:-1]) or parts[0] in SKIP_PARTS:
                continue
            yield path.relative_to(root).as_posix()


def file_hashes(root: Path = ROOT) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for rel in _iter_files(root):
        out[rel] = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    return out


def write_manifest(root: Path = ROOT) -> Dict[str, str]:
    hashes = file_hashes(root)
    MANIFEST_PATH.write_text(json.dumps({"version": __version__, "files": hashes}, indent=1, sort_keys=True) + "\n")
    return hashes


def check_manifest(root: Path = ROOT) -> Dict[str, object]:
    try:
        recorded = json.loads(MANIFEST_PATH.read_text())["files"]
    except (OSError, ValueError, KeyError):
        return {"checked": 0, "drift": [], "missing_manifest": True}
    actual = file_hashes(root)
    drift: List[str] = sorted(rel for rel in recorded if actual.get(rel) != recorded[rel])
    extra = sorted(rel for rel in actual if rel not in recorded)
    return {"checked": len(recorded), "drift": drift, "extra": extra}


def _git_commit() -> str:
    for key in ("RENDER_GIT_COMMIT", "GIT_COMMIT", "SOURCE_VERSION"):
        if os.environ.get(key):
            return os.environ[key]
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_info(static_dir: Path) -> Dict[str, object]:
    ui: Dict[str, object] = {}
    try:
        ui = json.loads((static_dir / "build.json").read_text())
    except (OSError, ValueError):
        pass
    integrity = check_manifest()
    return {
        "app": "blackthorn", "version": __version__, "commit": _git_commit(),
        "branch": os.environ.get("RENDER_GIT_BRANCH", ""), "ui": ui,
        "started_at": STARTED_AT, "uptime_seconds": int(time.time() - STARTED_AT),
        "python": sys.version.split()[0], "integrity": integrity,
        "overlay_stamp_present": (ROOT / ".blackthorn_overlay_stamp").exists(),
    }


if __name__ == "__main__":
    if "--write" in sys.argv:
        hashes = write_manifest()
        print(f"manifest written: {len(hashes)} files")
    else:
        print(json.dumps(check_manifest(), indent=1))
