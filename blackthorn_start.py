#!/usr/bin/env python3
"""Render start command for Blackthorn: ``python3 blackthorn_start.py``.

Everything that runs comes from this repository. This script does **not** download, extract
or patch anything at boot (an earlier setup pulled a code "overlay" out of Cloudflare D1 on
every start, which silently replaced what GitHub said was deployed).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# Render: keep HERMES_HOME inside the project so the environment built at deploy time is reused.
if Path("/opt/render/project/src").exists():
    os.environ.setdefault("HERMES_HOME", str(ROOT / ".hermes_home"))
    os.environ.setdefault("HERMES_DISABLE_LAZY_INSTALLS", "1")

# Single-user app on a public URL: explicit opt-in to bind publicly without the OAuth gate
# (set HERMES_FORCE_OAUTH_GATE=1 to turn the gate on again).
os.environ.setdefault("BLACKTHORN_PUBLIC_DASHBOARD", "1")


def _prepare_environment() -> None:
    from blackthorn import config

    # A session token that survives restarts (open tabs keep working) without living in the repo.
    token = config.stable_dashboard_token()
    if token:
        os.environ.setdefault("HERMES_DASHBOARD_SESSION_TOKEN", token)


def _check_web_dist() -> None:
    dist = Path(os.environ.get("HERMES_WEB_DIST") or ROOT / "hermes_cli" / "web_dist")
    if not (dist / "index.html").is_file() or not (dist / "build-info.json").is_file():
        sys.stderr.write(
            f"[Blackthorn] FATAL: no UI build found in {dist}.\n"
            "  The Render build command must run scripts/render_build.sh (it builds web/ from source).\n"
            "  Refusing to start rather than serve a missing or stale UI.\n"
        )
        raise SystemExit(1)


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    _prepare_environment()
    _check_web_dist()
    port = str(os.environ.get("PORT") or "10000")
    commit = (os.environ.get("RENDER_GIT_COMMIT") or "local")[:10]
    print(f"[Blackthorn] starting commit {commit} on 0.0.0.0:{port}", flush=True)
    os.execvp(
        sys.executable,
        [sys.executable, "-m", "hermes_cli.main", "dashboard", "--host", "0.0.0.0", "--port", port,
         "--no-open", "--skip-build"],
    )


if __name__ == "__main__":
    main()
