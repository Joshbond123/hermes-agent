#!/usr/bin/env python3
"""Blackthorn entrypoint for Render:  ``python3 blackthorn_start.py``

* Git is the only source of the code that runs. There is no overlay, no download-and-extract step and no
  credential in this file — secrets come from the Render environment (CLOUDFLARE_*, KAGGLE_*, TAVILY_*) or from D1.
* The build command's ``hermes_cli.main pm repair`` prepares a sealed dependency environment; this script
  activates it, then serves the lean Blackthorn app (``blackthorn.app``). The server is listening within seconds,
  so a cold start after the free instance sleeps is short.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# Keep HERMES_HOME inside the build output on Render so the dependency environment built there is the one used.
if Path("/opt/render/project/src").exists():
    os.environ.setdefault("HERMES_HOME", str(ROOT / ".hermes_home"))
    os.environ.setdefault("HERMES_DISABLE_LAZY_INSTALLS", "1")
os.environ.setdefault("HERMES_MODEL", "Qwen3.8-27B-Uncensored")


def activate_dependencies() -> None:
    try:
        from pm.environments import activate_dependencies as activate

        activate(ROOT)
    except Exception as exc:  # dev machines already have the packages in the current interpreter
        print(f"[Blackthorn] sealed environment not activated ({type(exc).__name__}: {exc}); using the current interpreter", flush=True)


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    activate_dependencies()
    from blackthorn import __version__

    port = int(os.environ.get("PORT") or 10000)
    commit = (os.environ.get("RENDER_GIT_COMMIT") or "dev")[:10]
    print(f"[Blackthorn] v{__version__} commit={commit} listening on 0.0.0.0:{port}", flush=True)
    import uvicorn

    uvicorn.run(
        "blackthorn.app:create_app", factory=True, host="0.0.0.0", port=port, log_level="info", access_log=False,
        timeout_keep_alive=75, timeout_graceful_shutdown=8, proxy_headers=True, forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
