#!/usr/bin/env python3
"""Blackthorn Render.com Entrypoint for Hermes Agent + Cloudflare D1 + Kaggle GPU."""

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# On Render, keep HERMES_HOME inside /opt/render/project/src/.hermes_home so the
# Python environment built during the 8GB build phase persists in the image layer
# rather than being recreated in the 512MB runtime container's RAM-backed overlayfs.
if Path("/opt/render/project/src").exists():
    os.environ.setdefault("HERMES_HOME", str(ROOT / ".hermes_home"))
    os.environ.setdefault("HERMES_DISABLE_LAZY_INSTALLS", "1")

os.environ.setdefault(
    "HERMES_DASHBOARD_SESSION_TOKEN",
    "${HERMES_DASHBOARD_SESSION_TOKEN}",
)
os.environ.setdefault("HERMES_MODEL", "Qwen3.8-27B-Uncensored")
os.environ.setdefault("HERMES_TUI_PROVIDER", "custom")
sys_node = shutil.which("node")
if sys_node:
    os.environ.setdefault("HERMES_NODE", sys_node)

CF_ACCOUNT_ID = "${CF_ACCOUNT_ID}"
CF_DB_ID = "${CF_DB_ID}"
CF_TOKEN = "${CF_API_TOKEN}"


def _d1_sql(sql: str) -> list:
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{CF_DB_ID}/query"
    req = urllib.request.Request(
        url,
        data=json.dumps({"sql": sql}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {CF_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["result"][0].get("results") or []


def ensure_overlay_extracted() -> None:
    """Pull the latest Blackthorn overlay from Cloudflare D1 if not already current."""
    for legacy in ("byterover", "holographic", "mem0", "openviking", "retaindb"):
        p = ROOT / "plugins" / "memory" / legacy
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

    stamp_file = ROOT / ".blackthorn_overlay_stamp"
    try:
        sha_rows = _d1_sql("SELECT value FROM state_meta WHERE key = 'blackthorn_overlay_sha256' LIMIT 1;")
        remote_sha = str(sha_rows[0]["value"]).strip() if sha_rows else ""
    except Exception:
        remote_sha = ""

    if (
        remote_sha
        and stamp_file.exists()
        and stamp_file.read_text(encoding="utf-8").strip() == remote_sha
        and (ROOT / "hermes_cli" / "web_dist" / "index.html").exists()
        and (ROOT / "hermes_cli" / "tui_dist" / "entry.js").exists()
    ):
        print(f"[Blackthorn] Overlay already current ({remote_sha[:12]}).")
        return

    print("[Blackthorn] Fetching latest overlay bundle from Cloudflare D1...")
    rows = _d1_sql("SELECT value FROM state_meta WHERE key LIKE 'blackthorn_overlay_0%' ORDER BY key ASC;")
    raw = base64.b64decode("".join(r["value"] for r in rows))
    actual_sha = hashlib.sha256(raw).hexdigest()
    tar_path = ROOT / "overlay.tar.gz"
    tar_path.write_bytes(raw)
    del raw
    del rows
    subprocess.run(["tar", "-xzf", str(tar_path)], cwd=str(ROOT), check=True)
    tar_path.unlink(missing_ok=True)
    stamp_file.write_text(actual_sha, encoding="utf-8")
    print(f"[Blackthorn] Overlay extracted successfully ({actual_sha[:12]}).")


def main() -> None:
    ensure_overlay_extracted()

    import cloudflare_d1_client as d1
    d1._ensure_kaggle_credentials()
    status = d1.get_kaggle_gpu_status(force_refresh=True)
    # Always write config.yaml even when GPU is OFF so model is always Qwen3.8-27B-Uncensored
    d1.update_hermes_model_endpoint(status.get("tunnel_url") or "", force_save=True)

    port = str(os.environ.get("PORT") or "10000")
    print(f"[Blackthorn] Starting Hermes Agent Dashboard on 0.0.0.0:{port}...")
    os.execvp(
        sys.executable,
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "dashboard",
            "--host",
            "0.0.0.0",
            "--port",
            port,
            "--no-open",
            "--skip-build",
        ],
    )


if __name__ == "__main__":
    main()
