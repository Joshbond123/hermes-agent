"""Guards that keep Git the single source of truth and keep credentials out of it."""

import json
import re
import subprocess
from pathlib import Path

import pytest

from blackthorn import version

ROOT = Path(__file__).resolve().parents[2]
SECRET_PATTERNS = [r"cfut_[A-Za-z0-9]{16,}", r"KGAT_[A-Za-z0-9]{16,}", r"tvly-[A-Za-z0-9_-]{16,}", r"github_pat_[A-Za-z0-9_]{20,}",
                   r"ghp_[A-Za-z0-9]{30,}", r"rnd_[A-Za-z0-9]{20,}", r"sk-qwen38-[A-Za-z0-9-]{6,}",
                   r"blackthorn-hermes-session-token-[a-f0-9]+", r"\b[0-9a-f]{32}\b.*\bd1\b|f932764f168d0f77"]


def tracked_product_files():
    names = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "blackthorn", "blackthorn_start.py", "cloudflare_d1_client.py",
                            "kaggle_cyber_ornith", "KAGGLE_COMPUTER_ARCHITECTURE.md", "docs"],
                           cwd=ROOT, capture_output=True, text=True).stdout.split()
    return [ROOT / n for n in names if (ROOT / n).is_file() and "node_modules" not in n]


def test_no_credentials_in_product_files():
    offenders = []
    for path in tracked_product_files():
        if path.suffix in {".png", ".woff", ".woff2", ".ico", ".map"}:
            continue
        text = path.read_text(errors="ignore")
        for pat in SECRET_PATTERNS:
            if re.search(pat, text):
                offenders.append((path.relative_to(ROOT).as_posix(), pat))
    assert not offenders, offenders


def test_manifest_matches_the_tree():
    """Run `python -m blackthorn.version --write` after changing shipped files, then commit the manifest."""
    status = version.check_manifest(ROOT)
    assert not status.get("missing_manifest"), "blackthorn/MANIFEST.json is missing"
    assert status["drift"] == [] and status["extra"] == [], status


def test_entrypoint_has_no_overlay_and_no_network_bootstrap():
    src = (ROOT / "blackthorn_start.py").read_text()
    for banned in ("overlay", "api.cloudflare.com", "tarfile", "tar -x", "urllib.request", "base64"):
        assert banned not in src, banned
    assert "blackthorn.app:create_app" in src


def test_ui_build_is_content_hashed_and_self_contained():
    static = ROOT / "blackthorn" / "static"
    index = (static / "index.html").read_text()
    assets = re.findall(r'(?:src|href)="/assets/([^"]+)"', index)
    assert assets, "index.html references no built assets"
    for name in assets:
        assert (static / "assets" / name).is_file(), name
        assert re.search(r"-[A-Za-z0-9_-]{8,}\.(js|css)$", name), f"{name} is not content-hashed"
    assert "http://" not in index and "https://" not in index, "the page must not load anything from a CDN"
    assert "?v=" not in index, "cache-busting query hacks are gone"
    info = json.loads((static / "build.json").read_text())
    assert info["files"], "build.json must list the shipped assets"
