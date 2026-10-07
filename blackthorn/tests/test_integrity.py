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
                   r"blackthorn-hermes-session-token-[a-f0-9]+"]
# Exact credential/identifier values are additionally blocked by the maintainer's local pre-commit hook, which reads them
# from a private file; they are deliberately not written down anywhere in the repository (not even here).


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
    """blackthorn_start.py must only start the app: no downloading, extracting or hashing code into the checkout."""
    import ast
    tree = ast.parse((ROOT / "blackthorn_start.py").read_text())
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not imported & {"tarfile", "urllib", "base64", "hashlib", "subprocess", "zipfile", "shutil", "requests", "httpx"}, imported
    literals = " ".join(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str))
    for banned in ("api.cloudflare.com", "tar -x", "state_meta", ".blackthorn_overlay_stamp"):
        assert banned not in literals, banned
    assert "blackthorn.app:create_app" in literals


def test_ui_build_is_content_hashed_and_self_contained():
    static = ROOT / "blackthorn" / "static"
    index = (static / "index.html").read_text()
    assets = re.findall(r'(?:src|href)="/assets/([^"]+)"', index)
    assert assets, "index.html references no built assets"
    for name in assets:
        assert (static / "assets" / name).is_file(), name
        assert re.search(r"-[A-Za-z0-9_-]{8,}\.(js|css)$", name), f"{name} is not content-hashed"
    external = re.findall(r'(?:src|href|action)=["\']https?://[^"\']+', index)
    assert not external, f"the page must not load anything from a CDN: {external}"
    assert "?v=" not in index, "cache-busting query hacks are gone"
    info = json.loads((static / "build.json").read_text())
    assert info["files"], "build.json must list the shipped assets"


def _ui_source_hash() -> str:
    """Must match sourceHash() in blackthorn/ui/vite.config.ts."""
    import hashlib
    ui = ROOT / "blackthorn" / "ui"
    entries = []

    def add(path: Path):
        rel = path.relative_to(ui).as_posix()
        entries.append(f"{rel}\n{hashlib.sha256(path.read_bytes()).hexdigest()}\n")

    for path in sorted((ui / "src").rglob("*")):
        if path.is_file():
            add(path)
    for name in ("index.html", "package.json", "package-lock.json", "vite.config.ts", "tsconfig.json"):
        add(ui / name)
    return hashlib.sha256("".join(sorted(entries)).encode()).hexdigest()


def test_committed_ui_build_matches_the_ui_source():
    """If you edit blackthorn/ui, run `npm run build` (and `python -m blackthorn.version --write`) before committing."""
    info = json.loads((ROOT / "blackthorn" / "static" / "build.json").read_text())
    assert info["source_hash"] == _ui_source_hash(), "blackthorn/static is stale: the UI source changed after the last build"
