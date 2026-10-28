"""Skills and tools live in the GitHub drive, not on a disposable microVM.

A PandaStack computer is temporary: it expires, hibernates, or is replaced during a
handover. Anything only stored on its disk is therefore lost. This module makes the
three cybersecurity repositories — Claude-Red, hackingtool-plugin and
Anthropic-Cybersecurity-Skills — plus the catalog reader (``skillctl.py``) into a
single versioned bundle that is:

1. **built** from the upstream repositories (once per upstream commit),
2. **published** to the private GitHub Computer-drive as a release asset,
3. **restored** onto every new computer *and* every lab VM before it is used,
4. **verified** by asking the restored catalog to index itself and search.

The drive is the authoritative store; a VM's ``/opt/blackthorn`` is only a cache that
can be thrown away at any moment. Restore is idempotent and cheap when the VERSION
marker already matches, so it can run on every ``ensure_ready`` without slowing a
normal turn.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import pathlib
import shutil
import tarfile
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional

CATALOG_DIR = "/opt/blackthorn/catalog"
BIN_DIR = "/opt/blackthorn/bin"
BUNDLE_PATH = "/opt/blackthorn/catalog.tar.gz"
VERSION_FILE = CATALOG_DIR + "/VERSION"
DRIVE_BUNDLE = "capabilities/catalog.tar.gz"
DRIVE_META = "capabilities/latest.json"
FORMAT = 2

# Same three upstream repositories the catalog indexes. ``dir`` must match
# skillctl.SOURCES so the restored tree works with no extra configuration.
SOURCES: Dict[str, Dict[str, str]] = {
    "anthropic": {"repo": "mukul975/Anthropic-Cybersecurity-Skills", "dir": "anthropic-cybersecurity-skills"},
    "claude-red": {"repo": "SnailSploit/Claude-Red", "dir": "claude-red"},
    "hackingtool": {"repo": "AKCodez/hackingtool-plugin", "dir": "hackingtool-plugin"},
}

# Only text a skill needs at run time. Upstream assets (images, screenshots, videos)
# are hundreds of megabytes and are never read by skillctl or by the model.
KEEP_SUFFIXES = (".md", ".json", ".py", ".txt", ".yaml", ".yml", ".sh", ".toml", ".cfg")
SKIP_PARTS = (".git", "assets", "images", "node_modules", "__pycache__", ".github")
MAX_FILE_BYTES = 2 * 1024 * 1024


class CapabilityError(RuntimeError):
    pass


def _wanted(name: str) -> bool:
    lowered = name.lower()
    if not lowered.endswith(KEEP_SUFFIXES):
        return False
    return True


def _has_skipped_part(path: str) -> bool:
    parts = [p for p in path.split("/") if p]
    return any(p in SKIP_PARTS for p in parts)


def build_bundle(fetch_repo: Callable[[str], bytes]) -> bytes:
    """Build the catalog tarball from zipped repositories.

    ``fetch_repo`` returns the raw bytes of a repository zip archive for a
    ``owner/name`` string, so tests can inject a fixture instead of the network.
    """
    buf = io.BytesIO()
    kept: List[str] = []
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=6) as tar:
        for meta in SOURCES.values():
            raw = fetch_repo(meta["repo"])
            if not raw:
                raise CapabilityError(f"could not download {meta['repo']}")
            with zipfile_or_raise(raw) as archive:
                for info in archive.infolist():
                    if info.is_dir() or info.file_size > MAX_FILE_BYTES:
                        continue
                    # Strip the "<repo>-<branch>/" prefix GitHub adds to archives.
                    parts = info.filename.split("/", 1)
                    rel = parts[1] if len(parts) == 2 else ""
                    if not rel or _has_skipped_part(rel) or not _wanted(rel):
                        continue
                    data = archive.read(info)
                    target = f"{meta['dir']}/{rel}"
                    entry = tarfile.TarInfo(target)
                    entry.size = len(data)
                    entry.mtime = int(time.time())
                    entry.mode = 0o644
                    tar.addfile(entry, io.BytesIO(data))
                    kept.append(target)
    if not kept:
        raise CapabilityError("the capability bundle would be empty")
    return buf.getvalue()


class zipfile_or_raise:
    """Tiny context manager so a corrupt archive yields a clear CapabilityError."""

    def __init__(self, raw: bytes):
        import zipfile
        try:
            self._archive = zipfile.ZipFile(io.BytesIO(raw))
        except Exception as exc:  # noqa: BLE001
            raise CapabilityError("a repository archive could not be read") from exc

    def __enter__(self):
        return self._archive

    def __exit__(self, *exc):
        self._archive.close()
        return False


def version_of(bundle: bytes, sources: Dict[str, str]) -> str:
    digest = hashlib.sha256(bundle).hexdigest()[:16]
    stamp = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:8]
    return f"v{FORMAT}-{stamp}-{digest}"


def _normalise(result: Dict[str, Any]) -> Dict[str, Any]:
    """One result shape for both transports.

    The computer client reports ``output`` and the lab reports ``stdout``/``stderr``; a
    restore that checks ``stdout`` therefore "failed" on a computer that had in fact
    extracted 43 MB of catalog perfectly. Normalise once, here.
    """
    data = dict(result or {})
    out = data.get("stdout")
    if out is None:
        out = data.get("output") or ""
    data["stdout"] = str(out)
    data.setdefault("stderr", "")
    return data


class _ComputerTarget:
    """Adapts the PandaStack computer client to the few operations install needs."""

    def __init__(self, computer: Any):
        self.computer = computer

    async def put(self, path: str, data: bytes) -> None:
        # A 7-43 MB bundle over a microVM's fs bridge sees occasional resets. One reset
        # must not leave the computer without its catalog, so retry the same bytes.
        last = ""
        for attempt in range(4):
            try:
                response = await self.computer._request("PUT", "/fs", params={"path": path},
                                                        content=data, timeout=300)
                if response.status_code in (200, 201, 204):
                    return
                last = f"HTTP {response.status_code}"
            except Exception as exc:  # noqa: BLE001 - transient transport failure
                last = f"{type(exc).__name__}: {str(exc)[:120]}"
            await asyncio.sleep(1.5 * (attempt + 1))
        raise CapabilityError(f"PandaStack file transfer failed after retries ({last})")

    async def run(self, command: str, timeout: float = 300) -> Dict[str, Any]:
        return _normalise(await self.computer._exec(command, timeout=timeout))


class _LabTarget:
    """Adapts the malware lab's sandbox to the same operations."""

    def __init__(self, lab: Any, sandbox_id: str):
        self.lab, self.sid = lab, sandbox_id

    async def put(self, path: str, data: bytes) -> None:
        await self.lab._put(path, data, sid=self.sid)

    async def run(self, command: str, timeout: float = 300) -> Dict[str, Any]:
        return _normalise(await self.lab._exec(command, timeout=timeout, sid=self.sid))


class CapabilityManager:
    """Builds, publishes and restores the drive-hosted capability catalog."""

    def __init__(self, repo: str, token: str, *, cache_dir: Optional[str] = None,
                 fetch_repo: Optional[Callable[[str], bytes]] = None, http: Any = None):
        self.repo = repo
        self.token = token
        self.cache_dir = pathlib.Path(cache_dir or tempfile.gettempdir()) / "blackthorn-capabilities"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fetch_repo = fetch_repo
        self._http = http
        self._bundle: Optional[bytes] = None
        self._version = ""
        self.last_status: Dict[str, Any] = {}

    # ------------------------------------------------------------------ drive
    def _drive(self):
        from .computer_drive import GitHubComputerDrive
        return GitHubComputerDrive(self.repo, self.token)

    def _fetch(self, repo: str) -> bytes:
        if self._fetch_repo is not None:
            return self._fetch_repo(repo)
        import httpx
        url = f"https://codeload.github.com/{repo}/zip/refs/heads/main"
        with httpx.Client(timeout=httpx.Timeout(60, read=300), follow_redirects=True) as client:
            response = client.get(url)
            if response.status_code == 404:
                response = client.get(f"https://codeload.github.com/{repo}/zip/refs/heads/master")
            response.raise_for_status()
            return response.content

    def source_stamps(self) -> Dict[str, str]:
        """Cheap upstream check: the head commit of each repository."""
        if self._fetch_repo is not None:
            return {name: "fixture" for name in SOURCES}
        import httpx
        out: Dict[str, str] = {}
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        with httpx.Client(timeout=30, headers=headers) as client:
            for name, meta in SOURCES.items():
                try:
                    response = client.get(f"https://api.github.com/repos/{meta['repo']}/commits",
                                          params={"per_page": 1})
                    out[name] = response.json()[0]["sha"] if response.status_code == 200 else "unknown"
                except Exception:  # noqa: BLE001 - an unreachable repo must not block restore
                    out[name] = "unknown"
        return out

    def published(self) -> Dict[str, Any]:
        """Metadata for the bundle already in the drive (empty when never published)."""
        target = self.cache_dir / "latest.json"
        try:
            with self._drive() as drive:
                drive.download(DRIVE_META, target)
            return json.loads(pathlib.Path(target).read_text())
        except Exception:  # noqa: BLE001 - absent metadata simply means "not published yet"
            return {}

    def publish(self, *, force: bool = False) -> Dict[str, Any]:
        """Build the bundle if upstream changed, upload it, and report what is live."""
        stamps = self.source_stamps()
        current = self.published()
        if not force and current.get("sources") == stamps and current.get("version"):
            self._version = str(current["version"])
            return {"changed": False, **current}
        bundle = build_bundle(self._fetch)
        version = version_of(bundle, stamps)
        if not force and current.get("version") == version:
            self._version = version
            return {"changed": False, **current}
        local = self.cache_dir / "catalog.tar.gz"
        local.write_bytes(bundle)
        meta = {"version": version, "sources": stamps, "bytes": len(bundle),
                "built_at": time.time(), "path": DRIVE_BUNDLE}
        meta_local = self.cache_dir / "latest.json"
        meta_local.write_text(json.dumps(meta, indent=1))
        with self._drive() as drive:
            drive.upload(str(local), DRIVE_BUNDLE)
            drive.upload(str(meta_local), DRIVE_META)
        self._bundle, self._version = bundle, version
        self.last_status = {"changed": True, **meta}
        return {"changed": True, **meta}

    # ------------------------------------------------------------------ restore
    async def ensure(self, target: Any, *, force: bool = False, kind: str = "computer") -> Dict[str, Any]:
        """Restore and verify the catalog on a computer or lab VM.

        Idempotent: when the VM already carries the published version, only a fast,
        real verification command runs.
        """
        adapter = target if hasattr(target, "put") and hasattr(target, "run") else None
        if adapter is None:
            adapter = _LabTarget(target[0], target[1]) if isinstance(target, tuple) else _ComputerTarget(target)
        meta = self.published()
        version = str(meta.get("version") or "")
        if not version:
            # Nothing published yet: build once so every VM afterwards is a download.
            meta = self.publish()
            version = str(meta.get("version") or "")
        if not version:
            raise CapabilityError("no capability bundle could be published to the drive")
        marker = await adapter.run(f"cat {VERSION_FILE} 2>/dev/null || true", timeout=60)
        present = str(marker.get("stdout") or "").strip()
        if not force and present == version:
            verify = await self.verify(adapter, timeout=180)
            return {**verify, "restored": False, "version": version, "kind": kind}
        local = self.cache_dir / "catalog.tar.gz"
        if not local.exists() or self._version != version:
            try:
                with self._drive() as drive:
                    drive.download(DRIVE_BUNDLE, local)
            except Exception as exc:  # noqa: BLE001
                raise CapabilityError("the drive did not return the capability bundle") from exc
            self._version = version
        bundle = local.read_bytes()
        await adapter.run(f"mkdir -p {CATALOG_DIR} {BIN_DIR}", timeout=60)
        await adapter.put(BUNDLE_PATH, bundle)
        # The bundle's members are the three source repos at the top level, so it must be
        # extracted *into* the catalog root. Extracting it beside the root left the skills
        # under /opt/blackthorn/<repo> and made the VERSION write fail on a fresh VM.
        extracted = await adapter.run(
            f"rm -rf {CATALOG_DIR} && mkdir -p {CATALOG_DIR} && "
            f"tar xzf {BUNDLE_PATH} -C {CATALOG_DIR} && "
            f"ls {CATALOG_DIR} >/dev/null && "
            f"printf '%s' '{version}' > {VERSION_FILE} && chmod -R a+rX {CATALOG_DIR} && "
            f"du -sh {CATALOG_DIR} >/dev/null && echo restored", timeout=600)
        if "restored" not in str(extracted.get("stdout") or ""):
            raise CapabilityError("the catalog could not be extracted: " +
                                  str(extracted.get("output") or "")[:200])
        await adapter.put(BIN_DIR + "/skillctl.py", _skillctl_source())
        await adapter.run("chmod 755 " + BIN_DIR + "/skillctl.py", timeout=60)
        verify = await self.verify(adapter, timeout=300)
        return {**verify, "restored": True, "version": version, "kind": kind, "bytes": len(bundle)}

    async def verify(self, adapter: Any, *, timeout: float = 300) -> Dict[str, Any]:
        """Prove the restored catalog works: index it and run a real search."""
        index = await adapter.run(
            f"BLACKTHORN_SKILL_ROOT={CATALOG_DIR} python3 {BIN_DIR}/skillctl.py registry", timeout=timeout)
        counts: Dict[str, Any] = {}
        raw = str(index.get("stdout") or "").strip().splitlines()
        for line in reversed(raw):
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict) and "counts" in parsed:
                counts = parsed["counts"]
                break
        search = await adapter.run(
            f"BLACKTHORN_SKILL_ROOT={CATALOG_DIR} python3 {BIN_DIR}/skillctl.py search "
            f"--q 'malware analysis' --limit 3 --kind all", timeout=min(timeout, 240))
        found: List[str] = []
        for line in reversed(str(search.get("stdout") or "").strip().splitlines()):
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict) and "results" in parsed:
                found = [str(r.get("id")) for r in parsed["results"]]
                break
        ok = bool(counts.get("skills")) and bool(counts.get("tools")) and bool(found)
        result = {"ok": ok, "skills": counts.get("skills", 0), "tools": counts.get("tools", 0),
                  "search_results": found[:5]}
        if not ok:
            result["error"] = (str(index.get("output") or "")[:200] or "the catalog did not index")
        self.last_status = result
        return result

    # ------------------------------------------------------------------ artifacts
    def upload_artifact(self, source: str, logical: str) -> Dict[str, Any]:
        """Store a task artifact in the drive (never only on the disposable VM)."""
        with self._drive() as drive:
            return drive.upload(source, logical)


def _skillctl_source() -> bytes:
    """The catalog reader shipped with the agent, restored next to the catalog."""
    here = pathlib.Path(__file__).resolve().parent / "computer_assets" / "skillctl.py"
    return here.read_bytes()


def from_env(http: Any = None, store: Any = None) -> Optional[CapabilityManager]:
    repo = (os.environ.get("COMPUTER_DRIVE_GITHUB_REPO") or "jadeniran21/Computer-drive").strip()
    token = (os.environ.get("COMPUTER_DRIVE_GITHUB_PAT") or "").strip()
    if not token:
        return None
    cache = os.environ.get("BLACKTHORN_CAPABILITY_CACHE") or None
    return CapabilityManager(repo, token, cache_dir=cache, http=http)


def script_path() -> str:
    return BIN_DIR + "/skillctl.py"
