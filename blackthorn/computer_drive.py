"""Private GitHub release-asset store for versioned computer workspace files.

This is an object store, NOT a POSIX filesystem or a replacement for a local
working directory. Only published releases with verified manifests are visible.
Never put credentials or sensitive content in release names, logs, or URLs.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import tempfile
import uuid
from typing import Iterator
from urllib.parse import quote

import httpx

MARKER = "blackthorn-computer-drive-v1"
MAX_CHUNK_BYTES = 128 * 1024 * 1024  # under GitHub's 2 GiB per-release-asset limit
MAX_ASSETS = 1000


class ComputerDriveError(RuntimeError):
    pass


def _safe_path(path: str) -> str:
    # Logical paths only: never allow absolute paths, traversal, or backslashes.
    if not path or path.startswith("/") or "\\" in path or any(
        p in ("", ".", "..") for p in path.split("/")
    ) or len(path) > 1024:
        raise ValueError("unsafe logical path")
    return path


class GitHubComputerDrive:
    def __init__(self, repo: str, token: str | None = None, chunk_bytes: int = MAX_CHUNK_BYTES):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("invalid GitHub owner/repo")
        self.token = token or os.environ.get("COMPUTER_DRIVE_GITHUB_PAT")
        if not self.token:
            raise ComputerDriveError("missing COMPUTER_DRIVE_GITHUB_PAT")
        if not 1 <= chunk_bytes < 2 * 1024**3:
            raise ValueError("chunk must be smaller than GitHub's release asset limit")
        self.repo = repo
        self.chunk_bytes = chunk_bytes
        self.api = f"https://api.github.com/repos/{repo}"
        self.client = httpx.Client(timeout=httpx.Timeout(30, read=300, write=300), headers={
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }, follow_redirects=False)

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def _check(response: httpx.Response, expected: tuple[int, ...]) -> dict:
        if response.status_code not in expected:
            # GitHub errors sometimes echo request data; don't print response bodies.
            raise ComputerDriveError(f"GitHub request returned HTTP {response.status_code}")
        if response.status_code == 204:
            return {}
        return response.json()

    def _releases(self) -> Iterator[dict]:
        page = 1
        while True:
            response = self.client.get(self.api + "/releases", params={"per_page": 100, "page": page})
            items = self._check(response, (200,))
            if not isinstance(items, list):
                raise ComputerDriveError("unexpected release listing")
            yield from items
            if len(items) < 100:
                return
            page += 1

    @staticmethod
    def _manifest(release: dict) -> dict | None:
        try:
            manifest = json.loads(release.get("body") or "")
            if manifest.get("format") != MARKER or release.get("draft"):
                return None
            if not isinstance(manifest["chunks"], list):
                return None
            _safe_path(manifest["path"])
            return manifest
        except (ValueError, KeyError, TypeError):
            return None

    def list_files(self) -> dict[str, dict]:
        records = {}
        for release in self._releases():
            manifest = self._manifest(release)
            if manifest and manifest["path"] not in records:
                records[manifest["path"]] = {
                    "size": manifest["size"], "sha256": manifest["sha256"],
                    "release_id": release["id"], "chunks": len(manifest["chunks"]),
                }
        return records

    def _find(self, path: str) -> tuple[dict, dict]:
        for release in self._releases():
            manifest = self._manifest(release)
            if manifest and manifest["path"] == path:
                return release, manifest
        raise FileNotFoundError(path)

    @staticmethod
    def _reader(source: pathlib.Path, offset: int, length: int):
        with source.open("rb") as file:
            file.seek(offset)
            remaining = length
            while remaining:
                block = file.read(min(1024 * 1024, remaining))
                if not block:
                    raise ComputerDriveError("source changed during upload")
                remaining -= len(block)
                yield block

    def upload(self, source: str | os.PathLike, destination: str) -> dict:
        path = _safe_path(destination)
        source = pathlib.Path(source)
        initial_size = source.stat().st_size
        count = (initial_size + self.chunk_bytes - 1) // self.chunk_bytes
        if count > MAX_ASSETS:
            raise ComputerDriveError("file exceeds 1000 assets per GitHub release")
        tag = "computer-drive-" + uuid.uuid4().hex
        draft = self._check(self.client.post(self.api + "/releases", json={
            "tag_name": tag, "name": tag, "body": "Upload pending; not visible to readers",
            "draft": True, "prerelease": True,
        }), (201,))
        release_id = draft["id"]
        chunks = []
        whole = hashlib.sha256()
        try:
            # Draft asset uploads are not exposed by list_files until publication.
            for i in range(count):
                length = min(self.chunk_bytes, max(0, initial_size - i * self.chunk_bytes))
                name = f"chunk-{i:06d}.bin"
                digest = hashlib.sha256()

                def data():
                    for block in self._reader(source, i * self.chunk_bytes, length):
                        digest.update(block)
                        whole.update(block)
                        yield block

                upload_url = draft["upload_url"].split("{", 1)[0]
                response = self.client.post(upload_url, params={"name": name},
                    headers={"Content-Type": "application/octet-stream", "Content-Length": str(length)},
                    content=data())
                asset = self._check(response, (201,))
                if asset.get("size") != length or asset.get("state") != "uploaded":
                    raise ComputerDriveError("asset was not confirmed uploaded")
                chunks.append({"asset_id": asset["id"], "size": length, "sha256": digest.hexdigest()})
            if source.stat().st_size != initial_size:
                raise ComputerDriveError("source changed during upload")
            manifest = {"format": MARKER, "path": path, "size": initial_size,
                        "sha256": whole.hexdigest(), "chunks": chunks}
            published = self._check(self.client.patch(self.api + f"/releases/{release_id}", json={
                "body": json.dumps(manifest, separators=(",", ":")), "draft": False,
            }), (200,))
            if published.get("draft") or self._manifest(published) != manifest:
                raise ComputerDriveError("publication not verified")
            return {"path": path, "size": initial_size, "sha256": whole.hexdigest(),
                    "release_id": release_id, "chunks": count}
        except Exception:
            # Reconcile ambiguous failures: only delete if GitHub confirms this
            # exact release is still a draft. Never delete a published object.
            try:
                check = self.client.get(self.api + f"/releases/{release_id}")
                if check.status_code == 200 and check.json().get("draft") is True:
                    self.client.delete(self.api + f"/releases/{release_id}")
            except (httpx.HTTPError, ValueError):
                pass  # Leave the draft for manual recovery if API is unavailable.
            raise

    def remove(self, path: str) -> int:
        """Delete all versions of one logical path. This is irreversible."""
        path = _safe_path(path)
        releases = [r for r in self._releases()
                    if (m := self._manifest(r)) and m["path"] == path]
        for release in releases:
            self._check(self.client.delete(self.api + f"/releases/{release['id']}"), (204,))
            tag = release.get("tag_name", "")
            if tag.startswith("computer-drive-"):
                response = self.client.delete(self.api + "/git/refs/tags/" + quote(tag, safe=""))
                if response.status_code not in (204, 404):
                    raise ComputerDriveError(f"tag removal returned HTTP {response.status_code}")
        return len(releases)

    def _asset_blocks(self, asset_id: int):
        url = self.api + f"/releases/assets/{asset_id}"
        with self.client.stream("GET", url, headers={"Accept": "application/octet-stream"}) as response:
            if response.status_code == 302:
                location = response.headers.get("location", "")
                if not location.startswith("https://"):
                    raise ComputerDriveError("unsafe release asset redirect")
                # A signed URL is fetched without forwarding the GitHub PAT.
                with httpx.Client(timeout=httpx.Timeout(30, read=300), follow_redirects=False) as downloader:
                    with downloader.stream("GET", location) as signed:
                        if signed.status_code != 200:
                            raise ComputerDriveError(f"asset download returned HTTP {signed.status_code}")
                        yield from signed.iter_bytes(chunk_size=1024 * 1024)
            else:
                if response.status_code != 200 or "json" in response.headers.get("content-type", ""):
                    raise ComputerDriveError(f"asset download returned HTTP {response.status_code}")
                yield from response.iter_bytes(chunk_size=1024 * 1024)

    def download(self, path: str, target: str | os.PathLike) -> dict:
        path = _safe_path(path)
        release, manifest = self._find(path)
        assets = {a["id"]: a for a in release.get("assets", [])}
        target = pathlib.Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".computer-drive-", delete=False) as out:
                temp_name = out.name
                whole = hashlib.sha256()
                total = 0
                for part in manifest["chunks"]:
                    asset = assets.get(part["asset_id"])
                    if not asset or asset.get("size") != part["size"]:
                        raise ComputerDriveError("missing or truncated release asset")
                    digest = hashlib.sha256()
                    downloaded = 0
                    for block in self._asset_blocks(part["asset_id"]):
                        downloaded += len(block)
                        if downloaded > part["size"]:
                            raise ComputerDriveError("release asset length mismatch")
                        digest.update(block)
                        whole.update(block)
                        out.write(block)
                    if downloaded != part["size"] or digest.hexdigest() != part["sha256"]:
                        raise ComputerDriveError("release asset checksum mismatch")
                    total += downloaded
                if total != manifest["size"] or whole.hexdigest() != manifest["sha256"]:
                    raise ComputerDriveError("whole-file checksum mismatch")
            os.replace(temp_name, target)
            temp_name = None
            return {"path": path, "size": total, "sha256": whole.hexdigest()}
        finally:
            if temp_name:
                pathlib.Path(temp_name).unlink(missing_ok=True)


def main(argv=None) -> int:
    """Small explicit file interface; credentials are read from the environment."""
    import argparse
    parser = argparse.ArgumentParser(description="Private GitHub computer object store")
    parser.add_argument("--repo", default=os.environ.get("COMPUTER_DRIVE_GITHUB_REPO", "jadeniran21/Computer-drive"))
    actions = parser.add_subparsers(dest="action", required=True)
    up = actions.add_parser("upload")
    up.add_argument("local_path")
    up.add_argument("logical_path")
    down = actions.add_parser("download")
    down.add_argument("logical_path")
    down.add_argument("local_path")
    actions.add_parser("list")
    rm = actions.add_parser("remove")
    rm.add_argument("logical_path")
    args = parser.parse_args(argv)
    with GitHubComputerDrive(args.repo) as drive:
        if args.action == "upload":
            result = drive.upload(args.local_path, args.logical_path)
        elif args.action == "download":
            result = drive.download(args.logical_path, args.local_path)
        elif args.action == "list":
            result = drive.list_files()
        else:
            result = {"removed_versions": drive.remove(args.logical_path)}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
