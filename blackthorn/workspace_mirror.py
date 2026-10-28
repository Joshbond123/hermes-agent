"""Durable workspace checkpoints via private GitHub release assets.

The VM is a temporary execution cache, not a GitHub mount. After each foreground
computer mutation, snapshot /workspace (including workspace-local pip/npm packages)
to one versioned GitHub object. A replacement restores the latest verified object
before executing a tool. Transfers use 16 MiB guest files so the PandaStack /fs
endpoint never has to return a multi-gigabyte response in one allocation.
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import shlex
import tempfile

from .computer_drive import GitHubComputerDrive
from .tools.base import ToolError

SNAPSHOT = "_system/blackthorn-workspace.tar.gz"
DIGEST_KEY = "pandastack_workspace_drive_digest"
RESTORED_KEY = "pandastack_workspace_restored_id"
CHUNK = 16 * 1024 * 1024
SCRATCH = "/workspace/.blackthorn-sync"

FINGERPRINT = r'''python3 - <<'PY'
import hashlib,json,os,stat
items=[]
for root,dirs,files in os.walk('/workspace',followlinks=False):
    dirs[:]=sorted(d for d in dirs if not (root == '/workspace' and d == '.blackthorn-sync'))
    for name in sorted(files+dirs):
        p=os.path.join(root,name)
        try: s=os.lstat(p)
        except FileNotFoundError: continue
        items.append((os.path.relpath(p,'/workspace'),s.st_mode,s.st_size,s.st_mtime_ns))
print(hashlib.sha256(json.dumps(items,sort_keys=True,separators=(',',':')).encode()).hexdigest())
PY'''


class WorkspaceMirror:
    def __init__(self, computer):
        self.computer = computer
        self._lock = asyncio.Lock()

    async def _run(self, cmd: str, timeout: int = 120) -> str:
        result = await self.computer._exec(cmd, timeout=timeout)
        if not result.get("ok"):
            raise ToolError("storage", "Workspace checkpoint command failed: " + str(result.get("output") or "")[-160:])
        return str(result.get("output") or "")

    def _drive_upload(self, source: pathlib.Path):
        with GitHubComputerDrive(self.computer.repo, self.computer.github_token) as drive:
            return drive.upload(source, self.computer.snapshot_path)

    def _drive_download(self, target: pathlib.Path):
        with GitHubComputerDrive(self.computer.repo, self.computer.github_token) as drive:
            return drive.download(self.computer.snapshot_path, target)

    async def _transfer(self, method: str, path: str, data: bytes | None = None) -> bytes:
        response = await self.computer.http.request(method, self.computer.base + "/fs",
            headers={"Authorization": "Bearer " + self.computer.key},
            params={"path": path}, content=data, timeout=120)
        if response.status_code not in (200, 201, 204):
            raise ToolError("storage", f"PandaStack workspace transfer returned HTTP {response.status_code}")
        return response.content

    async def restore(self, sid: str, *, force: bool = False) -> None:
        """Extract the newest verified archive into ``sid``.

        ``force=True`` bypasses the "already restored into this sandbox" shortcut so a
        freshly created handover sandbox receives the checkpoint even when the stored
        marker still names it (a marker can be stale after a crash mid-restore).
        """
        if not self.computer.store:
            return
        if not force and await self.computer.store.get_setting(RESTORED_KEY) == sid:
            return
        async with self._lock:
            if not force and await self.computer.store.get_setting(RESTORED_KEY) == sid:
                return
            with tempfile.TemporaryDirectory() as temp:
                archive = pathlib.Path(temp) / "workspace.tar.gz"
                try:
                    await asyncio.to_thread(self._drive_download, archive)
                except FileNotFoundError:
                    await self.computer.store.set_setting(RESTORED_KEY, sid)
                    return
                await self._run("mkdir -p " + shlex.quote(SCRATCH))
                names = []
                with archive.open("rb") as stream:
                    for i, chunk in enumerate(iter(lambda: stream.read(CHUNK), b"")):
                        name = f"part-{i:05d}"
                        await self._transfer("PUT", f"{SCRATCH}/{name}", chunk)
                        names.append(name)
                if not names:
                    raise ToolError("storage", "Empty workspace snapshot")
                await self._run(f"cat {SCRATCH}/part-* | tar -xzf - -C /workspace && rm -f {SCRATCH}/part-*", timeout=300)
            await self.computer.store.set_setting(RESTORED_KEY, sid)

    async def sync(self, *, force: bool = False) -> bool:
        """Return True only if the workspace is durably published.

        ``force=True`` is used by the lifecycle manager before a handover: an unchanged
        workspace whose digest is already published counts as checkpointed (DIGEST_KEY is
        only written after a successful upload), so no redundant gigabytes are pushed.
        """
        if not self.computer.store:
            return False
        async with self._lock:
            fingerprint = (await self._run(FINGERPRINT, timeout=90)).strip().splitlines()[-1]
            previous = await self.computer.store.get_setting(DIGEST_KEY)
            if fingerprint == previous:
                return bool(force and previous)
            await self._run(f"mkdir -p {SCRATCH} && rm -f {SCRATCH}/part-* {SCRATCH}/current.tar.gz && "
                            f"tar --sort=name --exclude='./.blackthorn-sync' -czf {SCRATCH}/current.tar.gz "
                            f"-C /workspace . && split -b {CHUNK} -d -a 5 {SCRATCH}/current.tar.gz {SCRATCH}/part-",
                            timeout=300)
            with tempfile.TemporaryDirectory() as temp:
                archive = pathlib.Path(temp) / "workspace.tar.gz"
                with archive.open("wb") as dst:
                    i = 0
                    while True:
                        name = f"{SCRATCH}/part-{i:05d}"
                        response = await self.computer.http.get(self.computer.base + "/fs",
                            headers={"Authorization": "Bearer " + self.computer.key},
                            params={"path": name}, timeout=120)
                        if response.status_code == 404:
                            break
                        if response.status_code != 200:
                            raise ToolError("storage", f"PandaStack workspace download returned HTTP {response.status_code}")
                        dst.write(response.content)
                        i += 1
                    if not i:
                        raise ToolError("storage", "Workspace checkpoint has no chunks")
                await asyncio.to_thread(self._drive_upload, archive)
            await self.computer.store.set_setting(DIGEST_KEY, fingerprint)
            await self.computer.store.set_setting(RESTORED_KEY, self.computer.sandbox_id)
            await self._run(f"rm -f {SCRATCH}/part-* {SCRATCH}/current.tar.gz")
            return True
