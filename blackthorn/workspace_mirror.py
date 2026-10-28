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
import logging
import pathlib
import shlex
import tempfile

from .computer_drive import GitHubComputerDrive
from .tools.base import ToolError

SNAPSHOT = "_system/blackthorn-workspace.tar.gz"
DIGEST_KEY = "pandastack_workspace_drive_digest"
RESTORED_KEY = "pandastack_workspace_restored_id"
log = logging.getLogger(__name__)
CHUNK = 16 * 1024 * 1024
SCRATCH = "/workspace/.blackthorn-sync"
# Rebuildable caches stay out of the durable checkpoint: the browser bundle alone is
# ~1.5 GB and would be re-uploaded to GitHub on every mutation. They are rehydrated by
# the tool that needs them (``browse.py::_ensure_runtime``) after a replacement.
EXCLUDES = tuple(x for x in (os.environ.get("BLACKTHORN_WORKSPACE_EXCLUDES")
                             or ".blackthorn-sync,.cache,__pycache__,.npm,_cacache,.tmp").split(",") if x)
MAX_ARCHIVE_BYTES = int(os.environ.get("BLACKTHORN_WORKSPACE_MAX_BYTES") or 1_600 * 1024 * 1024)

FINGERPRINT = r'''python3 - <<'PY'
import hashlib,json,os,stat
SKIP=set(__EXCLUDES__)
items=[]
for root,dirs,files in os.walk('/workspace',followlinks=False):
    dirs[:]=sorted(d for d in dirs if d not in SKIP)
    for name in sorted(files+dirs):
        p=os.path.join(root,name)
        try: s=os.lstat(p)
        except FileNotFoundError: continue
        items.append((os.path.relpath(p,'/workspace'),s.st_mode,s.st_size,s.st_mtime_ns))
print(hashlib.sha256(json.dumps(items,sort_keys=True,separators=(',',':')).encode()).hexdigest())
PY'''.replace("__EXCLUDES__", repr(list(EXCLUDES)))


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
                # A real workspace archive runs to hundreds of megabytes. Pushing the parts
                # one after another over a 120 s-per-request transfer took minutes and looked
                # like a hang, so recovery stalled on every replacement VM. Send a small,
                # bounded number of parts concurrently and log progress: same bytes, same
                # order, several times faster, and the operator can see it moving.
                parts: list[tuple[str, bytes]] = []
                with archive.open("rb") as stream:
                    for i, chunk in enumerate(iter(lambda: stream.read(CHUNK), b"")):
                        parts.append((f"part-{i:05d}", chunk))
                if not parts:
                    raise ToolError("storage", "Empty workspace snapshot")
                total = sum(len(c) for _, c in parts)
                done = 0
                gate = asyncio.Semaphore(4)

                async def push(name: str, chunk: bytes) -> None:
                    nonlocal done
                    async with gate:
                        await self._transfer("PUT", f"{SCRATCH}/{name}", chunk)
                        done += len(chunk)
                        log.info("workspace restore %d%% (%d/%d MiB)", int(done * 100 / total),
                                 done // (1024 * 1024), total // (1024 * 1024))

                await asyncio.gather(*(push(name, chunk) for name, chunk in parts))
                names = [name for name, _ in parts]
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
            fingerprint = (await self._run(FINGERPRINT, timeout=120)).strip().splitlines()[-1]
            previous = await self.computer.store.get_setting(DIGEST_KEY)
            if fingerprint == previous:
                return bool(force and previous)
            excludes = " ".join(f"--exclude='./{name}'" for name in EXCLUDES)
            # ``tar`` exits 1 with "file changed as we read it" whenever a background job,
            # a package install or the browser writes into /workspace while the checkpoint
            # is being taken. That is a *warning about a race*, not a failed backup: the
            # archive is still complete enough to restore, and the digest recorded below
            # describes exactly what was captured. Failing the user's tool call over it
            # would make every long task uncheckpointable.
            script = (f"mkdir -p {SCRATCH} && rm -f {SCRATCH}/part-* {SCRATCH}/current.tar.gz && "
                      f"tar --sort=name --warning=no-file-changed {excludes} -czf {SCRATCH}/current.tar.gz "
                      f"-C /workspace . ; rc=$? ; "
                      f"[ -s {SCRATCH}/current.tar.gz ] && gzip -t {SCRATCH}/current.tar.gz || exit $rc ; "
                      f"exit 0")
            await self._run(script, timeout=600)
            await self._run(f"split -b {CHUNK} -d -a 5 {SCRATCH}/current.tar.gz {SCRATCH}/part-", timeout=300)
            size = int((await self._run(f"stat -c %s {SCRATCH}/current.tar.gz", timeout=60)).strip() or 0)
            if size > MAX_ARCHIVE_BYTES:
                raise ToolError("storage", f"Workspace checkpoint is {size} bytes, over the "
                                           f"{MAX_ARCHIVE_BYTES} byte ceiling; move large assets out of /workspace")
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
