"""Checkpoint visibility is transactional; unchanged work skips GitHub uploads."""
import pathlib
from types import SimpleNamespace

import httpx
import pytest

from blackthorn.workspace_mirror import DIGEST_KEY, RESTORED_KEY, WorkspaceMirror


class Settings:
    def __init__(self): self.values = {}
    async def get_setting(self, key): return self.values.get(key, '')
    async def set_setting(self, key, value): self.values[key] = value


@pytest.mark.asyncio
async def test_checkpoint_upload_once_and_restore_after_replacement():
    store = Settings()
    sent = []
    class HTTP:
        async def get(self, url, **kw):
            path = kw['params']['path']
            return httpx.Response(200, content=b'archive-bytes') if path.endswith('part-00000') else httpx.Response(404)
    computer = SimpleNamespace(store=store, http=HTTP(), base='https://api.pandastack.ai/v1/sandboxes/test',
                               key='test', repo='test/repo', github_token='test', sandbox_id='old',
                               snapshot_path='_system/workspace.tar.gz')
    mirror = WorkspaceMirror(computer)
    async def run(cmd, timeout=120):
        return 'a' * 64 + '\n' if 'hashlib.sha256' in cmd else ''
    mirror._run = run
    mirror._drive_upload = lambda path: sent.append(path.read_bytes())
    assert await mirror.sync() is True
    assert sent == [b'archive-bytes'] and await store.get_setting(DIGEST_KEY) == 'a' * 64
    assert await mirror.sync() is False and len(sent) == 1
    computer.sandbox_id = 'replacement'
    mirror._drive_download = lambda path: path.write_bytes(sent[0])
    pieces = []
    async def transfer(method, path, data=None): pieces.append((method, data))
    mirror._transfer = transfer
    await mirror.restore('replacement')
    assert pieces == [('PUT', b'archive-bytes')]
    assert await store.get_setting(RESTORED_KEY) == 'replacement'


@pytest.mark.asyncio
async def test_failed_publish_does_not_claim_durability():
    store = Settings()
    class HTTP:
        async def get(self, url, **kw):
            return httpx.Response(200, content=b'archive') if kw['params']['path'].endswith('part-00000') else httpx.Response(404)
    computer = SimpleNamespace(store=store, http=HTTP(), base='https://example.test/v1/sandboxes/test',
                               key='test', repo='test/repo', github_token='test', sandbox_id='vm',
                               snapshot_path='_system/workspace.tar.gz')
    mirror = WorkspaceMirror(computer)
    async def run(cmd, timeout=120):
        return 'b' * 64 + '\n' if 'hashlib.sha256' in cmd else ''
    mirror._run = run
    def fail(_): raise RuntimeError('mock upload rejected')
    mirror._drive_upload = fail
    with pytest.raises(RuntimeError): await mirror.sync()
    assert await store.get_setting(DIGEST_KEY) == ''
    assert await store.get_setting(RESTORED_KEY) == ''
