"""Snapshot contract: changed/new/deleted/unchanged, corruption and interrupted retries."""
import ast
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / 'kaggle_cyber_ornith' / 'cyber_ornith_server.py'


def snapshot(tmp_path):
    names = {'_snapshot_entries', '_snapshot_digest', '_restore_snapshot'}
    tree = ast.parse(SOURCE.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    ws = tmp_path / 'computer'
    ws.mkdir()
    scope = {'os': os, 'json': json, 'shutil': shutil, 'hashlib': hashlib,
             'Path': Path, 'COMPUTER_WORKSPACE': str(ws)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope, ws


def make_source(scope, folder, generation):
    folder.mkdir(parents=True, exist_ok=True)
    entries = scope['_snapshot_entries'](str(folder))
    digest = scope['_snapshot_digest'](entries)
    marker = folder / '.blackthorn' / 'checkpoint.json'
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({'schema': 2, 'generation': generation, 'files': entries, 'digest': digest}))
    return digest


def test_incremental_reconciliation_and_deletion(tmp_path):
    scope, workspace = snapshot(tmp_path)
    first = tmp_path / 'first'
    (first / 'uploads').mkdir(parents=True)
    for name, content in {'keep.txt': 'unchanged', 'modified.txt': 'old', 'deleted.txt': 'remove'}.items():
        (first / 'uploads' / name).write_text(content)
    make_source(scope, first, 1)
    assert scope['_restore_snapshot'](str(first)) == (3, 3)
    original_stat = (workspace / 'uploads' / 'keep.txt').stat().st_mtime_ns
    (workspace / 'untracked.txt').write_text('preserve local work')
    second = tmp_path / 'second'
    (second / 'uploads').mkdir(parents=True)
    for name, content in {'keep.txt': 'unchanged', 'modified.txt': 'new', 'created.txt': 'created'}.items():
        (second / 'uploads' / name).write_text(content)
    digest = make_source(scope, second, 2)
    assert scope['_restore_snapshot'](str(second)) == (2, 3)
    assert (workspace / 'uploads' / 'keep.txt').stat().st_mtime_ns == original_stat
    assert (workspace / 'uploads' / 'modified.txt').read_text() == 'new'
    assert (workspace / 'uploads' / 'created.txt').read_text() == 'created'
    assert not (workspace / 'uploads' / 'deleted.txt').exists()
    assert (workspace / 'untracked.txt').read_text() == 'preserve local work'
    assert json.loads((workspace / '.blackthorn' / 'checkpoint.json').read_text())['digest'] == digest
    assert scope['_restore_snapshot'](str(second)) == (0, 3)


def test_corrupt_and_interrupted_snapshot_is_not_committed_and_retry_succeeds(tmp_path, monkeypatch):
    scope, workspace = snapshot(tmp_path)
    initial = tmp_path / 'initial'
    initial.mkdir()
    (initial / 'old.txt').write_text('retained until verified')
    make_source(scope, initial, 1)
    scope['_restore_snapshot'](str(initial))
    update = tmp_path / 'update'
    update.mkdir()
    (update / 'new.txt').write_text('new bytes')
    make_source(scope, update, 2)
    (update / 'new.txt').write_text('corrupt bytes')
    with pytest.raises(ValueError, match='mismatch'):
        scope['_restore_snapshot'](str(update))
    assert (workspace / 'old.txt').exists() and not (workspace / 'new.txt').exists()
    assert json.loads((workspace / '.blackthorn' / 'checkpoint.json').read_text())['generation'] == 1
    (update / 'new.txt').write_text('new bytes')
    original = shutil.copy2
    def interrupted(src, dst, *args, **kwargs):
        if str(dst).endswith('.checkpoint-part'):
            raise OSError('interrupted copy')
        return original(src, dst, *args, **kwargs)
    monkeypatch.setattr(shutil, 'copy2', interrupted)
    with pytest.raises(OSError):
        scope['_restore_snapshot'](str(update))
    assert (workspace / 'old.txt').exists() and not (workspace / 'new.txt').exists()
    monkeypatch.setattr(shutil, 'copy2', original)
    assert scope['_restore_snapshot'](str(update)) == (1, 1)
    assert not (workspace / 'old.txt').exists()
    assert (workspace / 'new.txt').read_text() == 'new bytes'


def test_all_user_files_deleted_do_not_reappear(tmp_path):
    scope, workspace = snapshot(tmp_path)
    old = tmp_path / 'old'
    old.mkdir()
    (old / 'last-user-file.txt').write_text('old')
    make_source(scope, old, 1)
    scope['_restore_snapshot'](str(old))
    empty = tmp_path / 'empty'
    empty.mkdir()
    make_source(scope, empty, 2)
    assert scope['_restore_snapshot'](str(empty)) == (0, 0)
    assert not (workspace / 'last-user-file.txt').exists()
    assert scope['_restore_snapshot'](str(empty)) == (0, 0)


def test_nested_symlink_cannot_redirect_restore_outside_workspace(tmp_path):
    scope, ws = snapshot(tmp_path)
    stage = tmp_path / 'stage'
    (stage / 'nested').mkdir(parents=True)
    (stage / 'nested' / 'payload.txt').write_text('payload')
    make_source(scope, stage, 1)
    (ws / 'nested').symlink_to(tmp_path / 'outside', target_is_directory=True)
    (tmp_path / 'outside').mkdir()
    with pytest.raises(ValueError, match='unsafe destination'):
        scope['_restore_snapshot'](str(stage))
    assert not (tmp_path / 'outside' / 'payload.txt').exists()
