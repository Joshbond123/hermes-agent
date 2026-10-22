"""Static triage tests use benign synthetic headers, never live malware."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'computer_assets' / 'sample_triage.py'


@pytest.fixture
def quarantine(tmp_path):
    directory = tmp_path / 'malware_samples'
    directory.mkdir()
    env = dict(os.environ, BLACKTHORN_WORKSPACE=str(tmp_path))
    def call(*args):
        proc = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True,
                              timeout=10, env=env)
        return proc.returncode, json.loads(proc.stdout)
    return directory, call


def test_preflight_refuses_to_claim_dynamic_vm(quarantine):
    _, call = quarantine
    code, result = call('--probe')
    assert code == 0 and result['ok'] and not result['dynamic_execution_allowed']
    assert not result['windows_vm_ready'] and not result['linux_vm_ready']


def test_benign_windows_header_is_classified_without_execution(quarantine):
    root, call = quarantine
    payload = bytearray(512)
    payload[:2] = b'MZ'
    payload[0x3c:0x40] = (128).to_bytes(4, 'little')
    payload[128:132] = b'PE\0\0'
    payload[132:134] = (0x8664).to_bytes(2, 'little')
    (root / 'synthetic-windows.bin').write_bytes(payload)
    code, result = call('--analyze', 'synthetic-windows.bin')
    assert code == 0 and result['format'] == 'Windows PE'
    assert result['architecture'] == 'x86-64' and result['bytes'] == 512
    assert not result['dynamic_execution_allowed']


def test_benign_linux_header_is_classified_without_execution(quarantine):
    root, call = quarantine
    payload = bytearray(64)
    payload[:6] = b'\x7fELF\x02\x01'
    payload[18:20] = (0x3e).to_bytes(2, 'little')
    (root / 'synthetic-linux.bin').write_bytes(payload)
    code, result = call('--analyze', 'synthetic-linux.bin')
    assert code == 0 and result['format'] == 'Linux ELF' and result['architecture'] == 'x86-64'


def test_path_traversal_symlink_and_execution_are_refused(quarantine, tmp_path):
    root, call = quarantine
    (tmp_path / 'private.txt').write_text('do not read')
    (root / 'shortcut').symlink_to(tmp_path / 'private.txt')
    for args in [('--analyze', '../private.txt'), ('--analyze', 'shortcut'),
                 ('--execute', 'shortcut'), ('--analyze', '/etc/passwd')]:
        code, result = call(*args)
        assert code == 2 and not result['ok'] and not result['dynamic_execution_allowed']
        assert 'do not read' not in str(result)


def test_oversized_sample_is_rejected_without_reading(quarantine):
    root, call = quarantine
    with (root / 'too-big.bin').open('wb') as fh:
        fh.truncate(32 * 1024 * 1024 + 1)
    code, result = call('--analyze', 'too-big.bin')
    assert code == 2 and 'at most 32 MiB' in result['error']


def test_agent_tool_is_registered_as_static_only_and_rejects_bad_path():
    from blackthorn.tools.security import sample_triage, specs
    import asyncio
    names = {spec.name: spec for spec in specs()}
    assert 'sample_triage' in names and 'Never executes' in names['sample_triage'].description
    result = asyncio.run(sample_triage({'sample_id': '../host'}, None))
    assert not result.ok
