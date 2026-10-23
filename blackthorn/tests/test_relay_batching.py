"""Exercise the exact self-contained relay code bundled into the Kaggle notebook."""
import ast
import queue
import threading
import time
from pathlib import Path

import pytest


def _batched():
    server = Path(__file__).resolve().parents[2] / 'kaggle_cyber_ornith/cyber_ornith_server.py'
    tree = ast.parse(server.read_text())
    code = next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RELAY_CLIENT_CODE' for t in node.targets))
    helper = next(node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef) and node.name == 'batched')
    module = ast.Module(body=[helper], type_ignores=[])
    namespace = {'queue': queue, 'threading': threading, 'time': time}
    exec(compile(module, '<relay-batched>', 'exec'), namespace)
    return namespace['batched']


def test_model_bursts_are_coalesced_without_losing_sse_bytes():
    batched = _batched()
    original = [f'data: {{"n":{i}}}\n\n'.encode() for i in range(200)]
    chunks = list(batched(lambda: iter(original), 'model'))
    assert b''.join(chunks) == b''.join(original)
    assert len(chunks) < len(original) / 4
    assert max(map(len, chunks)) <= 8192 + max(map(len, original))


def test_first_token_is_not_delayed_past_batch_window():
    batched = _batched()
    def source():
        yield b'data: first\n\n'
        time.sleep(0.17)
        yield b'data: second\n\n'
    start = time.monotonic()
    stream = batched(source, 'model')
    try:
        assert next(stream) == b'data: first\n\n'
        assert time.monotonic() - start < 0.13
        assert b''.join(stream) == b'data: second\n\n'
    finally:
        stream.close()


def test_large_computer_response_uses_sub_megabyte_requests():
    batched = _batched()
    original = [bytes([i % 251]) * 65536 for i in range(48)]
    chunks = list(batched(lambda: iter(original), 'computer'))
    assert b''.join(chunks) == b''.join(original)
    assert len(chunks) <= 8
    assert all(0 < len(chunk) <= 1_000_000 for chunk in chunks)


def test_reader_exception_is_not_silently_treated_as_success():
    def source():
        yield b'first'
        raise RuntimeError('broken stream')
    with pytest.raises(RuntimeError, match='broken stream'):
        list(_batched()(source, 'model'))
