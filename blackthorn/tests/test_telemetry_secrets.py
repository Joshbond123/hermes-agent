"""Regression guard: the public ntfy telemetry topic must never receive the gateway key."""

import ast
from pathlib import Path

SERVER = Path(__file__).resolve().parents[2] / "kaggle_cyber_ornith" / "cyber_ornith_server.py"


def _notify_workspace():
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "notify_workspace":
            return node
    raise AssertionError("notify_workspace not found")


def test_ntfy_payload_literal_has_no_api_key():
    fn = _notify_workspace()
    for node in ast.walk(fn):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                assert not (isinstance(key, ast.Constant) and key.value == "api_key"), \
                    "the payload dict must not carry api_key (ntfy topic is public)"


def test_api_key_is_stripped_before_publishing():
    src = ast.get_source_segment(SERVER.read_text(encoding="utf-8"), _notify_workspace())
    assert 'payload.pop("api_key", None)' in src
    assert src.index('payload.pop("api_key"') < src.index("https://ntfy.sh/")


def test_d1_write_still_stores_the_key():
    src = ast.get_source_segment(SERVER.read_text(encoding="utf-8"), _notify_workspace())
    assert "INSERT OR REPLACE INTO kaggle_gpu_state" in src and "API_KEY" in src
