"""Inrok tunnel names must not be fixed: a fixed name collides with the share left by the previous kernel (409)."""

import ast
import re
from pathlib import Path

SERVER = Path(__file__).resolve().parents[2] / "kaggle_cyber_ornith" / "cyber_ornith_server.py"


def _source():
    return SERVER.read_text(encoding="utf-8")


def test_default_name_has_a_per_boot_suffix():
    src = _source()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "inrok_tunnel_name")
    body = ast.get_source_segment(src, fn)
    assert "_INROK_BOOT_SUFFIX" in body
    assert 'return "blackthorn"' not in body and 'return "blackthorn-computer"' not in body


def test_boot_suffix_is_a_valid_random_label():
    m = re.search(r"^_INROK_BOOT_SUFFIX = (\S+)", _source(), re.M)
    assert m and "_secrets.token_hex(3)" in m.group(1)
    assert re.match(r"^[a-z0-9-]{1,40}$", "blackthorn-computer-" + "0" * 6)


def test_secrets_module_is_bound_under_the_name_used():
    src = _source()
    assert "import secrets as _secrets" in src
    assert "secrets.token_hex" not in src.replace("_secrets.token_hex", "")
