"""Pure unit tests: tool argument validation, path/URL/command safety, redaction."""

from __future__ import annotations

import pytest

from blackthorn.agent import redact as redact_mod
from blackthorn.agent.builtin_tools import check_command, check_url
from blackthorn.agent.kaggle_computer import WORKSPACE_ROOT, normalize_path
from blackthorn.agent.tools import (
    ToolError,
    canonical_key,
    clip,
    parse_arguments,
    validate_arguments,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "minLength": 1, "maxLength": 20},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300, "default": 60},
        "verbose": {"type": "boolean"},
        "mode": {"type": "string", "enum": ["a", "b"]},
    },
    "required": ["command"],
}


# ---- parse_arguments ----------------------------------------------------------
def test_parse_arguments_accepts_json_object_string():
    args, err = parse_arguments('{"command": "ls"}')
    assert err is None and args == {"command": "ls"}


@pytest.mark.parametrize("raw", ["", None, "   "])
def test_parse_arguments_empty_is_empty_object(raw):
    assert parse_arguments(raw) == ({}, None)


def test_parse_arguments_reports_invalid_json():
    args, err = parse_arguments('{"command": "ls"')
    assert args is None and "not valid JSON" in err


def test_parse_arguments_rejects_non_object():
    args, err = parse_arguments("[1, 2]")
    assert args is None and "JSON object" in err


def test_parse_arguments_handles_fenced_and_double_encoded():
    assert parse_arguments('```json\n{"command": "ls"}\n```')[0] == {"command": "ls"}
    assert parse_arguments('"{\\"command\\": \\"ls\\"}"')[0] == {"command": "ls"}


# ---- validate_arguments ------------------------------------------------------
def test_validate_fills_defaults_and_keeps_valid_values():
    clean, errors = validate_arguments(SCHEMA, {"command": "ls"})
    assert errors == [] and clean == {"command": "ls", "timeout_seconds": 60}


def test_validate_coerces_numeric_strings_and_booleans():
    clean, errors = validate_arguments(SCHEMA, {"command": "ls", "timeout_seconds": "30", "verbose": "true"})
    assert errors == [] and clean["timeout_seconds"] == 30 and clean["verbose"] is True


def test_validate_reports_missing_unknown_bounds_and_enum():
    _, errors = validate_arguments(SCHEMA, {"timeout_seconds": 999, "mode": "z", "bogus": 1})
    joined = " | ".join(errors)
    assert "missing required argument 'command'" in joined
    assert "unknown argument 'bogus'" in joined
    assert "must be <= 300" in joined
    assert "must be one of" in joined


def test_validate_rejects_too_long_and_wrong_type():
    _, errors = validate_arguments(SCHEMA, {"command": "x" * 21})
    assert any("too long" in e for e in errors)
    _, errors = validate_arguments(SCHEMA, {"command": ["ls"]})
    assert any("must be a string" in e for e in errors)


def test_canonical_key_is_order_independent():
    assert canonical_key("t", {"a": 1, "b": 2}) == canonical_key("t", {"b": 2, "a": 1})
    assert canonical_key("t", {"a": 1}) != canonical_key("t", {"a": 2})


def test_clip_marks_truncation():
    text, truncated = clip("x" * 500, 300)
    assert truncated and len(text) < 500 and "truncated" in text
    assert clip("short", 300) == ("short", False)


# ---- workspace paths -----------------------------------------------------------
@pytest.mark.parametrize(
    "given,expected",
    [
        ("fib.py", "fib.py"),
        ("./a/b/../c.txt", "a/c.txt"),
        (".", "."),
        ("", "."),
        (WORKSPACE_ROOT, "."),
        (f"{WORKSPACE_ROOT}/fib.py", "fib.py"),  # the baseline bug: absolute workspace path
        (f"{WORKSPACE_ROOT}/sub/dir/x.py", "sub/dir/x.py"),
        ("sub\\x.py", "sub/x.py"),
    ],
)
def test_normalize_path_accepts_workspace_paths(given, expected):
    assert normalize_path(given) == expected


@pytest.mark.parametrize("bad", ["/etc/passwd", "/tmp/x", "../secret", "a/../../b", "/kaggle/input/data"])
def test_normalize_path_rejects_escapes(bad):
    with pytest.raises(ToolError):
        normalize_path(bad)


# ---- command + url policy -------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        "rm -rf /",
        "rm -rf /*",
        "sudo rm -rf ~",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=/dev/sda",
        "shutdown -h now",
        ":(){ :|:& };:",
        "pkill -f ollama",
        "killall cloudflared",
        "kill -9 1",
    ],
)
def test_dangerous_commands_are_refused(cmd):
    with pytest.raises(ToolError):
        check_command(cmd)


@pytest.mark.parametrize(
    "cmd",
    ["ls -la", "python fib.py", "pip install requests", "rm -rf build/", "rm notes.txt", "nvidia-smi",
     "cat /proc/cpuinfo | head", "df -h /", "curl -s https://example.com | head"],
)
def test_ordinary_commands_are_allowed(cmd):
    check_command(cmd)


@pytest.mark.parametrize("url", ["http://localhost:8000", "http://127.0.0.1/x", "http://10.0.0.5/", "ftp://example.com",
                                 "http://169.254.169.254/latest/meta-data", "file:///etc/passwd", "http://foo.internal/"])
def test_unsafe_urls_are_refused(url):
    with pytest.raises(ToolError):
        check_url(url)


def test_public_urls_are_allowed():
    assert check_url(" https://example.com/page?q=1 ") == "https://example.com/page?q=1"


# ---- redaction ------------------------------------------------------------------
@pytest.mark.parametrize(
    "secret",
    [
        "sk-abcdefghijklmnopqrstuvwxyz123456",
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz",
        "KGAT_0123456789abcdef0123456789abcdef",
        "cfut_abcdefghijklmnopqrstuvwxyz0123456789ABCD",
        "AKIAABCDEFGHIJKLMNOP",
        "tvly-dev-abcdefghijklmnopqrstuvwx",
    ],
)
def test_redact_removes_credential_shapes(secret):
    out = redact_mod.redact(f"value is {secret} ok")
    assert secret not in out and "[REDACTED]" in out


def test_redact_handles_authorization_headers_and_assignments():
    assert "abcdef0123456789abcdef" not in redact_mod.redact("Authorization: Bearer abcdef0123456789abcdef")
    assert "hunter2hunter2" not in redact_mod.redact('password = "hunter2hunter2"')
    assert redact_mod.redact("nothing secret here") == "nothing secret here"


def test_preview_truncates_after_redacting():
    out = redact_mod.preview("sk-abcdefghijklmnopqrstuvwxyz123456 " + "x" * 5000, 100)
    assert len(out) <= 100 and "sk-abcdef" not in out
