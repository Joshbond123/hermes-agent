"""Model auto-off stops model accounts, but NEVER shuts down the computer role."""

import ast
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / "cloudflare_d1_client.py"


def _module():
    import cloudflare_d1_client as m
    return m


def test_daemon_auto_off_branch_uses_the_fleet_aware_stop():
    src = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.append(node.func.id)
    # the daemon's auto-off branch must call the helper, not the primary-only stop
    assert "_auto_off_stop" in calls
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "start_permanent_gpu_daemon")
    body = ast.get_source_segment(src, fn)
    assert "Kaggle GPU" in body and "_auto_off_stop()" in body
    assert "        turn_off_kaggle_gpu()\n        consecutive_dead" not in body


def test_auto_off_stops_backup_accounts_when_fleet_configured(monkeypatch):
    m = _module()
    monkeypatch.setattr(m, "_durable_active_runs", lambda: 0)
    calls = []
    monkeypatch.setattr(m, "turn_off_fleet", lambda blocking=True, source="", roles=(): calls.append(("fleet", blocking, roles)) or {"ok": True})
    monkeypatch.setattr(m, "turn_off_kaggle_gpu", lambda: calls.append(("single",)) or {"ok": True})
    import blackthorn.fleet as fleet
    monkeypatch.setattr(fleet, "accounts_from_env", lambda env: ["primary-account", "backup-account"])
    m._auto_off_stop()
    assert calls == [("fleet", True, ("model",))]


def test_auto_off_uses_the_single_session_when_no_fleet(monkeypatch):
    m = _module()
    monkeypatch.setattr(m, "_durable_active_runs", lambda: 0)
    calls = []
    monkeypatch.setattr(m, "turn_off_fleet", lambda blocking=True: calls.append(("fleet", blocking)) or {})
    monkeypatch.setattr(m, "turn_off_kaggle_gpu", lambda: calls.append(("single",)) or {"ok": True})
    import blackthorn.fleet as fleet
    monkeypatch.setattr(fleet, "accounts_from_env", lambda env: [])
    m._auto_off_stop()
    assert calls == [("single",)]
