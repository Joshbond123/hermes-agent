"""Server-side inactivity stops both fleet roles only after the workspace checkpoint."""

import ast
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / "cloudflare_d1_client.py"


def _module():
    import cloudflare_d1_client as m
    return m


def test_boot_watchdog_does_not_repush_on_empty_kaggle_status():
    """Kaggle returns an empty session status during accepted GPU allocation."""
    src = MODULE.read_text(encoding='utf-8')
    tree = ast.parse(src)
    daemon = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == 'start_permanent_gpu_daemon')
    body = ast.get_source_segment(src, daemon)
    assert 'role_kernel["slug"].split("/")[-1]' in body
    assert 'kernel_status not in ("COMPLETE", "ERROR", "CANCELLED"' in body
    assert 'kernel_status in ("RUNNING", "QUEUED", "PENDING")' not in body
    assert 'age < 3600 and state in ("BOOTING_KAGGLE_GPU"' in src


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
    monkeypatch.setattr(m, "_auto_off_roles_ready", lambda: True)
    calls = []
    monkeypatch.setattr(m, "turn_off_fleet", lambda blocking=True, source="", roles=(): calls.append(("fleet", blocking, roles)) or {"ok": True})
    monkeypatch.setattr(m, "turn_off_kaggle_gpu", lambda: calls.append(("single",)) or {"ok": True})
    import blackthorn.fleet as fleet
    monkeypatch.setattr(fleet, "accounts_from_env", lambda env: ["primary-account", "backup-account"])
    m._auto_off_stop()
    assert calls == [("fleet", True, ("model", "computer"))]


def test_auto_off_uses_the_single_session_when_no_fleet(monkeypatch):
    m = _module()
    monkeypatch.setattr(m, "_durable_active_runs", lambda: 0)
    monkeypatch.setattr(m, "_auto_off_roles_ready", lambda: True)
    calls = []
    monkeypatch.setattr(m, "turn_off_fleet", lambda blocking=True: calls.append(("fleet", blocking)) or {})
    monkeypatch.setattr(m, "turn_off_kaggle_gpu", lambda: calls.append(("single",)) or {"ok": True})
    import blackthorn.fleet as fleet
    monkeypatch.setattr(fleet, "accounts_from_env", lambda env: [])
    m._auto_off_stop()
    assert calls == [("single",)]


def test_auto_off_refuses_a_booting_computer_even_when_model_is_idle(monkeypatch):
    m = _module()
    monkeypatch.setattr(m, 'get_auto_off_minutes', lambda: 15)
    monkeypatch.setattr(m, '_STATUS_CACHE', {'active': True, 'booting': False})
    monkeypatch.setattr(m, '_published_status', lambda: 'HEARTBEAT_ONLINE')
    monkeypatch.setattr(m, '_COMPUTER_ACTIVITY', {'known': True, 'active_jobs': 0})
    monkeypatch.setattr(m, '_durable_active_runs', lambda: 0)
    monkeypatch.setattr(m, 'activity_snapshot', lambda: {'idle_seconds': 3600,
                             'active_tasks': 0, 'last_activity_at': 0})
    monkeypatch.setattr(m, 'd1_query', lambda sql, params=None: [
        {'id': 'primary', 'status': 'HEARTBEAT_ONLINE', 'tunnel_url': 'https://model.invalid'},
        {'id': 'computer', 'status': 'INSTALLING_DEPS', 'tunnel_url': ''}])
    decision = m.auto_off_decision()
    assert not decision['should_stop'] and 'Computer is still starting' in decision['reason']
    calls = []
    monkeypatch.setattr(m, 'turn_off_fleet', lambda **kw: calls.append(kw) or {})
    import blackthorn.fleet as fleet
    monkeypatch.setattr(fleet, 'accounts_from_env', lambda env: ['configured'])
    assert m._auto_off_stop()['model']['status'] == 'stop_refused'
    assert not calls
    assert m._auto_off_roles_ready() is False


def test_auto_off_allows_two_online_roles_or_a_deliberately_off_computer(monkeypatch):
    m = _module()
    rows = [{'id':'primary','status':'HEARTBEAT_ONLINE','tunnel_url':'https://model.invalid'},
            {'id':'computer','status':'HEARTBEAT_ONLINE','tunnel_url':'https://computer.invalid'}]
    monkeypatch.setattr(m, 'd1_query', lambda sql, params=None: rows)
    assert m._auto_off_roles_ready() is True
    rows[1]['status']='GPU_STOPPED_SAVING_QUOTA';rows[1]['tunnel_url']=''
    assert m._auto_off_roles_ready() is True
    rows[1]['status']='STARTING_GATEWAY'
    assert m._auto_off_roles_ready() is False
    monkeypatch.setattr(m, 'd1_query', lambda sql, params=None: (_ for _ in ()).throw(OSError('D1 down')))
    assert m._auto_off_roles_ready() is False
