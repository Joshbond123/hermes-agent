"""Every "turn the GPU on" entry point must respect the fleet. The legacy single-session path pushed a second
live model session (qwen3 slug on the primary account) next to the fleet's session when the boot watchdogs ran."""

import pytest


def _m():
    import cloudflare_d1_client as m
    return m


def test_turn_on_routes_to_fleet_when_accounts_are_configured(monkeypatch):
    m = _m()
    calls = []
    import blackthorn.fleet as fleet
    monkeypatch.setattr(fleet, "accounts_from_env", lambda env: ["a", "b", "c", "d"])
    monkeypatch.setattr(m, "turn_on_fleet", lambda blocking=False: calls.append(("fleet", blocking)) or {"ok": 1})
    monkeypatch.setattr(m, "_turn_on_kaggle_gpu_single", lambda blocking=False: calls.append(("single",)) or {})
    out = m.turn_on_kaggle_gpu()
    assert out == {"ok": 1}
    assert calls == [("fleet", False)]


def test_turn_on_uses_the_single_path_without_a_fleet(monkeypatch):
    m = _m()
    calls = []
    import blackthorn.fleet as fleet
    monkeypatch.setattr(fleet, "accounts_from_env", lambda env: [])
    monkeypatch.setattr(m, "turn_on_fleet", lambda blocking=False: calls.append(("fleet", blocking)) or {})
    monkeypatch.setattr(m, "_turn_on_kaggle_gpu_single", lambda blocking=False: calls.append(("single", blocking)) or {"s": 1})
    assert m.turn_on_kaggle_gpu(blocking=True) == {"s": 1}
    assert calls == [("single", True)]


def test_watchdog_paths_call_the_routed_entry_point():
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "cloudflare_d1_client.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "start_permanent_gpu_daemon")
    body = ast.get_source_segment(src, fn)
    assert "turn_on_kaggle_gpu()" in body
    # the watchdog must never call the single-session worker directly
    assert "_turn_on_kaggle_gpu_sync()" not in body
