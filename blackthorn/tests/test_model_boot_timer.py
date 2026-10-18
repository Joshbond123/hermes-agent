"""The model boot clock follows the actual fleet boot, not an older session."""

from types import SimpleNamespace

import cloudflare_d1_client as d1
from blackthorn.fleet import Account, FleetManager


def test_model_fleet_boot_resets_elapsed_clock_but_reuse_does_not(monkeypatch):
    from blackthorn import kaggle_bundle, fleet

    account = Account("example", "fake-token", "model", "primary")
    values = []
    rows = []
    monkeypatch.setattr(d1, "ensure_d1_schema", lambda: None)
    monkeypatch.setattr(d1, "_record_stop_event", lambda *a: None)
    monkeypatch.setattr(d1, "_set_meta", lambda key, val: values.append((key, float(val))))
    monkeypatch.setattr(d1, "_write_fleet_row", lambda *args, **kwargs: rows.append(args[:2]))
    monkeypatch.setattr(d1, "com11_start", lambda: {"ok": True, "status": "already_running"})
    monkeypatch.setattr(d1, "_fleet_role_online", lambda role: False)
    monkeypatch.setattr(fleet, "accounts_from_env", lambda env: [account])
    monkeypatch.setattr(fleet, "notebook_env_for", lambda *a: {})
    monkeypatch.setattr(fleet, "kernel_slug_for", lambda *a: "example/test-model")
    monkeypatch.setattr(kaggle_bundle, "build_notebook_text", lambda **kwargs: "fake notebook")
    monkeypatch.setattr(FleetManager, "decide", lambda self, role, **kw: SimpleNamespace(active="primary", reason="test"))
    monkeypatch.setattr(FleetManager, "status", lambda self: {})
    monkeypatch.setattr(FleetManager, "start_kernel", lambda self, *a, **kw: {"ok": True})

    result = d1.turn_on_fleet(blocking=True)
    assert result["model"]["status"] == "starting"
    assert len(values) == 1 and values[0][0] == "blackthorn_gpu_boot_started"
    assert ("primary", "BOOTING_KAGGLE_GPU") in rows

    # A healthy existing kernel must not reset the start time when it is reused.
    monkeypatch.setattr(d1, "_fleet_role_online", lambda role: role == "model")
    result = d1.turn_on_fleet(blocking=True)
    assert result["model"]["status"] == "already_online"
    assert len(values) == 1
