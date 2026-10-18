"""Fleet operations: both systems start together, backups stay cold, failover syncs first."""

import pytest

import cloudflare_d1_client as d1
from blackthorn.fleet import Account, FleetManager


ACCOUNTS = [
    Account("joshbond123", "tp", "model", "primary"),
    Account("teslaarymo", "tb", "model", "backup"),
    Account("josh787", "cp", "computer", "primary"),
    Account("alagbo", "cb", "computer", "backup"),
]


def _quota(remaining):
    return {"gpuQuota": {"timeUsed": f"{108000 - remaining}s", "totalTimeAllowed": "108000s"}}


@pytest.fixture()
def ops(monkeypatch):
    """Patch the fleet transport + row writes + notebook builders; record every action."""
    calls = {"starts": [], "rows": [], "rpcs": [], "mirrors": [], "texts": {}, "meta": [], "dispatches": []}
    quota = {"tp": 108000.0, "tb": 108000.0, "cp": 108000.0, "cb": 108000.0}

    def fake_rpc(method, body, token):
        calls["rpcs"].append((method, token))
        if method == "GetAcceleratorQuotaStatistics":
            return _quota(quota[token])
        return {"ok": True}

    monkeypatch.setattr(FleetManager, "_rpc", staticmethod(fake_rpc))
    monkeypatch.setattr(d1, "_write_fleet_row",
                        lambda row_id, status, **kwargs: calls["rows"].append((row_id, status)))
    monkeypatch.setattr(d1, "_set_meta", lambda key, value: calls["meta"].append((key, value)))
    monkeypatch.setattr(d1, "_record_stop_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(d1, "_fleet_role_online", lambda role: False)
    monkeypatch.setattr(d1, "com11_start", lambda: calls["dispatches"].append(True) or {"ok": True, "status": "dispatched"})
    monkeypatch.setattr(d1, "ensure_d1_schema", lambda: None)
    monkeypatch.setattr("blackthorn.fleet.accounts_from_env", lambda env: list(ACCOUNTS))
    monkeypatch.setattr("blackthorn.kaggle_bundle.build_notebook_text",
                        lambda server_py=None, env=None: calls["texts"].setdefault("model", "MODEL_NB") or "MODEL_NB")
    monkeypatch.setattr("blackthorn.kaggle_bundle.build_computer_notebook_text",
                        lambda server_py=None, env=None: calls["texts"].setdefault("computer", "COMP_NB") or "COMP_NB")

    def fake_start(self, account, *, slug, title, notebook_text, datasets=None):
        calls["starts"].append((account.user, slug, notebook_text))
        return {"ok": True}

    monkeypatch.setattr(FleetManager, "start_kernel", fake_start)
    def fake_rpc_bound(self, account, method, body=None):
        if method == "GetAcceleratorQuotaStatistics":
            return fake_rpc(method, body, account.token)
        calls["rpcs"].append((method, account.user))
        return {"ok": True}

    monkeypatch.setattr(FleetManager, "rpc", fake_rpc_bound)
    monkeypatch.setattr("blackthorn.fleet.mirror_state_dataset",
                        lambda *a, **k: calls["mirrors"].append(a) or {"ok": True})
    calls["quota"] = quota
    return calls


def test_turn_on_boots_both_systems_together_and_never_the_backups(ops):
    out = d1.turn_on_fleet(blocking=True)
    started = {user for user, _slug, _nb in ops["starts"]}
    assert started == {"joshbond123"}  # Kaggle runs only the selected model; Com11 is separate.
    assert out["model"]["account"] == "joshbond123"
    assert out["computer"]["account"] == "github-actions"
    assert ops["dispatches"] == [True]
    assert any(r == ("primary", "BOOTING_KAGGLE_GPU") for r in ops["rows"])
    assert any(r == ("computer", "COM11_BOOTING") for r in ops["rows"])
    assert not any(u in ("teslaarymo", "alagbo") for u in started)   # backups stay COLD


def test_backup_boots_only_when_the_primary_hits_thirty_minutes(ops):
    ops["quota"]["tp"] = 29 * 60                                     # model primary nearly spent
    out = d1.turn_on_fleet(blocking=True)
    started = {user for user, _s, _n in ops["starts"]}
    assert "teslaarymo" in started and "joshbond123" not in started   # only now does the backup run
    assert out["model"]["slot"] == "backup"
    assert out["model"]["reason"] == "primary_quota_low"


def test_model_failover_starts_backup_and_restarts_boot_timer(ops, monkeypatch):
    monkeypatch.setattr(d1, "_role_intended_online", lambda role: True)
    monkeypatch.setattr(d1, "_role_boot_in_progress", lambda role: False)
    monkeypatch.setattr(FleetManager, "find_role_kernel", lambda *a: None)
    monkeypatch.setattr(FleetManager, "stop_kernel_session", lambda *a: True)
    out = d1.failover_fleet_role("model", "primary", "backup")
    assert out["account"] == "teslaarymo"
    assert out["status"] == "starting"
    assert any(u == "teslaarymo" for u, _slug, _nb in ops["starts"])
    assert any(key == "blackthorn_gpu_boot_started" for key, _ in ops["meta"])
    assert not ops["mirrors"]  # Computer checkpoint does not apply to model failover.


def test_com11_dispatch_failure_is_reported_without_blocking_the_model(ops, monkeypatch):
    monkeypatch.setattr(d1, "com11_start", lambda: {"ok": False, "error": "GitHub dispatch unavailable"})
    out = d1.turn_on_fleet(blocking=True)
    assert out["model"]["status"] == "starting"
    assert out["computer"]["status"] == "start_failed"
    assert "dispatch unavailable" in out["computer"]["error"]
    assert ("computer", "BOOT_FAILED") in ops["rows"]
