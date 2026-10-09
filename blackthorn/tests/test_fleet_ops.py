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
    calls = {"starts": [], "rows": [], "rpcs": [], "mirrors": [], "texts": {}}
    quota = {"tp": 108000.0, "tb": 108000.0, "cp": 108000.0, "cb": 108000.0}

    def fake_rpc(method, body, token):
        calls["rpcs"].append((method, token))
        if method == "GetAcceleratorQuotaStatistics":
            return _quota(quota[token])
        return {"ok": True}

    monkeypatch.setattr(FleetManager, "_rpc", staticmethod(fake_rpc))
    monkeypatch.setattr(d1, "_write_fleet_row",
                        lambda row_id, status, tunnel_url="", api_key="", model="", gpu_info="":
                        calls["rows"].append((row_id, status)))
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
    out = d1.turn_on_fleet()
    started = {user for user, _slug, _nb in ops["starts"]}
    assert started == {"joshbond123", "josh787"}                     # both primaries, in one call
    assert out["model"]["account"] == "joshbond123"
    assert out["computer"]["account"] == "josh787"
    assert any(r == ("primary", "BOOTING_KAGGLE_GPU") for r in ops["rows"])
    assert any(r == ("computer", "BOOTING_KAGGLE_GPU") for r in ops["rows"])
    assert not any(u in ("teslaarymo", "alagbo") for u in started)   # backups stay COLD


def test_backup_boots_only_when_the_primary_hits_thirty_minutes(ops):
    ops["quota"]["tp"] = 29 * 60                                     # model primary nearly spent
    out = d1.turn_on_fleet()
    started = {user for user, _s, _n in ops["starts"]}
    assert "teslaarymo" in started and "joshbond123" not in started   # only now does the backup run
    assert out["model"]["slot"] == "backup"
    assert out["model"]["reason"] == "primary_quota_low"


def test_failover_syncs_state_before_booting_the_backup(ops):
    order = []
    ops["mirrors"].append  # keep fixture alive
    import blackthorn.fleet as fleet_mod

    def fake_mirror(*a, **k):
        order.append("sync")
        return {"ok": True}

    def fake_start(self, account, *, slug, title, notebook_text, datasets=None):
        order.append(f"boot:{account.user}")
        return {"ok": True}

    ops["starts"].clear()
    import cloudflare_d1_client as _d
    # patch where failover_fleet_role imports them from
    import blackthorn.fleet as fm
    orig_mirror = fm.mirror_state_dataset
    fm.mirror_state_dataset = fake_mirror
    try:
        import unittest.mock as mock
        with mock.patch.object(FleetManager, "start_kernel", fake_start), \
             mock.patch.object(FleetManager, "rpc",
                               lambda self, account, method, body=None: order.append(f"{method}:{account.user}") or {"ok": True}):
            out = _d.failover_fleet_role("computer", "primary", "backup")
    finally:
        fm.mirror_state_dataset = orig_mirror
    assert out["account"] == "alagbo"
    assert order.index("sync") < order.index("boot:alagbo")           # state first, then switch
    assert "StopKernel:josh787" in order                              # old primary retired afterwards
    assert order.index("boot:alagbo") < order.index("StopKernel:josh787")


def test_resource_exhaustion_is_reported_not_hidden(ops):
    """A start failure on one system must not fake success or block the other."""
    def boom(self, account, *, slug, title, notebook_text, datasets=None):
        if account.role == "computer":
            raise RuntimeError("Kaggle quota exhausted")
        return {"ok": True}

    import unittest.mock as mock
    with mock.patch.object(FleetManager, "start_kernel", boom):
        out = d1.turn_on_fleet()
    assert out["model"]["status"] == "starting"
    assert out["computer"]["status"] == "start_failed"
    assert "quota exhausted" in out["computer"]["error"]
    assert ("computer", "BOOT_FAILED") in ops["rows"]                 # the UI sees the real failure
