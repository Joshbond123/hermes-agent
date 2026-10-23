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
    out = d1.turn_on_fleet(blocking=True)
    started = {user for user, _slug, _nb in ops["starts"]}
    assert started == {"joshbond123", "josh787"}                     # both primaries, in one call
    assert out["model"]["account"] == "joshbond123"
    assert out["computer"]["account"] == "josh787"
    assert any(r == ("primary", "BOOTING_KAGGLE_GPU") for r in ops["rows"])
    assert any(r == ("computer", "BOOTING_KAGGLE_GPU") for r in ops["rows"])
    assert not any(u in ("teslaarymo", "alagbo") for u in started)   # backups stay COLD


def test_backup_boots_only_when_the_primary_hits_thirty_minutes(ops):
    ops["quota"]["tp"] = 29 * 60                                     # model primary nearly spent
    out = d1.turn_on_fleet(blocking=True)
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
        with mock.patch.object(_d, "_role_intended_online", lambda role: True), \
             mock.patch.object(_d, "_fleet_role_online", lambda role: True), \
             mock.patch.object(_d, "checkpoint_computer", lambda timeout=220: {"ok": True, "generation": 42, "digest": "verified"}), \
             mock.patch.object(_d, "_durable_active_runs", lambda: 0), \
             mock.patch.object(_d, "_role_boot_in_progress", lambda role: False), \
             mock.patch.object(FleetManager, "start_kernel", fake_start), \
             mock.patch.object(FleetManager, "stop_kernel_result",
                               lambda self, account, slug: order.append(f"stop:{account.user}") or {"state": "stopped"}), \
             mock.patch.object(FleetManager, "rpc",
                               lambda self, account, method, body=None: order.append(f"{method}:{account.user}") or {"status": "RUNNING"}):
            out = _d.failover_fleet_role("computer", "primary", "backup")
    finally:
        fm.mirror_state_dataset = orig_mirror
    assert out["account"] == "alagbo"
    assert order.index("sync") < order.index("boot:alagbo")           # state first, then switch
    assert out['ok'] and out['status'] == 'ready'
    assert "stop:josh787" in order
    assert order.index("boot:alagbo") < order.index("stop:josh787")


def test_resource_exhaustion_is_reported_not_hidden(ops):
    """A start failure on one system must not fake success or block the other."""
    def boom(self, account, *, slug, title, notebook_text, datasets=None):
        if account.role == "computer":
            raise RuntimeError("Kaggle quota exhausted")
        return {"ok": True}

    import unittest.mock as mock
    with mock.patch.object(FleetManager, "start_kernel", boom):
        out = d1.turn_on_fleet(blocking=True)
    assert out["model"]["status"] == "starting"
    assert out["computer"]["status"] == "start_failed"
    assert "quota exhausted" in out["computer"]["error"]
    assert ("computer", "BOOT_FAILED") in ops["rows"]                 # the UI sees the real failure


def test_recent_heartbeat_reuses_model_when_gateway_probe_times_out(ops, monkeypatch):
    import time
    monkeypatch.setattr(d1, '_fleet_role_online', lambda role: False)
    monkeypatch.setattr(d1, 'd1_query', lambda sql, params=None: (
        [{'status': 'HEARTBEAT_ONLINE', 'updated_at': time.time()}]
        if 'SELECT status, updated_at FROM kaggle_gpu_state' in sql and params == ['primary'] else
        [{'status': 'BOOTING_KAGGLE_GPU', 'updated_at': time.time()}]
        if 'SELECT status, updated_at FROM kaggle_gpu_state' in sql and params == ['computer'] else []))
    out = d1.turn_on_fleet(blocking=True)
    assert out['model']['status'] == 'reconnecting'
    assert out['computer']['status'] == 'already_starting'
    assert ops['starts'] == []


def test_computer_failover_refuses_failed_checkpoint_and_keeps_old(ops, monkeypatch):
    monkeypatch.setattr(d1, '_role_intended_online', lambda role: True)
    monkeypatch.setattr(d1, '_role_boot_in_progress', lambda role: False)
    monkeypatch.setattr(d1, '_fleet_role_online', lambda role: True)
    monkeypatch.setattr(d1, 'checkpoint_computer', lambda timeout=220: {'ok': False})
    monkeypatch.setattr(d1, '_fleet_manager_with_usage', lambda accounts: FleetManager(accounts))
    out = d1.failover_fleet_role('computer', 'primary', 'backup')
    assert out['status'] == 'checkpoint_failed' and not out['ok']
    assert not ops['starts']
    assert not any(method == 'DeleteKernel' for method, _account in ops['rpcs'])


def test_computer_failover_refuses_failed_dataset_mirror(ops, monkeypatch):
    monkeypatch.setattr(d1, '_role_intended_online', lambda role: True)
    monkeypatch.setattr(d1, '_role_boot_in_progress', lambda role: False)
    monkeypatch.setattr(d1, '_fleet_role_online', lambda role: False)
    monkeypatch.setattr(d1, '_fleet_manager_with_usage', lambda accounts: FleetManager(accounts))
    monkeypatch.setattr('blackthorn.fleet.mirror_state_dataset',
                        lambda *args, **kwargs: {'ok': False, 'error': 'unavailable'})
    out = d1.failover_fleet_role('computer', 'primary', 'backup')
    assert out['status'] == 'sync_failed' and not out['ok']
    assert not ops['starts']


def test_model_backup_waits_for_live_owner_and_does_not_stop_primary_early(ops, monkeypatch):
    monkeypatch.setattr(d1, '_role_intended_online', lambda role: True)
    monkeypatch.setattr(d1, '_role_boot_in_progress', lambda role: False)
    monkeypatch.setattr(d1, '_fleet_role_online', lambda role: True)
    monkeypatch.setattr(d1, '_durable_active_runs', lambda: 0)
    monkeypatch.setattr(d1, '_fleet_manager_with_usage', lambda accounts: FleetManager(accounts))
    events = []
    import blackthorn.fleet as fm
    def rpc(self, account, method, body=None):
        if method == 'GetKernelSessionStatus':
            events.append('owner')
            return {'status': 'RUNNING' if len(events) >= 2 else 'QUEUED'}
        return {'ok': True}
    monkeypatch.setattr(fm.FleetManager, 'rpc', rpc)
    monkeypatch.setattr(fm.FleetManager, 'stop_kernel_result',
                        lambda self, account, slug: events.append('old_stopped') or {'state': 'stopped'})
    monkeypatch.setattr(d1.time, 'sleep', lambda seconds: None)
    out = d1.failover_fleet_role('model', 'primary', 'backup')
    assert out['ok'] and out['status'] == 'ready'
    assert events == ['owner', 'owner', 'old_stopped']


def test_computer_recovery_syncs_backup_to_primary_before_automatic_return(ops, monkeypatch):
    """After backup has accepted writes, returning to primary cannot revive old files."""
    events = []
    monkeypatch.setattr(d1, '_computer_authoritative_slot', lambda: 'backup')
    monkeypatch.setattr(d1, '_fleet_role_online', lambda role: False)
    monkeypatch.setattr('blackthorn.fleet.mirror_state_dataset',
                        lambda old, new, **kwargs: events.append(('sync', old.slot, new.slot)) or {'ok': True, 'digest': 'fresh'})
    def fake_start(self, account, **kwargs):
        events.append(('boot', account.slot))
        return {'_slug': f'{account.user}/blackthorn-computer', '_session_running': True}
    monkeypatch.setattr(FleetManager, 'start_kernel', fake_start)
    d1.turn_on_fleet(blocking=True)
    assert events.index(('sync', 'backup', 'primary')) < events.index(('boot', 'primary'))


def test_watchdog_transition_retries_failed_model_and_computer_switches(ops, monkeypatch):
    accounts = list(ACCOUNTS)
    fleet = FleetManager(accounts)
    monkeypatch.setattr(d1, '_fleet_manager_with_usage', lambda a: FleetManager(a))
    checks = {'model': 0, 'computer': 0}
    def fake_switch(role, old, new):
        checks[role] += 1
        return {'ok': checks[role] > 1, 'status': 'ready' if checks[role] > 1 else 'sync_failed'}
    monkeypatch.setattr(d1, 'failover_fleet_role', fake_switch)
    for role in ('model', 'computer'):
        token = 'tp' if role == 'model' else 'cp'
        ops['quota'][token] = 29 * 60
        first = fleet.decide(role, force=True)
        assert first.active == 'backup'
        failed = d1._execute_fleet_transition(fleet, role, 'primary', 'backup', accounts)
        assert not failed['ok'] and fleet.state[role].active == 'primary'
        assert fleet.decide(role, force=True).active == 'backup'
        ready = d1._execute_fleet_transition(fleet, role, 'primary', 'backup', accounts)
        assert ready['ok'] and fleet.state[role].active == 'backup'
        assert checks[role] == 2


def test_old_model_is_deferred_during_an_agent_task_then_retired_safely(ops, monkeypatch):
    monkeypatch.setattr(d1, '_role_intended_online', lambda role: True)
    monkeypatch.setattr(d1, '_role_boot_in_progress', lambda role: False)
    monkeypatch.setattr(d1, '_fleet_role_online', lambda role: True)
    monkeypatch.setattr(d1, '_fleet_manager_with_usage', lambda accounts: FleetManager(accounts))
    monkeypatch.setattr(d1, '_durable_active_runs', lambda: 1)
    monkeypatch.setattr(d1, '_set_meta', lambda key, value: ops.setdefault('meta', []).append((key, value)))
    monkeypatch.setattr(FleetManager, 'rpc',
                        lambda self, account, method, body=None: {'status': 'RUNNING'} if method == 'GetKernelSessionStatus' else {'ok': True})
    out = d1.failover_fleet_role('model', 'primary', 'backup')
    assert out['ok'] and out['old_stopped']['state'] == 'deferred'
    assert not any(method == 'DeleteKernel' for method, _ in ops['rpcs'])
    assert ('bt_fleet_retire:model', 'primary') in ops['meta']
    fleet = FleetManager(list(ACCOUNTS))
    fleet.state['model'].active = 'backup'
    monkeypatch.setattr(d1, '_durable_active_runs', lambda: 0)
    monkeypatch.setattr(d1, 'd1_query', lambda sql, *a: [{'key':'bt_fleet_retire:model', 'value':'primary'}])
    stops = []
    monkeypatch.setattr(FleetManager, 'stop_kernel_result',
                        lambda self, account, slug: stops.append(account.slot) or {'state': 'stopped'})
    d1._retire_verified_old_slots(fleet)
    assert stops == ['primary']
    assert ('bt_fleet_retire:model', '') in ops['meta']


def test_explicit_checkpoint_is_not_success_until_source_dataset_reads_back(ops, monkeypatch):
    monkeypatch.setattr(d1, '_computer_link', lambda: ('https://test.invalid', 'test-only', 'ONLINE'))
    monkeypatch.setattr(d1, '_computer_request', lambda *args: {
        'ok': True, 'generation': 42, 'digest': 'live-digest', 'files': 2})
    monkeypatch.setattr(d1, '_computer_authoritative_slot', lambda: 'primary')
    checked = []
    monkeypatch.setattr('blackthorn.fleet.mirror_state_dataset',
                        lambda old, new, **kw: checked.append((old.slot, new.slot, kw)) or {
                            'ok': False, 'error': 'source dataset has not published the expected live checkpoint'})
    result = d1.checkpoint_computer()
    assert not result['ok'] and result['generation'] == 42
    assert checked and checked[0][0:2] == ('primary', 'primary')
    assert checked[0][2]['expected_generation'] == 42
    assert checked[0][2]['expected_digest'] == 'live-digest'


@pytest.mark.parametrize('role', ['model', 'computer'])
def test_live_shared_relay_failover_fails_closed_without_stopping_source(ops, monkeypatch, role):
    monkeypatch.setenv('BLACKTHORN_RELAY_URL', 'https://test.invalid/gpu-relay')
    monkeypatch.setattr(d1, '_role_intended_online', lambda r: True)
    monkeypatch.setattr(d1, '_role_boot_in_progress', lambda r: False)
    monkeypatch.setattr(d1, '_fleet_role_online', lambda r: True)
    monkeypatch.setattr(d1, '_fleet_manager_with_usage', lambda accounts: FleetManager(accounts))
    before = list(ops['rows'])
    outcome = d1.failover_fleet_role(role, 'primary', 'backup')
    assert not outcome['ok'] and outcome['status'] == 'shared_relay_handover_unavailable'
    assert ops['rows'] == before and not ops['starts'] and not ops['mirrors']
    assert not any(method == 'DeleteKernel' for method, _ in ops['rpcs'])
