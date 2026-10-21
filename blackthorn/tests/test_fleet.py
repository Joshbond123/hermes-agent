"""Fleet failover engine: real quotas, the 30-minute rule, hysteresis, and honest unknowns."""

import pytest

from blackthorn.fleet import (FAILOVER_BELOW_S, RETURN_ABOVE_S, RETURN_STABLE_CHECKS, Account,
                             FleetManager, accounts_from_env)

class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def make_fleet(clock, quota_map):
    """quota_map: user -> seconds remaining (updated in place to simulate the week burning)."""
    def fetcher(method, body, token):
        user = next(a.user for a in accounts if a.token == token)
        total = 108000.0
        used = total - quota_map[user]
        return {"quotaRefreshTime": "2026-10-10T00:00:00Z",
                "gpuQuota": {"timeUsed": f"{used}s", "timeReserved": "0s",
                             "timeTotal": None, "timeUsedSeconds": None,
                             "totalTimeAllowed": f"{total}s", "minimumTimeAllowed": "108000s",
                             "hasEverRun": True}}
    accounts = [
        Account("joshbond123", "tok-p", "model", "primary"),
        Account("teslaarymo", "tok-b", "model", "backup"),
        Account("josh787", "tok-cp", "computer", "primary"),
        Account("alagbo", "tok-cb", "computer", "backup"),
    ]
    return FleetManager(accounts, fetcher=fetcher, clock=clock), accounts, quota_map


def test_quota_is_parsed_from_kaggle_strings_not_invented():
    clock = FakeClock()
    fleet, accounts, _ = make_fleet(clock, {"joshbond123": 3600.0, "teslaarymo": 0.0,
                                            "josh787": 108000.0, "alagbo": 108000.0})
    q = fleet.quota(accounts[0])
    assert q["remaining_seconds"] == pytest.approx(3600.0)
    assert q["used_seconds"] == pytest.approx(104400.0)
    assert q["refresh_time"] == "2026-10-10T00:00:00Z"


def test_fails_over_when_primary_drops_to_thirty_minutes():
    clock = FakeClock()
    fleet, _, quota = make_fleet(clock, {"joshbond123": 31 * 60, "teslaarymo": 108000.0,
                                         "josh787": 108000.0, "alagbo": 108000.0})
    st = fleet.decide("model")
    assert st.active == "primary"
    quota["joshbond123"] = 29 * 60
    clock.advance(30)
    st = fleet.decide("model", force=True)
    assert st.active == "backup" and st.reason == "primary_quota_low"


def test_stays_on_primary_while_it_has_time():
    clock = FakeClock()
    fleet, _, _ = make_fleet(clock, {"joshbond123": 31 * 60, "teslaarymo": 108000.0,
                                     "josh787": 108000.0, "alagbo": 108000.0})
    for _ in range(5):
        assert fleet.decide("model").active == "primary"
        clock.advance(30)


def test_hard_down_switches_immediately_even_inside_flap_window():
    clock = FakeClock()
    fleet, _, _ = make_fleet(clock, {"joshbond123": 108000.0, "teslaarymo": 108000.0,
                                     "josh787": 108000.0, "alagbo": 108000.0})
    fleet.decide("model")
    fleet.mark_unhealthy("model", "primary", "kernel session died")
    st = fleet.decide("model")
    assert st.active == "backup" and st.reason == "primary_unhealthy"


def test_never_switches_into_an_exhausted_backup():
    clock = FakeClock()
    fleet, _, quota = make_fleet(clock, {"joshbond123": 10 * 60, "teslaarymo": 2 * 60,
                                         "josh787": 108000.0, "alagbo": 108000.0})
    st = fleet.decide("model", force=True)
    assert st.active == "primary"                       # burn the last of the primary first


def test_return_home_needs_hysteresis_and_interval():
    clock = FakeClock()
    fleet, _, quota = make_fleet(clock, {"joshbond123": 10 * 60, "teslaarymo": 108000.0,
                                         "josh787": 108000.0, "alagbo": 108000.0})
    fleet.decide("model", force=True)
    assert fleet.state["model"].active == "backup"
    # the primary refreshes its week (new quota) and stays healthy
    quota["joshbond123"] = 108000.0
    clock.advance(360)                                  # past the min-switch interval
    for i in range(RETURN_STABLE_CHECKS - 1):
        st = fleet.decide("model", force=True)
        assert st.active == "backup"                    # not yet: still inside the stable window
        clock.advance(30)
    st = fleet.decide("model", force=True)
    assert st.active == "primary" and st.reason == "primary_recovered"


def test_dead_backup_forces_the_primary_back_at_once():
    clock = FakeClock()
    fleet, _, quota = make_fleet(clock, {"joshbond123": 10 * 60, "teslaarymo": 108000.0,
                                         "josh787": 108000.0, "alagbo": 108000.0})
    fleet.decide("model", force=True)
    assert fleet.state["model"].active == "backup"
    fleet.mark_unhealthy("model", "backup", "backup tunnel died")
    st = fleet.decide("model", force=True)
    assert st.active == "primary" and st.reason == "backup_unavailable"


def test_unknown_quota_is_honest_not_zero():
    clock = FakeClock()
    fleet, accounts, _ = make_fleet(clock, {"joshbond123": 1.0, "teslaarymo": 1.0,
                                            "josh787": 1.0, "alagbo": 1.0})

    def broken(method, body, token):
        raise OSError("kaggle down")

    fleet._fetch = broken
    fleet.refresh_slots()
    view = fleet.status()["model"]["primary"]
    assert "remaining_seconds" not in view                # unknown, never "0 h left"


def test_recovery_clears_hard_down_after_the_kernel_serves_again():
    clock = FakeClock()
    fleet, _, _ = make_fleet(clock, {"joshbond123": 108000.0, "teslaarymo": 108000.0,
                                     "josh787": 108000.0, "alagbo": 108000.0})
    fleet.mark_unhealthy("model", "primary", "boom")
    fleet.mark_healthy("model", "primary")
    assert fleet.slots["joshbond123"].healthy is True
    assert fleet.decide("model").active == "primary"


def test_accounts_from_env_parses_pairs_and_falls_back_to_legacy():
    env = {"KAGGLE_FLEET_MODEL_PRIMARY": "u1:t1", "KAGGLE_FLEET_MODEL_BACKUP": "u2:t2",
           "KAGGLE_FLEET_COMPUTER_PRIMARY": "u3:t3", "KAGGLE_FLEET_COMPUTER_BACKUP": "u4:t4"}
    got = accounts_from_env(env)
    assert {(a.role, a.slot): a.user for a in got} == {("model", "primary"): "u1", ("model", "backup"): "u2",
                                                       ("computer", "primary"): "u3", ("computer", "backup"): "u4"}
    legacy = accounts_from_env({"KAGGLE_USERNAME": "old", "KAGGLE_API_TOKEN": "tok"})
    assert any(a.user == "old" and a.slot == "primary" for a in legacy)


def test_status_reports_all_four_slots_with_real_fields():
    clock = FakeClock()
    fleet, _, _ = make_fleet(clock, {"joshbond123": 108000.0, "teslaarymo": 108000.0,
                                     "josh787": 54000.0, "alagbo": 108000.0})
    fleet.refresh_slots()
    status = fleet.status()
    assert set(status) == {"model", "computer"}
    for role in status.values():
        assert set(role) >= {"active", "reason", "primary", "backup", "accounts", "events"}
        assert role["primary"]["remaining_seconds"] >= 0
    assert status["computer"]["primary"]["used_seconds"] == 54000


def test_partial_fleet_still_works():
    """Only the legacy single account configured: status/decide must never raise."""
    clock = FakeClock()
    fleet = FleetManager([Account("joshbond123", "tok", "model", "primary")],
                         fetcher=lambda m, b, t: {"gpuQuota": {"timeUsed": "1s", "totalTimeAllowed": "108000s"}},
                         clock=clock)
    fleet.refresh_slots()
    status = fleet.status()
    assert status["model"]["backup"] == {"configured": False}
    assert status["computer"]["primary"] == {"configured": False}
    assert fleet.decide("model").active == "primary"
    assert fleet.decide("computer").active == "primary"     # no accounts at all -> no crash


def test_mirror_uses_bearer_and_relabels_archive_for_backup(tmp_path):
    import io
    import json
    import zipfile
    from pathlib import Path
    from blackthorn.fleet import mirror_state_dataset
    calls = []
    def runner(cmd, env):
        calls.append((cmd[:3], 'KAGGLE_API_TOKEN' in env and 'KAGGLE_KEY' not in env))
        assert 'KAGGLE_KEY' not in env
        assert env['KAGGLE_API_TOKEN'] == ('primary-token' if cmd[2] == 'download' else 'backup-token')
        dest = Path(cmd[cmd.index('-p') + 1])
        if cmd[2] == 'download':
            with zipfile.ZipFile(dest / 'blackthorn-computer-state.zip', 'w') as z:
                z.writestr('workspace.zip', b'workspace bytes')
        else:
            metadata = json.loads((dest / 'dataset-metadata.json').read_text())
            assert metadata['id'] == 'backup/blackthorn-computer-state'
            assert metadata['isPrivate'] is True
            assert (dest / 'workspace.zip').read_bytes() == b'workspace bytes'
            assert not any(p.name == 'blackthorn-computer-state.zip' for p in dest.iterdir())
        return 0
    result = mirror_state_dataset(Account('primary', 'primary-token', 'computer', 'primary'),
                                  Account('backup', 'backup-token', 'computer', 'backup'), runner=runner)
    assert result['ok'] is True
    assert all(bearer for _cmd, bearer in calls)
    assert [cmd[2] for cmd, _ in calls] == ['download', 'create']


def test_backup_computer_restores_its_own_mirrored_dataset():
    from blackthorn.fleet import notebook_env_for
    backup = Account('backup', 'backup-token', 'computer', 'backup')
    env = notebook_env_for(backup, {'COMPUTER_STATE_DATASET': 'primary/blackthorn-computer-state'})
    assert env['COMPUTER_STATE_DATASET'] == 'backup/blackthorn-computer-state'


def test_mirror_preserves_flat_workspace_archive(tmp_path):
    import json
    import zipfile
    from pathlib import Path
    from blackthorn.fleet import mirror_state_dataset
    def runner(cmd, env):
        folder = Path(cmd[cmd.index('-p') + 1])
        if cmd[2] == 'download':
            with zipfile.ZipFile(folder / 'state.zip', 'w') as z:
                z.writestr('uploads/old-file.txt', 'preserved')
                z.writestr('.blackthorn/environment.json', '{}')
        else:
            assert (folder / 'uploads/old-file.txt').read_text() == 'preserved'
            assert (folder / '.blackthorn/environment.json').read_text() == '{}'
            assert json.loads((folder / 'dataset-metadata.json').read_text())['id'] == 'backup/blackthorn-computer-state'
        return 0
    result = mirror_state_dataset(Account('primary','token1','computer','primary'),
                                  Account('backup','token2','computer','backup'), runner=runner)
    assert result['ok'] is True


def test_delete_verified_by_listing_when_deleted_session_returns_403(monkeypatch):
    import urllib.error
    from blackthorn.fleet import FleetManager
    monkeypatch.setattr('time.sleep', lambda seconds: None)
    account = Account('primary', 'token', 'model', 'primary')
    state = {'deleted': False}
    def rpc(method, body, token):
        if method == 'DeleteKernel':
            state['deleted'] = True
            return {}
        if method == 'GetKernelSessionStatus':
            if state['deleted']:
                raise urllib.error.HTTPError('https://www.kaggle.com', 403, 'gone', {}, None)
            return {'status': 'RUNNING'}
        if method == 'ListKernels':
            return {'kernels': [] if state['deleted'] else [{'ref':'primary/blackthorn-model', 'title':'Blackthorn model'}]}
        raise AssertionError(method)
    fleet = FleetManager([account], fetcher=rpc)
    assert fleet.stop_kernel_result(account, 'primary/blackthorn-model')['state'] == 'stopped'


def test_stop_skips_already_absent_backup_without_false_error(monkeypatch):
    import urllib.error
    account = Account('backup', 'token', 'model', 'backup')
    calls = []
    def rpc(method, body, token):
        calls.append(method)
        if method == 'GetKernelSessionStatus':
            raise urllib.error.HTTPError('https://www.kaggle.com', 403, 'gone', {}, None)
        if method == 'ListKernels':
            return {'kernels': []}
        raise AssertionError('must not DeleteKernel for nonexistent backup')
    fleet = FleetManager([account], fetcher=rpc)
    assert fleet.stop_kernel_result(account, 'backup/blackthorn-model')['state'] == 'not_running'
    assert 'DeleteKernel' not in calls
