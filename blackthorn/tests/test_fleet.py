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
    backup = {}
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, 'w') as z:
        z.writestr('uploads/file.txt', 'saved')
        z.writestr('.blackthorn/environment.json', '{}')

    def runner(cmd, env):
        calls.append((cmd[2], env['KAGGLE_USERNAME']))
        assert 'KAGGLE_KEY' not in env and 'KAGGLE_API_TOKEN' in env
        assert env['KAGGLE_API_TOKEN'] == ('primary-token' if env['KAGGLE_USERNAME'] == 'primary' else 'backup-token')
        dest = Path(cmd[cmd.index('-p') + 1])
        if cmd[2] == 'download':
            if env['KAGGLE_USERNAME'] == 'backup' and 'archive' not in backup:
                return 1
            with zipfile.ZipFile(dest / 'dataset.zip', 'w') as z:
                z.writestr('workspace.zip', backup.get('archive', inner.getvalue()))
        else:
            metadata = json.loads((dest / 'dataset-metadata.json').read_text())
            assert metadata['id'] == 'backup/blackthorn-computer-state'
            assert metadata['isPrivate'] is True
            with zipfile.ZipFile(dest / 'workspace.zip') as z:
                assert z.read('uploads/file.txt') == b'saved'
                assert json.loads(z.read('.blackthorn/checkpoint.json'))['schema'] == 2
            backup['archive'] = (dest / 'workspace.zip').read_bytes()
        return 0
    result = mirror_state_dataset(Account('primary', 'primary-token', 'computer', 'primary'),
                                  Account('backup', 'backup-token', 'computer', 'backup'), runner=runner)
    assert result['ok'] and result['digest'] and result['generation']
    assert ('download', 'primary') in calls and ('download', 'backup') in calls
    assert ('create', 'backup') in calls
    again = mirror_state_dataset(Account('primary', 'primary-token', 'computer', 'primary'),
                                 Account('backup', 'backup-token', 'computer', 'backup'), runner=runner)
    assert again['ok'] and again['unchanged']
    assert [action for action, _ in calls].count('create') == 1


def test_backup_computer_restores_its_own_mirrored_dataset():
    from blackthorn.fleet import notebook_env_for
    backup = Account('backup', 'backup-token', 'computer', 'backup')
    env = notebook_env_for(backup, {'COMPUTER_STATE_DATASET': 'primary/blackthorn-computer-state'})
    assert env['COMPUTER_STATE_DATASET'] == 'backup/blackthorn-computer-state'


def test_mirror_preserves_flat_workspace_archive(tmp_path):
    import zipfile
    from pathlib import Path
    from blackthorn.fleet import mirror_state_dataset
    uploaded = {}
    def runner(cmd, env):
        folder = Path(cmd[cmd.index('-p') + 1])
        if cmd[2] == 'download':
            if env['KAGGLE_USERNAME'] == 'backup' and 'workspace' not in uploaded:
                return 1
            with zipfile.ZipFile(folder / 'state.zip', 'w') as z:
                if env['KAGGLE_USERNAME'] == 'backup':
                    z.writestr('workspace.zip', uploaded['workspace'])
                else:
                    z.writestr('uploads/old-file.txt', 'preserved')
                    z.writestr('.blackthorn/environment.json', '{}')
        else:
            with zipfile.ZipFile(folder / 'workspace.zip') as z:
                assert z.read('uploads/old-file.txt') == b'preserved'
                assert z.read('.blackthorn/environment.json') == b'{}'
            uploaded['workspace'] = (folder / 'workspace.zip').read_bytes()
        return 0
    result = mirror_state_dataset(Account('primary','token1','computer','primary'),
                                  Account('backup','token2','computer','backup'), runner=runner)
    assert result['ok'] and not result['unchanged']


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


def test_recently_deleted_kaggle_slug_conflict_retries_with_new_slug_and_title():
    import urllib.error
    account = Account('primary', 'token', 'model', 'primary')
    saves = []
    def rpc(method, body, token):
        if method == 'ListKernels':return {'kernels': []}
        if method == 'GetKernelSessionStatus':return {'status': 'STOPPED'}
        if method == 'SaveKernel':
            saves.append(dict(body))
            if len(saves)==1:
                raise urllib.error.HTTPError('https://www.kaggle.com',409,'title and slug reserved',{},None)
            return {'versionNumber':2}
        raise AssertionError(method)
    fleet=FleetManager([account],fetcher=rpc)
    # Save without a ref and an empty owner listing cannot identify the kernel
    # to create. Report the unconfirmed startup, never guess the old slug.
    with pytest.raises(Exception, match="not found in the authenticated owner listing"):
        fleet.start_kernel(account,slug='primary/blackthorn-model',
                           title='Blackthorn model (primary)',notebook_text='{}')
    assert len(saves)==2
    assert saves[0]['slug']!=saves[1]['slug']
    assert saves[0]['newTitle']!=saves[1]['newTitle']


def test_missing_cache_attachment_never_repushes_an_already_started_kernel():
    account = Account('primary', 'token', 'model', 'primary')
    saves = []
    def rpc(method, body, token):
        if method == 'ListKernels': return {'kernels': []}
        if method == 'GetKernelSessionStatus': return {'status': 'STOPPED'}
        if method == 'SaveKernel':
            saves.append(body)
            if len(saves) > 1: raise AssertionError('second save collides with newly created title')
            return {'versionNumber':1, 'invalidDatasetSources':['primary/cache-not-published-yet']}
        raise AssertionError(method)
    fleet = FleetManager([account], fetcher=rpc)
    with pytest.raises(Exception, match="not found in the authenticated owner listing"):
        fleet.start_kernel(account, slug='primary/blackthorn-model',
                           title='Blackthorn model (primary)', notebook_text='{}',
                           datasets=['primary/cache-not-published-yet'])
    assert len(saves) == 1


def test_delete_denied_but_kernel_disappears_from_listing(monkeypatch):
    import urllib.error
    monkeypatch.setattr('time.sleep', lambda seconds: None)
    account = Account('backup', 'token', 'model', 'backup')
    calls = {'delete': 0}
    def rpc(method, body, token):
        if method == 'GetKernelSessionStatus':
            raise urllib.error.HTTPError('https://www.kaggle.com', 403, 'gone', {}, None)
        if method == 'ListKernels':
            return {'kernels': [] if calls['delete'] >= 3 else [{'ref': 'backup/blackthorn-model', 'title': 'Blackthorn model'}]}
        if method == 'DeleteKernel':
            calls['delete'] += 1
            raise urllib.error.HTTPError('https://www.kaggle.com', 403, 'denied', {}, None)
        raise AssertionError(method)
    fleet = FleetManager([account], fetcher=rpc)
    assert fleet.stop_kernel_result(account, 'backup/blackthorn-model')['state'] == 'not_running'


def test_mirror_retries_transient_dataset_download_404(monkeypatch):
    import zipfile
    from pathlib import Path
    from blackthorn.fleet import mirror_state_dataset
    monkeypatch.setattr('time.sleep', lambda seconds: None)
    count = {'download': 0}
    uploaded = {}
    def runner(cmd, env):
        if cmd[2] == 'download':
            count['download'] += 1
            if count['download'] == 1 or (env['KAGGLE_USERNAME'] == 'backup' and not uploaded):
                return 1
            folder = Path(cmd[cmd.index('-p') + 1])
            with zipfile.ZipFile(folder / 'state.zip', 'w') as z:
                if env['KAGGLE_USERNAME'] == 'backup':
                    z.writestr('workspace.zip', uploaded['data'])
                else:
                    z.writestr('uploads/file.txt', 'saved')
        else:
            uploaded['data'] = (Path(cmd[cmd.index('-p') + 1]) / 'workspace.zip').read_bytes()
        return 0
    def fake_subprocess(cmd, **kwargs):
        import types
        rc = runner(cmd, kwargs['env'])
        return types.SimpleNamespace(returncode=rc, stderr='404 Client Error: Not Found' if rc else '', stdout='')
    monkeypatch.setattr('subprocess.run', fake_subprocess)
    result = mirror_state_dataset(Account('primary', 'token1', 'computer', 'primary'),
                                  Account('backup', 'token2', 'computer', 'backup'))
    assert result['ok'] and count['download'] >= 3


def test_empty_protobuf_listing_proves_backup_absent():
    import urllib.error
    account = Account('backup', 'token', 'model', 'backup')
    def rpc(method, body, token):
        if method == 'GetKernelSessionStatus':
            raise urllib.error.HTTPError('https://www.kaggle.com', 403, 'gone', {}, None)
        if method == 'ListKernels':
            return {}  # Kaggle omits empty repeated `kernels` in protobuf JSON.
        raise AssertionError('no deletion when an authenticated listing is empty')
    assert FleetManager([account], fetcher=rpc).stop_kernel_result(account, 'backup/blackthorn-model')['state'] == 'not_running'


def test_409_recovery_creates_session_for_saved_slug_and_never_assumes_version(monkeypatch):
    from blackthorn.fleet import FleetError
    monkeypatch.setattr('time.sleep', lambda _: None)
    account = Account('tester', 'private-token', 'computer', 'primary')
    seen = []

    def rpc(method, body, token):
        seen.append((method, dict(body or {})))
        if method == 'ListKernels':
            return ({'kernels': [{'title': [p['newTitle'] for m,p in seen if m == 'SaveKernel'][-1],
                                  'ref': 'tester/blackthorn-computer-tester-canonical'}]}
                    if sum(m == 'SaveKernel' for m,_ in seen) >= 2 else {'kernels': []})
        if method == 'GetKernelSessionStatus':
            return {'status': 'STOPPED'}
        if method == 'SaveKernel':
            if sum(name == 'SaveKernel' for name, _ in seen) == 1:
                raise FleetError('HTTP 409 reserved title')
            return {'versionNumber': 7, 'ref': 'unexpected-slug-only-ref'}
        if method == 'CreateKernelSession':
            return {'ok': True}  # accepted submission, not a running session
        raise AssertionError(method)

    fleet = FleetManager([account], fetcher=rpc)
    out = fleet.start_kernel(account, slug='tester/blackthorn-computer',
                             title='Blackthorn computer (tester)', notebook_text='{}')
    saved = [payload for method, payload in seen if method == 'SaveKernel'][-1]
    created = [payload for method, payload in seen if method == 'CreateKernelSession']
    checked = [payload for method, payload in seen[seen.index(('SaveKernel', saved))+1:] if method == 'GetKernelSessionStatus']
    assert created == [{'slug': 'tester/blackthorn-computer-tester-canonical'}]
    assert all(payload['kernelSlug'] == 'blackthorn-computer-tester-canonical' for payload in checked)
    assert out['_session_running'] is False and out['_session_pending'] is True
    assert out['_slug'] == created[0]['slug']
    assert not any(method == 'DeleteKernel' for method, _ in seen)


def test_live_kernel_is_reused_without_save_or_delete():
    account = Account('tester', 'private-token', 'model', 'primary')
    calls = []

    def rpc(method, body, token):
        calls.append(method)
        if method == 'ListKernels':
            return {'kernels': [{'ref': 'tester/blackthorn-model', 'title': 'Blackthorn model (tester)'}]}
        if method == 'GetKernelSessionStatus':
            return {'status': 'RUNNING'}
        raise AssertionError(f'no destructive {method} on reconnect')

    out = FleetManager([account], fetcher=rpc).start_kernel(
        account, slug='tester/blackthorn-model', title='Blackthorn model (tester)', notebook_text='{}')
    assert out['_reused'] and out['_session_running']
    assert calls == ['ListKernels', 'GetKernelSessionStatus']


def test_running_session_reserved_time_is_projected_without_double_counting():
    """Kaggle freezes timeUsed until session exit; reservation is NOT actual use."""
    clock = FakeClock(10000)
    account = Account('model_primary', 'token', 'model', 'primary')
    observed = {'used': 100000}
    def rpc(_method, _body, _token):
        return {'gpuQuota': {'timeUsed': f"{observed['used']}s",
                             'timeReserved': '32400s', 'totalTimeAllowed': '108000s'}}
    fleet = FleetManager([account], fetcher=rpc, clock=clock,
                         usage_baselines={account.user: (10000, 100000)})
    assert fleet.quota(account)['remaining_seconds'] == 8000
    clock.advance(6200)
    assert fleet.quota(account, force=True)['remaining_seconds'] == 1800
    observed['used'] = 102000  # provider began charging 2000 of those elapsed seconds
    assert fleet.quota(account, force=True)['remaining_seconds'] == 1800
    clock.advance(1)
    assert fleet.quota(account, force=True)['remaining_seconds'] == 1799


def test_mirror_does_not_advertise_unverified_backup_on_readback_mismatch(monkeypatch):
    import zipfile
    from pathlib import Path
    from blackthorn.fleet import mirror_state_dataset
    monkeypatch.setattr('time.sleep', lambda _: None)
    def runner(cmd, env):
        path = Path(cmd[cmd.index('-p') + 1])
        if cmd[2] == 'download':
            with zipfile.ZipFile(path / 'state.zip', 'w') as z:
                z.writestr('saved.txt', 'current' if env['KAGGLE_USERNAME'] == 'primary' else 'outdated')
        return 0  # even an apparently successful upload is NOT enough
    out = mirror_state_dataset(Account('primary','t1','computer','primary'),
                               Account('backup','t2','computer','backup'), runner=runner)
    assert not out['ok'] and 'identity did not match' in out['error']


def test_mirror_refuses_stale_download_after_live_checkpoint(monkeypatch):
    import hashlib
    import json
    import zipfile
    from pathlib import Path
    from blackthorn.fleet import mirror_state_dataset
    monkeypatch.setattr('time.sleep', lambda seconds: None)
    files = {'probe.txt': {'size': 5, 'sha256': hashlib.sha256(b'fresh').hexdigest()}}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    downloads = []
    def runner(cmd, env):
        if cmd[2] != 'download':
            raise AssertionError('must never upload an older source snapshot')
        downloads.append(cmd)
        folder = Path(cmd[cmd.index('-p') + 1])
        marker = {'schema': 2, 'files': files, 'digest': digest, 'generation': 41}
        with zipfile.ZipFile(folder / 'state.zip', 'w') as z:
            z.writestr('probe.txt', 'fresh')
            z.writestr('.blackthorn/checkpoint.json', json.dumps(marker))
        return 0
    primary = Account('primary', 'fake-token', 'computer', 'primary')
    backup = Account('backup', 'fake-token-2', 'computer', 'backup')
    result = mirror_state_dataset(primary, backup, runner=runner,
                                  expected_generation=42, expected_digest=digest)
    assert not result['ok'] and 'expected live checkpoint' in result['error']
    assert len(downloads) == 5
