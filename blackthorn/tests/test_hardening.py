"""Control-plane honesty: turn-off verdicts, secret redaction, and readiness that follows the relay link."""

import pytest

from blackthorn import api
from blackthorn.fleet import FleetError, FleetManager


def _manager_with_rpc(rpc):
    fm = FleetManager.__new__(FleetManager)
    fm.rpc = rpc
    return fm


def _acct(user):
    return type("Acct", (), {"user": user})()


def test_refused_delete_is_reported_as_failed_not_as_stopped(monkeypatch):
    """Kaggle denies kernels.delete while the session still runs: the result must say failed, never not_running."""
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda s: None)
    calls = []

    def rpc(account, method, body):
        calls.append(method)
        if method == "GetKernelSessionStatus":
            return {"status": "RUNNING"}
        raise FleetError("kernels.delete denied")

    fm = _manager_with_rpc(rpc)
    result = fm.stop_kernel_result(_acct("Josh787"), "Josh787/kernel")
    assert result["state"] == "failed"
    assert "kernels.delete denied" in result["error"]
    assert fm.stop_kernel_session(_acct("Josh787"), "Josh787/kernel") is False


def test_unreadable_status_is_failed_not_not_running():
    def rpc(account, method, body):
        raise FleetError("kernels.get denied")

    fm = _manager_with_rpc(rpc)
    result = fm.stop_kernel_result(_acct("Joshbond123"), "Joshbond123/kernel")
    assert result["state"] == "failed" and "kernels.get denied" in result["error"]


def test_a_session_that_is_not_running_is_reported_as_not_running():
    fm = _manager_with_rpc(lambda a, m, b: {"status": "COMPLETE"})
    assert fm.stop_kernel_result(_acct("x"), "x/k") == {"state": "not_running", "error": ""}


def test_successful_stop_is_verified_by_a_second_status_read(monkeypatch):
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda s: None)
    states = iter(["RUNNING", "COMPLETE"])

    def rpc(account, method, body):
        if method == "GetKernelSessionStatus":
            return {"status": next(states)}
        return {}

    fm = _manager_with_rpc(rpc)
    assert fm.stop_kernel_result(_acct("x"), "x/k")["state"] == "stopped"


def test_public_rows_never_carry_the_gateway_key_or_tunnel():
    row = {"status": "HEARTBEAT_ONLINE", "api_key": "sekret", "tunnel_url": "https://t.example", "gpu_info": "T4"}
    pub = api._public_row(row)
    assert "api_key" not in pub and "tunnel_url" not in pub
    assert pub["status"] == "HEARTBEAT_ONLINE" and pub["gpu_info"] == "T4"


class _Hub:
    def __init__(self, **health):
        self._h = health

    def health(self):
        return self._h


class _Req:
    def __init__(self, hubs):
        class _S:
            pass
        s = _S()
        s.relay_hubs = hubs
        s.bt = s
        self.app = type("A", (), {"state": type("St", (), {"bt": s})()})()


def test_online_is_withdrawn_when_the_relay_is_stalled(monkeypatch):
    monkeypatch.setattr(api, "svc", lambda request: request.app.state.bt)
    req = _Req({"model": _Hub(alive=True, stalled=True, consecutive_timeouts=2)})
    out = api._relay_overlay(req, {"online": True, "display_status": "Model ready"})
    assert out["online"] is False and "stalled" in out["display_status"]
    assert out["error"]


def test_online_is_kept_when_the_relay_is_answering(monkeypatch):
    monkeypatch.setattr(api, "svc", lambda request: request.app.state.bt)
    req = _Req({"model": _Hub(alive=True, stalled=False, consecutive_timeouts=0)})
    out = api._relay_overlay(req, {"online": True, "display_status": "Model ready"})
    assert out["online"] is True and out["relay"]["alive"] is True


def test_computer_readiness_is_checked_separately(monkeypatch):
    monkeypatch.setattr(api, "svc", lambda request: request.app.state.bt)
    req = _Req({"model": _Hub(alive=True, stalled=False), "computer": _Hub(alive=True, stalled=True)})
    out = api._relay_overlay(req, {"online": True, "computer": {"online": True}})
    assert out["online"] is True                       # model is fine
    assert out["computer"]["online"] is False          # computer link is stalled


def test_turn_off_message_claims_only_what_was_verified():
    import inspect
    from blackthorn import gpu
    src = inspect.getsource(gpu.GpuService.turn_off)
    assert "did not confirm" in src and "still reports the session running" not in src


def test_an_accepted_delete_that_cannot_be_read_back_is_unverified_not_failed():
    import time as _t
    calls = {"status": 0}

    def rpc(account, method, body):
        if method == "GetKernelSessionStatus":
            calls["status"] += 1
            if calls["status"] == 1:
                return {"status": "RUNNING"}
            raise FleetError("kernels.get denied")
        return {}

    import pytest as _p
    mp = _p.MonkeyPatch()
    mp.setattr(_t, "sleep", lambda s: None)
    try:
        fm = _manager_with_rpc(rpc)
        result = fm.stop_kernel_result(_acct("Joshbond123"), "Joshbond123/k")
    finally:
        mp.undo()
    assert result["state"] == "unverified"
