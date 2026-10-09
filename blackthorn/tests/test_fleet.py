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
