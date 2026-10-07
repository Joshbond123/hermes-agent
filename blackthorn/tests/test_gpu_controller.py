"""The real cloudflare_d1_client.get_kaggle_gpu_status under the scenario that misreported the quota in production:
a freshly started process that finds the GPU already online."""

import json
import time

import cloudflare_d1_client as d1


class _Resp:
    def __init__(self, payload): self._b = json.dumps(payload).encode()
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _prepare(monkeypatch, status="HEARTBEAT_ONLINE"):
    for name, value in {"_STATUS_CACHE": None, "_STATUS_CACHE_TS": 0.0, "_QUOTA_CACHE": None, "_QUOTA_CACHE_TS": 0.0,
                        "_STATUS_REFRESH_IN_FLIGHT": False}.items():
        monkeypatch.setattr(d1, name, value)
    row = {"id": "primary", "status": status, "tunnel_url": "https://tunnel.example", "api_key": "k", "model": "m",
           "gpu_info": "Tesla T4\nTesla T4", "updated_at": time.time(), "detail": ""}
    rpc_calls = []

    def fake_query(sql, params=None, timeout=15.0):
        return [row] if "FROM kaggle_gpu_state" in sql and sql.strip().upper().startswith("SELECT *") else []

    def fake_rpc(method, body):
        rpc_calls.append(method)
        if method == "GetAcceleratorQuotaStatistics":
            return {"gpuQuota": {"timeUsed": "71094.4s", "totalTimeAllowed": "108000s", "timeReserved": "0s"},
                    "quotaRefreshTime": "2026-10-10T00:00:00Z"}
        return {"status": "RUNNING"}

    monkeypatch.setattr(d1, "d1_query", fake_query)
    monkeypatch.setattr(d1, "_kaggle_rpc", fake_rpc)
    monkeypatch.setattr(d1, "_ensure_kaggle_credentials", lambda: None)
    monkeypatch.setattr(d1, "ensure_d1_schema", lambda: None)
    monkeypatch.setattr(d1, "update_hermes_model_endpoint", lambda *a, **k: None)
    monkeypatch.setattr(d1.urllib.request, "urlopen", lambda *a, **k: _Resp({"status": "online", "model_loaded": True, "gpu": "Tesla T4\nTesla T4"}))
    return rpc_calls


def test_quota_is_never_reported_as_zero_when_it_is_simply_unknown(monkeypatch):
    calls = _prepare(monkeypatch)
    first = d1.get_kaggle_gpu_status(False)                    # an HTTP poll right after process start
    assert first["status"] == "HEARTBEAT_ONLINE" and first["active"] is True
    assert "quota" not in first, "unknown quota must be omitted, not shown as '0 h used'"
    assert "GetAcceleratorQuotaStatistics" not in calls        # polls never block on a Kaggle round trip


def test_watchdog_tick_loads_the_real_quota_and_polls_serve_it_from_cache(monkeypatch):
    calls = _prepare(monkeypatch)
    tick = d1.get_kaggle_gpu_status(True)                      # what the permanent watchdog does every 20 s
    assert tick["quota"]["used_hours"] == 19.75 and tick["quota"]["total_hours"] == 30.0
    assert tick["quota"]["refresh_time"] == "2026-10-10T00:00:00Z" and tick["quota"]["remaining_hours"] == 10.25
    d1._STATUS_CACHE_TS = 0.0                                  # expire the status cache so the fast path recomputes
    before = calls.count("GetAcceleratorQuotaStatistics")
    poll = d1.get_kaggle_gpu_status(False)
    assert poll["quota"]["used_hours"] == 19.75
    assert calls.count("GetAcceleratorQuotaStatistics") == before          # served from the cache
    public = d1.public_gpu_status(poll)
    assert public["quota"]["used_hours"] == 19.75 and "tunnel_url" not in public and "api_key" not in public


def test_public_status_omits_unknown_values_instead_of_inventing_them():
    out = d1.public_gpu_status({"active": True, "booting": False, "status": "ONLINE", "model_loaded": None, "quota": None, "tunnel_url": "x"})
    assert "model_loaded" not in out and "quota" not in out and out["online"] is True


# ---------------------------------------------------------------------------------------------------------------
# Regression: a confirmed-dead tunnel URL was remembered and resurrected, which overwrote the real progress of the
# recovery boot with TUNNEL_ERROR ("Connection Lost" while the GPU was actually starting).
# ---------------------------------------------------------------------------------------------------------------
import urllib.error


def _capture(monkeypatch, row, health):
    """Run the real status function with a controllable D1 row and health probe; record every D1 write + probed URL."""
    for name, value in {"_STATUS_CACHE": None, "_STATUS_CACHE_TS": 0.0, "_QUOTA_CACHE": None, "_QUOTA_CACHE_TS": 0.0,
                        "_STATUS_REFRESH_IN_FLIGHT": False}.items():
        monkeypatch.setattr(d1, name, value)
    writes, probed = [], []

    def fake_query(sql, params=None, timeout=15.0):
        if sql.strip().upper().startswith("SELECT * FROM KAGGLE_GPU_STATE"):
            return [dict(row)]
        if sql.strip().upper().startswith(("UPDATE", "INSERT", "DELETE")):
            writes.append((sql, list(params or [])))
        return []

    def fake_urlopen(req, timeout=None):
        url = getattr(req, "full_url", str(req))
        probed.append(url)
        return health(url)

    monkeypatch.setattr(d1, "d1_query", fake_query)
    monkeypatch.setattr(d1, "_kaggle_rpc", lambda m, b: {"status": "RUNNING"})
    monkeypatch.setattr(d1, "_ensure_kaggle_credentials", lambda: None)
    monkeypatch.setattr(d1, "ensure_d1_schema", lambda: None)
    monkeypatch.setattr(d1, "update_hermes_model_endpoint", lambda *a, **k: None)
    monkeypatch.setattr(d1.urllib.request, "urlopen", fake_urlopen)
    return writes, probed


def _dead(url):
    raise urllib.error.HTTPError(url, 530, "tunnel gone", {}, None)


def _row(status, url="", age=5.0):
    return {"id": "primary", "status": status, "tunnel_url": url, "api_key": "k", "model": "m", "gpu_info": "T4", "updated_at": time.time() - age, "detail": ""}


def test_a_boot_in_progress_is_never_overwritten_by_a_remembered_dead_url(monkeypatch):
    monkeypatch.setattr(d1, "_LAST_SAVED_TUNNEL_URL", "https://dead.example")      # left over from the session that died
    writes, probed = _capture(monkeypatch, _row("DOWNLOADING_MODEL"), _dead)
    state = d1.get_kaggle_gpu_status(True)
    assert state["status"] == "DOWNLOADING_MODEL" and state["booting"] is True        # the real boot stage is reported
    assert not [w for w in writes if "TUNNEL_ERROR" in str(w)], writes                 # and never replaced by an error
    assert not [u for u in probed if "dead.example" in u], probed                       # the dead URL is not even probed
    assert not [w for w in writes if "SET tunnel_url" in w[0]], writes                  # and never written back into D1


def test_a_gpu_that_was_online_and_lost_its_tunnel_is_marked_once_and_the_url_is_forgotten(monkeypatch):
    monkeypatch.setattr(d1, "_LAST_SAVED_TUNNEL_URL", "https://t.example")
    writes, _ = _capture(monkeypatch, _row("HEARTBEAT_ONLINE", "https://t.example"), _dead)
    state = d1.get_kaggle_gpu_status(True)
    assert state["status"] == "TUNNEL_ERROR" and state["active"] is False
    marks = [w for w in writes if "TUNNEL_ERROR" in str(w[1])]
    assert len(marks) == 1 and "AND status IN (" in marks[0][0] and "'HEARTBEAT_ONLINE'" in marks[0][0]   # guarded write
    assert d1._LAST_SAVED_TUNNEL_URL == ""                                                                  # forgotten for good
    # the next look at the (now error) row must not bring the dead URL back or write it into D1
    writes2, probed2 = _capture(monkeypatch, _row("TUNNEL_ERROR"), _dead)
    again = d1.get_kaggle_gpu_status(True)
    assert again["status"] == "TUNNEL_ERROR" and not [u for u in probed2 if "t.example" in u]               # the dead URL is not probed
    assert not [w for w in writes2 if "SET tunnel_url" in w[0]]                                              # nor written back to D1


def test_a_new_tunnel_published_by_the_notebook_is_trusted_and_remembered(monkeypatch):
    monkeypatch.setattr(d1, "_LAST_SAVED_TUNNEL_URL", "")
    ok = lambda url: _Resp({"status": "online", "model_loaded": True, "gpu": "T4"})   # noqa: E731
    _capture(monkeypatch, _row("HEARTBEAT_ONLINE", "https://fresh.example"), ok)
    state = d1.get_kaggle_gpu_status(True)
    assert state["status"] == "HEARTBEAT_ONLINE" and state["active"] is True and state["model_loaded"] is True
