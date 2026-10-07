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
