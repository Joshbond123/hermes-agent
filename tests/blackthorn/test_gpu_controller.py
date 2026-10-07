"""GPU controller: instant responses, background work, honest failures."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from blackthorn.gpu import GpuController

pytestmark = pytest.mark.asyncio


class Kaggle:
    """Fake Kaggle RPC that behaves like a session lifecycle."""

    def __init__(self):
        self.calls = []
        self.session = "NONE"
        self.push_delay = 0.0
        self.fail_push = False
        self.payloads = []

    def __call__(self, method, body):
        self.calls.append(method)
        if method == "GetKernelSessionStatus":
            return {"status": self.session}
        if method == "SaveKernel":
            time.sleep(self.push_delay)
            if self.fail_push:
                raise RuntimeError("HTTP 400 bad notebook")
            self.payloads.append(body)
            self.session = "RUNNING"
            return {"versionNumber": 3}
        if method == "DeleteKernel":
            self.session = "NONE"
            return {}
        if method == "GetAcceleratorQuotaStatistics":
            return {"gpuQuota": {"timeUsed": "68699.5s", "totalTimeAllowed": "108000s"},
                    "quotaRefreshTime": "2026-10-10T00:00:00Z"}
        return {}


class Probe:
    def __init__(self):
        self.health = None
        self.calls = 0

    async def __call__(self, url):
        self.calls += 1
        return self.health


def make(fake_d1, **kw):
    kaggle, probe = Kaggle(), Probe()
    ctl = GpuController(rpc=kaggle, probe=probe, build_notebook=lambda key: "NOTEBOOK:" + key)
    return ctl, kaggle, probe


def set_row(fake_d1, status, tunnel="", key="k", detail="", updated=None):
    fake_d1.rows("DELETE FROM kaggle_gpu_state")
    fake_d1.rows("INSERT INTO kaggle_gpu_state (id, status, tunnel_url, api_key, model, gpu_info, updated_at, detail) "
                 "VALUES ('primary', ?, ?, ?, 'qwen', 'Tesla T4', ?, ?)", [status, tunnel, key, updated or time.time(), detail])


async def test_off_snapshot_includes_quota_and_auto_off(fake_d1):
    ctl, kaggle, _ = make(fake_d1)
    set_row(fake_d1, "GPU_STOPPED_SAVING_QUOTA")
    snap = await ctl.refresh(force=True)
    assert snap["state"] == "off"
    assert snap["quota"]["used_hours"] == 19.08 and snap["quota"]["remaining_hours"] == 10.92
    assert snap["auto_off"]["minutes"] == 15 and snap["auto_off"]["choices"] == [0, 5, 10, 15, 30, 60]


async def test_turn_on_returns_immediately_even_when_kaggle_is_slow(fake_d1):
    ctl, kaggle, _ = make(fake_d1)
    kaggle.push_delay = 1.0
    set_row(fake_d1, "GPU_STOPPED_SAVING_QUOTA")
    t0 = time.monotonic()
    result = await ctl.turn_on()
    assert time.monotonic() - t0 < 0.6, "the HTTP handler must not wait for the Kaggle push"
    assert result["ok"] and result["state"] == "starting"
    await asyncio.wait_for(ctl._transition, 10)
    assert kaggle.payloads and json.loads(json.dumps(kaggle.payloads[0]))["enableGpu"] is True
    assert kaggle.payloads[0]["text"].startswith("NOTEBOOK:bt-"), "a fresh per-boot gateway key is injected"
    row = fake_d1.rows("SELECT status, api_key FROM kaggle_gpu_state")[0]
    assert row["status"] == "BOOTING_KAGGLE_GPU" and row["api_key"].startswith("bt-")


async def test_turn_on_is_idempotent_while_starting(fake_d1):
    ctl, kaggle, _ = make(fake_d1)
    set_row(fake_d1, "GPU_STOPPED_SAVING_QUOTA")
    await ctl.turn_on()
    await asyncio.wait_for(ctl._transition, 10)
    again = await ctl.turn_on()
    assert again["note"] == "already starting" and len(kaggle.payloads) == 1


async def test_push_failure_is_visible_as_an_error_not_a_stuck_spinner(fake_d1):
    ctl, kaggle, _ = make(fake_d1)
    kaggle.fail_push = True
    set_row(fake_d1, "GPU_STOPPED_SAVING_QUOTA")
    await ctl.turn_on()
    await asyncio.wait_for(ctl._transition, 30)
    snap = await ctl.refresh(force=True)
    assert snap["state"] == "error" and "refused the notebook push" in snap["error"]["message"]


async def test_full_boot_to_ready_and_route(fake_d1):
    ctl, kaggle, probe = make(fake_d1)
    set_row(fake_d1, "GPU_STOPPED_SAVING_QUOTA")
    assert await ctl.route() is None
    await ctl.turn_on()
    await asyncio.wait_for(ctl._transition, 10)
    # notebook reports progress
    set_row(fake_d1, "DOWNLOADING_MODEL", detail=json.dumps({"done": 8e9, "total": 16e9}))
    kaggle.session = "RUNNING"
    snap = await ctl.refresh(force=True)
    assert snap["state"] == "starting" and snap["progress"]["fraction"] == 0.5
    assert await ctl.route() is None, "not ready: no chat traffic is routed yet"
    # model comes up
    set_row(fake_d1, "MODEL_READY_AND_WARMED", tunnel="https://x.trycloudflare.com/v1", key="gw-key")
    probe.health = {"status": "online", "model_loaded": True}
    route = await ctl.route()
    assert route is not None and route.url == "https://x.trycloudflare.com" and route.api_key == "gw-key"
    assert route.model == "qwen"
    assert ctl.snapshot()["state"] == "ready"


async def test_status_snapshot_is_instant_and_never_does_io(fake_d1):
    ctl, kaggle, probe = make(fake_d1)
    set_row(fake_d1, "GPU_STOPPED_SAVING_QUOTA")
    await ctl.refresh(force=True)
    calls_before, rpc_before = fake_d1.calls, len(kaggle.calls)
    t0 = time.perf_counter()
    for _ in range(1000):
        ctl.snapshot()
    assert time.perf_counter() - t0 < 0.5
    assert fake_d1.calls == calls_before and len(kaggle.calls) == rpc_before


async def test_turn_off_stops_the_session_and_records_off(fake_d1):
    ctl, kaggle, _ = make(fake_d1)
    kaggle.session = "RUNNING"
    set_row(fake_d1, "MODEL_READY_AND_WARMED", tunnel="https://x", key="k")
    result = await ctl.turn_off()
    assert result["ok"] and result["state"] == "stopping"
    await asyncio.wait_for(ctl._transition, 10)
    assert "DeleteKernel" in kaggle.calls
    row = fake_d1.rows("SELECT status, tunnel_url, api_key FROM kaggle_gpu_state")[0]
    assert row == {"status": "GPU_STOPPED_SAVING_QUOTA", "tunnel_url": "", "api_key": ""}
    assert (await ctl.refresh(force=True))["state"] == "off"


async def test_restart_stops_then_pushes_a_fresh_notebook(fake_d1):
    ctl, kaggle, _ = make(fake_d1)
    kaggle.session = "RUNNING"
    set_row(fake_d1, "LOADING_MODEL", updated=time.time() - 900)
    await ctl.restart()
    await asyncio.wait_for(ctl._transition, 30)
    assert kaggle.calls.index("DeleteKernel") < kaggle.calls.index("SaveKernel")
    assert fake_d1.rows("SELECT status FROM kaggle_gpu_state")[0]["status"] == "BOOTING_KAGGLE_GPU"


async def test_stalled_boot_is_reported(fake_d1):
    ctl, kaggle, _ = make(fake_d1)
    kaggle.session = "RUNNING"
    set_row(fake_d1, "LOADING_MODEL", updated=time.time() - 800)
    snap = await ctl.refresh(force=True)
    assert snap["state"] == "starting" and snap["stalled"] is True


async def test_auto_off_after_idle_but_not_while_a_turn_is_running(fake_d1):
    clock = [1000.0]
    kaggle, probe = Kaggle(), Probe()
    ctl = GpuController(rpc=kaggle, probe=probe, build_notebook=lambda k: "NB", clock=lambda: clock[0])
    kaggle.session = "RUNNING"
    set_row(fake_d1, "MODEL_READY_AND_WARMED", tunnel="https://x", updated=clock[0])
    probe.health = {"status": "online", "model_loaded": True}
    await ctl.set_auto_off_minutes(5)
    ctl.mark_activity()
    snap = await ctl.refresh(force=True)
    assert snap["state"] == "ready" and snap["auto_off"]["remaining_s"] == 300
    clock[0] += 200
    await ctl._maybe_auto_off(await ctl.refresh(force=True))
    assert ctl._transition is None, "not idle long enough"
    clock[0] += 200  # 400 s idle > 5 min
    ctl.task_started()
    await ctl._maybe_auto_off(await ctl.refresh(force=True))
    assert ctl._transition is None, "a running turn keeps the GPU alive"
    ctl.task_finished()
    clock[0] += 400
    await ctl._maybe_auto_off(await ctl.refresh(force=True))
    await asyncio.wait_for(ctl._transition, 10)
    assert fake_d1.rows("SELECT status FROM kaggle_gpu_state")[0]["status"] == "GPU_STOPPED_SAVING_QUOTA"
    assert "without activity" in fake_d1.rows("SELECT value FROM state_meta WHERE key='blackthorn_auto_off_last_reason'")[0]["value"]


async def test_auto_off_never_setting_keeps_the_gpu(fake_d1):
    clock = [1000.0]
    kaggle, probe = Kaggle(), Probe()
    ctl = GpuController(rpc=kaggle, probe=probe, build_notebook=lambda k: "NB", clock=lambda: clock[0])
    kaggle.session = "RUNNING"
    set_row(fake_d1, "ONLINE", tunnel="https://x", updated=clock[0])
    probe.health = {"status": "online", "model_loaded": True}
    await ctl.set_auto_off_minutes(0)
    clock[0] += 99999
    await ctl._maybe_auto_off(await ctl.refresh(force=True))
    assert ctl._transition is None
    with pytest.raises(ValueError):
        await ctl.set_auto_off_minutes(7)


async def test_legacy_table_without_detail_column_is_upgraded(fake_d1):
    fake_d1.rows("DROP TABLE kaggle_gpu_state")
    fake_d1.rows("CREATE TABLE kaggle_gpu_state (id TEXT PRIMARY KEY, status TEXT NOT NULL, tunnel_url TEXT NOT NULL "
                 "DEFAULT '', api_key TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT 'm', gpu_info TEXT NOT NULL "
                 "DEFAULT '', updated_at REAL NOT NULL)")
    fake_d1.rows("INSERT INTO kaggle_gpu_state (id, status, updated_at) VALUES ('primary', 'GPU_STOPPED_SAVING_QUOTA', 1)")
    ctl, kaggle, _ = make(fake_d1)
    assert (await ctl.refresh(force=True))["state"] == "off"  # legacy select works
    await ctl.turn_on()
    await asyncio.wait_for(ctl._transition, 10)
    assert fake_d1.rows("SELECT detail FROM kaggle_gpu_state")[0]["detail"] is not None  # column was added


async def test_tunnel_failure_report_forces_a_fresh_check(fake_d1):
    ctl, kaggle, probe = make(fake_d1)
    kaggle.session = "RUNNING"
    set_row(fake_d1, "ONLINE", tunnel="https://x")
    probe.health = {"status": "online", "model_loaded": True}
    await ctl.refresh(force=True)
    probe.health = None
    ctl.report_tunnel_failure("530")
    snap = await ctl.refresh()  # not forced, but the report invalidated the snapshot
    assert probe.calls >= 2 and snap["state"] != "ready"


async def test_verify_inference_runs_a_real_completion(fake_d1, upstream):
    ctl, kaggle, probe = make(fake_d1)
    kaggle.session = "RUNNING"
    set_row(fake_d1, "ONLINE", tunnel=upstream.url, key="gw")
    probe.health = {"status": "online", "model_loaded": True}
    await ctl.refresh(force=True)
    result = await ctl.verify_inference()
    assert result["ok"] is True and result["reply"] == "OK" and result["latency_ms"] >= 0
    assert upstream.requests[-1]["max_tokens"] == 24 and upstream.requests[-1]["stream"] is False
    upstream.verify_reply = ""
    bad = await ctl.verify_inference()
    assert bad["ok"] is False
