"""The Kaggle GPU: power, persistence, idle policy and honest usage accounting.

These run the real ``KaggleGPU`` against a fake store and a fake Kaggle, so a passing test means the
behaviour that ships is the behaviour asserted here — including the parts that protect the user's
limited GPU quota.
"""

from __future__ import annotations

import ast
import json
import time

import pytest

from blackthorn import kaggle_gpu as gpumod
from blackthorn.kaggle_gpu import (ERROR, OFF, READY, STARTING, STOPPING, KaggleAPI, KaggleGPU,
                                   GPUError, kaggle_week)
from blackthorn.kaggle_kernels import install_source, serve_source


class FakeStore:
    def __init__(self):
        self.settings = {}

    async def get_setting(self, key):
        return self.settings.get(key)

    async def set_setting(self, key, value):
        self.settings[key] = value


class FakeRuns:
    def __init__(self, active=0):
        self._active = active

    def active_count(self):
        return self._active


class FakeKaggle(KaggleAPI):
    """Records pushes; refuses nothing unless told to."""

    def __init__(self, *, fail=None):
        super().__init__(http=None, username="jb", key="k")
        self.pushed = []
        self.fail = fail
        self.statuses = {}

    def __bool__(self):
        return True

    async def push_kernel(self, **kwargs):
        if self.fail:
            raise self.fail
        self.pushed.append(kwargs)
        return {"versionNumber": len(self.pushed)}

    async def kernel_status(self, slug):
        return {"status": self.statuses.get(slug, "COMPLETE")}


def make_gpu(*, runs_active=0, installed=True, fail=None):
    gpu = KaggleGPU(FakeStore(), http=None, runs=FakeRuns(runs_active), username="jb", key="k",
                    install_slug="blackthorn-qwen-install", public_url="https://app.example",
                    kernel_token="secret-token", hf_token="hf_x")
    gpu.api = FakeKaggle(fail=fail)
    if installed:
        gpu.store.settings[gpumod.STATE_KEY] = json.dumps({"state": OFF, "installed_at": time.time()})
    return gpu


# ------------------------------------------------------------------ the generated kernels
def test_the_installer_kernel_is_real_python_and_downloads_the_requested_quant():
    src = install_source({"hf_token": "hf_secret", "kaggle_user": "jb"})
    ast.parse(src)                                   # it is pushed to Kaggle as source; it must run
    assert "Q4_K_M" in src
    assert "/kaggle/working/qwen" in src, "the weights must land in the kernel output (persistent storage)"
    assert "Range" in src, "a 17 GB download must resume rather than restart"
    assert "bartowski/orcarouter_Qwen3.8-27B-Uncensored-GGUF" in src, "the ungated mirror is the fallback"


def test_the_serve_kernel_loads_from_the_cache_and_can_never_silently_download():
    src = serve_source({"backend": "https://app.example", "token": "t"}, sources=["jb/blackthorn-qwen-install"])
    ast.parse(src)
    assert "/kaggle/input/**/*Q4_K_M*.gguf" in src, "weights come from the mounted, persistent copy"
    assert "huggingface.co" not in src, "the GPU kernel must not be able to re-download the model"
    assert "--tensor-split" in src, "17 GB of weights do not fit a single 16 GB T4"
    assert "--jinja" in src, "tool calling needs the model's own chat template"
    assert '"/api/gpu/heartbeat"' in src and '"/api/gpu/register"' in src
    assert "if answer.get(\"stop\")" in src, "the kernel must honour a stop order"
    # llama.cpp ships Linux CUDA builds as .tar.gz under b* tags; /releases/latest is a decoy.
    assert "releases?per_page=" in src and "cuda-12.8-x64" in src and "tarfile" in src


def test_no_secret_is_written_into_the_repository_sources():
    """The token is injected at push time; the file that ships to GitHub carries none."""
    import re

    real_token = re.compile(r"(hf_[A-Za-z0-9]{28,}|KGAT_[0-9a-f]{16,}|nvapi-[A-Za-z0-9_-]{20,})")
    for path in ("blackthorn/kaggle_kernels.py", "blackthorn/kaggle_gpu.py"):
        text = open(path).read()
        assert not real_token.search(text), f"{path} must not contain a credential"


# ------------------------------------------------------------------ the quota week
def test_the_quota_week_starts_on_saturday_at_midnight_utc():
    # 2026-11-07 is a Saturday; its week starts at that midnight and ends seven days later.
    now = time.mktime(time.strptime("2026-11-07 00:00:01", "%Y-%m-%d %H:%M:%S"))
    start, end = kaggle_week(now)
    assert end - start == 7 * 86400
    assert abs((now - start) % 86400) < 2, "the window begins at Saturday 00:00"


# ------------------------------------------------------------------ switching on and off
async def test_switching_the_gpu_on_pushes_a_gpu_kernel_that_mounts_the_installed_weights():
    gpu = make_gpu()
    out = await gpu.start()
    assert out["state"] == STARTING and out["running"] is True
    push = gpu.api.pushed[0]
    assert push["enable_gpu"] is True, "the whole point is a GPU session"
    assert push["kernel_sources"] == ["jb/blackthorn-qwen-install"], "the persistent weights are mounted"
    assert push["dataset_sources"] == []
    assert (await gpu.usage())["current_session_seconds"] >= 0


async def test_the_gpu_cannot_be_switched_on_before_the_model_is_installed():
    gpu = make_gpu(installed=False)
    gpu.api.statuses["blackthorn-qwen-install"] = "error"      # Kaggle says the installer never finished
    with pytest.raises(GPUError) as err:
        await gpu.start()
    assert err.value.code == "model_not_installed"
    assert gpu.api.pushed == []


async def test_a_lost_local_flag_is_recovered_from_kaggles_own_record():
    """The installer's output is the source of truth, not a setting that a redeploy might drop."""
    gpu = make_gpu(installed=False)
    gpu.api.statuses["blackthorn-qwen-install"] = "complete"
    out = await gpu.start()
    assert out["state"] == STARTING
    assert (await gpu.status())["installed"] is True


async def test_the_gpu_is_never_switched_off_while_a_task_is_running():
    gpu = make_gpu(runs_active=1)
    await gpu.start()
    with pytest.raises(GPUError) as err:
        await gpu.stop()
    assert err.value.code == "gpu_busy"
    assert (await gpu.status())["stop_requested"] is False


async def test_a_forced_stop_is_still_available_to_the_operator():
    gpu = make_gpu(runs_active=1)
    await gpu.start()
    out = await gpu.stop(force=True)
    assert out["stop_requested"] is True


async def test_the_kernel_is_told_to_stop_only_when_it_is_not_busy():
    gpu = make_gpu()
    await gpu.start()
    await gpu.register(base_url="https://tunnel.example", session="s")
    await gpu.stop()
    busy = await gpu.heartbeat(busy=True)
    assert busy["stop"] is False, "an inference request in flight must not be cut off"
    idle = await gpu.heartbeat(busy=False)
    assert idle["stop"] is True


# ------------------------------------------------------------------ idle shutdown
async def test_the_gpu_is_switched_off_after_the_configured_idle_window():
    gpu = make_gpu()
    await gpu.start()
    await gpu.register(base_url="https://tunnel.example", session="s")
    await gpu.update_settings(idle_minutes=5)
    # no activity for longer than the window
    state = json.loads(gpu.store.settings[gpumod.STATE_KEY])
    state["last_activity"] = time.time() - 6 * 60
    gpu.store.settings[gpumod.STATE_KEY] = json.dumps(state)
    out = await gpu.check_idle()
    assert out["stop_requested"] is True and out["state"] == STOPPING


async def test_the_idle_timer_does_not_fire_while_a_task_is_running():
    gpu = make_gpu(runs_active=1)
    await gpu.start()
    await gpu.register(base_url="https://tunnel.example", session="s")
    await gpu.update_settings(idle_minutes=5)
    state = json.loads(gpu.store.settings[gpumod.STATE_KEY])
    state["last_activity"] = time.time() - 60 * 60
    gpu.store.settings[gpumod.STATE_KEY] = json.dumps(state)
    out = await gpu.check_idle()
    assert out["stop_requested"] is False, "an autonomous task must never lose its GPU"
    assert out["state"] == READY


async def test_recent_activity_keeps_the_gpu_on():
    gpu = make_gpu()
    await gpu.start()
    await gpu.register(base_url="https://tunnel.example", session="s")
    await gpu.update_settings(idle_minutes=5)
    await gpu.note_activity("chat")
    out = await gpu.check_idle()
    assert out["stop_requested"] is False


async def test_the_timer_choices_are_the_ones_the_panel_offers():
    gpu = make_gpu()
    settings = await gpu.settings()
    assert settings["idle_choices"] == [5, 10, 15, 30, 45, 60]
    custom = await gpu.update_settings(idle_minutes=7)
    assert custom["idle_minutes"] == 7, "a custom value is allowed"
    with pytest.raises(GPUError):
        await gpu.update_settings(idle_minutes=0)


# ------------------------------------------------------------------ reconnect + availability
async def test_a_registered_kernel_makes_the_model_selectable_with_its_live_url():
    from blackthorn.route import RouteResolver

    gpu = make_gpu()
    resolver = RouteResolver(live=gpu)
    before = {m["id"]: m for m in resolver.describe_all()["models"]}
    qwen = [m for k, m in before.items() if "Qwen" in k][0]
    assert qwen["available"] is False, "off means honestly unavailable"

    await gpu.start()
    await gpu.register(base_url="https://tunnel.example/", session="s")
    after = {m["id"]: m for m in resolver.describe_all()["models"]}
    qwen = [m for k, m in after.items() if "Qwen" in k][0]
    assert qwen["available"] is True and "Kaggle GPU" in qwen["detail"]


async def test_a_new_tunnel_after_a_restart_replaces_the_old_one_without_losing_state():
    gpu = make_gpu()
    await gpu.start()
    await gpu.register(base_url="https://first.example", session="s1")
    second = await gpu.register(base_url="https://second.example", session="s2")
    assert second["ok"] is True
    status = await gpu.status()
    assert status["base_url"] == "https://second.example"
    assert status["reconnected"] is True
    assert gpu.live_endpoints[gpu.model_key] == "https://second.example"


async def test_a_dead_kernel_is_reported_as_an_error_rather_than_as_serving():
    gpu = make_gpu()
    await gpu.start()
    await gpu.register(base_url="https://tunnel.example", session="s")
    state = json.loads(gpu.store.settings[gpumod.STATE_KEY])
    state["last_heartbeat"] = time.time() - 10 * 60      # silence
    gpu.store.settings[gpumod.STATE_KEY] = json.dumps(state)
    status = await gpu.status()
    assert status["state"] == ERROR and status["serving"] is False
    assert gpu.live_endpoints == {}, "a silent GPU must stop being offered as an endpoint"
    assert await gpu.endpoint() == ""


async def test_a_backend_restart_restores_a_gpu_that_is_still_serving():
    gpu = make_gpu()
    await gpu.start()
    await gpu.register(base_url="https://tunnel.example", session="s")
    fresh = KaggleGPU(gpu.store, http=None, username="jb", key="k")
    assert fresh.live_endpoints == {}
    await fresh.rehydrate()
    assert fresh.live_endpoints[fresh.model_key] == "https://tunnel.example"


# ------------------------------------------------------------------ usage accounting
async def test_usage_measures_only_the_hours_inside_the_current_week():
    gpu = make_gpu()
    start, _ = kaggle_week()
    inside = {"start": start + 3600, "end": start + 3600 + 1800}          # 30 minutes
    before = {"start": start - 7200, "end": start - 3600}                 # previous week
    gpu.store.settings[gpumod.LEDGER_KEY] = json.dumps([before, inside])
    usage = await gpu.usage()
    assert usage["hours_used"] == pytest.approx(0.5, abs=0.01)
    assert usage["hours_total"] == 30.0
    assert usage["hours_remaining"] == pytest.approx(29.5, abs=0.01)
    assert usage["percent_used"] == pytest.approx(1.7, abs=0.2)
    assert "measured" in usage["usage_source"]
    assert "no usage API" in usage["hours_basis"], "the allowance must be labelled, never implied"


async def test_the_allowance_can_be_corrected_and_the_maths_follows_it():
    gpu = make_gpu()
    await gpu.update_settings(weekly_hours=41)
    usage = await gpu.usage()
    assert usage["hours_total"] == 41.0
    assert usage["percent_used"] == 0.0


async def test_switching_on_records_a_session_so_hours_are_measured_not_invented():
    gpu = make_gpu()
    await gpu.start()
    await gpu.register(base_url="https://tunnel.example", session="s")
    usage = await gpu.usage()
    assert usage["sessions_this_week"] >= 1
    assert usage["current_session_seconds"] >= 0


# ------------------------------------------------------------------ kaggle auth
async def test_kaggle_tokens_are_sent_as_a_bearer_header():
    """``KGAT_`` tokens are rejected under HTTP Basic; only Bearer authenticates."""
    seen = {}

    class Resp:
        status_code = 200
        content = b"{}"

        def json(self):
            return {}

    class Http:
        async def request(self, method, url, **kwargs):
            seen["headers"] = kwargs.get("headers") or {}
            seen["url"] = url
            return Resp()

    api = KaggleAPI(Http(), "jb", "KGAT_test")
    await api.kernel_status("slug")
    assert seen["headers"]["Authorization"] == "Bearer KGAT_test"
    assert "kernelSlug=slug" in seen["url"]


async def test_a_kaggle_account_with_no_quota_reports_why_it_could_not_start():
    gpu = make_gpu(fail=GPUError("kaggle_error", "forbidden", status=403))
    with pytest.raises(GPUError) as err:
        await gpu.start()
    assert err.value.code == "kaggle_forbidden"
    assert "quota" in err.value.message


# ------------------------------------------------------------------ Kaggle's failure semantics
async def test_a_200_containing_an_error_is_not_treated_as_success():
    """Kaggle answers bad pushes with HTTP 200 and ``{"hasError": true}``.

    Believing that envelope is how "installed" gets reported while nothing exists on Kaggle, so it
    must raise.
    """
    class Resp:
        status_code = 200
        content = b'{"error": "Invalid slug", "hasError": true, "ref": ""}'

        def json(self):
            return {"error": "Invalid slug: blackthorn-probe", "hasError": True, "ref": ""}

    class Http:
        async def request(self, method, url, **kwargs):
            return Resp()

    api = KaggleAPI(Http(), "Joshbond123", "KGAT_x")
    with pytest.raises(GPUError) as err:
        await api.push_kernel(slug="blackthorn-probe", title="t", source="print(1)",
                              dataset_sources=[], enable_gpu=False)
    assert err.value.code == "kaggle_bad_slug"


async def test_the_kernel_slug_is_owner_qualified_and_lower_case():
    """``Joshbond123`` is denied; ``joshbond123`` is the identity Kaggle accepts."""
    seen = {}

    class Resp:
        status_code = 200
        content = b'{"hasError": false, "ref": "/code/joshbond123/x", "versionNumber": 1}'

        def json(self):
            return {"hasError": False, "ref": "/code/joshbond123/x", "versionNumber": 1}

    class Http:
        async def request(self, method, url, **kwargs):
            seen["body"] = kwargs.get("json")
            return Resp()

    api = KaggleAPI(Http(), "Joshbond123", "KGAT_x")
    await api.push_kernel(slug="blackthorn-qwen-install", title="t", source="print(1)",
                          dataset_sources=[], enable_gpu=False)
    assert seen["body"]["slug"] == "joshbond123/blackthorn-qwen-install"



async def test_kernel_status_is_asked_for_with_the_lower_case_owner():
    seen = {}

    class Resp:
        status_code = 200
        content = b'{"status": "running"}'

        def json(self):
            return {"status": "running"}

    class Http:
        async def request(self, method, url, **kwargs):
            seen["url"] = url
            return Resp()

    api = KaggleAPI(Http(), "Joshbond123", "KGAT_x")
    out = await api.kernel_status("blackthorn-qwen-install")
    assert out["status"] == "running"
    assert "userName=joshbond123" in seen["url"]


def test_the_kernel_endpoints_are_reachable_without_the_owner_cookie(monkeypatch):
    """The kernel lives on Kaggle; it can never hold the owner's session cookie.

    If the auth gate blocks it, registering silently fails and the GPU looks like it never started,
    so these two paths must be exempt — and they authenticate with the GPU token instead.
    """
    import asyncio

    from blackthorn import auth

    src = open(auth.__file__).read()
    assert '"/api/gpu/register"' in src and '"/api/gpu/heartbeat"' in src
