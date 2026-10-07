"""Offline tests for the code that runs on Kaggle (cyber_ornith_server.py): the real-progress reporter and the download
paths it instruments. External tools (curl, ollama) are simulated; nothing here touches the network."""

import importlib.util
import json
import os
import subprocess
import zipfile
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "kaggle_cyber_ornith" / "cyber_ornith_server.py"


@pytest.fixture
def srv(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("cyber_ornith_server_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    posts = []
    monkeypatch.setattr(mod, "_d1_exec", lambda sql, params: posts.append((sql, params)))
    monkeypatch.setattr(mod, "log", lambda *a, **k: None)
    mod.posts = posts
    mod.tmp = tmp_path
    return mod


def detail_of(post):
    return json.loads(post[1][0])


def test_tick_publishes_real_bytes_and_refreshes_updated_at_only_while_moving(srv):
    size = {"n": 100}
    srv.progress_stage("DOWNLOADING_MODEL")
    t0 = srv.PROGRESS.since
    with srv.progress_probe("Downloading the model", 1000, lambda: size["n"]):
        assert srv.progress_tick(t0 + 10) is True
        d = detail_of(srv.posts[-1])
        assert (d["bytes_done"], d["bytes_total"], d["label"]) == (100, 1000, "Downloading the model")
        assert "updated_at = ?" in srv.posts[-1][0] and "status = ?" in srv.posts[-1][0]   # guarded: never clobbers a newer stage
        size["n"] = 250
        assert srv.progress_tick(t0 + 20) is True and detail_of(srv.posts[-1])["bytes_done"] == 250
        # the transfer freezes: after the stall window we STOP refreshing updated_at so the watchdog can heal a real hang
        assert srv.progress_tick(t0 + 20 + srv.PROGRESS_STALL_SECONDS + 1) is False
        assert "updated_at" not in srv.posts[-1][0] and detail_of(srv.posts[-1])["stalled"] is True
        size["n"] = 400                                                                    # it moves again: alive again
        assert srv.progress_tick(t0 + 20 + srv.PROGRESS_STALL_SECONDS + 5) is True


def test_non_measurable_stage_is_alive_for_a_bounded_time(srv):
    srv.progress_stage("LOADING_MODEL")
    t0 = srv.PROGRESS.since
    assert srv.progress_tick(t0 + 30) is True and "bytes_done" not in detail_of(srv.posts[-1])
    assert srv.progress_tick(t0 + srv.STAGE_MAX_SECONDS + 1) is False


def test_nothing_is_published_outside_the_boot_stages(srv):
    srv.progress_stage("HEARTBEAT_ONLINE")
    assert srv.progress_tick() is False and srv.posts == []


def test_reporter_failures_never_propagate(srv, monkeypatch):
    srv.progress_stage("DOWNLOADING_MODEL")

    def boom(*a):
        raise OSError("d1 down")

    monkeypatch.setattr(srv, "_d1_exec", boom)
    with srv.progress_probe("x", 10, lambda: (_ for _ in ()).throw(RuntimeError("probe broke"))):
        assert srv.progress_tick() is False                                                # swallowed, not raised


def test_notify_workspace_announces_the_stage_to_the_reporter(srv, monkeypatch):
    monkeypatch.setattr(srv.urllib.request, "urlopen", lambda *a, **k: None)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a")
    monkeypatch.setenv("CLOUDFLARE_D1_DATABASE_ID", "d")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")
    srv.notify_workspace("LOADING_MODEL")
    assert srv.PROGRESS.status == "LOADING_MODEL" and srv.PROGRESS.probe is None


def test_hf_download_reports_the_growing_part_file(srv, monkeypatch):
    monkeypatch.setattr(srv, "MODEL_BYTES", {srv.MODEL_QUANT: 1000})
    dest = str(srv.tmp / "model.gguf")
    srv.progress_stage("DOWNLOADING_MODEL")

    def fake_curl(cmd, **kw):
        partial = cmd[cmd.index("-o") + 1]
        for n in (300, 650, 1000):                                                         # curl writes; the reporter samples
            Path(partial).write_bytes(b"x" * n)
            srv.progress_tick(srv.PROGRESS.since + 10 * n / 100)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(srv.subprocess, "run", fake_curl)
    path, how = srv.download_model(dest)
    assert (path, how) == (dest, "verified") and os.path.getsize(dest) == 1000
    done = [detail_of(p)["bytes_done"] for p in srv.posts]
    assert done == [300, 650, 1000] and all(detail_of(p)["bytes_total"] == 1000 for p in srv.posts)
    assert srv.PROGRESS.probe is None                                                      # the probe is detached afterwards


def test_dataset_download_reports_download_then_unpack(srv, monkeypatch):
    monkeypatch.setattr(srv, "MODEL_BYTES", {srv.MODEL_QUANT: 2000})
    dest = str(srv.tmp / "m.gguf")
    srv.progress_stage("DOWNLOADING_MODEL")

    def fake_curl(cmd, **kw):
        zip_part = cmd[cmd.index("-o") + 1]
        with zipfile.ZipFile(zip_part, "w", zipfile.ZIP_STORED) as z:
            z.writestr("model.gguf", b"g" * 2000)
        srv.progress_tick(srv.PROGRESS.since + 5)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(srv.subprocess, "run", fake_curl)
    real_copy = srv.shutil.copyfileobj

    def sampling_copy(src, dst, length=0):
        real_copy(src, dst, length)
        dst.flush()
        srv.progress_tick(srv.PROGRESS.since + 6)

    monkeypatch.setattr(srv.shutil, "copyfileobj", sampling_copy)
    assert srv.download_from_cache_dataset(dest) == dest and os.path.getsize(dest) == 2000
    labels = [detail_of(p)["label"] for p in srv.posts]
    assert labels == ["Downloading the model archive", "Unpacking the model"]
    assert detail_of(srv.posts[-1])["bytes_done"] == 2000


def test_ollama_pull_reports_the_growing_model_store(srv, monkeypatch):
    monkeypatch.setattr(srv, "PERSIST_MODELS_DIR", str(srv.tmp / "ollama_models"))
    monkeypatch.setattr(srv, "MODEL_BYTES", {srv.MODEL_QUANT: 5000})
    os.makedirs(srv.PERSIST_MODELS_DIR + "/blobs")
    srv.progress_stage("DOWNLOADING_MODEL")

    def fake_pull(cmd, **kw):
        Path(srv.PERSIST_MODELS_DIR + "/blobs/sha256-abc-partial").write_bytes(b"z" * 1234)
        srv.progress_tick(srv.PROGRESS.since + 12)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(srv.subprocess, "run", fake_pull)
    assert srv.ollama_pull_model("ollama") is True
    d = detail_of(srv.posts[-1])
    assert d["bytes_done"] == 1234 and d["bytes_total"] == 5000 and "Ollama" in d["label"]


def test_server_source_has_no_literal_credentials_and_generates_a_per_boot_key(srv):
    assert srv.API_KEY.startswith("bt-") and len(srv.API_KEY) >= 40
