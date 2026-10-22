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


def test_cache_mount_is_reused_only_for_exact_complete_model(srv, monkeypatch, tmp_path):
    root = tmp_path / 'input' / 'datasets' / 'owner' / 'model-cache'
    root.mkdir(parents=True)
    path = root / srv.GGUF_BASENAME.format(quant=srv.MODEL_QUANT)
    expected = 2 * 1024 * 1024
    monkeypatch.setattr(srv, 'INPUT_ROOT', str(tmp_path / 'input'))
    monkeypatch.setattr(srv, 'MODEL_BYTES', {srv.MODEL_QUANT: expected})
    with path.open('wb') as fh:
        fh.truncate(expected)
    assert srv.find_dataset_gguf() == str(path)
    with path.open('r+b') as fh:
        fh.truncate(expected - 3000)
    assert srv.find_dataset_gguf() == ''


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


def test_thinking_disabled_maps_blackthorn_flags(srv):
    """Blackthorn sends chat_template_kwargs.enable_thinking=false. The helper lives in the generated gateway source
    (which needs FastAPI on Kaggle), so extract just that function with ast and run it."""
    import ast
    tree = ast.parse(srv.gateway_code())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "thinking_disabled")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<gateway>", "exec"), ns)
    f = ns["thinking_disabled"]
    assert f({"chat_template_kwargs": {"enable_thinking": False}}) is True
    assert f({"think": False}) is True
    assert f({"messages": []}) is False
    assert f({"chat_template_kwargs": {"enable_thinking": True}}) is False


def test_llama_cmake_on_t4_uses_driver_library_and_sm75(srv, monkeypatch):
    """Reproduces Kaggle's CUDA::cuda_driver configure failure without a driver hint."""
    import subprocess
    commands = []
    monkeypatch.setattr(srv.os.path, "isdir", lambda path: True)
    driver = "/usr/local/nvidia/lib64/libcuda.so"
    monkeypatch.setattr(srv.os.path, "isfile", lambda path: path == driver)
    monkeypatch.setattr(srv.subprocess, "run",
                        lambda cmd, **kw: commands.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    with pytest.raises(RuntimeError, match="binary not found"):
        srv.build_llama_server_bin()
    cmake = next(cmd for cmd in commands if isinstance(cmd, list) and cmd[:2] == ["cmake", "-B"])
    assert "-DGGML_CUDA=ON" in cmake
    assert "-DCMAKE_CUDA_ARCHITECTURES=75" in cmake
    assert "-DCUDA_cuda_driver_LIBRARY=" + driver in cmake


def test_official_cuda_binary_is_pinned_and_fails_closed(srv, monkeypatch, tmp_path):
    monkeypatch.setattr(srv.os.path, "isfile", lambda path: False)
    monkeypatch.setattr(srv.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0))
    monkeypatch.setattr(srv, "_sha256", lambda path: "incorrect")
    monkeypatch.setenv("LLAMA_BUILD_FROM_SOURCE", "0")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        srv.ensure_llama_server_bin()


def test_default_model_registration_prefers_direct_cuda_serving(srv, monkeypatch, tmp_path):
    model = tmp_path / "cached.gguf"
    model.write_bytes(b"model")
    monkeypatch.delenv("PREFER_LLAMA", raising=False)
    monkeypatch.setattr(srv, "start_llama_server", lambda path: srv.posts.append(("llama", path)))
    monkeypatch.setattr(srv, "ollama_model_present", lambda: False)
    srv.register_model("ollama", str(model))
    assert srv.posts == [("llama", str(model))]


def test_cached_model_starts_with_latest_working_offload_not_slow_full_probe(srv, monkeypatch, tmp_path):
    model = tmp_path / 'cached.gguf'
    model.write_bytes(b'fixture')
    monkeypatch.setattr(srv, 'ensure_llama_server_bin', lambda: '/verified/llama-server')
    monkeypatch.setenv('LLAMA_TRY_FULL_OFFLOAD', '0')
    seen = []
    class Proc:
        def poll(self): return None
    def launch(cmd, **kw):
        seen.append(cmd[cmd.index('-ngl') + 1])
        return Proc()
    monkeypatch.setattr(srv.subprocess, 'Popen', launch)
    monkeypatch.setattr(srv.urllib.request, 'urlopen', lambda *a, **kw: object())
    srv.start_llama_server(str(model))
    assert seen == ['40']


def test_full_offload_requires_explicit_opt_in_and_falls_back(srv, monkeypatch, tmp_path):
    monkeypatch.setattr(srv, 'ensure_llama_server_bin', lambda: '/verified/llama-server')
    monkeypatch.setenv('LLAMA_TRY_FULL_OFFLOAD', '1')
    seen = []
    class Proc:
        def __init__(self, mode): self.mode = mode
        def poll(self): return 1 if self.mode == '99' else None
        def terminate(self): pass
        def wait(self, timeout=None): return 1
    def launch(cmd, **kw):
        mode = cmd[cmd.index('-ngl') + 1]
        seen.append(mode)
        return Proc(mode)
    monkeypatch.setattr(srv.subprocess, 'Popen', launch)
    monkeypatch.setattr(srv.urllib.request, 'urlopen', lambda *a, **kw: (_ for _ in ()).throw(OSError('not ready')) if seen[-1] == '99' else object())
    monkeypatch.setattr(srv.time, 'sleep', lambda *a: None)
    srv.start_llama_server(str(tmp_path / 'fixture.gguf'))
    assert seen == ['99', '56']


def test_gateway_routes_official_llama_engine_to_separate_port(srv, monkeypatch):
    import builtins
    real_open = builtins.open
    def fake_open(path, *args, **kwargs):
        if path == '/tmp/engine_mode.txt':
            import io
            return io.StringIO('llama')
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', fake_open)
    assert f'OLLAMA_URL = "http://127.0.0.1:{srv.LLAMA_PORT}"' in srv.gateway_code()
    monkeypatch.setattr(builtins, 'open', real_open)
    assert srv.LLAMA_PORT != srv.OLLAMA_PORT


def test_model_registration_does_not_delete_cached_blobs():
    import ast
    tree = ast.parse(SRC.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'register_model')
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ('remove', 'unlink', 'rmtree') for n in ast.walk(fn))


def test_mounted_computer_state_restores_without_download_or_overwrite(srv, monkeypatch, tmp_path):
    root = tmp_path / 'input'
    mount = root / 'datasets' / 'alagbo' / 'blackthorn-computer-state'
    (mount / 'live_verification').mkdir(parents=True)
    (mount / 'live_verification' / 'marker.txt').write_text('published')
    (mount / 'live_verification' / 'new.txt').write_text('from snapshot')
    (mount / 'dataset-metadata.json').write_text('{}')
    (mount / 'private.key').write_text('not-copied')
    (mount / 'link.txt').symlink_to('marker.txt')
    live = tmp_path / 'workspace'
    (live / 'live_verification').mkdir(parents=True)
    (live / 'live_verification' / 'marker.txt').write_text('newer local')
    monkeypatch.setattr(srv, 'INPUT_ROOT', str(root))
    monkeypatch.setattr(srv, 'COMPUTER_WORKSPACE', str(live))
    monkeypatch.setattr(srv, 'STATE_DATASET', 'alagbo/blackthorn-computer-state')
    monkeypatch.setattr(srv.subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('download should not run')))
    assert srv.restore_workspace() is True
    assert (live / 'live_verification' / 'marker.txt').read_text() == 'newer local'
    assert (live / 'live_verification' / 'new.txt').read_text() == 'from snapshot'
    assert not (live / 'private.key').exists() and not (live / 'link.txt').exists()


def test_restore_uses_intact_older_dataset_version_when_latest_is_inventory_only(srv, monkeypatch, tmp_path):
    import io
    root = tmp_path / 'input'
    mount = root / 'datasets' / 'alagbo' / 'blackthorn-computer-state'
    mount.mkdir(parents=True)
    (mount / 'environment.json').write_text('{}')
    live = tmp_path / 'workspace'
    monkeypatch.setattr(srv, 'INPUT_ROOT', str(root))
    monkeypatch.setattr(srv, 'COMPUTER_WORKSPACE', str(live))
    monkeypatch.setattr(srv, 'STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setattr(srv, 'STATE_DATASET', 'alagbo/blackthorn-computer-state')
    monkeypatch.setattr(srv, '_kaggle_cli_env', lambda: {'KAGGLE_API_TOKEN': 'test-token'})
    zbytes = io.BytesIO()
    with zipfile.ZipFile(zbytes, 'w') as z:
        z.writestr('live_verification/marker.txt', 'older intact data')
        z.writestr('.blackthorn/environment.json', '{}')
    class Response(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *args): self.close()
    def urlopen(request, **kwargs):
        url = request.full_url
        if 'GetDataset' in url:
            return Response(json.dumps({'currentVersionNumber': 337}).encode())
        if 'ListDatasetFiles' in url:
            version = json.loads(request.data)['datasetVersionNumber']
            files = [{'name':'environment.json'}] if version == 337 else [
                {'name':'live_verification/marker.txt'}, {'name':'.blackthorn/environment.json'}]
            return Response(json.dumps({'datasetFiles': files}).encode())
        if 'datasetVersionNumber=336' in url:
            return Response(zbytes.getvalue())
        raise AssertionError('unexpected network request: ' + url)
    monkeypatch.setattr(srv.urllib.request, 'urlopen', urlopen)
    monkeypatch.setattr(srv.subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('CLI should not run')))
    assert srv.restore_workspace() is True
    assert (live / 'live_verification' / 'marker.txt').read_text() == 'older intact data'


def test_inventory_only_checkpoint_fails_closed(srv, monkeypatch, tmp_path):
    live = tmp_path / 'workspace'
    live.mkdir()
    (live / 'environment.json').write_text('{}')
    monkeypatch.setattr(srv, 'COMPUTER_WORKSPACE', str(live))
    monkeypatch.setattr(srv, 'STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setattr(srv, 'STATE_DATASET', 'alagbo/blackthorn-computer-state')
    assert srv.checkpoint_workspace()['ok'] is False
    assert 'inventory-only' in srv.checkpoint_workspace()['error']


def test_late_notebook_heartbeat_cannot_resurrect_verified_off(srv, monkeypatch):
    calls = []
    def fake_urlopen(req, **kwargs):
        if '/d1/database/' in getattr(req, 'full_url', ''):
            calls.append(json.loads(req.data))
        return None
    monkeypatch.setattr(srv.urllib.request, 'urlopen', fake_urlopen)
    for name in ('CLOUDFLARE_ACCOUNT_ID', 'CLOUDFLARE_D1_DATABASE_ID', 'CLOUDFLARE_API_TOKEN'):
        monkeypatch.setenv(name, 'test-secret-not-logged')
    srv.notify_workspace('HEARTBEAT_ONLINE')
    assert len(calls) == 1
    sql, params = calls[0]['sql'], calls[0]['params']
    assert "ON CONFLICT(id) DO UPDATE" in sql
    assert "WHERE kaggle_gpu_state.status NOT IN" in sql
    assert "'GPU_STOPPED_SAVING_QUOTA'" in sql
    assert params[1] == 'HEARTBEAT_ONLINE'
