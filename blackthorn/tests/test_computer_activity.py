"""Auto-off must see background work on the computer, and nothing it does may kill a busy computer.

Root cause this guards (verified live 2026-10-10): the idle clock counted chat turns only. An installer running
as a background job on the computer produced no chat activity, so auto-off stopped the Kaggle session after
its idle limit. Kaggle recorded those stops as cancelled kernels.
"""

import os
import subprocess
import sys
import time
import urllib.error

import pytest

import cloudflare_d1_client as d1c
import kaggle_cyber_ornith.cyber_ornith_server as srv


def _tagged():
    """Run the gateway's own _tagged_processes (extracted from the generated source; the full gateway creates
    /kaggle paths at import time, which only exist on the computer)."""
    import ast
    tree = ast.parse(srv.gateway_code(), "gateway_server.py")
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_tagged_processes")
    ns = {"os": os}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "gateway_server.py", "exec"), ns)
    return ns["_tagged_processes"]()


# ---------------------------------------------------------------- computer side: tagged process census

def test_tagged_processes_counts_background_children_and_ignores_untagged():
    tag = {**os.environ, "BLACKTHORN_EXEC_TAG": "1"}
    plain = {k: v for k, v in os.environ.items() if k != "BLACKTHORN_EXEC_TAG"}
    tagged = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], env=tag)
    untagged = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], env=plain)
    try:
        time.sleep(0.5)
        pids = {p["pid"] for p in _tagged()}
        assert tagged.pid in pids
        assert untagged.pid not in pids
    finally:
        tagged.kill()
        untagged.kill()
        tagged.wait()
        untagged.wait()


def test_tagged_processes_drops_exited_children():
    tag = {**os.environ, "BLACKTHORN_EXEC_TAG": "1"}
    done = subprocess.Popen([sys.executable, "-c", "pass"], env=tag)
    done.wait()
    assert done.pid not in {p["pid"] for p in _tagged()}


def test_gateway_has_tagging_activity_and_checkpoint_routes():
    gw = srv.gateway_code()
    compile(gw, "gateway_server.py", "exec")           # generated gateway is valid Python
    assert "\x00" not in gw                            # escapes survive the template string
    assert 'env["BLACKTHORN_EXEC_TAG"] = "1"' in gw
    assert '@app.get("/computer/activity")' in gw
    assert '@app.post("/computer/checkpoint")' in gw
    assert 'split(b"\\0")' in gw


def test_checkpoint_skips_caches_and_rebuildable_trees(tmp_path, monkeypatch):
    ws = tmp_path / "blackthorn_workspace"
    (ws / "notes").mkdir(parents=True)
    (ws / "notes" / "plan.md").write_text("keep")
    for rel in ("tools/venv/bin", "tools/home/.cache/camoufox", "security_catalog/x", "tmp", "__pycache__"):
        (ws / rel).mkdir(parents=True, exist_ok=True)
        (ws / rel / "big.bin").write_text("skip")
    (ws / "tools" / "registry.json").write_text("{}")
    (ws / "partial.part").write_text("skip")
    monkeypatch.setattr(srv, "COMPUTER_WORKSPACE", str(ws))
    monkeypatch.setattr(srv, "CHECKPOINT_MAX_FILE_BYTES", 10)
    (ws / "huge.bin").write_bytes(b"x" * 100)

    names = os.listdir(ws)
    skipped = srv._checkpoint_ignore(str(ws), names)
    assert "tmp" in skipped and "huge.bin" in skipped and "partial.part" in skipped
    assert "notes" not in skipped

    tools_skipped = srv._checkpoint_ignore(str(ws / "tools"), os.listdir(ws / "tools"))
    assert "venv" in tools_skipped
    assert "registry.json" not in tools_skipped

    home_skipped = srv._checkpoint_ignore(str(ws / "tools" / "home"), [".cache"]) if (ws / "tools" / "home").exists() else set()
    assert home_skipped in (set(), {".cache"})


def test_cache_slug_matches_controller_dataset_name():
    assert d1c.CACHE_DATASET_NAME == srv.CACHE_DATASET_SLUG.split("/", 1)[1]


def test_model_download_deletion_is_gone():
    # Automatic deletion of model weights on boot violated the "never delete data automatically" rule.
    assert not hasattr(srv, "cleanup_model_downloads")


# ---------------------------------------------------------------- controller side: auto-off decision

@pytest.fixture
def ready_primary(monkeypatch):
    monkeypatch.setattr(d1c, "get_auto_off_minutes", lambda: 5)
    monkeypatch.setattr(d1c, "_published_status", lambda: "HEARTBEAT_ONLINE")
    monkeypatch.setattr(d1c, "_STATUS_CACHE", {})
    monkeypatch.setattr(d1c, "_ACTIVE_TASKS", 0)
    monkeypatch.setattr(d1c, "_LAST_ACTIVITY_TS", time.time() - 3600)   # long idle on the chat side
    monkeypatch.setattr(d1c, "_computer_link", lambda: ("https://computer.example", "k", "HEARTBEAT_ONLINE"))
    fresh = {"active_jobs": 0, "known": True, "fails": 0, "checked_at": time.time(), "error": ""}
    monkeypatch.setattr(d1c, "_COMPUTER_ACTIVITY", dict(fresh))
    return fresh


def test_idle_chat_but_busy_computer_is_not_stopped(ready_primary, monkeypatch):
    monkeypatch.setattr(d1c, "_COMPUTER_ACTIVITY", {**ready_primary, "active_jobs": 2})
    decision = d1c.auto_off_decision()
    assert decision["should_stop"] is False
    assert decision["computer_jobs"] == 2
    assert "background job" in decision["reason"]


def test_idle_everywhere_is_stopped(ready_primary):
    decision = d1c.auto_off_decision()
    assert decision["should_stop"] is True
    assert decision["computer_jobs"] == 0


def test_unknown_computer_activity_blocks_stop_a_bounded_number_of_times(ready_primary, monkeypatch):
    monkeypatch.setattr(d1c, "_COMPUTER_ACTIVITY", {**ready_primary, "known": False, "fails": 1, "error": "timeout"})
    assert d1c.auto_off_decision()["should_stop"] is False
    monkeypatch.setattr(d1c, "_COMPUTER_ACTIVITY", {**ready_primary, "known": False, "fails": 3, "error": "timeout"})
    assert d1c.auto_off_decision()["should_stop"] is True   # bounded: never blocks forever


def test_refresh_counts_jobs_from_the_computer(monkeypatch):
    monkeypatch.setattr(d1c, "_computer_link", lambda: ("https://computer.example", "k", "HEARTBEAT_ONLINE"))
    monkeypatch.setattr(d1c, "_COMPUTER_ACTIVITY", {"active_jobs": 0, "known": False, "fails": 2, "checked_at": 0.0,
                                                   "error": ""})
    monkeypatch.setattr(d1c, "_computer_request", lambda path, payload, timeout: {"active_jobs": 4})
    out = d1c.refresh_computer_activity(force=True)
    assert out["active_jobs"] == 4 and out["known"] is True and out["fails"] == 0


def test_refresh_http_error_is_unknown_not_idle(monkeypatch):
    monkeypatch.setattr(d1c, "_computer_link", lambda: ("https://computer.example", "k", "HEARTBEAT_ONLINE"))
    monkeypatch.setattr(d1c, "_COMPUTER_ACTIVITY", {"active_jobs": 0, "known": True, "fails": 0, "checked_at": 0.0,
                                                   "error": ""})

    def _404(path, payload, timeout):
        raise urllib.error.HTTPError(path, 404, "old computer build", hdrs=None, fp=None)

    monkeypatch.setattr(d1c, "_computer_request", _404)
    out = d1c.refresh_computer_activity(force=True)
    assert out["known"] is False and out["fails"] == 1 and "404" in out["error"]


def test_checkpoint_before_stop_reports_but_never_raises(monkeypatch):
    monkeypatch.setattr(d1c, "_computer_link", lambda: ("", "", ""))
    assert d1c.checkpoint_computer()["ok"] is False

    monkeypatch.setattr(d1c, "_computer_link", lambda: ("https://computer.example", "k", "HEARTBEAT_ONLINE"))

    def _boom(path, payload, timeout):
        raise OSError("connection reset")

    monkeypatch.setattr(d1c, "_computer_request", _boom)
    result = d1c.checkpoint_computer()
    assert result["ok"] is False and result["error"] == "OSError"


def test_turn_off_checkpoints_before_clearing_the_computer_row():
    import ast
    from pathlib import Path
    src = (Path(d1c.__file__)).read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "turn_off_fleet")
    body = ast.get_source_segment(src, fn)
    assert body.index("checkpoint_computer()") < body.index("_write_fleet_row(row_id")


def test_busy_computer_resets_the_idle_clock(monkeypatch):
    monkeypatch.setattr(d1c, "_computer_link", lambda: ("https://computer.example", "k", "HEARTBEAT_ONLINE"))
    monkeypatch.setattr(d1c, "_COMPUTER_ACTIVITY", {"active_jobs": 0, "known": False, "fails": 0, "checked_at": 0.0,
                                                   "error": ""})
    monkeypatch.setattr(d1c, "_LAST_ACTIVITY_TS", time.time() - 3600)
    monkeypatch.setattr(d1c, "_computer_request", lambda path, payload, timeout: {"active_jobs": 1})
    d1c.refresh_computer_activity(force=True)
    assert time.time() - d1c._LAST_ACTIVITY_TS < 5
