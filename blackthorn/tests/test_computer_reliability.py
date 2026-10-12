"""Recovery regression tests against the production computer + lifecycle implementations."""
import io
import json
import os
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import cloudflare_d1_client as d1
import kaggle_cyber_ornith.cyber_ornith_server as srv
from blackthorn.route import RouteError
from blackthorn.tools.computer import ComputerClient


def zip_bytes(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_cli_auth_uses_bearer_not_legacy_basic(monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_abc")
    monkeypatch.setenv("KAGGLE_KEY", "old-key")
    env = srv._kaggle_cli_env()
    assert env["KAGGLE_API_TOKEN"] == "KGAT_abc"
    assert "KAGGLE_KEY" not in env


def test_nested_real_cli_zip_restores_without_overwriting_or_unpacking_user_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "STATE_DATASET", "alagbo/blackthorn-computer-state")
    monkeypatch.setattr(srv, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(srv, "COMPUTER_WORKSPACE", str(tmp_path / "workspace"))
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "project.txt").write_text("user's newer edit")
    inner = zip_bytes({"project.txt": "older saved edit", "src/code.py": "print(123)",
                       "download.zip": zip_bytes({"artifact": "keep zipped"})})
    outer = zip_bytes({"workspace.zip": inner, "manifest.json": json.dumps({"generation": 45})})

    def cli(cmd, **kw):
        if "download" in cmd:
            dest = cmd[cmd.index("-p") + 1]
            os.makedirs(dest, exist_ok=True)
            with open(os.path.join(dest, "dataset.zip"), "wb") as f:
                f.write(outer)
        return type("R", (), {"returncode": 0, "stderr": "", "stdout": ""})()
    monkeypatch.setattr(srv.subprocess, "run", cli)
    assert srv.restore_workspace() is True
    assert (ws / "src" / "code.py").read_text() == "print(123)"
    assert (ws / "project.txt").read_text() == "user's newer edit"
    assert (ws / "download.zip").is_file()
    assert not (ws / "artifact").exists()


def test_snapshot_does_not_follow_symlinks_or_include_credentials(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "safe.txt").write_text("keep")
    (ws / ".env").write_text("SECRET=abc")
    (ws / "token.pem").write_text("private")
    (ws / "jump").symlink_to(tmp_path)
    (ws / ".ssh").mkdir()
    (ws / ".ssh" / "id_rsa").write_text("private")
    monkeypatch.setattr(srv, "COMPUTER_WORKSPACE", str(ws))
    excluded = srv._checkpoint_ignore(str(ws), ["safe.txt", ".env", "token.pem", "jump", ".ssh"])
    assert excluded == {".env", "token.pem", "jump", ".ssh"}


def test_missing_packages_are_only_installed_if_absent(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    lock = ws / ".blackthorn" / "environment.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"pip": {"present": "1", "missing": "2"}, "apt": ["present-apt", "missing-apt"]}))
    monkeypatch.setattr(srv, "COMPUTER_WORKSPACE", str(ws))
    monkeypatch.setattr(srv, "_environment_inventory", lambda: {"pip": {"present": "1"}, "apt": ["present-apt"]})
    commands = []
    def run(cmd, **kwargs):
        commands.append(cmd)
        return type("R", (), {"returncode": 0})()
    monkeypatch.setattr(srv.subprocess, "run", run)
    result = srv.restore_environment()
    assert result["ok"] is True and result["pip_installed"] == 1 and result["apt_installed"] == 1
    assert all("present==1" not in c and "present-apt" not in c for c in commands)
    assert any("missing==2" in c for c in commands)
    assert any("missing-apt" in c for c in commands)


def test_local_and_public_gateway_are_probed_independently(monkeypatch):
    key = srv.API_KEY
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/computer/activity" and self.headers.get("Authorization") == "Bearer " + key:
                self.send_response(200); self.end_headers(); self.wfile.write(b'{"active_jobs":0}')
            else:
                self.send_response(502); self.end_headers(); self.wfile.write(b'bad gateway')
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert srv._computer_gateway_ok(url)
        assert srv._tunnel_answers(url)
        assert not srv._tunnel_answers(url + "/wrong-path")
        server.shutdown(); server.server_close(); thread.join(timeout=3)
        assert not srv._computer_gateway_ok(url)
    finally:
        server.shutdown(); server.server_close()


def test_model_auto_off_never_stops_busy_computer(monkeypatch):
    from blackthorn import fleet
    stops = []
    monkeypatch.setattr(fleet, "accounts_from_env", lambda env: ["model", "computer"])
    monkeypatch.setattr(d1, "_durable_active_runs", lambda: 0)
    monkeypatch.setattr(d1, "turn_off_fleet", lambda blocking, source, roles: stops.append(roles) or {})
    d1._auto_off_stop()
    assert stops == [("model",)]
    monkeypatch.setattr(d1, "_durable_active_runs", lambda: 1)
    result = d1._auto_off_stop()
    assert result["model"]["status"] == "stop_refused" and stops == [("model",)]


def test_explicit_shutdown_refuses_active_task_and_unknown_computer(monkeypatch):
    from blackthorn import fleet
    monkeypatch.setattr(fleet, "accounts_from_env", lambda env: [
        fleet.Account("m", "tok-m", "model", "primary"),
        fleet.Account("c", "tok-c", "computer", "primary")])
    monkeypatch.setattr(d1, "ensure_d1_schema", lambda: None)
    monkeypatch.setattr(d1, "_record_stop_event", lambda *args: None)
    monkeypatch.setattr(d1, "_durable_active_runs", lambda: 1)
    result = d1.turn_off_fleet()
    assert result["model"]["status"] == "stop_refused"
    assert result["computer"]["status"] == "stop_refused"
    monkeypatch.setattr(d1, "_durable_active_runs", lambda: 0)
    monkeypatch.setattr(d1, "activity_snapshot", lambda: {"active_tasks": 0})
    monkeypatch.setattr(d1, "refresh_computer_activity", lambda **kw: {"known": False, "active_jobs": 0})
    result = d1.turn_off_fleet()
    assert result["computer"]["status"] == "stop_refused"


async def test_enrolled_offline_computer_never_routes_to_model(stack):
    await stack.executor.query(
        "INSERT OR REPLACE INTO kaggle_gpu_state (id, status, tunnel_url, api_key, model, gpu_info, updated_at) "
        "VALUES ('computer', 'TUNNEL_ERROR', '', 'computer-key', 'computer', 'offline', ?)", [time.time()])
    client = ComputerClient(stack.services.resolver, stack.services.http, store=stack.services.store)
    with pytest.raises(RouteError, match="workspace has not been replaced"):
        await client._endpoint()


def test_model_turn_on_reuses_live_computer_instead_of_restarting_it(monkeypatch):
    from blackthorn import fleet as fm
    accounts = [fm.Account("m", "tm", "model", "primary"),
                fm.Account("c", "tc", "computer", "primary")]
    pushes = []
    rows = []

    class FakeFleet:
        def __init__(self, _accounts):
            self.by_slot = {(a.role, a.slot): a for a in accounts}
            self.state = {role: type("State", (), {"active": "primary", "reason": "initial"})()
                          for role in ("model", "computer")}
        def decide(self, role, force=False):
            return self.state[role]
        def start_kernel(self, account, **kwargs):
            pushes.append(account.role)
            return {"versionNumber": 1}
        def status(self):
            return {}

    monkeypatch.setattr(fm, "accounts_from_env", lambda env: accounts)
    monkeypatch.setattr(fm, "FleetManager", FakeFleet)
    monkeypatch.setattr(d1, "_fleet_role_online", lambda role: role == "computer")
    monkeypatch.setattr(d1, "ensure_d1_schema", lambda: None)
    monkeypatch.setattr(d1, "_record_stop_event", lambda *args: None)
    monkeypatch.setattr(d1, "_write_fleet_row", lambda role, *args, **kw: rows.append(role))
    monkeypatch.setattr(fm, "notebook_env_for", lambda account, env: {})
    import blackthorn.kaggle_bundle as kb
    monkeypatch.setattr(kb, "build_notebook_text", lambda env: "model notebook")
    monkeypatch.setattr(kb, "build_computer_notebook_text", lambda env: "computer notebook")
    result = d1.turn_on_fleet(blocking=True)
    assert result["computer"]["status"] == "already_online"
    assert result["model"]["status"] == "starting"
    assert pushes == ["model"] and rows == ["primary", "primary"]


def test_invalid_primary_kaggle_token_fails_over_to_valid_computer_backup():
    from blackthorn.fleet import Account, FleetManager
    primary = Account("p", "bad", "computer", "primary")
    backup = Account("b", "good", "computer", "backup")
    def fetcher(method, body, token):
        if token == "bad":
            raise __import__("urllib").error.HTTPError("kaggle", 401, "Unauthenticated", {},
                io.BytesIO(b'{"code":401,"message":"Unauthenticated"}'))
        return {"gpuQuota": {"timeUsed": "0s", "timeReserved": "0s", "totalTimeAllowed": "108000s"}}
    fleet = FleetManager([primary, backup], fetcher=fetcher)
    state = fleet.decide("computer", force=True)
    assert state.active == "backup" and state.reason == "primary_unhealthy"


async def test_explicit_model_only_control_never_targets_computer(monkeypatch):
    from blackthorn.api import OffBody
    from blackthorn.gpu import GpuService
    from blackthorn import fleet
    assert OffBody(confirm=True, role="model").role == "model"
    with pytest.raises(Exception):
        OffBody(confirm=True, role="invalid")
    calls = []
    class FakeD1:
        def turn_off_fleet(self, blocking, source, roles):
            calls.append((blocking, source, roles))
            return {"model": {"status": "stopped"}}
    monkeypatch.setattr(fleet, "accounts_from_env", lambda env: ["fleet configured"])
    gpu = GpuService(d1_module=FakeD1())
    async def status(refresh=False):
        return {"online": False}
    gpu.status = status
    result = await gpu.turn_off(role="model")
    assert result["turn_off_ok"] is True
    assert calls == [(True, "user turn-off", ("model",))]
