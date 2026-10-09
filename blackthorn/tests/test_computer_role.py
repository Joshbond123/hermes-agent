"""Computer role: persistence semantics (never lose local state) and computer-host resolution."""

import io
import json
import os
import zipfile

import httpx
import pytest

import kaggle_cyber_ornith.cyber_ornith_server as srv
from blackthorn.kaggle_bundle import build_computer_notebook_text
from blackthorn.tools.computer import ComputerClient, ToolError


# --------------------------------------------------------------------------- persistence
def _zip_with(files):
    """Snapshot zips are flat: files at the root, manifest.json alongside."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
        zf.writestr("manifest.json", json.dumps({"generation": 42, "files": len(files)}))
    return buf.getvalue()


def _fake_cli(zip_bytes):
    """Stub the kaggle CLI: 'datasets download' drops the zip into -p; everything else succeeds."""
    def run(cmd, **kw):
        class R:
            returncode = 0
            stderr = ""
            stdout = ""

        if "download" in cmd:
            idx = cmd.index("-p")
            dest = cmd[idx + 1]
            os.makedirs(dest, exist_ok=True)
            with open(os.path.join(dest, "state.zip"), "wb") as fh:
                fh.write(zip_bytes)
        return R()

    return run


def test_restore_brings_back_snapshot_files_but_never_overwrites_newer_local(tmp_path, monkeypatch):
    snap = _zip_with({"hello.txt": "from snapshot", "scripts/run.sh": "#!/bin/sh"})
    monkeypatch.setattr(srv, "STATE_DATASET", "josh787/blackthorn-computer-state")
    monkeypatch.setattr(srv, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(srv, "STATE_MANIFEST", str(tmp_path / "state" / "manifest.json"))
    monkeypatch.setattr(srv, "COMPUTER_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setattr(srv.subprocess, "run", _fake_cli(snap))
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "hello.txt").write_text("local and newer")     # must survive

    assert srv.restore_workspace() is True
    ws = tmp_path / "ws"
    assert (ws / "hello.txt").read_text() == "local and newer"        # never overwritten
    assert (ws / "scripts" / "run.sh").exists()                      # missing files restored


def test_restore_without_dataset_or_snapshot_is_honest(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "STATE_DATASET", "")
    assert srv.restore_workspace() is False

    def boom(cmd, **kw):
        class R:
            returncode = 1
            stderr = "403 forbidden"
            stdout = ""

        return R()

    monkeypatch.setattr(srv, "STATE_DATASET", "josh787/blackthorn-computer-state")
    monkeypatch.setattr(srv, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(srv.subprocess, "run", boom)
    assert srv.restore_workspace() is False                          # fresh workspace, no crash


def test_checkpoint_writes_a_generation_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "STATE_DATASET", "josh787/blackthorn-computer-state")
    monkeypatch.setattr(srv, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(srv, "STATE_MANIFEST", str(tmp_path / "state" / "manifest.json"))
    monkeypatch.setattr(srv, "COMPUTER_WORKSPACE", str(tmp_path / "ws"))
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "a.txt").write_text("data")
    calls = {}

    def fake_run(cmd, **kw):
        calls["cmd"] = cmd

        class R:
            returncode = 0
            stderr = ""
            stdout = ""

        return R()

    monkeypatch.setattr(srv.subprocess, "run", fake_run)
    out = srv.checkpoint_workspace("test")
    assert out["ok"] and out["files"] == 1 and out["generation"] > 0
    assert calls["cmd"][1] in ("datasets",)                          # went through the kaggle CLI
    manifest = json.loads((tmp_path / "state" / "manifest.json").read_text())
    assert manifest["generation"] == out["generation"]


# --------------------------------------------------------------------------- computer host resolution
async def test_computer_client_prefers_the_computer_row(stack):
    await stack.executor.query(
        "INSERT OR REPLACE INTO kaggle_gpu_state (id, status, tunnel_url, api_key, model, gpu_info, updated_at) "
        "VALUES ('computer', 'ONLINE', 'https://computer.example/gpu-relay', 'ckey', 'kali', 'cpu', ?)", [1.0])
    seen = {}

    class Spy:
        async def request(self, method, url, **kw):
            seen["url"] = url

            class R:
                status_code = 200
                headers = {"content-type": "application/json"}

                def json(self):
                    return {"ok": True}

                async def aread(self):
                    return b"{}"

            return R()

    client = ComputerClient(stack.services.resolver, Spy(), store=stack.services.store)
    await client.call("/computer/info", {})
    assert seen["url"].startswith("https://computer.example/gpu-relay/computer/")


async def test_computer_client_falls_back_to_the_model_host_without_a_computer_row(stack):
    seen = {}

    class Spy:
        async def request(self, method, url, **kw):
            seen["url"] = url

            class R:
                status_code = 200
                headers = {"content-type": "application/json"}

                def json(self):
                    return {"ok": True}

                async def aread(self):
                    return b"{}"

            return R()

    client = ComputerClient(stack.services.resolver, Spy(), store=stack.services.store)
    await client.call("/computer/info", {})
    assert stack.backend_srv.url in seen["url"]                       # model host answered


# --------------------------------------------------------------------------- bundle
def test_computer_notebook_forces_the_computer_role():
    import base64
    import re
    text = build_computer_notebook_text(env={"CLOUDFLARE_API_TOKEN": "t", "CLOUDFLARE_ACCOUNT_ID": "a",
                                             "CLOUDFLARE_D1_DATABASE_ID": "d", "KAGGLE_USERNAME": "u",
                                             "KAGGLE_API_TOKEN": "k"})
    blob = re.search(r"base64\.b64decode\('([A-Za-z0-9+/=]+)'\)", text).group(1)
    injected = json.loads(base64.b64decode(blob))
    assert injected["ORNITH_ROLE"] == "computer"                     # role baked into the notebook env
