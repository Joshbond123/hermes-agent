"""The computer installer must trust the live system, not its registry, and keep caches off the workspace."""

import importlib.util
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "computer_assets" / "install_stack.py"


@pytest.fixture
def inst(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("install_stack_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.WS = tmp_path / "ws"
    mod.TOOLS = mod.WS / "tools"
    mod.HOME = mod.TOOLS / "home"
    mod.VENV = mod.TOOLS / "venv"
    mod.REGISTRY = mod.TOOLS / "registry.json"
    mod.CACHE = tmp_path / "cache"
    return mod


def test_apt_base_reinstalls_what_the_system_lost(inst, monkeypatch):
    installed = {"git"}
    monkeypatch.setattr(inst, "apt_installed", lambda pkg: pkg in installed)
    calls = []
    monkeypatch.setattr(inst, "run", lambda cmd, timeout, cwd=None: calls.append(cmd) or {"cmd": cmd, "rc": 0, "seconds": 0, "tail": ""})
    monkeypatch.setattr(inst, "save", lambda reg: None)
    reg = {"base": [{"pkg": "git", "rc": 0}]}   # the registry says git is done, but the runtime is fresh
    inst.apt_base(reg)
    assert not any(c.endswith(" git") for c in calls)           # system already has it
    assert any(c.endswith(" nmap") for c in calls)              # registry-only "done" is not trusted


def test_installed_tool_is_reproved_and_reinstalled_when_gone(inst, monkeypatch, tmp_path):
    cat = tmp_path / "security_catalog"
    (cat / "hackingtool-plugin" / "plugins" / "hackingtool" / "data").mkdir(parents=True)
    import json
    (cat / "hackingtool-plugin" / "plugins" / "hackingtool" / "data" / "tools.json").write_text(
        json.dumps({"tools": [{"id": "demo", "title": "Demo", "install_commands": ["echo x"],
                               "run_commands": ["demo --version"]}]}))
    monkeypatch.setattr(inst, "WS", tmp_path)
    monkeypatch.setattr(inst, "TOOLS", tmp_path / "tools")
    monkeypatch.setattr(inst, "save", lambda reg: None)
    reinstalled = []
    monkeypatch.setattr(inst, "install_tool", lambda tool, reg: reinstalled.append(tool["id"]) or {
        "id": tool["id"], "status": "installed", "steps": [], "probe": None})
    probes = {"rc": 127, "tail": "demo: not found"}
    monkeypatch.setattr(inst, "run", lambda cmd, timeout, cwd=None: dict(probes, cmd=cmd, seconds=0))
    reg = {"tools": [{"id": "demo", "status": "installed"}]}
    inst.catalog(reg)
    assert reinstalled == ["demo"]                                # probe failed on this runtime -> reinstall


def test_kaggle_cancelling_packages_are_never_installed(inst, monkeypatch):
    calls = []
    monkeypatch.setattr(inst, "apt_installed", lambda pkg: False)
    monkeypatch.setattr(inst, "run", lambda cmd, timeout, cwd=None: calls.append(cmd) or {"cmd": cmd, "rc": 0, "seconds": 0, "tail": ""})
    monkeypatch.setattr(inst, "save", lambda reg: None)
    reg = {"base": []}
    inst.apt_base(reg)
    for pkg in ("hydra", "john", "hashcat", "aircrack-ng"):
        assert not any(c.endswith(" " + pkg) for c in calls), pkg
        assert any(b["pkg"] == pkg and b["status"] == "skipped_kaggle_session_cancel" for b in reg["base"])
    assert any(c.endswith(" nmap") for c in calls)            # scanners that work on Kaggle are still installed


def test_caches_are_not_on_the_workspace(inst):
    e = inst.env()
    for key in ("PIP_CACHE_DIR", "XDG_CACHE_HOME", "GOCACHE", "GOMODCACHE", "TMPDIR"):
        assert str(inst.WS) not in e[key]
        assert str(inst.CACHE) in e[key]


def test_camoufox_fetch_does_not_put_tmp_under_workspace():
    text = SRC.read_text(encoding="utf-8")
    assert "TMPDIR={WS}/tmp" not in text
