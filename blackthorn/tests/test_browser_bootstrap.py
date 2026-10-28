"""Browser runtime bootstrap: disk-backed scratch space and loud dependency failures.

These tests encode what the live PandaStack guest actually does:
  * ``/tmp`` is a 1 GB tmpfs, so downloading the ~1.5 GB Camoufox bundle there dies
    with ``No space left on device`` unless TMPDIR/XDG_CACHE_HOME move to real disk;
  * a replacement sandbox must rehydrate the browser without touching the network
    when the bundle is already on disk;
  * a failed download must raise (and say why) instead of surfacing later as a
    mysterious browser crash.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import types

import pytest

from blackthorn.computer_assets import browse


def _load():
    return importlib.reload(browse)


def _fake_usage(sizes):
    """shutil.disk_usage replacement keyed by path prefix."""
    real = shutil.disk_usage

    def usage(path):
        for prefix, gb in sizes.items():
            if str(path).startswith(prefix):
                return types.SimpleNamespace(total=int(gb * 1024 ** 3), used=0, free=int(gb * 1024 ** 3))
        return real(path)

    return usage


def test_scratch_space_moves_off_a_small_tmpfs(tmp_path, monkeypatch):
    monkeypatch.setenv("BLACKTHORN_BROWSER_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("BLACKTHORN_BROWSER_TMP", str(tmp_path / "tmp"))
    monkeypatch.setattr(shutil, "disk_usage", _fake_usage({str(tmp_path): 40.0, "/tmp": 0.9}))
    notes: list = []
    root = browse._prepare_disk(notes)
    assert root == str(tmp_path / "cache")
    assert browse.os.environ["XDG_CACHE_HOME"] == str(tmp_path / "cache")
    assert browse.os.environ["TMPDIR"].startswith(str(tmp_path))
    assert any("browser scratch" in note for note in notes)


def test_small_tmpdir_is_never_chosen_when_disk_is_available(tmp_path, monkeypatch):
    monkeypatch.delenv("BLACKTHORN_BROWSER_CACHE", raising=False)
    monkeypatch.setenv("BLACKTHORN_BROWSER_TMP", str(tmp_path / "scratch"))
    monkeypatch.setattr(shutil, "disk_usage", _fake_usage({str(tmp_path): 30.0, "/workspace/.cache": 25.0,
                                                           "/tmp": 0.5}))
    assert browse._disk_dir("/tmp", minimum_gb=2.0) != "/tmp"


def test_runtime_is_reused_without_network_when_bundle_is_present(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    (cache / "camoufox" / "browsers").mkdir(parents=True)
    (cache / "camoufox" / "browsers" / "camoufox-bin").write_text("binary")
    monkeypatch.setenv("BLACKTHORN_BROWSER_CACHE", str(cache))
    monkeypatch.setenv("BLACKTHORN_BROWSER_TMP", str(cache))
    monkeypatch.delenv("BLACKTHORN_BROWSER_NO_BOOTSTRAP", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "camoufox" else None)
    ctypes_util = __import__("ctypes.util", fromlist=["util"])
    monkeypatch.setattr(ctypes_util, "find_library", lambda name: "lib-" + name)
    calls = []
    monkeypatch.setattr(browse, "_run",
                        lambda cmd, timeout, env=None: calls.append(cmd) or types.SimpleNamespace(
                            returncode=0, stderr="", stdout=""))
    notes: list = []
    browse._ensure_runtime(notes)
    assert [c for c in calls if "pip" in c or "apt-get" in c or "fetch" in c] == [], \
        "an installed browser must not trigger pip/apt/network work"
    assert browse._browser_present(str(cache)) is True


def test_missing_bundle_triggers_fetch_and_reports_failure(monkeypatch, tmp_path):
    import subprocess as real_subprocess
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("BLACKTHORN_BROWSER_CACHE", str(cache))
    monkeypatch.setenv("BLACKTHORN_BROWSER_TMP", str(cache))
    monkeypatch.delenv("BLACKTHORN_BROWSER_NO_BOOTSTRAP", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "camoufox" else None)

    def run(cmd, timeout, env=None):
        if "camoufox" in cmd and "pip" not in cmd:
            return types.SimpleNamespace(returncode=28, stderr="Error: [Errno 28] No space left on device", stdout="")
        return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(browse, "_run", run)
    notes: list = []
    with pytest.raises(RuntimeError) as err:
        browse._ensure_runtime(notes)
    assert "No space left on device" in str(err.value)


def test_pip_failure_raises_with_stderr(monkeypatch, tmp_path):
    monkeypatch.setenv("BLACKTHORN_BROWSER_CACHE", str(tmp_path))
    monkeypatch.setenv("BLACKTHORN_BROWSER_TMP", str(tmp_path))
    monkeypatch.delenv("BLACKTHORN_BROWSER_NO_BOOTSTRAP", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    def run(cmd, timeout, env=None):
        return types.SimpleNamespace(returncode=1, stderr="ERROR: could not find a version that satisfies camoufox",
                                     stdout="")

    monkeypatch.setattr(browse, "_run", run)
    with pytest.raises(RuntimeError) as err:
        browse._ensure_runtime([])
    assert "could not find a version" in str(err.value)


def test_bootstrap_can_be_disabled(monkeypatch):
    monkeypatch.setenv("BLACKTHORN_BROWSER_NO_BOOTSTRAP", "1")
    notes: list = []
    browse._ensure_runtime(notes)
    assert notes == ["runtime bootstrap skipped by configuration"]


def test_missing_shared_libraries_are_installed_before_launch(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    binary = cache / "camoufox" / "browsers" / "official" / "camoufox-bin"
    binary.parent.mkdir(parents=True)
    binary.write_text("binary")
    monkeypatch.setenv("BLACKTHORN_BROWSER_CACHE", str(cache))
    monkeypatch.setenv("BLACKTHORN_BROWSER_TMP", str(cache))
    monkeypatch.delenv("BLACKTHORN_BROWSER_NO_BOOTSTRAP", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "camoufox" else None)
    state = {"installed": False}
    commands: list = []

    def run(cmd, timeout, env=None):
        commands.append(cmd)
        if cmd[:2] == ["apt-get", "install"]:
            state["installed"] = True
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_ldd(cmd, timeout, env=None):
        if state["installed"]:
            return types.SimpleNamespace(returncode=0, stdout="\tlibgtk-3.so.0 => /usr/lib/libgtk-3.so.0", stderr="")
        return types.SimpleNamespace(
            returncode=0,
            stdout="\tlibgtk-3.so.0 => not found\n\tlibnss3.so => not found\n", stderr="")

    monkeypatch.setattr(browse, "_run", run)
    monkeypatch.setattr(browse, "_missing_libs_ldd", fake_ldd, raising=False)
    monkeypatch.setattr(browse, "_missing_libs", lambda binary: [] if state["installed"] else
                        ["libgtk-3.so.0", "libnss3.so"])
    notes: list = []
    browse._ensure_runtime(notes)
    assert any(cmd[:2] == ["apt-get", "install"] for cmd in commands)
    assert any("Firefox system libraries" in note for note in notes)


def test_unresolvable_libraries_raise_instead_of_a_mystery_crash(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    binary = cache / "camoufox" / "browsers" / "official" / "camoufox-bin"
    binary.parent.mkdir(parents=True)
    binary.write_text("binary")
    monkeypatch.setenv("BLACKTHORN_BROWSER_CACHE", str(cache))
    monkeypatch.setenv("BLACKTHORN_BROWSER_TMP", str(cache))
    monkeypatch.delenv("BLACKTHORN_BROWSER_NO_BOOTSTRAP", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "camoufox" else None)
    monkeypatch.setattr(browse, "_run", lambda cmd, timeout, env=None: types.SimpleNamespace(
        returncode=1, stdout="", stderr="E: Unable to locate package libgtk-3-0t64"))
    monkeypatch.setattr(browse, "_missing_libs", lambda binary: ["libgtk-3.so.0"])
    with pytest.raises(RuntimeError) as err:
        browse._ensure_runtime([])
    assert "libgtk-3.so.0" in str(err.value)
