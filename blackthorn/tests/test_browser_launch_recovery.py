"""The browser must survive the failures the operator actually saw.

Two real incidents are pinned here:

1. ``BrowserType.launch: ENOENT ... mkdtemp '/workspace/.cache/tmp/playwright-artifacts-...'``
   - Playwright creates its profile with ``mkdtemp`` inside ``TMPDIR``. After a sandbox
     replacement ``/workspace/.cache`` is rehydrated lazily; if the launch wins that race the
     browser never starts. The tool must create *and prove* a usable temp root before every
     launch, and retry with a fresh one when it still fails.
2. ``BrowserType.launch: Failed to launch the browser process`` on a fresh VM because the GTK /
   NSS / ALSA libraries were only checked *after* the first launch.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import tempfile
import types

import pytest


ASSET = pathlib.Path(__file__).resolve().parent.parent / "computer_assets" / "browse.py"


def _browse_module():
    spec = importlib.util.spec_from_file_location("browse_under_test", ASSET)
    module = importlib.util.module_from_spec(spec)
    sys.modules["browse_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def browse():
    return _browse_module()


def test_temp_root_is_created_and_exported_before_a_launch(browse, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.delenv("BLACKTHORN_BROWSER_TMP", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "gone" / "tmp"))
    os.rmdir(str(tmp_path / "gone")) if (tmp_path / "gone").exists() else None

    notes: list = []
    root = browse._ensure_temp_roots(str(cache), notes)

    assert pathlib.Path(root).is_dir()
    # Not merely present: a real mkdtemp inside it must work, which is what Playwright needs.
    with tempfile.TemporaryDirectory(dir=root):
        pass
    assert os.environ["TMPDIR"] == root
    assert os.environ["XDG_CACHE_HOME"] == str(cache)


def test_temp_root_falls_back_when_the_workspace_cache_is_unwritable(browse, monkeypatch, tmp_path):
    monkeypatch.setenv("BLACKTHORN_BROWSER_TMP", "/proc/definitely-not-writable")
    notes: list = []
    root = browse._ensure_temp_roots(str(tmp_path / "cache"), notes)
    assert root != "/proc/definitely-not-writable"
    assert root.startswith(str(tmp_path / "cache")), "the workspace cache is preferred when it works"
    with tempfile.TemporaryDirectory(dir=root):
        pass


def test_launch_retries_after_recreating_the_temp_root(browse, monkeypatch, tmp_path):
    """The exact incident: mkdtemp ENOENT on the first attempt, success on the retry."""
    cache = tmp_path / "cache"
    attempts = {"n": 0}

    class FakeCamoufox:
        def __init__(self, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                # Reproduce Playwright's message and delete the root underneath it.
                import shutil
                shutil.rmtree(str(cache), ignore_errors=True)
                raise RuntimeError(
                    "BrowserType.launch: ENOENT: no such file or directory, "
                    f"mkdtemp '{cache}/tmp/playwright-artifacts-bzyvzE'")
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setitem(sys.modules, "camoufox.sync_api",
                        types.SimpleNamespace(Camoufox=FakeCamoufox))
    notes: list = []
    with browse._launch(str(cache), {"headless": True}, notes) as browser:
        assert isinstance(browser, FakeCamoufox)

    assert attempts["n"] == 2, "a temp-root failure must be retried once, not reported"
    assert any("temp" in n for n in notes)
    assert pathlib.Path(os.environ["TMPDIR"]).is_dir()


def test_a_library_failure_installs_libraries_then_retries(browse, monkeypatch, tmp_path):
    attempts = {"n": 0}
    installed = {"called": 0}

    class FakeCamoufox:
        def __init__(self, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("BrowserType.launch: Failed to launch the browser process! "
                                   "error while loading shared libraries: libgtk-3.so.0")
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    monkeypatch.setitem(sys.modules, "camoufox.sync_api", types.SimpleNamespace(Camoufox=FakeCamoufox))
    monkeypatch.setattr(browse, "_install_linux_libs",
                        lambda binary: (installed.__setitem__("called", installed["called"] + 1), (True, "installed"))[1])
    monkeypatch.setattr(browse, "_browser_binary", lambda root: "/x/camoufox-bin")
    monkeypatch.setattr(browse, "_ensure_temp_roots", lambda root, notes: "/tmp")

    browse._launch(str(tmp_path / "cache"), {"headless": True}, [])
    assert installed["called"] == 1 and attempts["n"] == 2


def test_a_hopeless_library_failure_is_reported_with_the_reason(browse, monkeypatch, tmp_path):
    class FakeCamoufox:
        def __init__(self, **kwargs):
            raise RuntimeError("BrowserType.launch: Failed to launch the browser process! "
                               "error while loading shared libraries: libnss3.so")
    monkeypatch.setitem(sys.modules, "camoufox.sync_api", types.SimpleNamespace(Camoufox=FakeCamoufox))
    monkeypatch.setattr(browse, "_install_linux_libs", lambda binary: (False, "apt-get failed: no network"))
    monkeypatch.setattr(browse, "_browser_binary", lambda root: "/x/camoufox-bin")
    monkeypatch.setattr(browse, "_ensure_temp_roots", lambda root, notes: "/tmp")

    with pytest.raises(RuntimeError) as err:
        browse._launch(str(tmp_path / "cache"), {"headless": True}, [])
    assert "missing browser system libraries" in str(err.value)
    assert "no network" in str(err.value)


def test_missing_libs_checks_the_whole_bundle_not_just_the_binary(browse, monkeypatch, tmp_path):
    """Child processes and plugins need libraries the main binary does not link."""
    root = tmp_path / "bundle"
    root.mkdir()
    main = root / "camoufox-bin"
    main.write_text("#!/bin/false\n")
    lib = root / "libxul.so"
    lib.write_text("x" * 8192)

    def fake_elf(path, limit=40):
        return [str(main), str(lib)]

    def fake_run(cmd, timeout=0):
        target = cmd[-1]
        out = "\tlibnspr4.so => not found\n" if target == str(lib) else "\tlibc.so.6 => /lib/libc.so.6\n"
        return types.SimpleNamespace(stdout=out, returncode=0)

    monkeypatch.setattr(browse, "_elf_files", fake_elf)
    monkeypatch.setattr(browse, "_run", fake_run)
    assert browse._missing_libs(str(main)) == ["libnspr4.so"]


def test_launch_errors_are_classified(browse):
    assert browse._classify_launch_error(
        RuntimeError("ENOENT: no such file or directory, mkdtemp '/w/.cache/tmp/playwright-1'")) == "temp"
    assert browse._classify_launch_error(
        RuntimeError("CamoufoxNotInstalled: run camoufox fetch")) == "bundle"
    assert browse._classify_launch_error(RuntimeError("Profile is in use")) == "other"


def test_bundle_libraries_are_preferred_over_the_system_ones(browse, monkeypatch, tmp_path):
    """Ubuntu's NSS (3.110) is older than the bundle's (3.117): the loader must use the bundle's."""
    bundle = tmp_path / "camoufox"
    bundle.mkdir()
    (bundle / "libnss3.so").write_text("x")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib/x86_64-linux-gnu")
    notes: list = []
    browse._prefer_bundle_libraries(str(bundle / "camoufox-bin"), notes)
    parts = os.environ["LD_LIBRARY_PATH"].split(":")
    assert parts[0] == str(bundle), "the bundle must come first on the search path"
    assert "/usr/lib/x86_64-linux-gnu" in parts
    # idempotent: a second call must not accumulate entries
    browse._prefer_bundle_libraries(str(bundle / "camoufox-bin"), notes)
    assert os.environ["LD_LIBRARY_PATH"].count(str(bundle)) == 1


def test_a_library_the_bundle_ships_is_not_reported_as_missing(browse, monkeypatch, tmp_path):
    """libnspr4.so inside the bundle is a path problem, not an apt problem."""
    root = tmp_path / "bundle"
    root.mkdir()
    main = root / "camoufox-bin"
    main.write_text("#!/bin/false\n")
    (root / "libnspr4.so").write_text("x" * 8192)
    (root / "libnss3.so").write_text("x" * 8192)

    def fake_run(cmd, timeout=0, env=None):
        return types.SimpleNamespace(stdout="\tlibnspr4.so => not found\n\tlibgtk-3.so.0 => not found\n",
                                     returncode=0, stderr="")

    monkeypatch.setattr(browse, "_run", fake_run)
    monkeypatch.setattr(browse, "_elf_files", lambda root, limit=40: [])
    assert browse._missing_libs(str(main)) == ["libgtk-3.so.0"]


def test_the_library_sweep_is_skipped_once_a_launch_has_been_proven(browse, tmp_path):
    cache, binary = str(tmp_path / "cache"), "/b/camoufox-bin"
    assert not browse._libs_verified(cache, binary)
    browse._mark_libs_verified(cache, binary)
    assert browse._libs_verified(cache, binary)
    assert not browse._libs_verified(cache, "/other/camoufox-bin"), "a different bundle must re-check"
    browse._forget_libs_verified(cache)
    assert not browse._libs_verified(cache, binary), "a library failure must invalidate the stamp"


def test_a_symbol_version_error_is_classified_as_a_library_problem(browse):
    exc = RuntimeError("/x/camoufox-bin: /x/libsmime3.so: version `NSS_3.117' not found")
    assert browse._classify_launch_error(exc) == "libs"
