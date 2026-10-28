"""The one-time browser start-up must happen before a user action needs it.

A fresh Camoufox bundle installs the fingerprint model and extracts its addons on the
first launch. Doing that inside the first ``open`` made a perfectly healthy tool call
hit its time budget (observed live: the first two opens of a new guest timed out while
the third succeeded). The warm-up runs once per cache and is then stamped.
"""
from __future__ import annotations

import importlib
import sys
import types

import pytest

from blackthorn.computer_assets import browse


class _Page:
    def __init__(self):
        self.urls = []

    def goto(self, url, **_kw):
        self.urls.append(url)
        return None


class _Context:
    def __init__(self, browser):
        self.browser = browser
        self.closed = False
        self.pages = []

    def new_page(self):
        page = _Page()
        self.pages.append(page)
        return page

    def close(self):
        self.closed = True


class _Browser:
    def __init__(self, launches):
        self.launches = launches
        self.contexts = []

    def new_context(self, **_kw):
        context = _Context(self)
        self.contexts.append(context)
        return context


@pytest.fixture()
def fake_camoufox(monkeypatch):
    launches: list = []
    module = types.SimpleNamespace()

    class Camoufox:
        def __init__(self, **kw):
            self.kw = kw
            launches.append(kw)

        def __enter__(self):
            return _Browser(launches)

        def __exit__(self, *exc):
            return False

    module.Camoufox = Camoufox
    monkeypatch.setitem(sys.modules, "camoufox", types.ModuleType("camoufox"))
    monkeypatch.setitem(sys.modules, "camoufox.sync_api", module)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    return launches


def test_warmup_launches_once_and_is_stamped(tmp_path, fake_camoufox):
    notes: list = []
    browse._warm_browser(str(tmp_path), notes)
    assert len(fake_camoufox) == 1, "the browser must actually be launched once"
    assert (tmp_path / "camoufox" / ".blackthorn-warm").exists()
    assert any("warmed" in note for note in notes)

    browse._warm_browser(str(tmp_path), notes)          # second call: stamped, no relaunch
    assert len(fake_camoufox) == 1


def test_warmup_failure_is_reported_but_not_fatal(tmp_path, monkeypatch):
    class Boom:
        def __init__(self, **kw):
            raise RuntimeError("XPCOMGlueLoad error for file camoufox-bin")

    monkeypatch.setitem(sys.modules, "camoufox", types.ModuleType("camoufox"))
    monkeypatch.setitem(sys.modules, "camoufox.sync_api", types.SimpleNamespace(Camoufox=Boom))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    notes: list = []
    browse._warm_browser(str(tmp_path), notes)          # must not raise
    assert any("warm-up failed" in note for note in notes)
    assert not (tmp_path / "camoufox" / ".blackthorn-warm").exists(), "no stamp after a failed warm-up"


def test_warmup_is_skipped_when_camoufox_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    notes: list = []
    browse._warm_browser(str(tmp_path), notes)
    assert notes == []
