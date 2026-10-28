"""Autonomous capability use and interruption resistance.

Covers the two behaviours the operator asked for directly:

* the agent must be able to *decide* a task needs a skill or a tool, so the
  discovery and execution tools are always offered — never keyword-gated;
* a running task must not be destroyed by a computer rotation, an expiry or a
  dropped chat connection.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from blackthorn.agent import AgentRun as AgentRunner, Deps
from blackthorn.prompts import BASE_PROMPT


class FakeRun:
    def __init__(self):
        self.id = "run_test"
        self.session_id = "s1"
        self.cancel_event = asyncio.Event()

    def emit(self, *a, **k):
        return {}


@dataclass
class FakeSettings:
    context_tokens: int = 8192
    completion_reserve_tokens: int = 1200
    max_repeat_calls: int = 3
    tool_result_chars: int = 4000


def _runner(user_message: str, turns=None) -> AgentRunner:
    from blackthorn.tools import default_registry
    deps = Deps(settings=FakeSettings(), store=None, resolver=None, registry=default_registry(),
                http=None, tavily=None, computer=None)
    return AgentRunner(FakeRun(), deps, system="sys", history=[], user_message=user_message,
                       turns=turns or [])


def _offered(runner: AgentRunner) -> set:
    return {s["function"]["name"] for s in runner._active_schemas}


@pytest.mark.parametrize("message", [
    "Audit this server for weak SSH configuration and report what you find",
    "Here is a suspicious binary, tell me what it does",
    "Check whether this invoice email is a scam",
])
def test_discovery_tools_are_always_offered(message):
    offered = _offered(_runner(message))
    for tool in ("skill_search", "skill_load", "tool_search", "cyber_run"):
        assert tool in offered, f"{tool} must be available for the agent to choose it itself"


@pytest.mark.parametrize("message", [
    "Here is a suspicious binary, tell me what it does",
    "Set up a Windows characterisation run and execute the sample end to end",
])
def test_lab_tools_are_always_offered_without_the_word_malware(message):
    offered = _offered(_runner(message))
    for tool in ("lab_detonate", "lab_exec", "lab_os", "lab_close"):
        assert tool in offered


def test_a_plain_greeting_still_offers_discovery_but_not_a_flood():
    offered = _offered(_runner("hi"))
    assert "skill_search" in offered and "tool_search" in offered
    assert "browse" not in offered, "unrelated heavy tools stay undisclosed"


def test_the_prompt_tells_the_agent_to_pick_its_own_capabilities():
    assert "tool_search" in BASE_PROMPT and "skill_search" in BASE_PROMPT
    assert "Decide yourself" in BASE_PROMPT


# --------------------------------------------------------------------------- rotation safety
class BusyRuns:
    def __init__(self, busy: bool):
        self.busy = busy

    def active_count(self) -> int:
        return 1 if self.busy else 0


class FakeComputer:
    def __init__(self):
        self.handovers: List[str] = []

    async def handover(self, *, reason="manual"):
        self.handovers.append(reason)
        return {"ok": True, "sandbox_id": "new"}


def _snapshot(due=True):
    return {"sandbox_id": "sid-1", "status": "running", "seconds_left": 12.0, "handover_due": due}


@pytest.mark.asyncio
async def test_an_active_run_is_never_rotated_out_from_under_itself(monkeypatch):
    from blackthorn.lifecycle import SandboxLifecycle

    class Store:
        def __init__(self):
            self.values = {}

        async def get_setting(self, key):
            return self.values.get(key, "")

        async def set_setting(self, key, value):
            self.values[key] = value

    computer = FakeComputer()
    manager = SandboxLifecycle(Store(), computer)
    manager.runs = BusyRuns(True)
    monkeypatch.setattr(manager, "snapshot", lambda: _async(_snapshot()))
    manager._last_tick = -1e9
    result = await manager.tick()
    assert result["action"] == "deferred" and result["reason"] == "active_run"
    assert computer.handovers == [], "no handover while a turn is executing"


@pytest.mark.asyncio
async def test_the_same_lifecycle_still_rotates_when_the_agent_is_idle(monkeypatch):
    from blackthorn.lifecycle import SandboxLifecycle

    class Store:
        def __init__(self):
            self.values = {}

        async def get_setting(self, key):
            return self.values.get(key, "")

        async def set_setting(self, key, value):
            self.values[key] = value

    computer = FakeComputer()
    manager = SandboxLifecycle(Store(), computer)
    manager.runs = BusyRuns(False)
    monkeypatch.setattr(manager, "snapshot", lambda: _async(_snapshot()))
    monkeypatch.setattr(manager, "handover", lambda **kw: _async({"ok": True, "sandbox_id": "new"}))
    manager._last_tick = -1e9
    result = await manager.tick()
    assert result["action"] == "handover"


def _async(value):
    async def inner():
        return value
    return inner()


# --------------------------------------------------------------------------- deadlock guard
async def test_the_wake_lock_can_be_re_entered_by_its_owner():
    """ensure_ready() holds the lock while calling _exec, which takes it again.

    A plain asyncio.Lock self-deadlocks there: a freshly created or replaced computer
    hung on its first command with no network traffic and no error. The lock must let
    its own task back in and still exclude other tasks.
    """
    from blackthorn.panda_computer import _ReentrantLock

    lock = _ReentrantLock()
    order: List[str] = []

    async def outer():
        async with lock:
            order.append("outer")
            async with lock:                    # nested: must not block
                order.append("nested")
            await asyncio.sleep(0.05)

    async def other():
        await asyncio.sleep(0.01)
        async with lock:
            order.append("other")

    await asyncio.wait_for(asyncio.gather(outer(), other()), timeout=2)
    assert order[0:2] == ["outer", "nested"] and order[-1] == "other", order


# --------------------------------------------------------------------------- quota failover
class QuotaPanda:
    """A provider whose first account has burned today's transfer allowance."""

    def __init__(self):
        self.created: List[str] = []
        self.deleted: List[str] = []
        self.probes = 0

    class _Resp:
        def __init__(self, status, payload=None):
            self.status_code = status
            self._payload = payload if payload is not None else {}
            self.text = ""

        def json(self):
            return self._payload

    async def post(self, url, **kw):
        key = (kw.get("headers") or {}).get("Authorization", "").split()[-1]
        if url.endswith("/sandboxes"):
            if key == "pds_spent":
                return self._Resp(429, {"detail": "outbound data transfer allowance"})
            sid = "12345678-1234-1234-1234-12345678" + ("0001" if key == "pds_spent" else "0002")
            self.created.append(sid)
            return self._Resp(201, {"id": sid})
        if url.endswith("/lifecycle"):
            return self._Resp(200, {"persistent": True, "ttl_seconds": 3600})
        if url.endswith("/exec"):
            self.probes += 1
            return self._Resp(200, {"exit_code": 0, "stdout": "437", "stderr": ""})
        return self._Resp(200, {})

    async def get(self, url, **kw):
        return self._Resp(200, [])          # nothing owned yet: start() will create one

    async def request(self, method, url, **kw):
        if url.endswith("/lifecycle"):
            return self._Resp(200, {"persistent": True, "ttl_seconds": 3600})
        if method == "GET":
            return self._Resp(200, [])
        if method == "DELETE":
            self.deleted.append(url.rsplit("/", 1)[-1])
            return self._Resp(204)
        return await self.post(url, **kw)


class MemoryStore:
    def __init__(self):
        self.values = {}

    async def get_setting(self, key):
        return self.values.get(key, "")

    async def set_setting(self, key, value):
        self.values[key] = value


@pytest.mark.asyncio
async def test_an_exhausted_account_never_switches_the_owners_power_off():
    """A 429 on the first key used to set intent=off and strand the studio.

    The pool must move to a healthy account and the owner's intent must survive the
    failure untouched - that is the difference between "today's quota ran out on one
    account" and "the computer is off".
    """
    from blackthorn.panda_control import INTENT, PandaControl

    store = MemoryStore()
    await store.set_setting(INTENT, "on")
    panda = QuotaPanda()
    control = PandaControl(store, panda, ["pds_spent", "pds_fresh"])
    status = await control.start()
    assert panda.created == ["12345678-1234-1234-1234-123456780002"], panda.created
    assert await store.get_setting(INTENT) == "on", "the owner's power intent must not be flipped"
    assert status is not None


@pytest.mark.asyncio
async def test_the_lifecycle_and_the_computer_both_hand_the_pool_over():
    """A string key must never silently downgrade rotation to a single account."""
    from blackthorn.panda_control import PandaControl

    class Pool:
        def __init__(self):
            self.states = {"aaaa": type("S", (), {"fingerprint": "aaaa", "key": "pds_aaaa"})(),
                           "bbbb": type("S", (), {"fingerprint": "bbbb", "key": "pds_bbbb"})()}
            self.active = type("S", (), {"key": "pds_aaaa", "fingerprint": "aaaa"})()

        async def ensure_active(self):
            return self.active

        async def restore_active(self):
            return None

        def active_key(self):
            return "pds_aaaa"

        def invalidate(self):
            return None

        async def rotate(self, reason=""):
            return True

    control = PandaControl(MemoryStore(), QuotaPanda(), "pds_aaaa", pool=Pool())
    keys = await control._keys()
    assert [fp for _, fp in keys] == ["aaaa", "bbbb"], "both accounts must stay in rotation"


@pytest.mark.asyncio
async def test_a_reset_connection_is_retried_instead_of_failing_the_command():
    """The provider resets connections under load; one reset must not fail a healthy task.

    A live run lost its capability search to exactly this: the VM was fine, the connection
    was not, and the agent was told "could not be reached".
    """
    import httpx
    from blackthorn.panda_computer import PandaComputerClient

    class FlakyClient(httpx.AsyncClient):
        pass

    attempts = {"n": 0}

    class Transport:
        async def handle_async_request(self, request):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise httpx.RemoteProtocolError("Server disconnected without sending a response")
            return httpx.Response(200, json={"stdout": "ok", "stderr": "", "exit_code": 0},
                                  request=request)

    http = httpx.AsyncClient(transport=Transport())
    client = object.__new__(PandaComputerClient)
    client.http = http
    client.key = "pds_test"
    client.base = "https://api.pandastack.ai/v1/sandboxes/abc"
    client.store = None
    client._awake_checked_at = 0.0
    result = await PandaComputerClient._request(client, "POST", "/exec", payload={"cmd": "true"})
    assert result.status_code == 200 and attempts["n"] == 2

    # A non-idempotent call must not silently run twice.
    attempts["n"] = 0
    with pytest.raises(Exception):
        await PandaComputerClient._request(client, "POST", "/exec", payload={"cmd": "true"}, retries=0)
