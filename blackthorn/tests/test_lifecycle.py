"""Sandbox lifecycle tests: proactive handover, verification, rollback and recovery.

The provider is mocked with the documented PandaStack REST shapes so the manager's
decision logic is tested against real semantics (TTL in seconds, overlapping
sandboxes, exec probe, delete).
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from blackthorn.lifecycle import SandboxLifecycle

OLD = "11111111-1111-1111-1111-111111111111"
NEW = "22222222-2222-2222-2222-222222222222"
BASE = "/v1/sandboxes"


class Store:
    def __init__(self, **values):
        self.values = dict(values)

    async def get_setting(self, key):
        return self.values.get(key, "")

    async def set_setting(self, key, value):
        self.values[key] = value


class Mirror:
    def __init__(self, fail_restore=False):
        self.synced = []
        self.restored = []
        self.fail_restore = fail_restore

    async def sync(self, force=False):
        self.synced.append(force)
        return True

    async def restore(self, sid, force=False):
        if self.fail_restore:
            raise RuntimeError("archive extraction failed")
        self.restored.append((sid, force))


class Computer:
    def __init__(self, mirror):
        self.mirror = mirror
        self.key = "pds_test"
        self.sandbox_id = OLD
        self.base = "https://api.pandastack.ai/v1/sandboxes/" + OLD
        self.http = None


def provider(*, created_minutes_ago=58, records=None, probe_ok=True, delete_ok=True, ttl=3600):
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - created_minutes_ago * 60))
    state = {"records": records if records is not None else [
        {"id": OLD, "status": "running", "created_at": created, "metadata": {"purpose": "blackthorn-production-computer",
                                                                             "role": "primary"}}]}
    calls = []

    def handler(request: httpx.Request):
        path, method = request.url.path, request.method
        calls.append((method, path))
        if path == BASE and method == "GET":
            return httpx.Response(200, json=state["records"])
        if path == BASE and method == "POST":
            payload = json.loads(request.content)
            assert payload["ttl_seconds"] == 3600
            record = {"id": NEW, "status": "running",
                      "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "metadata": payload["metadata"]}
            state["records"].append(record)
            return httpx.Response(201, json=record)
        if path.endswith("/lifecycle"):
            return httpx.Response(200, json={"ttl_seconds": ttl, "persistent": False})
        if path.endswith("/exec"):
            if not probe_ok:
                return httpx.Response(200, json={"stdout": "", "exit_code": 1})
            return httpx.Response(200, json={"stdout": "437\n", "exit_code": 0})
        if method == "DELETE":
            if not delete_ok:
                return httpx.Response(500, json={"error": "busy"})
            state["records"] = [r for r in state["records"] if f"/{r['id']}" != path.replace(BASE, "")]
            return httpx.Response(204)
        if method == "GET":
            match = next((r for r in state["records"] if path.endswith(r["id"])), None)
            return httpx.Response(200, json=match) if match else httpx.Response(404, json={})
        return httpx.Response(404, json={})

    return httpx.MockTransport(handler), calls, state


def build(transport, *, window_s=300, fail_restore=False):
    mirror = Mirror(fail_restore=fail_restore)
    computer = Computer(mirror)
    store = Store(**{"pandastack_computer_sandbox_id": OLD, "pandastack_computer_intent": "on"})
    computer.http = httpx.AsyncClient(transport=transport)
    return SandboxLifecycle(store, computer, window_s=window_s), computer, store


@pytest.mark.asyncio
async def test_snapshot_reports_provider_timing():
    transport, _calls, _state = provider(created_minutes_ago=58)
    manager, _computer, _store = build(transport)
    snap = await manager.snapshot()
    assert snap["sandbox_id"] == OLD and snap["online"] is True
    assert 0 < snap["seconds_left"] <= 300
    assert snap["handover_due"] is True
    await _computer.http.aclose()


@pytest.mark.asyncio
async def test_handover_checkpoints_creates_verifies_promotes_and_deletes():
    transport, calls, records = provider(created_minutes_ago=58)
    manager, computer, store = build(transport)
    result = await manager.tick(activity=True)
    assert result["action"] == "handover" and result["ok"], result
    assert result["phase"] == "promoted"
    assert computer.mirror.synced == [True], "a durable checkpoint is published first"
    assert (NEW, True) in computer.mirror.restored, "the replacement receives the checkpoint"
    assert await store.get_setting("pandastack_computer_sandbox_id") == NEW
    assert computer.sandbox_id == NEW and computer.base.endswith(NEW)
    assert ("DELETE", f"{BASE}/{OLD}") in calls, "the old sandbox is released"
    assert records["records"] and records["records"][0]["id"] == NEW
    log = json.loads(await store.get_setting("pandastack_lifecycle_log"))
    phases = [e.get("phase") for e in log if e.get("kind") == "handover"]
    assert phases[-1] == "promoted" and phases[0] == "created"
    await computer.http.aclose()


@pytest.mark.asyncio
async def test_a_completed_handover_is_not_repeated():
    transport, calls, _records = provider(created_minutes_ago=58)
    manager, computer, _store = build(transport)
    await manager.tick(activity=True)
    manager._last_tick = 0
    creates = [c for c in calls if c == ("POST", BASE)]
    second = await manager.handover(reason="ttl")
    assert second["phase"] in ("already_completed", "not_due"), second
    assert len([c for c in calls if c == ("POST", BASE)]) == len(creates), "no second sandbox is created"
    await computer.http.aclose()


@pytest.mark.asyncio
async def test_failed_verification_rolls_back_to_the_live_sandbox():
    transport, _calls, records = provider(created_minutes_ago=58, probe_ok=False)
    manager, computer, store = build(transport)
    result = await manager.tick(activity=True)
    assert result["action"] == "handover" and not result["ok"]
    assert result["phase"] == "verify_failed"
    assert computer.sandbox_id == OLD and computer.base.endswith(OLD), "the task keeps its original VM"
    assert await store.get_setting("pandastack_computer_sandbox_id") == OLD
    state = json.loads(await store.get_setting("pandastack_lifecycle_state"))
    assert state["phase"] == "verify_failed"
    assert any(r["id"] == OLD for r in records["records"])
    await computer.http.aclose()


@pytest.mark.asyncio
async def test_restore_failure_is_reported_not_hidden():
    transport, _calls, _records = provider(created_minutes_ago=58)
    manager, computer, _store = build(transport, fail_restore=True)
    result = await manager.tick(activity=True)
    assert not result["ok"] and result["phase"] == "verify_failed"
    assert "archive" in result["detail"]
    await computer.http.aclose()


@pytest.mark.asyncio
async def test_not_due_means_no_work():
    transport, calls, _records = provider(created_minutes_ago=10)
    manager, computer, _store = build(transport)
    result = await manager.tick(activity=True)
    assert result["action"] == "none"
    assert ("POST", BASE) not in calls
    await computer.http.aclose()


@pytest.mark.asyncio
async def test_expired_sandbox_is_recreated_and_restored():
    transport, calls, _records = provider(records=[])
    manager, computer, store = build(transport)
    result = await manager.tick(activity=True)
    assert result["action"] == "recovered" and result["ok"]
    assert (NEW, True) in computer.mirror.restored
    assert await store.get_setting("pandastack_computer_sandbox_id") == NEW
    await computer.http.aclose()


@pytest.mark.asyncio
async def test_checkpoint_records_what_is_durable():
    transport, _calls, _records = provider(created_minutes_ago=5)
    manager, computer, store = build(transport)
    await store.set_setting("pandastack_workspace_drive_digest", "abc123")
    event = await manager.checkpoint(reason="unit")
    assert event["published"] is True and event["workspace"] == "abc123"
    assert computer.mirror.synced == [True]
    await computer.http.aclose()
