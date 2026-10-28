"""Controller invariants. These unit tests do not replace live provider probes."""
import json

import httpx
import pytest

from blackthorn.panda_control import PandaControl, INTENT, SANDBOX, PURPOSE
from blackthorn.panda_computer import PandaComputerClient
from blackthorn.tools.base import ToolError


class Settings:
    def __init__(self):
        self.values = {}

    async def get_setting(self, key):
        return self.values.get(key, "")

    async def set_setting(self, key, value):
        self.values[key] = value


@pytest.mark.asyncio
async def test_start_probe_reuse_and_stop_only_managed_sandbox():
    own = "12345678-1234-1234-1234-123456789abc"
    other = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    records = [{"id": other, "status": "running", "metadata": {"purpose": "unrelated"}}]
    actions = []

    def answer(request):
        actions.append((request.method, request.url.path))
        if request.url.path == "/v1/sandboxes" and request.method == "GET":
            return httpx.Response(200, json=records)
        if request.url.path == "/v1/sandboxes" and request.method == "POST":
            import json
            payload = json.loads(request.content)
            assert payload["ttl_seconds"] == 3600 and not payload.get("persistent")
            records.append({"id": own, "status": "running", "metadata": {"purpose": PURPOSE}})
            return httpx.Response(201, json=records[-1])
        if request.url.path == f"/v1/sandboxes/{own}/lifecycle":
            if request.method == "PATCH":
                return httpx.Response(200, json={"ttl_seconds": 3600, "persistent": False})
            return httpx.Response(200, json={"ttl_seconds": 300 if not any(m == "PATCH" for m, _ in actions) else 3600,
                                             "persistent": False})
        if request.url.path == f"/v1/sandboxes/{own}/exec":
            return httpx.Response(200, json={"stdout": "437\n", "exit_code": 0})
        if request.url.path == f"/v1/sandboxes/{own}/hibernate":
            records[-1]["status"] = "hibernated"
            return httpx.Response(200, json=records[-1])
        if request.url.path == f"/v1/sandboxes/{own}/wake":
            records[-1]["status"] = "running"
            return httpx.Response(200, json=records[-1])
        if request.url.path == f"/v1/sandboxes/{own}" and request.method == "DELETE":
            records[:] = [x for x in records if x["id"] != own]
            return httpx.Response(204)
        raise AssertionError(f"unexpected request {request.method} {request.url.path}")

    store = Settings()
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as http:
        controller = PandaControl(store, http, "private-test-key")
        assert (await controller.start())["online"]
        assert store.values[SANDBOX] == own and store.values[INTENT] == "on"
        assert (await controller.start())["online"]
        assert sum(m == "POST" and p == "/v1/sandboxes" for m, p in actions) == 1
        assert (await controller.stop())["status"] == "off"
        assert store.values[SANDBOX] == own and store.values[INTENT] == "off"
        assert (await controller.start())["online"]
        assert sum(m == "POST" and p == "/v1/sandboxes" for m, p in actions) == 1
        assert (await controller.destroy())["status"] == "off"
        assert records == [{"id": other, "status": "running", "metadata": {"purpose": "unrelated"}}]


@pytest.mark.asyncio
async def test_agent_route_refuses_powered_off_computer_before_network():
    store = Settings()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: (_ for _ in ()).throw(AssertionError("network used")))) as http:
        client = PandaComputerClient(http, api_key="provider-test", github_token="drive-test", store=store, mirror=False)
        with pytest.raises(ToolError) as error:
            await client.call("/computer/exec", {"command": "echo hello"})
        assert error.value.kind == "offline"

@pytest.mark.asyncio
async def test_agent_task_wakes_automatically_hibernated_computer():
    sid = "12345678-1234-1234-1234-123456789abc"
    store = Settings()
    store.values.update({INTENT: "on", SANDBOX: sid})
    actions = []
    state = {"status": "hibernated"}

    def answer(request):
        actions.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == "/v1/sandboxes":
            return httpx.Response(200, json=[{"id": sid, "status": state["status"], "metadata": {"purpose": PURPOSE}}])
        if request.method == "GET" and request.url.path.endswith("/lifecycle"):
            return httpx.Response(200, json={"persistent": False, "ttl_seconds": 3600})
        if request.method == "POST" and request.url.path.endswith("/wake"):
            state["status"] = "running"
            return httpx.Response(200, json={"status": "running"})
        if request.method == "POST" and request.url.path.endswith("/exec"):
            assert state["status"] == "running"
            value = "437\n" if b"19*23" in request.content else "45\n"
            return httpx.Response(200, json={"stdout": value, "stderr": "", "exit_code": 0})
        raise AssertionError(f"unexpected request {request.method} {request.url.path}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as http:
        client = PandaComputerClient(http, api_key="provider-test", github_token="drive-test", store=store, mirror=False)
        out = await client.call("/computer/exec", {"command": "echo 45"})
        assert out["ok"] and out["output"].strip() == "45"
        assert any(path.endswith("/wake") for _, path in actions)
        assert not any(method == "POST" and path == "/v1/sandboxes" for method, path in actions)
        fetched = await client.call("/computer/fetch_url", {"url": "https://example.com"})
        assert fetched["ok"] and fetched["text"].strip() == "45"
        assert actions[-1][1].endswith("/exec")

@pytest.mark.asyncio
async def test_expired_vm_is_replaced_only_once_across_concurrent_workers():
    import asyncio
    from blackthorn.sql import SqliteExecutor
    from blackthorn.store import ChatStore
    db = SqliteExecutor()
    await db.query('CREATE TABLE state_meta (key TEXT, value TEXT)')
    store = ChatStore(db)
    await store.set_setting(INTENT, 'on')
    sid = '12345678-1234-1234-1234-123456789abc'
    records = []
    creates = []

    def answer(request):
        path = request.url.path
        if path == '/v1/sandboxes' and request.method == 'GET':
            return httpx.Response(200, json=records)
        if path == '/v1/sandboxes' and request.method == 'POST':
            creates.append(1)
            records.append({'id': sid, 'status': 'running', 'metadata': {'purpose': PURPOSE}})
            return httpx.Response(201, json=records[-1])
        if path.endswith('/lifecycle'):
            return httpx.Response(200, json={'persistent': False, 'ttl_seconds': 3600})
        if path.endswith('/exec'):
            return httpx.Response(200, json={'stdout': '437\n', 'exit_code': 0})
        raise AssertionError(path)

    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as http:
            a, b = PandaControl(store, http, 'test'), PandaControl(store, http, 'test')
            states = await asyncio.gather(a.ensure_for_task(), b.ensure_for_task())
        assert all(x['online'] for x in states)
        assert len(creates) == 1
        assert await store.get_setting(SANDBOX) == sid
    finally:
        await db.aclose()


async def test_reconciliation_never_deletes_a_sandbox_a_handover_depends_on():
    """During a handover two sandboxes legitimately exist: the source and its replacement."""
    store = Settings()
    state = {"source": "old-vm", "replacement": "new-vm"}
    await store.set_setting("pandastack_lifecycle_state", json.dumps(state))
    deleted = []

    class Control(PandaControl):
        async def _owned(self):
            return [{"id": "old-vm", "status": "running", "metadata": {"purpose": PURPOSE, "role": "primary"}},
                    {"id": "new-vm", "status": "running", "metadata": {"purpose": PURPOSE, "role": "replacement"}},
                    {"id": "leaked-vm", "status": "running", "metadata": {"purpose": PURPOSE, "role": "primary"}}]

        async def delete(self, sid):
            deleted.append(sid)
            return True

    released = await Control(store, None, "test-key").reconcile_extras()
    assert released == ["leaked-vm"], released
    assert set(deleted) == {"leaked-vm"}
