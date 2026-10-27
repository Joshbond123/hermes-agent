"""Controller invariants. These unit tests do not replace live provider probes."""
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
            records.append({"id": own, "status": "running", "metadata": {"purpose": PURPOSE}})
            return httpx.Response(201, json=records[-1])
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
        client = PandaComputerClient(http, api_key="provider-test", github_token="drive-test", store=store)
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
            return httpx.Response(200, json=[{"id": sid, "status": state["status"]}])
        if request.method == "POST" and request.url.path.endswith("/wake"):
            state["status"] = "running"
            return httpx.Response(200, json={"status": "running"})
        if request.method == "POST" and request.url.path.endswith("/exec"):
            assert state["status"] == "running"
            return httpx.Response(200, json={"stdout": "45\n", "stderr": "", "exit_code": 0})
        raise AssertionError(f"unexpected request {request.method} {request.url.path}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as http:
        client = PandaComputerClient(http, api_key="provider-test", github_token="drive-test", store=store)
        out = await client.call("/computer/exec", {"command": "echo 45"})
        assert out["ok"] and out["output"].strip() == "45"
        assert [path.rsplit("/", 1)[-1] for _, path in actions] == ["sandboxes", "wake", "exec"]
        fetched = await client.call("/computer/fetch_url", {"url": "https://example.com"})
        assert fetched["ok"] and fetched["text"].strip() == "45"
        assert actions[-1][1].endswith("/exec")
