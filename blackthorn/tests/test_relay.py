"""The durable GPU relay: request/response round-trip, auth, and honest failure modes."""

import asyncio

import httpx
import pytest

import blackthorn.relay as relay_mod

pytestmark = pytest.mark.asyncio

KEY = "test-key"


class Notebook:
    """What the notebook-side relay client does, in-process: pull, execute, push."""

    def __init__(self, stack, key: str = KEY):
        self.stack = stack
        self.key = key
        self.seen = []
        self.respond = {"status": 200, "content_type": "application/json", "body": b'{"ok": true}', "chunks": None}
        self.push = True

    async def run_once(self) -> int:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(f"{self.stack.url}/api/kaggle-relay/pull", headers={"X-Blackthorn-Key": self.key})
            if r.status_code != 200:
                return r.status_code
            job = r.json()
            self.seen.append(job)
            if not self.push:
                return 200
            resp = self.respond
            chunks = resp.get("chunks") if resp.get("chunks") is not None else [resp.get("body", b"")]

            async def gen():
                for ch in chunks:
                    yield ch

            await c.post(f"{self.stack.url}/api/kaggle-relay/push/{job['id']}",
                         headers={"X-Blackthorn-Key": self.key,
                                  "X-Relay-Status": str(resp["status"]),
                                  "X-Relay-Content-Type": resp["content_type"]},
                         content=gen())
            return 200


async def _proxy(stack, path="v1/models", method="POST", body=b'{"q": 1}', headers=None):
    async with httpx.AsyncClient(timeout=30) as c:
        return await c.request(method, f"{stack.url}/gpu-relay/{path}", content=body, headers=headers or {})


async def test_relay_round_trip_preserves_method_path_body_status_and_streams_the_reply(stack):
    nb = Notebook(stack)
    nb.respond = {"status": 503, "content_type": "application/json", "body": b'{"upstream": "busy"}'}
    call = asyncio.create_task(_proxy(stack, path="v1/chat/completions", body=b'{"stream": true}',
                                      headers={"Authorization": "Bearer test-key"}))
    await asyncio.sleep(0.05)
    assert await nb.run_once() == 200
    resp = await call
    assert resp.status_code == 503
    assert resp.content == b'{"upstream": "busy"}'
    assert resp.headers["content-type"].startswith("application/json")
    job = nb.seen[0]
    assert job["method"] == "POST" and job["path"] == "v1/chat/completions"
    import base64
    assert base64.b64decode(job["body_b64"]) == b'{"stream": true}'
    assert job["headers"].get("authorization") == "Bearer test-key"      # the gateway key passes through
    assert "host" not in job["headers"] and "content-length" not in job["headers"]


async def test_relay_forwards_chunked_replies_intact(stack):
    nb = Notebook(stack)
    nb.respond = {"status": 200, "content_type": "text/event-stream",
                  "chunks": [b"data: one\n\n", b"data: two\n\n", b"data: [DONE]\n\n"], "body": b""}
    call = asyncio.create_task(_proxy(stack))
    await asyncio.sleep(0.05)
    await nb.run_once()
    resp = await call
    assert resp.status_code == 200
    assert resp.content == b"data: one\n\ndata: two\n\ndata: [DONE]\n\n"
    assert resp.headers["content-type"].startswith("text/event-stream")


async def test_relay_pull_requires_the_per_boot_key(stack):
    async with httpx.AsyncClient(timeout=30) as c:
        bad = await c.get(f"{stack.url}/api/kaggle-relay/pull", headers={"X-Blackthorn-Key": "wrong"})
        assert bad.status_code == 401
        none = await c.get(f"{stack.url}/api/kaggle-relay/pull")
        assert none.status_code == 401
    nb = Notebook(stack, key="wrong")
    assert await nb.run_once() == 401


async def test_relay_proxy_fails_fast_when_no_notebook_is_connected(stack):
    stack.services.relay_hub = relay_mod.RelayHub()
    stack.services.relay_hub.last_seen = 0.0                     # nobody has been around for ages
    resp = await _proxy(stack)
    assert resp.status_code == 502
    assert resp.json()["detail"]["code"] == "gpu_unreachable"


async def test_relay_proxy_times_out_if_the_notebook_never_pushes(stack, monkeypatch):
    monkeypatch.setattr(relay_mod, "HEADERS_TIMEOUT_S", 0.25)
    nb = Notebook(stack)
    nb.push = False                                              # pulls the job, then vanishes
    call = asyncio.create_task(_proxy(stack))
    await asyncio.sleep(0.05)
    await nb.run_once()
    resp = await call
    assert resp.status_code == 502
    assert resp.json()["detail"]["code"] == "gpu_unreachable"


async def test_relay_quiet_poll_returns_204(stack, monkeypatch):
    monkeypatch.setattr(relay_mod, "PULL_WAIT_S", 0.2)
    nb = Notebook(stack)
    assert await nb.run_once() == 204
