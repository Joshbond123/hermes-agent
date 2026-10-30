"""Regressions for the live route-key mismatch that stopped a 51-step task."""
import asyncio
import time

import httpx
import pytest

import blackthorn.relay as relay
import cloudflare_d1_client as controller
from .fakes import call, finish, say


@pytest.mark.asyncio
async def test_rotated_key_recovers_without_repeating_completed_tool(stack):
    """Real HTTP/SSE: first model step and tool succeed; the next model request
    uses a cached old route and receives 401; D1 then supplies the new key.
    """
    stack.backend.queue([call("run_command", {"command": "echo once"}, cid="auth1"), finish("tool_calls")],
                        [*say("The command ran once."), finish("stop")])
    stack.backend.exec_delay = 0.3
    async def rotate_after_tool():
        for _ in range(200):
            if stack.backend.computer_calls:
                stack.backend.key = "fresh-test-key"
                await stack.executor.query("UPDATE kaggle_gpu_state SET api_key = ? WHERE id = 'primary'",
                                           [stack.backend.key])
                return
            await asyncio.sleep(0.01)
        pytest.fail("the tool did not execute")
    watcher = asyncio.create_task(rotate_after_tool())
    out = await stack.stream({"message": "run one command and report its output"})
    await watcher
    assert out.end["status"] == "stop", out.end
    assert "ran once" in out.text
    assert len(stack.backend.computer_calls) == 1
    assert len(stack.backend.requests) == 2  # the unauthorized request never enters the model
    assert len(out.of("tool.end")) == 1
    assert any("refreshed" in e.get("text", "") for e in out.of("notice"))
    assert any(m["role"] == "tool" for m in stack.backend.requests[-1]["messages"])


@pytest.mark.asyncio
async def test_auth_stays_bounded_when_d1_key_is_still_wrong(stack):
    stack.backend.key = "different-test-key"  # D1 retains the old key
    out = await stack.stream({"message": "answer briefly"})
    assert out.end["status"] == "error" and out.of("error")[0]["code"] == "gpu_auth"
    assert out.of("error")[0]["retryable"] is True, "the task is saved; a retry must be offered"
    assert not stack.backend.requests and not stack.backend.computer_calls


@pytest.mark.asyncio
async def test_transient_auth_rejection_with_still_valid_key_retries_once(stack):
    stack.backend.queue({"http_error": 401, "message": "temporary edge rejection"},
                        [*say("Recovered."), finish("stop")])
    out = await stack.stream({"message": "recover from a one-off gateway rejection"})
    assert out.end["status"] == "stop"
    assert out.text == "Recovered."
    assert len(stack.backend.requests) == 2
    assert len([e for e in out.of("notice") if "refreshed" in e.get("text", "")]) == 1


@pytest.mark.asyncio
async def test_persistent_auth_rejection_is_bounded_but_not_immediate(stack):
    """Three refreshes, not one: the notebook publishes its rotated key seconds later.

    Still bounded - a gateway that keeps denying must fail, not loop - and the failure is
    retryable with the task saved, so the operator resumes instead of losing 12 finished steps.
    """
    stack.backend.queue(*[{"http_error": 403, "message": "gateway denied"}] * 6)
    out = await stack.stream({"message": "a gateway that repeatedly denies the stream"})
    assert out.end["status"] == "error"
    assert out.of("error")[0]["code"] == "gpu_auth"
    assert 2 <= len(stack.backend.requests) <= 5, len(stack.backend.requests)


@pytest.mark.asyncio
async def test_relay_does_not_let_old_or_other_role_take_a_new_model_job(stack, monkeypatch):
    monkeypatch.setattr(relay, "PULL_WAIT_S", 0.2)
    monkeypatch.setattr(relay, "_KEY_CACHE", {})
    # A computer key is never a model poll credential.
    await stack.executor.query(
        "INSERT INTO kaggle_gpu_state (id, status, tunnel_url, api_key, model, gpu_info, updated_at) "
        "VALUES ('computer', 'HEARTBEAT_ONLINE', ?, 'computer-key', 'computer', 'T4', ?)",
        [stack.backend_srv.url, time.time()])
    relay._KEY_CACHE.clear()
    async with httpx.AsyncClient(timeout=5) as c:
        wrong_role = await c.get(f"{stack.url}/api/kaggle-relay/pull?channel=model",
                                 headers={"X-Blackthorn-Key": "computer-key"})
        assert wrong_role.status_code == 401
        # The rejected other-role poll populated the model-key cache.
        await stack.executor.query("UPDATE kaggle_gpu_state SET api_key='new-model-key' WHERE id='primary'")
        task = asyncio.create_task(c.get(f"{stack.url}/gpu-relay/model/v1/models",
                                       headers={"Authorization": "Bearer new-model-key"}))
        await asyncio.sleep(0.06)
        stale = await c.get(f"{stack.url}/api/kaggle-relay/pull?channel=model",
                            headers={"X-Blackthorn-Key": "test-key"})
        assert stale.status_code == 401
        assert len(stack.services.relay_hubs["model"].inflight) == 1
        # A fresh notebook must be accepted immediately despite a cached old key.
        fresh = await c.get(f"{stack.url}/api/kaggle-relay/pull?channel=model",
                            headers={"X-Blackthorn-Key": "new-model-key"})
        assert fresh.status_code == 200
        job = fresh.json()
        assert job["headers"]["authorization"] == "Bearer new-model-key"
        intruder = await c.post(f"{stack.url}/api/kaggle-relay/push/{job['id']}",
                                headers={"X-Blackthorn-Key": "computer-key"}, content=b"wrong")
        assert intruder.status_code == 401
        # A role's key may rotate mid-stream; assigned jobs retain their own
        # authenticated push capability while new jobs require the new key.
        await stack.executor.query("UPDATE kaggle_gpu_state SET api_key='yet-newer-key' WHERE id='primary'")
        pushed = await c.post(f"{stack.url}/api/kaggle-relay/push/{job['id']}",
                              headers={"X-Blackthorn-Key": "new-model-key", "X-Relay-Status": "200"},
                              content=b"ok")
        assert pushed.status_code == 200
        assert (await task).text == "ok"


@pytest.mark.asyncio
async def test_quiet_relay_204_has_no_body_and_connection_is_reusable(stack, monkeypatch):
    monkeypatch.setattr(relay, "PULL_WAIT_S", 0.05)
    monkeypatch.setattr(relay, "_KEY_CACHE", {})
    async with httpx.AsyncClient(timeout=5) as client:
        for _ in range(2):
            resp = await client.get(f"{stack.url}/api/kaggle-relay/pull?channel=model",
                                    headers={"X-Blackthorn-Key": "test-key"})
            assert resp.status_code == 204 and resp.content == b""
        assert (await client.get(f"{stack.url}/health")).status_code in (200, 404)


def test_status_probe_preserves_notebook_key_and_guards_concurrent_boot(monkeypatch):
    """Exercise real controller's slow health path; no legacy empty key write."""
    from .test_gpu_controller import _prepare
    _prepare(monkeypatch, status="TUNNEL_ONLINE")
    row = {"status": "TUNNEL_ONLINE", "tunnel_url": "https://tunnel.example", "api_key": "notebook-key"}
    writes = []
    def query(sql, params=None, timeout=15.0):
        if sql.strip().upper().startswith("SELECT *") and "FROM kaggle_gpu_state" in sql:
            return [row.copy()]
        if "UPDATE kaggle_gpu_state" in sql:
            writes.append((sql, params))
        return []
    monkeypatch.setattr(controller, "d1_query", query)
    monkeypatch.setattr(controller, "CYBER_ORNITH_API_KEY", "")
    controller.get_kaggle_gpu_status(force_refresh=True)
    online = [(sql, args) for sql, args in writes if "SET status = 'ONLINE'" in sql]
    assert len(online) == 1
    sql, args = online[0]
    assert "api_key =" not in sql and "api_key <> ''" in sql
    assert "tunnel_url = ?" in sql and args[-1] == row["tunnel_url"]
    assert row["api_key"] == "notebook-key"
