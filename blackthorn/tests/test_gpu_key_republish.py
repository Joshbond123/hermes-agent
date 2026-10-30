"""The GPU credential must be recoverable without the operator.

Two proven defects are pinned here.

1. **The notebook could never republish its key.** Its heartbeat upsert refused to write while the
   row read GPU_STOPPED/STOPPING (so a late heartbeat cannot resurrect a role) - but that also
   blocked a *newly started* session, whose fresh key was therefore never published. The agent's
   refresh kept returning the same dead credential, which is exactly the operator's
   "the GPU gateway kept rejecting the refreshed credential after 3 refresh rounds".
   A new boot now wins over a row written before it started.
2. **The controller forwarded the caller's stale key.** The relay hop now authenticates the caller
   against the published key and forwards the authoritative one, and answers a stale credential
   with a distinct retryable code instead of a bare 401.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import time

import pytest

import blackthorn.relay as relay

SRC = pathlib.Path(__file__).resolve().parents[2] / "kaggle_cyber_ornith" / "cyber_ornith_server.py"


@pytest.fixture()
def srv(monkeypatch):
    spec = importlib.util.spec_from_file_location("cyber_ornith_boot_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    posts = []

    def capture(request, *a, **k):
        if "d1/database" in getattr(request, "full_url", ""):
            body = json.loads(request.data.decode())
            posts.append((body["sql"], body["params"]))
        return None

    for key in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_D1_DATABASE_ID", "CLOUDFLARE_API_TOKEN"):
        monkeypatch.setenv(key, "test")
    monkeypatch.setattr(mod, "_d1_exec", lambda sql, params: posts.append((sql, params)))
    monkeypatch.setattr(mod, "log", lambda *a, **k: None)
    monkeypatch.setattr(mod.urllib.request, "urlopen", capture)
    mod.posts = posts
    return mod


def test_a_new_session_may_publish_over_a_row_a_stop_closed(srv):
    """The upsert must be able to overwrite a closed row written before this boot."""
    srv.notify_workspace("ONLINE", "https://tunnel.example")
    sql, params = srv.posts[-1]
    assert "GPU_STOPPED_SAVING_QUOTA" in sql and "STOPPING_KAGGLE_GPU" in sql
    # ... but only when that row predates this process.
    assert "OR kaggle_gpu_state.updated_at < ?" in sql, "a restarted session could never republish"
    assert params[-1] == srv.BOOT_EPOCH, "the boot epoch must reach the SQL guard"
    assert params[3] == srv.API_KEY, "the key this boot actually serves is what gets published"


def test_boot_epoch_is_taken_once_at_import(srv):
    first = srv.BOOT_EPOCH
    time.sleep(0.01)
    assert srv.BOOT_EPOCH == first and first <= time.time()


# --------------------------------------------------------------------------- relay hop
class _Req:
    def __init__(self, headers, app=None, method="POST"):
        self.headers = headers
        self.app = app
        self.method = method

    async def body(self):
        return b'{"model":"m","messages":[]}'


@pytest.mark.asyncio
async def test_relay_forwards_the_published_key_not_the_callers_stale_one(monkeypatch):
    calls = []

    def fake_submit(method, path, headers, body):
        calls.append(headers)
        raise RuntimeError("stop here - the headers are what matters")

    monkeypatch.setattr(relay, "_row_keys", _publish("bt-new"))
    monkeypatch.setattr(relay, "_hub", _fake_hub(fake_submit))
    with pytest.raises(RuntimeError):
        await relay.relay_proxy(_Req({"authorization": "Bearer bt-new"}), "model/v1/chat/completions")
    assert calls[0]["authorization"] == "Bearer bt-new"


@pytest.mark.asyncio
async def test_a_stale_caller_is_told_to_refresh_and_nothing_is_forwarded(monkeypatch):
    forwarded = []
    monkeypatch.setattr(relay, "_row_keys", _publish("bt-new"))
    monkeypatch.setattr(relay, "_hub", _fake_hub(lambda *a: forwarded.append(a)))
    with pytest.raises(relay.HTTPException) as err:
        await relay.relay_proxy(_Req({"authorization": "Bearer bt-old"}), "model/v1/chat/completions")
    assert err.value.status_code == 401
    assert err.value.detail["code"] == "stale_key"
    assert not forwarded, "a rejected request must not reach the model"


@pytest.mark.asyncio
async def test_no_published_key_means_service_unavailable_not_a_silent_401(monkeypatch):
    monkeypatch.setattr(relay, "_row_keys", _publish(""))
    monkeypatch.setattr(relay, "_hub", _fake_hub(lambda *a: None))
    with pytest.raises(relay.HTTPException) as err:
        await relay.relay_proxy(_Req({"authorization": "Bearer bt-x"}), "model/v1/chat/completions")
    assert err.value.status_code == 503
    assert err.value.detail["code"] == "gpu_key_unavailable"


@pytest.mark.asyncio
async def test_get_probes_still_pass_through_without_a_caller_key(monkeypatch):
    forwarded = []

    def fake_submit(method, path, headers, body):
        forwarded.append(headers)
        raise RuntimeError("done")

    monkeypatch.setattr(relay, "_row_keys", _publish("bt-new"))
    monkeypatch.setattr(relay, "_hub", _fake_hub(fake_submit))
    with pytest.raises(RuntimeError):
        await relay.relay_proxy(_Req({}, method="GET"), "model/health")
    assert forwarded[0]["authorization"] == "Bearer bt-new"


def _fake_hub(submit):
    class _Hub:
        def alive(self):
            return True
        def submit(self, *a):
            return submit(*a)
    return lambda request, channel="model": _Hub()


def _publish(key):
    async def _row_keys(request, channel="model", *, force=False):
        return (key,) if key else ()
    return _row_keys


@pytest.mark.asyncio
async def test_agent_reopens_a_row_a_stop_closed_so_the_live_session_can_republish(stack):
    """The operator's loop, end to end.

    A stop lands while the session is still alive, so the published key is the previous boot's.
    The notebook is then forbidden from publishing its current key, every refresh returns the same
    credential, and the old code ended the task with "kept rejecting the refreshed credential".
    The agent must re-open the row instead, let the live session republish, and continue with no
    work lost and no error surfaced.
    """
    import asyncio

    from .fakes import finish, say

    await stack.set_gpu(key="stale-key")            # online, but the published key is the old one
    stack.backend.key = "live-key"                  # what the running session actually serves
    stack.backend.queue([*say("Recovered on the live credential."), finish("stop")])

    async def stop_lands_then_notebook_republishes():
        # 1. a verified shutdown writes the row closed while the session keeps serving
        await asyncio.sleep(0.4)
        await stack.executor.query(
            "UPDATE kaggle_gpu_state SET status='GPU_STOPPED_SAVING_QUOTA' WHERE id='primary'")
        # 2. once the agent re-opens the row, the live session publishes the key it serves
        for _ in range(1200):
            row = await stack.executor.query("SELECT status FROM kaggle_gpu_state WHERE id='primary'")
            if str(row.rows[0]["status"]).upper() == "RECOVERING_KEY":
                await stack.executor.query(
                    "UPDATE kaggle_gpu_state SET status='HEARTBEAT_ONLINE', api_key=? WHERE id='primary'",
                    [stack.backend.key])
                return True
            await asyncio.sleep(0.02)
        return False

    watcher = asyncio.create_task(stop_lands_then_notebook_republishes())
    out = await stack.stream({"message": "answer briefly"})
    repaired = await watcher

    assert repaired, "the agent never re-opened the row the stop had closed"
    assert out.end["status"] == "stop", out.end
    assert "Recovered" in out.text
    assert any("re-opened" in e.get("text", "") for e in out.of("notice"))
    assert not out.of("error"), "a stale credential must never end the task"
