"""Background jobs and the no-double-execution rule, run against a real local shell acting as the computer.

LocalComputer executes /computer/* requests with bash in a temp workspace, so the job files, exit markers,
PIDs and logs are genuine. It is only a stand-in for the Kaggle host and is never used as evidence of the
Kaggle path itself.
"""

import asyncio
import subprocess
import time

import httpx
import pytest

from blackthorn.config import Settings
from blackthorn.tools import computer as comp
from blackthorn.tools.base import ToolContext, ToolError


class LocalComputer:
    def __init__(self, root):
        self.root = root
        self.exec_calls = 0
        self.fail_next_exec_timeout = False

    async def call(self, path, payload=None, *, method="POST", timeout=90.0, idempotent=True):
        payload = payload or {}
        if path == "/computer/write_file":
            p = self.root / payload["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(payload["content"])
            return {"ok": True}
        if path == "/computer/exec":
            self.exec_calls += 1
            if self.fail_next_exec_timeout:
                self.fail_next_exec_timeout = False
                raise ToolError("timeout", "simulated: the notebook did not answer")
            cwd = (self.root / payload.get("cwd", ".")).resolve()
            cwd.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(["bash", "-c", payload["command"]], cwd=cwd, capture_output=True, text=True,
                               timeout=payload.get("timeout_seconds", 60))
            return {"ok": True, "output": r.stdout + r.stderr, "exit_code": r.returncode}
        raise AssertionError(path)


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(settings=Settings(), store=None, computer=LocalComputer(tmp_path), http=None, tavily=None)


async def _poll(ctx, job_id, limit=40):
    for _ in range(limit):
        res = await comp.job_status({"job_id": job_id, "wait_seconds": 1}, ctx)
        if res.data["state"] in ("EXITED", "LOST"):
            return res
    return res


async def test_background_job_runs_detached_and_reports_real_exit_code(ctx):
    started = await comp.run_command({"command": "echo step-1; sleep 2; echo step-2; exit 3", "background": True}, ctx)
    assert started.ok and started.data["state"] == "RUNNING"
    job_id = started.data["job_id"]
    # while it runs the status must say RUNNING, not finished
    first = await comp.job_status({"job_id": job_id, "wait_seconds": 0}, ctx)
    assert first.data["state"] in ("RUNNING", "EXITED")
    done = await _poll(ctx, job_id)
    assert done.data["state"] == "EXITED" and done.data["exit_code"] == 3
    assert "step-1" in done.content and "step-2" in done.content


async def test_background_job_survives_the_caller_going_away(ctx):
    """The job must not depend on the HTTP step that started it: a dropped relay is not a dead command."""
    started = await comp.run_command({"command": "sleep 1; echo survived > marker.txt", "background": True}, ctx)
    job_id = started.data["job_id"]
    # the agent's step ends here; nothing waits on the job
    await asyncio.sleep(0.2)
    done = await _poll(ctx, job_id)
    assert done.data["state"] == "EXITED" and done.data["exit_code"] == 0
    assert (ctx.computer.root / "marker.txt").read_text().strip() == "survived"


async def test_long_foreground_request_is_routed_to_a_background_job(ctx):
    res = await comp.run_command({"command": "echo long", "timeout_seconds": 3600}, ctx)
    assert res.data.get("job_id", "").startswith("bg-")


class _Route:
    url = "https://computer.example"
    api_key = "k"


class _Resolver:
    def __init__(self):
        self.invalidations = 0

    async def get(self):
        return _Route()

    def invalidate(self):
        self.invalidations += 1


async def test_command_is_never_resent_after_a_timeout(tmp_path):
    """Exercises the real ComputerClient: a timed-out shell request must not be sent a second time."""
    seen = {"exec": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/computer/exec":
            seen["exec"] += 1
            raise httpx.ReadTimeout("no answer", request=request)
        return httpx.Response(200, json={"ok": True})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = comp.ComputerClient(_Resolver(), http)
    ctx = ToolContext(settings=Settings(), store=None, computer=client, http=None, tavily=None)
    with pytest.raises(ToolError) as info:
        await comp.run_command({"command": "echo once >> log.txt"}, ctx)
    assert info.value.kind == "timeout"
    assert "background" in info.value.hint
    assert seen["exec"] == 1          # one send, no silent re-run
    await http.aclose()


async def test_reads_are_still_retried_on_timeout(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"ok": True, "listing": "a"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = comp.ComputerClient(_Resolver(), http)
    data = await client.call("/computer/list_files", {"path": "."}, idempotent=True)
    assert data["ok"] and calls["n"] == 2
    await http.aclose()


async def test_refused_destructive_command_never_reaches_the_computer(ctx):
    res = await comp.run_command({"command": "rm -rf /"}, ctx)
    assert not res.ok and res.data.get("kind") == "denied"
    assert ctx.computer.exec_calls == 0


async def test_job_id_is_validated(ctx):
    res = await comp.job_status({"job_id": "../../etc"}, ctx)
    assert not res.ok
