"""GitHub Codespaces as Blackthorn's computer: lifecycle, exec channel, power management, usage.

Why this shape
--------------
GitHub's REST API can create/start/stop/delete a codespace but it cannot run a command in one.
The documented way to reach a process inside a codespace from the internet is a forwarded port:
a process listening in the container is published at ``https://<codespace>-<port>.app.github.dev``.
So a small agent (``devcontainer/blackthorn_agent.py``, committed to the computer repository)
starts with the codespace, and this module talks to it over that HTTPS port.

Authentication has no shared secret to leak or rotate:

1. The in-codespace agent boots, reads the ``GITHUB_TOKEN`` that GitHub itself injects, and
   registers with the app: ``POST /api/codespaces/register`` carrying that token.
2. The app verifies it against GitHub — the token must belong to the configured owner *and* be
   able to read that very codespace — and only then stores the session token the agent generated.
3. Every later command carries that session token.

Nothing secret lives in the repository, nothing is written to a public port that is not
authenticated, and a token that does not belong to the owner's own codespace cannot register.

Power management (requirement 4)
--------------------------------
* ``ensure_running`` starts the codespace only when work actually needs it, and reuses the one
  codespace that already exists for this repo (never a second one).
* ``note_activity`` is called for every tool that touches the computer, so the idle clock is
  driven by real work rather than by a guess.
* ``maybe_stop`` stops it only when the agent is idle, no run is executing and no tool call is in
  flight; a stop is never issued while a command is running.
* Stops keep the container, so files, installed packages and task progress survive; ``start``
  brings the same codespace back and the agent re-registers automatically.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx

GITHUB_API = "https://api.github.com"


class CodespaceError(RuntimeError):
    """A codespace operation failed; ``code`` is stable for the UI and for tests."""

    def __init__(self, code: str, message: str, *, status: int = 0):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _int_env(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(float(raw)) if raw else default
    except ValueError:
        return default


class CodespaceAPI:
    """Thin, honest wrapper over the documented Codespaces endpoints."""

    def __init__(self, http: httpx.AsyncClient, token: str, *, repo: str, owner: str = "",
                 machine: str = "", idle_timeout_minutes: int = 30, retention_minutes: int = 4320,
                 name_prefix: str = "blackthorn"):
        self.http = http
        self.token = token
        self.repo = repo
        self.owner = owner or (repo.split("/")[0] if "/" in repo else "")
        self.machine = machine
        self.idle_timeout_minutes = idle_timeout_minutes
        self.retention_minutes = retention_minutes
        self.name_prefix = name_prefix

    # -- plumbing ------------------------------------------------------------
    @property
    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "Blackthorn-Codespaces"}

    async def _call(self, method: str, path: str, *, json_body: Any = None, timeout: float = 60.0,
                    allow: tuple = (200, 201, 202, 204)) -> Any:
        url = path if path.startswith("http") else f"{GITHUB_API}{path}"
        try:
            resp = await self.http.request(method, url, headers=self.headers, json=json_body, timeout=timeout)
        except httpx.HTTPError as exc:
            raise CodespaceError("github_unreachable", f"GitHub API request failed: {type(exc).__name__}") from exc
        if resp.status_code not in allow:
            detail = ""
            try:
                detail = str((resp.json() or {}).get("message") or "")[:200]
            except ValueError:
                detail = str(resp.text)[:200]
            raise CodespaceError("github_error", f"GitHub API {method} {path.split('?')[0]} returned "
                                f"HTTP {resp.status_code}: {detail}", status=resp.status_code)
        if resp.status_code == 204 or not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError:
            return {}

    # -- identity ------------------------------------------------------------
    async def whoami(self) -> Dict[str, Any]:
        return await self._call("GET", "/user")

    async def repo_machines(self) -> Dict[str, Any]:
        return await self._call("GET", f"/repos/{self.repo}/codespaces/machines")

    # -- listing / lookup ----------------------------------------------------
    async def list_all(self) -> List[Dict[str, Any]]:
        """Every codespace this token can see, newest first.

        Used to guarantee there is never more than one instance for this computer: the manager
        always works from this list instead of trusting a cached name.
        """
        out: List[Dict[str, Any]] = []
        page = 1
        while page <= 5:
            data = await self._call("GET", f"/user/codespaces?per_page=100&page={page}")
            batch = data.get("codespaces") or []
            out.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return out

    async def find(self) -> Optional[Dict[str, Any]]:
        """The single codespace that serves this computer (or None)."""
        mine = []
        for cs in await self.list_all():
            repo = ((cs.get("repository") or {}).get("full_name") or "")
            name = str(cs.get("name") or "")
            if repo and repo.lower() != self.repo.lower():
                continue
            if not repo and not name.startswith(self.name_prefix):
                continue
            mine.append(cs)
        if not mine:
            return None
        # Prefer a running one, then the most recently used.
        mine.sort(key=lambda c: (str(c.get("state")) == "Available",
                                 str(c.get("last_used_at") or "")), reverse=True)
        return mine[0]

    async def duplicates(self) -> List[Dict[str, Any]]:
        """Any extra codespaces for this repo beyond the one in use (they are deleted)."""
        mine = []
        for cs in await self.list_all():
            repo = ((cs.get("repository") or {}).get("full_name") or "")
            name = str(cs.get("name") or "")
            if (repo and repo.lower() == self.repo.lower()) or (not repo and name.startswith(self.name_prefix)):
                mine.append(cs)
        if len(mine) <= 1:
            return []
        mine.sort(key=lambda c: str(c.get("last_used_at") or ""), reverse=True)
        return mine[1:]

    async def get(self, name: str) -> Dict[str, Any]:
        return await self._call("GET", f"/user/codespaces/{name}")

    async def create(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {"idle_timeout_minutes": self.idle_timeout_minutes,
                                "retention_period_minutes": self.retention_minutes}
        if self.machine:
            body["machine"] = self.machine
        return await self._call("POST", f"/repos/{self.repo}/codespaces", json_body=body, timeout=120.0)

    async def start(self, name: str) -> Dict[str, Any]:
        return await self._call("POST", f"/user/codespaces/{name}/start", timeout=120.0,
                                allow=(200, 201, 202, 204, 409))

    async def stop(self, name: str) -> Dict[str, Any]:
        return await self._call("POST", f"/user/codespaces/{name}/stop", timeout=120.0,
                                allow=(200, 201, 202, 204, 409))

    async def delete(self, name: str) -> None:
        await self._call("DELETE", f"/user/codespaces/{name}", timeout=120.0)

    async def ports(self, name: str) -> List[Dict[str, Any]]:
        data = await self._call("GET", f"/user/codespaces/{name}/ports")
        return data.get("ports") or []

    async def publish_port(self, name: str, port: int, visibility: str = "public") -> Dict[str, Any]:
        """Make the agent's port reachable from the internet.

        Public visibility is required because GitHub offers no token-authenticated way to reach a
        private forwarded port from a server. The port itself only answers requests that carry the
        per-session token the agent generated at boot, so exposure is not the same as access.
        """
        return await self._call("PUT", f"/user/codespaces/{name}/ports/{port}",
                                json_body={"visibility": visibility}, allow=(200, 201, 202, 204))


class UsageLedger:
    """Real Codespaces consumption, recorded from observed running intervals.

    GitHub's billing API answers for personal accounts only with a token that carries the "Plan"
    permission; without it the documented endpoints return 403 and there is no other public
    source for the monthly figure. Rather than invent a number, the app measures what it can see
    for certain — when the codespace was actually running — and reports which source it used, so
    the panel never shows a fabricated figure.
    """

    KEY = "codespaces_usage_intervals"

    def __init__(self, store: Any):
        self.store = store

    async def _load(self) -> List[Dict[str, Any]]:
        try:
            raw = await self.store.get_setting(self.KEY)
            return json.loads(raw) if raw else []
        except Exception:  # noqa: BLE001 - usage accounting must never break a run
            return []

    async def _save(self, intervals: List[Dict[str, Any]]) -> None:
        try:
            await self.store.set_setting(self.KEY, json.dumps(intervals[-500:]))
        except Exception:  # noqa: BLE001
            pass

    async def open_interval(self, name: str) -> None:
        intervals = await self._load()
        if intervals and intervals[-1].get("end") is None:
            return
        intervals.append({"name": name, "start": time.time(), "end": None})
        await self._save(intervals)

    async def close_interval(self) -> None:
        intervals = await self._load()
        if intervals and intervals[-1].get("end") is None:
            intervals[-1]["end"] = time.time()
            await self._save(intervals)

    async def seconds_this_month(self, now: Optional[float] = None) -> float:
        now = now or time.time()
        month = time.gmtime(now).tm_mon
        year = time.gmtime(now).tm_year
        total = 0.0
        for item in await self._load():
            start = float(item.get("start") or 0)
            if not start:
                continue
            started = time.gmtime(start)
            if (started.tm_mon, started.tm_year) != (month, year):
                continue
            end = float(item["end"]) if item.get("end") else now
            total += max(0.0, end - start)
        return total
