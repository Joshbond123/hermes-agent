"""The Kaggle GPU: a real on/off GPU that serves Qwen3.8-27B-Uncensored to Blackthorn.

Why it looks like this
----------------------
Kaggle gives a free GPU inside a *notebook session* with a weekly quota. It publishes no API for
power, and cannot cancel a session once it starts, so the only honest model is:

* **ON** — push a private kernel with ``enableGpu`` set. Kaggle starts a GPU session for us.
* **OFF** — the kernel itself must exit. The backend therefore keeps an "order" the kernel polls;
  the kernel shuts its server down and ends, which is what actually stops the GPU meter.
* **Model files** — a Kaggle *dataset* (persistent storage), mounted read-only at
  ``/kaggle/input/...``. It is built once by an installer kernel and reused for ever, so a GPU
  that comes back after days away loads weights from Kaggle's disk, not from the internet.
* **Quota** — Kaggle exposes no usage API at all. The panel therefore shows hours *measured* from
  the intervals this app actually observed, against a weekly allowance that is labelled as
  Kaggle's published minimum (30 h, resetting Saturday 00:00 UTC) and can be corrected in the UI.
  No figure in the panel is invented.

Idle shutdown is enforced here, not in the browser: the browser may be closed, so a task's
activity is recorded server-side, and a stop is never ordered while a chat run is executing or
while the kernel reports an inference request in flight.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .kaggle_kernels import install_source, serve_source

log = logging.getLogger("blackthorn.kaggle_gpu")

KAGGLE_API = "https://www.kaggle.com/api/v1"
STATE_KEY = "kaggle_gpu_state"
LEDGER_KEY = "kaggle_gpu_intervals"
SETTINGS_KEY = "kaggle_gpu_settings"

OFF, STARTING, LOADING, READY, STOPPING, ERROR = "OFF", "STARTING", "LOADING", "READY", "STOPPING", "ERROR"
TIMER_CHOICES = (5, 10, 15, 30, 45, 60)
DEFAULT_TIMER_MINUTES = 15
# Kaggle publishes no quota number; ~30 h/week is its stated minimum and it floats upward with
# demand. It is a *label* on the panel, never a measurement.
DEFAULT_WEEKLY_HOURS = 30.0
HEARTBEAT_GRACE_SECONDS = 180      # a READY gpu with no heartbeat for this long is presumed gone
START_TIMEOUT_SECONDS = 2400       # kernel boot + weight load from disk, measured by the registry


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def kaggle_week(now: Optional[float] = None) -> Tuple[float, float]:
    """The quota week containing ``now``: Saturday 00:00 UTC to the next Saturday 00:00 UTC."""
    t = time.gmtime(now if now is not None else time.time())
    # 1970-01-01 was a Thursday, so weekday(): Mon=0 .. Sun=6. Saturday is 5.
    days_since_saturday = (t.tm_wday - 5) % 7
    start_of_day = time.struct_time((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, t.tm_wday, t.tm_yday, 0))
    start = time.mktime(start_of_day) - time.timezone - days_since_saturday * 86400
    return start, start + 7 * 86400


class GPUError(RuntimeError):
    def __init__(self, code: str, message: str, *, status: int = 0):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class KaggleAPI:
    """The three Kaggle calls this feature needs, with the auth Kaggle actually accepts.

    ``KGAT_`` tokens are answered with 401 under HTTP Basic; a ``Bearer`` header works, so that is
    what is sent. Every failure is raised as :class:`GPUError` with Kaggle's own message.
    """

    def __init__(self, http: httpx.AsyncClient, username: str, key: str):
        self.http = http
        self.username = username
        # Kaggle's API is case-sensitive about identity: ``userName=Joshbond123`` answers
        # "Permission 'kernels.get' was denied" while the lower-case handle works. Every owner-qualified
        # call therefore uses the handle exactly as Kaggle stores it.
        self.owner = (username or "").lower()
        self.key = key

    @property
    def configured(self) -> bool:
        return bool(self.username and self.key)

    async def _request(self, method: str, path: str, *, body: Any = None) -> Any:
        if not self.configured:
            raise GPUError("kaggle_unconfigured",
                           "No Kaggle credentials are configured on this server (KAGGLE_USERNAME/KAGGLE_KEY).")
        resp = await self.http.request(method, f"{KAGGLE_API}{path}", json=body, timeout=60.0,
                                       headers={"Authorization": f"Bearer {self.key}",
                                                "Content-Type": "application/json",
                                                "User-Agent": "Blackthorn-KaggleGPU"})
        if resp.status_code == 429:
            raise GPUError("kaggle_rate_limited", "Kaggle is rate limiting requests; try again shortly.", status=429)
        if resp.status_code >= 400:
            detail = ""
            try:
                detail = str((resp.json() or {}).get("message") or "")
            except Exception:  # noqa: BLE001
                detail = (resp.text or "")[:200]
            raise GPUError("kaggle_error", f"Kaggle returned HTTP {resp.status_code}: {detail}",
                           status=resp.status_code)
        if not resp.content:
            return {}
        try:
            data = resp.json()
        except ValueError:
            return {}
        # Kaggle reports failures inside a 200 body (``{"error": "...", "hasError": true}``). Treating
        # that as success is how a push "succeeds" while nothing is ever created, so it is an error.
        if isinstance(data, dict) and (data.get("hasError") or data.get("error")):
            message = str(data.get("error") or "unknown error")
            code = "kaggle_rejected"
            if "slug" in message.lower():
                code = "kaggle_bad_slug"
            raise GPUError(code, f"Kaggle rejected the request: {message}", status=400)
        return data

    async def push_kernel(self, *, slug: str, title: str, source: str, dataset_sources: List[str],
                          enable_gpu: bool, kernel_sources: Optional[List[str]] = None) -> Dict[str, Any]:
        owner = self.owner or (self.username or "").lower()
        body = {"slug": f"{owner}/{slug}" if owner and not slug.startswith(f"{owner}/") else slug,
                "newTitle": title, "text": source, "language": "python",
                "kernelType": "script", "isPrivate": True, "enableGpu": bool(enable_gpu),
                "enableInternet": True, "enableTpu": False,
                "datasetDataSources": dataset_sources, "competitionDataSources": [],
                "kernelDataSources": list(kernel_sources or []), "categoryIds": []}
        return await self._request("POST", "/kernels/push", body=body)

    async def kernel_status(self, slug: str) -> Dict[str, Any]:
        """RUNNING / QUEUED / COMPLETE / ERROR, as Kaggle reports it."""
        return await self._request("GET", f"/kernels/status?userName={self.owner}&kernelSlug={slug}")

    async def datasets_for(self, owner: str, slug: str) -> Dict[str, Any]:
        return await self._request("GET", f"/datasets/list?user={owner or self.owner}&search={slug}&pageSize=10")


class KaggleGPU:
    """Power state, endpoint registry, idle policy and measured usage for the Kaggle GPU."""

    def __init__(self, store: Any, http: httpx.AsyncClient, *, runs: Any = None,
                 username: str = "", key: str = "", dataset: str = "", serve_slug: str = "blackthorn-qwen-serve",
                 install_slug: str = "blackthorn-qwen-install", public_url: str = "",
                 kernel_token: str = "", hf_token: str = "",
                 model_key: str = "orcarouter/Qwen3.8-27B-Uncensored"):
        self.store = store
        self.http = http
        self.runs = runs
        self.api = KaggleAPI(http, username, key)
        self.username = username
        self.dataset = dataset                     # e.g. "joshbond123/blackthorn-qwen-27b-q4km"
        self.serve_slug = serve_slug
        self.install_slug = install_slug
        self.public_url = public_url.rstrip("/")   # how the kernel reaches this backend
        self.kernel_token = kernel_token
        self.hf_token = hf_token
        self._lock = asyncio.Lock()
        self.model_key = model_key
        # Read synchronously by the router on every request: {model key: live base url}.
        self.live_endpoints: Dict[str, str] = {}

    # ------------------------------------------------------------------ persisted state
    async def _state(self) -> Dict[str, Any]:
        try:
            raw = await self.store.get_setting(STATE_KEY)
            return json.loads(raw) if raw else {}
        except Exception:  # noqa: BLE001 - state must never break chat
            return {}

    def _sync_live(self, state: Dict[str, Any]) -> None:
        url = str(state.get("base_url") or "")
        if state.get("state") == READY and url and self._fresh(state):
            self.live_endpoints[self.model_key] = url
        else:
            self.live_endpoints.pop(self.model_key, None)

    async def rehydrate(self) -> None:
        """After a restart, remember a GPU that is still serving so inference keeps working."""
        state = await self._state()
        self._sync_live(state)

    async def _save(self, state: Dict[str, Any]) -> None:
        state["updated_at"] = time.time()
        self._sync_live(state)
        try:
            await self.store.set_setting(STATE_KEY, json.dumps(state))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not persist gpu state: %s", exc)

    async def settings(self) -> Dict[str, Any]:
        try:
            raw = await self.store.get_setting(SETTINGS_KEY)
            data = json.loads(raw) if raw else {}
        except Exception:  # noqa: BLE001
            data = {}
        timer = data.get("idle_minutes", DEFAULT_TIMER_MINUTES)
        try:
            timer = int(timer)
        except (TypeError, ValueError):
            timer = DEFAULT_TIMER_MINUTES
        weekly = data.get("weekly_hours", DEFAULT_WEEKLY_HOURS)
        try:
            weekly = float(weekly)
        except (TypeError, ValueError):
            weekly = DEFAULT_WEEKLY_HOURS
        return {"idle_minutes": max(1, min(600, timer)), "weekly_hours": max(1.0, min(200.0, weekly)),
                "idle_choices": list(TIMER_CHOICES)}

    async def update_settings(self, *, idle_minutes: Optional[int] = None,
                              weekly_hours: Optional[float] = None) -> Dict[str, Any]:
        data = await self.settings()
        if idle_minutes is not None:
            if int(idle_minutes) not in TIMER_CHOICES and not (1 <= int(idle_minutes) <= 600):
                raise GPUError("bad_timer", f"The idle timer must be one of {list(TIMER_CHOICES)} or 1-600 minutes.")
            data["idle_minutes"] = int(idle_minutes)
        if weekly_hours is not None:
            if not (1 <= float(weekly_hours) <= 200):
                raise GPUError("bad_allowance", "Weekly GPU hours must be between 1 and 200.")
            data["weekly_hours"] = float(weekly_hours)
        try:
            await self.store.set_setting(SETTINGS_KEY, json.dumps({"idle_minutes": data["idle_minutes"],
                                                                  "weekly_hours": data["weekly_hours"]}))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not persist gpu settings: %s", exc)
        return data

    # ------------------------------------------------------------------ activity + ledger
    async def note_activity(self, reason: str = "") -> None:
        """Real work touches this; heartbeats deliberately do not."""
        state = await self._state()
        if not state:
            return
        state["last_activity"] = time.time()
        state["last_activity_reason"] = reason
        await self._save(state)

    async def _intervals(self) -> List[Dict[str, Any]]:
        try:
            raw = await self.store.get_setting(LEDGER_KEY)
            return json.loads(raw) if raw else []
        except Exception:  # noqa: BLE001
            return []

    async def _save_intervals(self, intervals: List[Dict[str, Any]]) -> None:
        try:
            await self.store.set_setting(LEDGER_KEY, json.dumps(intervals[-500:]))
        except Exception:  # noqa: BLE001
            pass

    async def _open_interval(self) -> None:
        intervals = await self._intervals()
        if intervals and intervals[-1].get("end") is None:
            return
        intervals.append({"start": time.time(), "end": None})
        await self._save_intervals(intervals)

    async def _close_interval(self) -> None:
        intervals = await self._intervals()
        if intervals and intervals[-1].get("end") is None:
            intervals[-1]["end"] = time.time()
            await self._save_intervals(intervals)

    # ------------------------------------------------------------------ power
    async def start(self) -> Dict[str, Any]:
        """Push the serve kernel with the GPU enabled and return immediately.

        The weights are already in a Kaggle dataset, so this path never downloads them; the kernel
        only builds/loads llama.cpp and announces its tunnel.
        """
        async with self._lock:
            if not self.api.configured:
                raise GPUError("kaggle_unconfigured",
                               "Kaggle is not configured: set KAGGLE_USERNAME and KAGGLE_KEY on the server.")
            if not self._installed(await self._state()):
                raise GPUError("model_not_installed",
                               "Qwen is not installed on Kaggle yet. Run the installer once (POST /api/gpu/install) — "
                               "it downloads the weights one time into persistent Kaggle storage.")
            state = await self._state()
            if state.get("state") in (STARTING, LOADING, READY) and self._fresh(state):
                return await self.status()
            # The weights live in the installer kernel's output, which Kaggle keeps permanently and
            # mounts read-only. Mounting it is what makes a switch-on fast and download-free.
            cache = self.dataset or f"{self.api.owner}/{self.install_slug}"
            source = serve_source({"backend": self.public_url, "token": self.kernel_token,
                                   "model_id": "orcarouter/Qwen3.8-27B-Uncensored",
                                   "ctx": int(_env("QWEN_CTX", "16384"))},
                                  sources=[cache] if "/" in cache and cache.startswith(self.username) else [])
            dataset_sources = [cache] if self.dataset and "/" in cache and not cache.startswith(self.username) else []
            kernel_sources = [cache] if not dataset_sources else []
            try:
                await self.api.push_kernel(slug=self.serve_slug, title="Blackthorn Qwen serve",
                                           source=source, dataset_sources=dataset_sources,
                                           kernel_sources=kernel_sources, enable_gpu=True)
            except GPUError as exc:
                if exc.status in (403, 401):
                    raise GPUError("kaggle_forbidden",
                                   "Kaggle refused to start the kernel — the account may have no GPU quota "
                                   "left this week, or the API key lacks permission.", status=exc.status) from exc
                raise
            now = time.time()
            await self._save({"state": STARTING, "started_at": now, "last_activity": now,
                              "last_activity_reason": "gpu_on", "base_url": "", "session": "",
                              "stop_requested": False, "error": "", "phase": "pushing kernel to Kaggle"})
            await self._open_interval()
            return await self.status()

    async def stop(self, *, reason: str = "manual", force: bool = False) -> Dict[str, Any]:
        """Order the kernel to finish. Kaggle cannot cancel a session, so the kernel stops itself."""
        async with self._lock:
            state = await self._state()
            if state.get("state") in (OFF, "", None) and not state.get("stop_requested"):
                return await self.status()
            if not force:
                busy = self.busy_reason()
                if busy:
                    raise GPUError("gpu_busy", f"The GPU is still working ({busy}); it will stop when that finishes.")
            state["stop_requested"] = True
            state["stop_reason"] = reason
            state["state"] = STOPPING
            await self._save(state)
            return await self.status()

    def busy_reason(self) -> str:
        """True when work is in flight — a stop must never be issued then."""
        active = 0
        try:
            active = int(self.runs.active_count()) if self.runs is not None else 0
        except Exception:  # noqa: BLE001
            active = 0
        return f"{active} task(s) running" if active else ""

    def _installed(self, state: Dict[str, Any]) -> bool:
        """True once the weights are known to be in Kaggle's persistent storage."""
        return bool(self.dataset) or bool(state.get("installed_at"))

    def _fresh(self, state: Dict[str, Any]) -> bool:
        beat = float(state.get("last_heartbeat") or 0)
        return bool(beat) and (time.time() - beat) < HEARTBEAT_GRACE_SECONDS

    # ------------------------------------------------------------------ kernel channel
    async def register(self, *, base_url: str, session: str, version: str = "", loaded_seconds: float = 0.0,
                       weights_source: str = "") -> Dict[str, Any]:
        state = await self._state()
        previous = state.get("base_url") or ""
        state.update({"state": READY, "base_url": base_url.rstrip("/"), "session": session,
                      "kernel_version": version, "last_heartbeat": time.time(),
                      "ready_at": time.time(), "error": "", "stop_requested": False,
                      "loaded_seconds": round(float(loaded_seconds), 1),
                      "weights_source": weights_source, "phase": "serving",
                      "reconnected": bool(previous and previous != base_url)})
        await self._save(state)
        log.info("kaggle gpu registered: %s (%s)", base_url, weights_source)
        return {"ok": True, "state": READY}

    async def heartbeat(self, *, busy: bool = False, base_url: str = "") -> Dict[str, Any]:
        """The kernel asks whether it should stop. Busy inference overrides any pending order."""
        state = await self._state()
        state["last_heartbeat"] = time.time()
        if base_url:
            state["base_url"] = base_url.rstrip("/")
        if state.get("state") in (STARTING, LOADING):
            state["state"] = READY
        await self._save(state)
        stop = bool(state.get("stop_requested")) and not busy and not self.busy_reason()
        if busy:
            state["stop_deferred"] = time.time()
            await self._save(state)
        return {"stop": stop, "state": state.get("state"), "reason": state.get("stop_reason", "")}

    async def check_idle(self) -> Dict[str, Any]:
        """Called periodically by the app: order a stop when the GPU has been idle too long."""
        state = await self._state()
        if state.get("state") not in (READY, LOADING, STARTING):
            return await self.status()
        settings = await self.settings()
        idle_limit = settings["idle_minutes"] * 60
        last = float(state.get("last_activity") or state.get("ready_at") or time.time())
        idle_for = time.time() - last
        if idle_for < idle_limit:
            return await self.status()
        if self.busy_reason():
            # Work is running: the clock keeps running but the GPU stays up.
            state["last_activity"] = time.time()
            state["idle_extension"] = "task in progress"
            await self._save(state)
            return await self.status()
        try:
            await self.stop(reason=f"idle {int(idle_for // 60)} min")
        except GPUError:
            pass
        return await self.status()

    # ------------------------------------------------------------------ views
    async def endpoint(self) -> str:
        """The live URL for inference, or "" when the GPU is not serving."""
        state = await self._state()
        if state.get("state") != READY:
            return ""
        if not self._fresh(state):
            return ""
        return str(state.get("base_url") or "")

    async def status(self) -> Dict[str, Any]:
        state = await self._state()
        settings = await self.settings()
        phase = state.get("state") or OFF
        if phase in (STARTING, LOADING) and time.time() - float(state.get("started_at") or 0) > START_TIMEOUT_SECONDS:
            phase = ERROR
            state["error"] = ("Kaggle did not bring the GPU up in time. It may be queued behind other "
                              "notebooks, or the weekly GPU quota may be exhausted.")
        if phase in (READY, STARTING, LOADING) and state.get("last_heartbeat") and not self._fresh(state):
            phase = ERROR
            state["error"] = "The GPU session stopped sending heartbeats; it probably ended on Kaggle's side."
        # Whatever the reason the GPU stopped serving, it must not be offered to the router.
        self._sync_live({**state, "state": phase})
        session_seconds = 0.0
        if state.get("started_at") and phase in (READY, STARTING, LOADING, STOPPING):
            session_seconds = max(0.0, time.time() - float(state["started_at"]))
        settings_out = {"idle_minutes": settings["idle_minutes"], "idle_choices": settings["idle_choices"],
                        "weekly_hours": settings["weekly_hours"]}
        return {"provider": "Kaggle GPU",
                "state": phase,
                "running": phase in (READY, LOADING, STARTING),
                "serving": phase == READY and bool(state.get("base_url")) and self._fresh(state),
                "base_url": state.get("base_url", "") if phase == READY else "",
                "phase": state.get("phase", ""),
                "loaded_seconds": state.get("loaded_seconds"),
                "weights_source": state.get("weights_source", ""),
                "reconnected": bool(state.get("reconnected")),
                "session_seconds": round(session_seconds),
                "session_started_at": state.get("started_at"),
                "last_activity": state.get("last_activity"),
                "last_activity_reason": state.get("last_activity_reason", ""),
                "stop_requested": bool(state.get("stop_requested")),
                "error": state.get("error", ""),
                "dataset": self.dataset,
                "kaggle_user": self.username,
                "busy": self.busy_reason(),
                "settings": settings_out,
                "updated_at": int(time.time())}

    async def usage(self) -> Dict[str, Any]:
        """Weekly GPU hours: measured consumption against a labelled allowance."""
        settings = await self.settings()
        start, end = kaggle_week()
        intervals = await self._intervals()
        used = 0.0
        sessions = 0
        for item in intervals:
            s = float(item.get("start") or 0)
            e = float(item.get("end") or time.time())
            if e <= start or s >= end:
                continue
            sessions += 1
            used += max(0.0, min(e, end) - max(s, start))
        current = 0.0
        state = await self._state()
        if state.get("started_at") and state.get("state") in (READY, STARTING, LOADING, STOPPING):
            current = max(0.0, time.time() - float(state["started_at"]))
        allowance = settings["weekly_hours"]
        total_used = used / 3600.0
        return {"provider": "Kaggle GPU",
                "week_start": int(start), "week_end": int(end),
                "week_label": time.strftime("%Y-%m-%d", time.gmtime(start)),
                "hours_total": round(allowance, 2),
                "hours_used": round(total_used, 2),
                "hours_remaining": round(max(0.0, allowance - total_used), 2),
                "percent_used": round(min(100.0, (total_used / allowance) * 100.0), 1) if allowance else 0.0,
                "current_session_seconds": round(current),
                "sessions_this_week": sessions,
                "usage_source": "measured from observed GPU sessions",
                "hours_basis": (f"Kaggle's published minimum is {DEFAULT_WEEKLY_HOURS:g} GPU hours per week "
                                f"(resets Saturday 00:00 UTC). Kaggle exposes no usage API, so this "
                                f"allowance is a labelled setting you can correct to the figure in your "
                                f"account; hours used are measured here."),
                "updated_at": int(time.time())}

    async def install(self) -> Dict[str, Any]:
        """Download the weights once into persistent Kaggle storage.

        This runs on a CPU session, so it costs no GPU quota. Kaggle keeps a completed kernel's
        output permanently (up to 20 GB) and it can be mounted by later kernels, which is exactly the
        "install once, never download again" behaviour asked for.
        """
        if not self.api.configured:
            raise GPUError("kaggle_unconfigured", "Kaggle is not configured on this server.")
        source = install_source({"hf_token": self.hf_token, "kaggle_user": self.username})
        await self.api.push_kernel(slug=self.install_slug, title="Blackthorn Qwen install",
                                   source=source, dataset_sources=[], enable_gpu=False)
        state = await self._state()
        state.update({"install_started_at": time.time(), "install_kernel": f"{self.api.owner}/{self.install_slug}"})
        await self._save(state)
        return {"ok": True, "kernel": f"{self.api.owner}/{self.install_slug}",
                "store": f"kernel output of {self.api.owner}/{self.install_slug}",
                "gpu_quota": "none — CPU session",
                "note": "One download into persistent Kaggle storage. Every later GPU session mounts it "
                        "and downloads nothing."}

    async def mark_installed(self) -> Dict[str, Any]:
        """Record that the installer finished, so the GPU can be switched on."""
        state = await self._state()
        state["installed_at"] = time.time()
        await self._save(state)
        return await self.status()

    async def kernel_state(self) -> Dict[str, Any]:
        """What Kaggle itself says about our kernels (used by the panel and by tests)."""
        out: Dict[str, Any] = {}
        for name, slug in (("serve", self.serve_slug), ("install", self.install_slug)):
            try:
                out[name] = (await self.api.kernel_status(slug)).get("status") or "unknown"
            except GPUError as exc:
                out[name] = f"error: {exc.message[:80]}"
        return out
