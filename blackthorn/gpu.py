"""Kaggle GPU controller.

* A supervisor task refreshes a snapshot in the background, so ``GET /api/kaggle-gpu/status``
  answers from memory in microseconds instead of running a chain of network calls
  (the old endpoint took 36–75 s while booting).
* ``turn_on`` / ``turn_off`` / ``restart`` return immediately with the new state; the slow
  Kaggle calls run in the background and any failure is written to the state, never lost.
* The state shown is *derived from evidence* (see :mod:`blackthorn.gpu_state`).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import httpx

from blackthorn import config, notebook
from blackthorn import http as bt_http
from blackthorn.d1 import D1Error, aget_meta, aset_meta, d1
from blackthorn.gpu_state import SESSION_ALIVE, Evidence, derive, parse_detail
from blackthorn.route import Route

log = logging.getLogger("blackthorn.gpu")

AUTO_OFF_CHOICES = (0, 5, 10, 15, 30, 60)
KEY_AUTO_OFF = "blackthorn_auto_off_minutes"
KEY_AUTO_OFF_REASON = "blackthorn_auto_off_last_reason"
KEY_BOOT_STARTED = "blackthorn_gpu_boot_started"
QUOTA_TTL = 120.0
SESSION_TTL = 20.0

_ROW_SQL = (
    "SELECT status, tunnel_url, api_key, model, gpu_info, detail, updated_at "
    "FROM kaggle_gpu_state WHERE id = 'primary' LIMIT 1"
)
_ROW_SQL_LEGACY = (
    "SELECT status, tunnel_url, api_key, model, gpu_info, '' AS detail, updated_at "
    "FROM kaggle_gpu_state WHERE id = 'primary' LIMIT 1"
)
_UPSERT = (
    "INSERT INTO kaggle_gpu_state (id, status, tunnel_url, api_key, model, gpu_info, detail, updated_at) "
    "VALUES ('primary', ?, ?, ?, ?, ?, ?, ?) "
    "ON CONFLICT(id) DO UPDATE SET status = excluded.status, tunnel_url = excluded.tunnel_url, "
    "api_key = excluded.api_key, model = excluded.model, gpu_info = excluded.gpu_info, "
    "detail = excluded.detail, updated_at = excluded.updated_at"
)


def kaggle_rpc(method: str, body: Dict[str, Any], timeout: float = 20.0) -> Dict[str, Any]:
    """Blocking call to Kaggle's kernels API (run it in a worker thread)."""
    resp = httpx.post(
        f"https://api.kaggle.com/v1/kernels.KernelsApiService/{method}",
        json=body,
        headers={"Authorization": f"Bearer {config.kaggle_token()}", "User-Agent": "kaggle-api/v1.7.0"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json() if resp.content else {}


async def default_probe(url: str) -> Optional[Dict[str, Any]]:
    try:
        resp = await bt_http.client().get(url.rstrip("/") + "/health", timeout=4.0)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if isinstance(data, dict) else None
    except (httpx.HTTPError, ValueError):
        return None


class GpuController:
    def __init__(
        self,
        *,
        rpc: Callable[[str, Dict[str, Any]], Dict[str, Any]] = kaggle_rpc,
        probe: Callable[[str], Any] = default_probe,
        build_notebook: Optional[Callable[[str], str]] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._rpc = rpc
        self._probe = probe
        self._build_notebook = build_notebook or notebook.notebook_for_push
        self._now = clock
        self._snapshot: Dict[str, Any] = {"state": "unknown", "message": "Checking the GPU…", "updated_at": 0.0}
        self._row: Dict[str, Any] = {}
        self._task: Optional[asyncio.Task] = None
        self._transition: Optional[asyncio.Task] = None
        self._refresh_lock = asyncio.Lock()
        self._schema_ready = False
        self._probe_failures = 0
        self._status_seen = ("", 0.0)
        self._session: Optional[str] = None
        self._session_at = 0.0
        self._quota: Optional[Dict[str, Any]] = None
        self._quota_at = 0.0
        self._boot_started = 0.0
        self._last_activity = clock()
        self._active = 0
        self._last_inference: Optional[Dict[str, Any]] = None
        self._verifying = False
        self._auto_off_cache = (0.0, config.DEFAULT_AUTO_OFF_MINUTES)
        self._last_status_request = 0.0
        self._was_ready = False

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._supervise(), name="gpu-supervisor")

    async def stop(self) -> None:
        for t in (self._task, self._transition):
            if t is not None:
                t.cancel()
        for t in (self._task, self._transition):
            if t is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t

    async def _supervise(self) -> None:
        while True:
            try:
                snap = await self.refresh(force=True)
                await self._maybe_auto_off(snap)
                state = snap.get("state")
                idle_for_ui = self._now() - self._last_status_request > 300
                delay = {"starting": 3.0, "stopping": 3.0, "ready": 10.0}.get(state, 25.0)
                if idle_for_ui and state in ("off", "unknown", "error"):
                    delay = 90.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the supervisor must never die
                log.warning("gpu supervisor iteration failed: %s", exc)
                delay = 15.0
            await asyncio.sleep(delay)

    # ------------------------------------------------------------------ activity / auto-off
    def mark_activity(self) -> None:
        self._last_activity = self._now()

    def task_started(self) -> None:
        self._active += 1
        self.mark_activity()

    def task_finished(self) -> None:
        self._active = max(0, self._active - 1)
        self.mark_activity()

    async def auto_off_minutes(self) -> int:
        stamp, value = self._auto_off_cache
        if self._now() - stamp < 60:
            return value
        try:
            raw = await aget_meta(KEY_AUTO_OFF, "")
            value = int(raw) if raw.strip() else config.DEFAULT_AUTO_OFF_MINUTES
        except (ValueError, D1Error):
            value = config.DEFAULT_AUTO_OFF_MINUTES
        self._auto_off_cache = (self._now(), value)
        return value

    async def set_auto_off_minutes(self, minutes: int) -> int:
        if minutes not in AUTO_OFF_CHOICES:
            raise ValueError(f"minutes must be one of {AUTO_OFF_CHOICES}")
        await aset_meta(KEY_AUTO_OFF, str(minutes))
        self._auto_off_cache = (self._now(), minutes)
        return minutes

    async def _maybe_auto_off(self, snap: Dict[str, Any]) -> None:
        if snap.get("state") != "ready" or self._active:
            return
        minutes = await self.auto_off_minutes()
        if minutes <= 0:
            return
        idle = self._now() - self._last_activity
        if idle >= minutes * 60:
            with contextlib.suppress(Exception):
                await aset_meta(KEY_AUTO_OFF_REASON, f"Stopped after {minutes} min without activity")
            log.info("auto-off: idle %.0fs >= %d min — stopping the GPU", idle, minutes)
            await self.turn_off(reason="auto-off")

    # ------------------------------------------------------------------ storage
    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with contextlib.suppress(Exception):
            await d1.aexec("ALTER TABLE kaggle_gpu_state ADD COLUMN detail TEXT NOT NULL DEFAULT ''")
        self._schema_ready = True

    async def _read_row(self) -> Dict[str, Any]:
        try:
            rows = await d1.aquery(_ROW_SQL)
        except D1Error:
            rows = await d1.aquery(_ROW_SQL_LEGACY)
        return rows[0] if rows else {}

    async def _write_row(self, status: str, *, tunnel_url: str = "", api_key: Optional[str] = None,
                         gpu_info: str = "", detail: Optional[Dict[str, Any]] = None) -> None:
        await self._ensure_schema()
        key = api_key if api_key is not None else str(self._row.get("api_key") or "")
        await d1.aexec(_UPSERT, [status, tunnel_url, key, config.model_alias(), gpu_info,
                                 json.dumps(detail or {}), self._now()])

    # ------------------------------------------------------------------ evidence + snapshot
    async def _session_status(self, force: bool = False) -> Optional[str]:
        if not force and self._now() - self._session_at < SESSION_TTL:
            return self._session
        try:
            data = await asyncio.to_thread(
                self._rpc, "GetKernelSessionStatus",
                {"userName": config.kaggle_username(), "kernelSlug": config.KAGGLE_KERNEL_SLUG})
            status = str(data.get("status") or data.get("sessionStatus") or "").upper() or "NONE"
            self._session, self._session_at = status, self._now()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (403, 404):  # Kaggle answers 403 for a kernel that no longer exists
                self._session, self._session_at = "NONE", self._now()
            else:
                log.debug("kaggle session status unavailable: %s", exc)
                self._session_at = self._now()
        except Exception as exc:  # noqa: BLE001 - unknown is acceptable, we then rely on the probe
            log.debug("kaggle session status unavailable: %s", exc)
            self._session_at = self._now()
        return self._session

    async def _quota_info(self) -> Optional[Dict[str, Any]]:
        if self._quota is not None and self._now() - self._quota_at < QUOTA_TTL:
            return self._quota
        try:
            data = await asyncio.to_thread(self._rpc, "GetAcceleratorQuotaStatistics", {})
            gpu = data.get("gpuQuota") or {}

            def secs(value: Any) -> float:
                text = str(value or "0").rstrip("s")
                try:
                    return float(text)
                except ValueError:
                    return 0.0

            used = secs(gpu.get("timeUsed"))
            total = secs(gpu.get("totalTimeAllowed")) or 108000.0
            if total <= 21600:
                total = 108000.0
            self._quota = {
                "used_hours": round(used / 3600, 2),
                "total_hours": round(total / 3600, 1),
                "remaining_hours": max(0.0, round((total - used) / 3600, 2)),
                "used_pct": min(100.0, round(used / total * 100, 1)),
                "refresh_time": str(data.get("quotaRefreshTime") or ""),
            }
            self._quota_at = self._now()
        except Exception as exc:  # noqa: BLE001
            log.debug("quota unavailable: %s", exc)
            self._quota_at = self._now() - QUOTA_TTL + 20  # retry soon, keep the last value
        return self._quota

    async def refresh(self, force: bool = False) -> Dict[str, Any]:
        if not force and self._now() - self._snapshot.get("updated_at", 0) < 2.0:
            return self._snapshot
        async with self._refresh_lock:
            if not force and self._now() - self._snapshot.get("updated_at", 0) < 2.0:
                return self._snapshot
            row = await self._read_row()
            self._row = row
            raw = str(row.get("status") or "").upper()
            now = self._now()
            if raw != self._status_seen[0]:
                self._status_seen = (raw, now)
            detail = parse_detail(row.get("detail"))
            tunnel = str(row.get("tunnel_url") or "").rstrip("/")
            row_updated = float(row.get("updated_at") or 0.0)
            want_session = raw not in ("", "OFF", "GPU_STOPPED_SAVING_QUOTA") or now - self._session_at > 60
            session = await self._session_status(force=True) if want_session else self._session
            health = await self._probe(tunnel) if tunnel else None
            if tunnel and health is None:
                self._probe_failures += 1
            else:
                self._probe_failures = 0
            if self._boot_started == 0.0 and raw not in ("", "OFF", "GPU_STOPPED_SAVING_QUOTA"):
                with contextlib.suppress(Exception):
                    self._boot_started = float(await aget_meta(KEY_BOOT_STARTED, "0") or 0)
            ev = Evidence(
                row_status=raw, row_updated_at=row_updated, tunnel_url=tunnel,
                gpu_info=str(row.get("gpu_info") or ""), model=str(row.get("model") or config.model_alias()),
                detail=detail, session_status=session, health=health, probe_failures=self._probe_failures,
                boot_started_at=self._boot_started,
                status_since=float(detail.get("stage_started") or 0) or self._status_seen[1],
            )
            snap = derive(ev, now)
            snap["quota"] = await self._quota_info()
            minutes = await self.auto_off_minutes()
            idle = max(0.0, now - self._last_activity)
            snap["auto_off"] = {
                "minutes": minutes, "enabled": minutes > 0, "idle_s": int(idle),
                "remaining_s": max(0, int(minutes * 60 - idle)) if minutes > 0 and snap["state"] == "ready" else None,
                "choices": list(AUTO_OFF_CHOICES),
            }
            snap["last_inference"] = self._last_inference
            snap["busy"] = self._active > 0
            snap["updated_at"] = now
            snap["transitioning"] = bool(self._transition and not self._transition.done())
            snap["api_key_present"] = bool(row.get("api_key"))
            self._snapshot = snap
            ready = snap["state"] == "ready"
            if ready and not self._was_ready:
                self._schedule_verify()
            self._was_ready = ready
            return snap

    def snapshot(self) -> Dict[str, Any]:
        """Latest snapshot — never blocks."""
        self._last_status_request = self._now()
        snap = dict(self._snapshot)
        if snap.get("updated_at"):
            snap["age_s"] = round(self._now() - snap["updated_at"], 1)
        return snap

    # ------------------------------------------------------------------ route for the chat engine
    async def route(self) -> Optional[Route]:
        snap = self._snapshot
        if snap.get("state") != "ready" or self._now() - snap.get("updated_at", 0) > 20:
            snap = await self.refresh(force=True)
        if snap.get("state") != "ready":
            return None
        url = str(self._row.get("tunnel_url") or "").rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        key = str(self._row.get("api_key") or "")
        if not url or not key:
            return None
        return Route(url=url, api_key=key, model=str(self._row.get("model") or config.model_alias()))

    def report_tunnel_failure(self, why: str) -> None:
        """The engine hit a dead tunnel: re-check right away instead of waiting for the next tick."""
        self._probe_failures = max(self._probe_failures, 1)
        self._snapshot["updated_at"] = 0.0

    # ------------------------------------------------------------------ transitions
    def _spawn(self, coro_fn: Callable[[], Any]) -> None:
        if self._transition and not self._transition.done():
            return
        self._transition = asyncio.create_task(coro_fn(), name="gpu-transition")

    async def turn_on(self) -> Dict[str, Any]:
        snap = await self.refresh(force=True)
        if snap["state"] in ("ready", "starting") and not snap.get("stalled"):
            return {**self.snapshot(), "ok": True, "note": "already " + snap["state"]}
        if self._transition and not self._transition.done():
            return {**self.snapshot(), "ok": True, "note": "a start/stop is already in progress"}
        self._boot_started = self._now()
        gateway_key = notebook.new_gateway_key()
        await self._write_row("BOOTING_KAGGLE_GPU", api_key=gateway_key, detail={"stage_started": self._now()},
                              gpu_info="Allocating 2x NVIDIA Tesla T4…")
        with contextlib.suppress(Exception):
            await aset_meta(KEY_BOOT_STARTED, str(self._boot_started))
        self._spawn(lambda: self._boot(gateway_key))
        snap = await self.refresh(force=True)
        return {**self.snapshot(), "ok": True, "note": "starting"}

    async def _boot(self, gateway_key: str) -> None:
        try:
            await self._write_row("PUSHING_NOTEBOOK", api_key=gateway_key, detail={"stage_started": self._now()})
            await asyncio.to_thread(self._push_kernel, gateway_key)
            await self._write_row("BOOTING_KAGGLE_GPU", api_key=gateway_key, detail={"stage_started": self._now()})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface every failure in the state
            log.error("GPU boot failed: %s", exc)
            with contextlib.suppress(Exception):
                await self._write_row("BOOT_FAILED", api_key=gateway_key, gpu_info=f"FAILED: {str(exc)[:200]}",
                                      detail={"error": str(exc)[:300]})
        finally:
            self._snapshot["updated_at"] = 0.0

    def _wait_session_gone(self, seconds: float) -> bool:
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                data = self._rpc("GetKernelSessionStatus",
                                 {"userName": config.kaggle_username(), "kernelSlug": config.KAGGLE_KERNEL_SLUG})
                if str(data.get("status") or "").upper() not in SESSION_ALIVE:
                    return True
            except Exception:  # noqa: BLE001
                return True
            time.sleep(3)
        return False

    def _push_kernel(self, gateway_key: str) -> None:
        """Blocking: stop a stale session, push the notebook (retrying), confirm it was accepted."""
        slug, user = config.KAGGLE_KERNEL_SLUG, config.kaggle_username()
        try:
            data = self._rpc("GetKernelSessionStatus", {"userName": user, "kernelSlug": slug})
            if str(data.get("status") or "").upper() in SESSION_ALIVE:
                log.info("stopping a stale Kaggle session before the push")
                for _ in range(3):
                    with contextlib.suppress(Exception):
                        self._rpc("DeleteKernel", {"userName": user, "kernelSlug": slug})
                    if self._wait_session_gone(40):
                        break
        except Exception as exc:  # noqa: BLE001
            log.debug("pre-push session check: %s", exc)
        payload = {
            "slug": config.kaggle_kernel_ref(), "newTitle": config.KAGGLE_KERNEL_TITLE,
            "text": self._build_notebook(gateway_key), "language": "python", "kernelType": "notebook",
            "isPrivate": True, "enableGpu": True, "enableTpu": False, "enableInternet": True,
            "machineShape": "NvidiaTeslaT4",
        }
        last = ""
        for attempt in range(1, 4):
            try:
                pushed = self._rpc("SaveKernel", payload) or {}
                if pushed.get("versionNumber") or pushed.get("ref"):
                    log.info("Kaggle kernel pushed (version %s)", pushed.get("versionNumber") or "?")
                    return
                status = self._rpc("GetKernelSessionStatus", {"userName": user, "kernelSlug": slug})
                if str(status.get("status") or "").upper() in SESSION_ALIVE:
                    return
                last = "Kaggle accepted the push but returned no version"
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
            if attempt < 3:
                time.sleep(4 * attempt)
        raise RuntimeError(f"Kaggle refused the notebook push after 3 attempts ({last})")

    async def turn_off(self, reason: str = "user") -> Dict[str, Any]:
        if self._transition and not self._transition.done():
            return {**self.snapshot(), "ok": True, "note": "a start/stop is already in progress"}
        await self._write_row("STOPPING_KAGGLE_GPU", detail={"stage_started": self._now(), "reason": reason})
        self._spawn(self._stop_session)
        await self.refresh(force=True)
        return {**self.snapshot(), "ok": True, "note": "stopping"}

    async def _stop_session(self) -> None:
        try:
            user, slug = config.kaggle_username(), config.KAGGLE_KERNEL_SLUG
            await asyncio.to_thread(lambda: self._rpc("DeleteKernel", {"userName": user, "kernelSlug": slug}))
            await asyncio.to_thread(self._wait_session_gone, 45)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("stopping the Kaggle session failed: %s", exc)
        finally:
            with contextlib.suppress(Exception):
                await self._write_row("GPU_STOPPED_SAVING_QUOTA", api_key="", gpu_info="OFF", detail={})
            self._boot_started = 0.0
            self._last_inference = None
            self._session_at = 0.0
            self._snapshot["updated_at"] = 0.0

    async def restart(self) -> Dict[str, Any]:
        """Stop then start — the recovery for a stalled or errored boot."""
        if self._transition and not self._transition.done():
            return {**self.snapshot(), "ok": True, "note": "a start/stop is already in progress"}

        async def _cycle() -> None:
            await self._stop_session()
            self._boot_started = self._now()
            key = notebook.new_gateway_key()
            await self._write_row("BOOTING_KAGGLE_GPU", api_key=key, detail={"stage_started": self._now()})
            with contextlib.suppress(Exception):
                await aset_meta(KEY_BOOT_STARTED, str(self._boot_started))
            await self._boot(key)

        await self._write_row("STOPPING_KAGGLE_GPU", detail={"stage_started": self._now(), "reason": "restart"})
        self._spawn(_cycle)
        await self.refresh(force=True)
        return {**self.snapshot(), "ok": True, "note": "restarting"}

    # ------------------------------------------------------------------ verification + logs
    def _schedule_verify(self) -> None:
        if not self._verifying:
            asyncio.create_task(self.verify_inference(), name="gpu-verify")

    async def verify_inference(self) -> Dict[str, Any]:
        """Run one tiny real completion and record whether the model actually answers."""
        if self._verifying:
            return self._last_inference or {"ok": None, "error": "verification already running"}
        self._verifying = True
        try:
            route = await self.route()
            if route is None:
                self._last_inference = {"ok": False, "error": "the GPU is not ready", "at": self._now()}
                return self._last_inference
            started = time.monotonic()
            try:
                resp = await bt_http.client().post(
                    route.url + "/v1/chat/completions",
                    json={"model": route.model, "stream": False, "max_tokens": 24, "reasoning_effort": "none",
                          "messages": [{"role": "user", "content": "Reply with the single word: OK"}]},
                    headers={"Authorization": f"Bearer {route.api_key}"}, timeout=180.0)
                data = resp.json() if resp.status_code == 200 else {}
                text = str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
                usage = data.get("usage") or {}
                ok = resp.status_code == 200 and bool(text)
                self._last_inference = {
                    "ok": ok, "latency_ms": int((time.monotonic() - started) * 1000),
                    "completion_tokens": usage.get("completion_tokens"), "reply": text[:40], "at": self._now(),
                    "error": None if ok else f"HTTP {resp.status_code}" if resp.status_code != 200 else "empty reply",
                }
            except Exception as exc:  # noqa: BLE001
                self._last_inference = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:160], "at": self._now()}
            self._snapshot["last_inference"] = self._last_inference
            return self._last_inference
        finally:
            self._verifying = False

    async def logs(self, limit: int = 120) -> Dict[str, Any]:
        url = str(self._row.get("tunnel_url") or "").rstrip("/")
        key = str(self._row.get("api_key") or "")
        if not url:
            return {"lines": [], "note": "No GPU session is running."}
        try:
            resp = await bt_http.client().get(url + "/logs", headers={"Authorization": f"Bearer {key}"}, timeout=10.0)
            lines = [str(x) for x in (resp.json().get("lines") or [])]
            return {"lines": lines[-max(1, min(int(limit), 400)):]}
        except Exception as exc:  # noqa: BLE001
            return {"lines": [], "note": f"Logs unavailable ({type(exc).__name__})"}


#: process-wide controller
controller = GpuController()
