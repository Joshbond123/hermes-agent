"""Sandbox lifecycle management: prepare, verify and complete a handover before the TTL expires.

PandaStack gives each sandbox a provider TTL (this account supports one hour and
rejects ``persistent=true``). While a task is running the agent's heartbeat touches
the VM, but a long model prefill, a paused run or a provider-side reap can still
take the sandbox away. The lifecycle manager makes replacement *proactive*:

1. read the sandbox record (``created_at``) and its ``/lifecycle`` TTL;
2. at ``window_s`` (default 300 s ≈ five minutes) before the deadline, publish a
   durable checkpoint of ``/workspace`` to the private GitHub drive;
3. create an overlapping replacement sandbox (the provider allows several at once);
4. restore the verified archive into the replacement and probe it with real work;
5. promote the replacement (persist the new sandbox ID, point the client at it),
   then delete the old sandbox so only one billable VM remains;
6. record the event so the next tick — or a restarted worker — does not repeat it.

Every step is idempotent and bounded. A failure is recorded and retried on a later
tick with backoff; it never raises into the agent loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

log = logging.getLogger("blackthorn.lifecycle")

STATE_KEY = "pandastack_lifecycle_state"      # per-sandbox handover bookkeeping
LOG_KEY = "pandastack_lifecycle_log"          # rolling history, no secrets
FIRST_SEEN_KEY = "pandastack_sandbox_first_seen"
DEFAULT_WINDOW_S = 300.0
MIN_SECONDS_BETWEEN_TICKS = 15.0


def _iso_to_epoch(value: str) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        from datetime import datetime
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo else parsed.timestamp()
    except ValueError:
        return None


class SandboxLifecycle:
    """Driver for one deployment's production computer."""

    def __init__(self, store: Any, computer: Any, *, window_s: float = DEFAULT_WINDOW_S,
                 clock=time.time, sleeper=asyncio.sleep):
        self.store = store
        self.computer = computer
        self.window_s = float(window_s)
        self.clock = clock
        self.sleeper = sleeper
        self._lock = asyncio.Lock()
        self._last_tick = 0.0
        self._failures = 0
        self._backoff_until = 0.0

    # ------------------------------------------------------------------ provider access
    def _control(self):
        from .panda_control import PandaControl
        key = getattr(self.computer, "key", None)
        if callable(key):
            key = key()
        return PandaControl(self.store, self.computer.http, key)

    async def _sid(self) -> str:
        return str(await self.store.get_setting("pandastack_computer_sandbox_id") or "")

    async def _state(self) -> Dict[str, Any]:
        try:
            raw = await self.store.get_setting(STATE_KEY)
            return json.loads(raw) if raw else {}
        except Exception:
            return {}

    async def _save_state(self, value: Dict[str, Any]) -> None:
        try:
            await self.store.set_setting(STATE_KEY, json.dumps(value))
        except Exception as exc:  # bookkeeping must never fail a task
            log.warning("lifecycle state save failed: %s", type(exc).__name__)

    async def _append_log(self, event: Dict[str, Any]) -> None:
        try:
            history = []
            raw = await self.store.get_setting(LOG_KEY)
            if raw:
                try:
                    history = json.loads(raw)
                except ValueError:
                    history = []
            history = (history + [event])[-100:]
            await self.store.set_setting(LOG_KEY, json.dumps(history))
        except Exception:
            pass

    # ------------------------------------------------------------------ observation
    async def snapshot(self) -> Dict[str, Any]:
        """Current sandbox timing, as reported by the provider."""
        sid = await self._sid()
        if not sid:
            return {"sandbox_id": "", "status": "missing", "online": False}
        control = self._control()
        try:
            record = await control.describe(sid)
        except Exception as exc:
            return {"sandbox_id": sid, "status": "unreachable", "online": False,
                    "error": type(exc).__name__}
        life = {}
        if record:
            try:
                life = await control.lifetime(sid)
            except Exception:
                life = {}
        created = _iso_to_epoch(record.get("created_at") or record.get("created") or "")
        if created is None:
            # Provider did not report a creation time: fall back to when we first saw it.
            raw = await self.store.get_setting(FIRST_SEEN_KEY)
            try:
                seen = json.loads(raw) if raw else {}
            except ValueError:
                seen = {}
            if seen.get("sandbox_id") != sid:
                seen = {"sandbox_id": sid, "at": self.clock()}
                await self.store.set_setting(FIRST_SEEN_KEY, json.dumps(seen))
            created = float(seen.get("at") or self.clock())
        ttl = life.get("ttl_seconds")
        try:
            ttl_s = float(ttl) if ttl is not None else 3600.0
        except (TypeError, ValueError):
            ttl_s = 3600.0
        if life.get("persistent") is True:
            ttl_s = 0.0
        deadline = created + ttl_s if ttl_s > 0 else created + 3600.0
        remaining = deadline - self.clock()
        if not record:
            # The provider no longer lists this sandbox: it was reaped.
            return {"sandbox_id": sid, "status": "missing", "online": False,
                    "created_at": created, "ttl_seconds": ttl_s, "seconds_left": 0.0,
                    "handover_due": True}
        status = str(record.get("status") or "unknown").lower()
        return {"sandbox_id": sid, "status": status, "online": status in ("running", "hibernated"),
                "created_at": created, "ttl_seconds": ttl_s, "persistent": bool(life.get("persistent")),
                "deadline": deadline, "seconds_left": round(remaining, 1),
                "window_s": self.window_s,
                "handover_due": remaining <= self.window_s}

    # ------------------------------------------------------------------ checkpointing
    async def checkpoint(self, reason: str = "manual") -> Dict[str, Any]:
        """Publish the workspace to the private drive and record what is durable."""
        mirror = getattr(self.computer, "mirror", None)
        published = False
        detail = ""
        if mirror is not None:
            try:
                published = bool(await mirror.sync(force=True))
            except Exception as exc:
                detail = f"{type(exc).__name__}: {str(exc)[:200]}"
                log.warning("checkpoint failed: %s", detail)
        else:
            detail = "no workspace mirror configured"
        event = {"at": self.clock(), "kind": "checkpoint", "reason": reason,
                 "sandbox_id": await self._sid(), "published": published,
                 "detail": detail, "workspace": await self.store.get_setting("pandastack_workspace_drive_digest")}
        await self._append_log(event)
        return event

    # ------------------------------------------------------------------ handover
    async def handover(self, *, reason: str = "ttl", prepare_only: bool = False) -> Dict[str, Any]:
        """Create, verify and promote a replacement sandbox. Idempotent per sandbox ID."""
        async with self._lock:
            sid = await self._sid()
            state = await self._state()
            if not sid:
                return {"ok": False, "phase": "no_sandbox", "reason": "no managed sandbox"}
            if state.get("completed_for") == sid:
                return {"ok": True, "phase": "already_completed", "sandbox_id": sid}
            if reason != "manual":
                # Never replace a sandbox that still has a healthy TTL left; the periodic
                # driver may call this more than once for the same deadline.
                snap = await self.snapshot()
                if snap.get("sandbox_id") == sid and not snap.get("handover_due"):
                    return {"ok": True, "phase": "not_due", "sandbox_id": sid,
                            "seconds_left": snap.get("seconds_left")}
            if state.get("replacement") and state.get("source") == sid and not prepare_only:
                # A previous tick already created the replacement: finish the handover.
                return await self._finish(sid, state, reason)

            control = self._control()
            checkpoint = await self.checkpoint(reason=f"handover:{reason}")
            if not checkpoint.get("published") and checkpoint.get("detail"):
                await self._save_state({"source": sid, "phase": "checkpoint_failed",
                                        "detail": checkpoint.get("detail"), "at": self.clock()})
                return {"ok": False, "phase": "checkpoint_failed", "detail": checkpoint.get("detail")}

            try:
                replacement = await control.create(role="replacement",
                                                   metadata={"handover_from": sid, "reason": reason})
            except Exception as exc:
                self._failures += 1
                self._backoff_until = self.clock() + min(120.0, 10.0 * (2 ** min(self._failures, 3)))
                await self._save_state({"source": sid, "phase": "create_failed",
                                        "detail": f"{type(exc).__name__}: {str(exc)[:200]}",
                                        "at": self.clock()})
                await self._append_log({"at": self.clock(), "kind": "handover_failed", "phase": "create",
                                        "sandbox_id": sid, "detail": type(exc).__name__})
                return {"ok": False, "phase": "create_failed"}
            await self._save_state({"source": sid, "replacement": replacement, "phase": "created",
                                    "at": self.clock()})
            await self._append_log({"at": self.clock(), "kind": "handover", "phase": "created",
                                    "sandbox_id": sid, "replacement": replacement, "reason": reason})
            if prepare_only:
                return {"ok": True, "phase": "prepared", "sandbox_id": sid, "replacement": replacement}
            return await self._finish(sid, await self._state(), reason)

    async def _finish(self, sid: str, state: Dict[str, Any], reason: str) -> Dict[str, Any]:
        replacement = str(state.get("replacement") or "")
        if not replacement:
            return {"ok": False, "phase": "no_replacement"}
        control = self._control()
        # Point the client at the replacement before restoring so mirror transfers go to the new VM.
        previous_base = getattr(self.computer, "base", "")
        previous_sid = getattr(self.computer, "sandbox_id", "")
        try:
            self.computer.sandbox_id = replacement
            self.computer.base = "https://api.pandastack.ai/v1/sandboxes/" + replacement
            mirror = getattr(self.computer, "mirror", None)
            if mirror is not None:
                await mirror.restore(replacement, force=True)
            if not await control.probe(replacement):
                raise RuntimeError("replacement sandbox failed its live exec probe")
        except Exception as exc:
            # Roll back the client so the (still alive) original keeps serving the task.
            self.computer.sandbox_id, self.computer.base = previous_sid, previous_base
            self._failures += 1
            self._backoff_until = self.clock() + min(120.0, 10.0 * (2 ** min(self._failures, 3)))
            await self._save_state({"source": sid, "replacement": replacement, "phase": "verify_failed",
                                    "detail": f"{type(exc).__name__}: {str(exc)[:200]}", "at": self.clock()})
            await self._append_log({"at": self.clock(), "kind": "handover_failed", "phase": "verify",
                                    "sandbox_id": sid, "replacement": replacement,
                                    "detail": type(exc).__name__})
            return {"ok": False, "phase": "verify_failed", "detail": str(exc)[:200]}

        await self.store.set_setting("pandastack_computer_sandbox_id", replacement)
        try:
            await self.store.set_setting("pandastack_workspace_restored_id", replacement)
        except Exception:
            pass
        deleted = False
        if sid and sid != replacement:
            try:
                deleted = await control.delete(sid)
            except Exception as exc:
                log.warning("old sandbox delete failed: %s", type(exc).__name__)
        self._failures = 0
        self._backoff_until = 0.0
        # ``completed_for`` names the sandbox the handover *produced*, so a repeat call for
        # the same target is recognised instead of creating another billable VM.
        await self._save_state({"source": sid, "replacement": replacement, "completed_for": replacement,
                                "phase": "promoted", "at": self.clock()})
        await self._append_log({"at": self.clock(), "kind": "handover", "phase": "promoted",
                                "sandbox_id": sid, "replacement": replacement, "reason": reason,
                                "old_deleted": deleted})
        return {"ok": True, "phase": "promoted", "sandbox_id": replacement, "replaced": sid,
                "old_deleted": deleted}

    async def _recover_after_loss(self) -> Dict[str, Any]:
        """The sandbox is already gone: create a fresh one and restore the checkpoint."""
        control = self._control()
        from .panda_control import SANDBOX
        sid = await control.create(role="primary", metadata={"reason": "post_expiry_recovery"})
        await self.store.set_setting(SANDBOX, sid)
        self.computer.sandbox_id = sid
        self.computer.base = "https://api.pandastack.ai/v1/sandboxes/" + sid
        mirror = getattr(self.computer, "mirror", None)
        if mirror is not None:
            await mirror.restore(sid, force=True)
        if not await control.probe(sid):
            raise RuntimeError("replacement sandbox failed its live exec probe")
        await self._append_log({"at": self.clock(), "kind": "recovered", "sandbox_id": sid,
                                "reason": "expired"})
        return {"ok": True, "phase": "recovered", "sandbox_id": sid}

    # ------------------------------------------------------------------ periodic driver
    def _runs_busy(self) -> bool:
        """True while an agent turn is actually executing on this computer.

        Rotating (or hibernating) a sandbox mid-turn is the commonest cause of an
        interrupted task: the tool call in flight dies with the VM. The lifecycle
        waits for the turn to finish instead, and the provider TTL is the backstop.
        """
        probe = getattr(self, "runs", None) or getattr(self.computer, "runs", None)
        if probe is None:
            return False
        try:
            if hasattr(probe, "active_count"):
                return int(probe.active_count()) > 0
        except Exception:  # noqa: BLE001
            return False
        return False

    async def tick(self, *, activity: bool = False) -> Dict[str, Any]:
        """One lifecycle step. Cheap when nothing is due; never raises."""
        now = self.clock()
        if now < self._backoff_until:
            return {"action": "backoff", "seconds": round(self._backoff_until - now, 1)}
        if now - self._last_tick < MIN_SECONDS_BETWEEN_TICKS:
            return {"action": "skipped"}
        self._last_tick = now
        try:
            snap = await self.snapshot()
        except Exception as exc:
            return {"action": "error", "detail": type(exc).__name__}
        if not snap.get("sandbox_id") or snap.get("status") in ("missing", "unreachable"):
            try:
                return {"action": "recovered", **(await self._recover_after_loss())}
            except Exception as exc:
                self._failures += 1
                self._backoff_until = now + min(120.0, 10.0 * (2 ** min(self._failures, 3)))
                return {"action": "recover_failed", "detail": f"{type(exc).__name__}: {str(exc)[:160]}"}
        if not snap.get("handover_due"):
            return {"action": "none", "seconds_left": snap.get("seconds_left")}
        if self._runs_busy():
            # A turn is running: keep it alive. Never trade a working task for a fresh VM.
            await self._append_log({"at": self.clock(), "kind": "handover_deferred",
                                    "reason": "active_run", "sandbox_id": snap.get("sandbox_id"),
                                    "seconds_left": snap.get("seconds_left")})
            return {"action": "deferred", "reason": "active_run", "seconds_left": snap.get("seconds_left")}
        result = await self.handover(reason="ttl" if activity else "ttl_idle")
        return {"action": "handover", **result}


async def lifecycle_daemon(store: Any, computer: Any, *, interval: float = 60.0,
                           window_s: float = DEFAULT_WINDOW_S, activity=None, max_ticks: Optional[int] = None) -> None:
    """Background loop started by the app; every tick is bounded and logged."""
    manager = SandboxLifecycle(store, computer, window_s=window_s)
    ticks = 0
    while True:
        busy = False
        if activity is not None:
            try:
                busy = bool(await activity())
            except Exception:
                busy = False
        try:
            outcome = await manager.tick(activity=busy)
            if outcome.get("action") not in ("none", "skipped"):
                log.warning("sandbox lifecycle: %s", json.dumps(outcome)[:400])
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a lifecycle failure must never kill the app
            log.error("sandbox lifecycle tick failed: %s", type(exc).__name__)
        ticks += 1
        if max_ticks is not None and ticks >= max_ticks:
            return
        await asyncio.sleep(interval)
