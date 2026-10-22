"""Multi-account Kaggle fleet: roles, real quotas, kernel lifecycle, and failover decisions.

Four accounts form two roles:

* **model** — ``black797`` (primary) and ``temilolo`` (backup) host the AI model.
* **computer** — ``alagbo`` (primary) and ``teslaarymo`` (backup) are the agent's computer.

Both accounts of a role are kept as synchronized clones (same notebook text, same kernels,
same persistent datasets). The controller decides which account is *active* from real
``GetAcceleratorQuotaStatistics`` numbers and observed health — never from timers alone —
and switches at the 30-minute threshold with hysteresis so the fleet cannot flap.

All state is data in / data out: pass a ``fetcher`` to make the whole engine testable.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("blackthorn.fleet")

KAGGLE_RPC = "https://api.kaggle.com/v1/kernels.KernelsApiService/{method}"

# Failover policy (seconds of weekly GPU allowance)
FAILOVER_BELOW_S = 30 * 60            # switch away when the active account has <= 30 min left
BACKUP_MIN_S = 5 * 60                 # never switch INTO an account with < 5 min left
RETURN_ABOVE_S = 90 * 60              # come home only when the primary has >= 90 min free
RETURN_STABLE_CHECKS = 3              # ...and has stayed healthy that many checks in a row
MIN_SWITCH_INTERVAL_S = 5 * 60        # soft flap guard (bypassed when the active side is hard-down)
QUOTA_CACHE_S = 25.0

ROLES = ("model", "computer")


class FleetError(RuntimeError):
    pass


@dataclass
class Account:
    user: str
    token: str
    role: str                  # "model" | "computer"
    slot: str                  # "primary" | "backup"
    label: str = ""

    @property
    def slug(self) -> str:
        return f"{self.user}/{self.role}-{'primary' if self.slot == 'primary' else 'backup'}"


@dataclass
class SlotState:
    healthy: Optional[bool] = None       # None = unknown yet
    remaining_s: Optional[float] = None  # None = unknown (never invent numbers)
    used_s: Optional[float] = None
    total_s: Optional[float] = None
    refresh_time: str = ""
    kernel_running: Optional[bool] = None
    last_checked: float = 0.0
    last_error: str = ""

    def view(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"healthy": self.healthy, "kernel_running": self.kernel_running,
                               "last_checked": round(self.last_checked, 1), "last_error": self.last_error[:160]}
        if self.remaining_s is not None:
            out.update(remaining_seconds=int(self.remaining_s), used_seconds=int(self.used_s or 0),
                       total_seconds=int(self.total_s or 0), refresh_time=self.refresh_time)
        return out


@dataclass
class RoleState:
    active: str = "primary"
    reason: str = "initial"
    since: float = field(default_factory=time.time)
    last_switch: float = 0.0
    primary_stable: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)

    def event(self, text: str) -> None:
        self.events.append({"t": time.time(), "text": text[:200]})
        del self.events[:-50]           # ring buffer


class FleetManager:
    """Kaggle RPC + failover engine for one role pair (or the whole four-account fleet)."""

    def __init__(self, accounts: List[Account], *,
                 fetcher: Optional[Callable[[str, Dict[str, Any], str], Dict[str, Any]]] = None,
                 clock: Callable[[], float] = time.time,
                 store: Any = None):
        self.accounts = {a.user: a for a in accounts}
        self.by_slot: Dict[Tuple[str, str], Account] = {(a.role, a.slot): a for a in accounts}
        self._fetch = fetcher or self._rpc
        self._clock = clock
        self._store = store                       # optional: persists active slot + events (state_meta)
        self._lock = threading.Lock()
        self._quota_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self.state: Dict[str, RoleState] = {r: RoleState() for r in ROLES}
        self.slots: Dict[str, SlotState] = {u: SlotState() for u in self.accounts}

    # ------------------------------------------------------------------ transport
    @staticmethod
    def _rpc(method: str, body: Dict[str, Any], token: str) -> Dict[str, Any]:
        req = urllib.request.Request(
            KAGGLE_RPC.format(method=method),
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                     "User-Agent": "kaggle-api/v1.7.0"},
            method="POST")
        with urllib.request.urlopen(req, timeout=25) as resp:
            return json.loads(resp.read().decode("utf-8")) or {}

    def rpc(self, account: Account, method: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            return self._fetch(method, body or {}, account.token)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:160]
            raise FleetError(f"{account.user}:{method} HTTP {exc.code}: {detail}") from exc
        except Exception as exc:  # noqa: BLE001 - surface any transport failure as a FleetError
            raise FleetError(f"{account.user}:{method} {type(exc).__name__}: {exc}") from exc

    # ------------------------------------------------------------------ quotas
    @staticmethod
    def _seconds(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            return float(str(value).rstrip("s"))
        except ValueError:
            return 0.0

    def quota(self, account: Account, *, force: bool = False) -> Dict[str, Any]:
        """Real weekly GPU allowance. Never fabricates numbers: raises on failure."""
        now = self._clock()
        cached = self._quota_cache.get(account.user)
        if not force and cached and now - cached[0] < QUOTA_CACHE_S:
            return cached[1]
        raw = self.rpc(account, "GetAcceleratorQuotaStatistics", {})
        gpu = raw.get("gpuQuota") or {}
        out = {
            "used_seconds": self._seconds(gpu.get("timeUsed")),
            "reserved_seconds": self._seconds(gpu.get("timeReserved")),
            "total_seconds": self._seconds(gpu.get("totalTimeAllowed")),
            "refresh_time": str(raw.get("quotaRefreshTime") or ""),
        }
        out["remaining_seconds"] = max(0.0, out["total_seconds"] - out["used_seconds"])
        self._quota_cache[account.user] = (now, out)
        return out

    def refresh_slots(self) -> None:
        """Update every slot's quota+health view. Safe to call from a watchdog."""
        for user, account in self.accounts.items():
            slot = self.slots[user]
            slot.last_checked = self._clock()
            try:
                q = self.quota(account)
                slot.remaining_s = q["remaining_seconds"]
                slot.used_s = q["used_seconds"]
                slot.total_s = q["total_seconds"]
                slot.refresh_time = q["refresh_time"]
                slot.healthy = True
                slot.last_error = ""
            except FleetError as exc:
                slot.healthy = slot.healthy if slot.remaining_s is not None else None
                slot.last_error = str(exc)

    # ------------------------------------------------------------------ kernel lifecycle
    def find_role_kernel(self, account: Account) -> Optional[Dict[str, str]]:
        """The account's kernel for this role, found by its reserved title prefix."""
        try:
            listing = self.rpc(account, "ListKernels", {"pageSize": 30, "user": account.user})
        except FleetError:
            return None
        for k in (listing.get("kernels") or listing.get("results") or []):
            if str(k.get("title") or "").startswith(f"Blackthorn {account.role}"):
                return {"slug": str(k.get("ref") or k.get("slug") or ""), "title": str(k.get("title") or "")}
        return None

    def purge_running_sessions(self, account: Account, *, keep_slug: str = "") -> int:
        """Retire only stale Blackthorn sessions, never unrelated user notebooks."""
        keep_name = keep_slug.split("/", 1)[-1] if keep_slug else ""
        stopped = 0
        try:
            listing = self.rpc(account, "ListKernels", {"pageSize": 100, "user": account.user})
        except FleetError as exc:
            log.warning("purge_running_sessions: ListKernels failed for %s: %s", account.user, exc)
            return 0
        kernels = listing.get("kernels") or listing.get("results") or []
        for k in kernels:
            ref = str(k.get("ref") or k.get("slug") or "")
            name = ref.split("/", 1)[-1] if ref else ""
            title = str(k.get("title") or "")
            if not name:
                continue
            if keep_name and name == keep_name:
                continue
            is_ours = title.startswith(f"Blackthorn {account.role}") or name.startswith(f"blackthorn-{account.role}")
            if not is_ours:
                continue
            try:
                self.rpc(account, "DeleteKernel", {"userName": account.user, "kernelSlug": name})
                stopped += 1
                log.info("purged running session %s/%s (title=%r)", account.user, name, title[:60])
            except FleetError as exc:
                low = str(exc).lower()
                if "not found" in low or "404" in low or "does not exist" in low:
                    continue
                log.warning("purge DeleteKernel %s/%s: %s", account.user, name, str(exc)[:120])
        if stopped:
            import time as _t
            _t.sleep(4)  # let Kaggle release the concurrent slots
        return stopped

    def stop_kernel_session(self, account: Account, slug: str) -> bool:
        """Boolean view of :meth:`stop_kernel_result` (True only when a running session was verified stopped)."""
        return self.stop_kernel_result(account, slug)["state"] == "stopped"

    def stop_kernel_result(self, account: Account, slug: str) -> Dict[str, str]:
        """Stop a running session and say exactly what happened.

        ``state`` is one of:
        - ``stopped``     the session was running, deletion was accepted and the status now reads not-running
        - ``not_running`` there was no running session to stop
        - ``failed``      Kaggle refused (for example ``kernels.delete`` denied) or the result could not be verified

        Failures are never folded into ``not_running``: a refused stop means the GPU is still billing quota.
        Kaggle keeps executing the version a session started with, so a re-push also needs this.
        """
        name = slug.split("/", 1)[1] if "/" in slug else slug
        status_readable = True
        sess: Dict[str, Any] = {}
        try:
            sess = self.rpc(account, "GetKernelSessionStatus", {"userName": account.user, "kernelSlug": name})
        except FleetError as exc:
            # Some tokens lack kernels.get but still have kernels.delete. Do not give up —
            # attempt DeleteKernel and report unverified/stopped from that outcome.
            status_readable = False
            last_error = f"could not read the session status: {str(exc)[:160]}"
            sess = {}
        if status_readable and str(sess.get("status") or "").upper() not in ("RUNNING", "QUEUED", "PENDING"):
            return {"state": "not_running", "error": ""}
        if not status_readable:
            try:
                listing = self.rpc(account, "ListKernels", {"pageSize": 100, "user": account.user})
                # Kaggle protobuf JSON omits an empty repeated `kernels` field,
                # so a successful ListKernels response may be plain {}.
                if not isinstance(listing, dict):
                    raise FleetError("kernel listing unavailable")
                kernels = listing.get("kernels") or listing.get("results") or []
                if not any(str(k.get("ref") or k.get("slug") or "").split("/")[-1] == name or
                           str(k.get("title") or "").startswith(f"Blackthorn {account.role}")
                           for k in kernels):
                    return {"state": "not_running", "error": ""}
            except FleetError:
                pass  # An unreadable listing proves nothing; attempt deletion.
        last_error = ""
        accepted = False

        def missing_after_delete() -> bool:
            """A successful DeleteKernel can make GetKernelSessionStatus return 403.

            A readable listing that no longer contains the very kernel we saw is
            independent confirmation; a listing error or a still-present kernel is not.
            """
            try:
                listing = self.rpc(account, "ListKernels", {"pageSize": 100, "user": account.user})
                if not isinstance(listing, dict):
                    return False
                kernels = listing.get("kernels") or listing.get("results") or []
                return not any(str(k.get("ref") or k.get("slug") or "").split("/")[-1] == name or
                               str(k.get("title") or "").startswith(f"Blackthorn {account.role}")
                               for k in kernels)
            except FleetError:
                return False

        for _ in range(3):
            try:
                self.rpc(account, "DeleteKernel", {"userName": account.user, "kernelSlug": name})
                accepted = True
            except FleetError as exc:
                last_error = str(exc)[:160]
                # Already gone / never existed — treat as not running.
                low = last_error.lower()
                if "not found" in low or "404" in low or "does not exist" in low:
                    return {"state": "not_running", "error": ""}
            import time as _t
            _t.sleep(3)
            if not status_readable:
                # Cannot re-read status (permission). If delete was accepted, trust the stop.
                if accepted:
                    return ({"state": "stopped", "error": ""} if missing_after_delete() else
                            {"state": "unverified", "error": "stop accepted; session status cannot be read (kernels.get denied)"})
                continue
            try:
                again = self.rpc(account, "GetKernelSessionStatus", {"userName": account.user, "kernelSlug": name})
            except FleetError as exc:
                if accepted:
                    return ({"state": "stopped", "error": ""} if missing_after_delete() else
                            {"state": "unverified", "error": f"Kaggle accepted the stop; the status read was refused: {str(exc)[:120]}"})
                return {"state": "failed", "error": f"could not verify the stop: {str(exc)[:160]}"}
            if str(again.get("status") or "").upper() not in ("RUNNING", "QUEUED", "PENDING"):
                return {"state": "stopped", "error": ""}
        if accepted:
            return {"state": "unverified", "error": "Kaggle accepted the stop but the session still reads as running" if status_readable else "stop accepted; status unreadable"}
        # A backup token may be denied kernels.delete when Kaggle has already
        # removed the kernel. Recheck the listing *after* the retries rather than
        # treating that 403 as evidence that a session is still billing.
        if missing_after_delete():
            return {"state": "not_running", "error": ""}
        return {"state": "failed", "error": last_error or "the session is still running after 3 stop attempts"}

    def start_kernel(self, account: Account, *, slug: str, title: str, notebook_text: str,
                     datasets: Optional[List[str]] = None) -> Dict[str, Any]:
        # Kaggle titles are account-unique and slugs get normalized: if this account already
        # has a kernel for this role (by title), update THAT kernel instead of creating a twin.
        existing = self.find_role_kernel(account)
        if existing and existing.get("slug"):
            slug = existing["slug"]
            title = existing.get("title") or title
            try:
                session = self.rpc(account, "GetKernelSessionStatus", {
                    "userName": account.user, "kernelSlug": slug.split("/", 1)[-1]})
                if str(session.get("status") or "").upper() in ("RUNNING", "QUEUED", "PENDING", "STARTING"):
                    return {"_session_running": True, "_slug": slug, "_reused": True}
            except FleetError:
                # An unreadable status is not proof of a stopped kernel. Do not
                # destructively replace it on a transient Kaggle API failure.
                raise FleetError("Existing kernel status unavailable; refusing destructive restart")
        try:
            self.stop_kernel_session(account, slug)          # stale sessions ignore pushes
        except Exception:  # noqa: BLE001 - stopping is best effort
            pass
        # Never purge unrelated user notebooks to free capacity. A quota failure
        # must be reported, not "fixed" by deleting someone else's work.
        payload: Dict[str, Any] = {
            "slug": slug, "newTitle": title, "text": notebook_text,
            "language": "python", "kernelType": "notebook", "isPrivate": True,
            "enableGpu": True, "enableTpu": False, "enableInternet": True,
            "machineShape": "NvidiaTeslaT4",
        }
        if datasets:
            payload["datasetDataSources"] = datasets
        try:
            out = self.rpc(account, "SaveKernel", payload)
        except FleetError as exc:
            if "409" not in str(exc) and "already in use" not in str(exc):
                raise
            # Kaggle can reserve BOTH a title and a slug for minutes after
            # DeleteKernel. A renamed title on the old slug still gets HTTP 409.
            import secrets
            suffix = f"{int(self._clock())}-{secrets.token_hex(3)}"
            payload["newTitle"] = f"Blackthorn {account.role} ({account.user}) {suffix}"[:100]
            payload["slug"] = f"{account.user}/blackthorn-{account.role}-{suffix}"
            out = self.rpc(account, "SaveKernel", payload)
        invalid = [x for x in (out.get("invalidDatasetSources") or []) if x]
        if invalid:
            log.warning("Kaggle ignored %s unavailable dataset attachment(s) for %s",
                        len(invalid), account.role)
        # SaveKernel may choose a new ref after a 409; CREATE/CHECK that ref.
        full_slug = payload["slug"]
        name = full_slug.split("/", 1)[-1]

        def _session_running() -> bool:
            try:
                sess = self.rpc(account, "GetKernelSessionStatus",
                                {"userName": account.user, "kernelSlug": name})
            except FleetError:
                return False
            return str((sess or {}).get("status") or "").upper() in ("RUNNING", "QUEUED", "PENDING", "STARTING")

        running = _session_running()
        creation_error = ""
        if not running:
            try:
                self.rpc(account, "CreateKernelSession", {"slug": full_slug})
            except FleetError as exc:
                creation_error = str(exc)[:160]
                log.warning("CreateKernelSession for %s failed: %s", full_slug, creation_error)
            import time as _t
            for _ in range(5):
                _t.sleep(2)
                if _session_running():
                    running = True
                    break
        # versionNumber proves only a saved notebook; a successful create reply
        # proves only accepted submission. Neither proves the session is running.
        out["_session_running"] = running
        out["_slug"] = full_slug
        if not running:
            out["_session_pending"] = True
            if creation_error:
                out["_session_error"] = creation_error
            log.warning("Session not confirmed for %s after save/create", full_slug)
        return out

    def kernel_status(self, account: Account, slug: str) -> Optional[Dict[str, Any]]:
        try:
            out = self.rpc(account, "GetKernel", {"slug": slug})
            return out or None
        except FleetError:
            return None

    # ------------------------------------------------------------------ failover engine
    def decide(self, role: str, *, force: bool = False) -> RoleState:
        """Refresh both slots of a role and decide the active one. Returns the role state."""
        if role not in ROLES:
            raise FleetError(f"unknown role {role!r}")
        with self._lock:
            state = self.state[role]
            primary = self.by_slot.get((role, "primary"))
            backup = self.by_slot.get((role, "backup"))
            if primary is None and backup is None:
                return state
            if primary is None:                       # backup-only fleet: it is simply active
                state.active = "backup"
                if state.reason == "initial":
                    state.reason = "backup_only"
                return state
            for account in (primary, backup):
                if account is None:
                    continue
                slot = self.slots[account.user]
                slot.last_checked = self._clock()
                try:
                    q = self.quota(account, force=force)
                    slot.remaining_s, slot.used_s = q["remaining_seconds"], q["used_seconds"]
                    slot.total_s, slot.refresh_time = q["total_seconds"], q["refresh_time"]
                    if slot.healthy is not False:          # a hard-down mark is sticky until recovery
                        slot.healthy = True
                    slot.last_error = ""
                except FleetError as exc:
                    slot.last_error = str(exc)
                    # A permanently rejected token cannot host this role. Prefer a
                    # healthy backup; do not misread transient network errors as
                    # proof of a dead account and flap between paid sessions.
                    if "HTTP 401" in slot.last_error or "HTTP 403" in slot.last_error:
                        slot.healthy = False
                    elif slot.remaining_s is None:
                        slot.healthy = None
            if backup is None:                        # single-slot role: the primary just runs
                for account in (primary,):
                    slot = self.slots[account.user]
                    slot.last_checked = self._clock()
                    try:
                        q = self.quota(account, force=force)
                        slot.remaining_s, slot.used_s = q["remaining_seconds"], q["used_seconds"]
                        slot.total_s, slot.refresh_time = q["total_seconds"], q["refresh_time"]
                        if slot.healthy is not False:
                            slot.healthy = True
                        slot.last_error = ""
                    except FleetError as exc:
                        slot.last_error = str(exc)
                state.active = "primary"
                return state
            p, b = self.slots[primary.user], self.slots[backup.user]
            now = self._clock()

            def hard_down(slot: SlotState) -> bool:
                return slot.healthy is False

            def has_time(slot: SlotState) -> bool:
                return slot.remaining_s is None or slot.remaining_s > BACKUP_MIN_S

            def switch(to: str, reason: str) -> None:
                if state.active != to:
                    state.active = to
                    state.reason = reason
                    state.since = now
                    state.last_switch = now
                    state.event(f"active -> {to}: {reason}")
                    self._persist(role)

            # the backup is unusable -> force the primary back if it has any life in it
            if state.active == "backup" and (hard_down(b) or (b.remaining_s is not None and b.remaining_s <= BACKUP_MIN_S)):
                if p.remaining_s is None or p.remaining_s > BACKUP_MIN_S:
                    switch("primary", "backup_unavailable")
                    return state

            # primary has <= 30 min (or is hard-down) -> move to the backup
            if state.active == "primary":
                p_low = p.remaining_s is not None and p.remaining_s <= FAILOVER_BELOW_S
                if hard_down(p) or p_low:
                    if has_time(b) and not hard_down(b):
                        soft = (now - state.last_switch) < MIN_SWITCH_INTERVAL_S
                        if hard_down(p) or not soft:
                            switch("backup", "primary_unhealthy" if hard_down(p) else "primary_quota_low")
                            return state
                else:
                    state.primary_stable = (state.primary_stable + 1) if p.healthy else 0

            # running on backup -> come home only when the primary is genuinely safe again
            if state.active == "backup":
                p_ok = p.healthy and (p.remaining_s is None or p.remaining_s >= RETURN_ABOVE_S)
                state.primary_stable = (state.primary_stable + 1) if p_ok else 0
                if state.primary_stable >= RETURN_STABLE_CHECKS and (now - state.last_switch) >= MIN_SWITCH_INTERVAL_S:
                    switch("primary", "primary_recovered")
            return state

    def mark_healthy(self, role: str, slot: str) -> None:
        """A hard-down slot proved itself again (kernel up, tunnel serving)."""
        account = self.by_slot.get((role, slot))
        if not account:
            return
        st = self.slots[account.user]
        if st.healthy is False:
            st.healthy = True
            st.last_error = ""
            self.state[role].event(f"{slot} recovered")

    def mark_unhealthy(self, role: str, slot: str, error: str) -> None:
        """Record a hard failure observed by the caller (kernel crash, tunnel dead...)."""
        account = self.by_slot.get((role, slot))
        if not account:
            return
        st = self.slots[account.user]
        st.healthy = False
        st.last_error = error[:160]
        self.state[role].event(f"{slot} marked unhealthy: {error[:80]}")

    def status(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for role in ROLES:
            st = self.state[role]
            p_acct, b_acct = self.by_slot.get((role, "primary")), self.by_slot.get((role, "backup"))
            out[role] = {
                "active": st.active, "reason": st.reason,
                "active_since": round(st.since, 1), "events": st.events[-8:],
                "primary": self.slots[p_acct.user].view() if p_acct else {"configured": False},
                "backup": self.slots[b_acct.user].view() if b_acct else {"configured": False},
                "accounts": {"primary": p_acct.user if p_acct else None,
                             "backup": b_acct.user if b_acct else None},
            }
        return out

    # ------------------------------------------------------------------ persistence hooks
    def _persist(self, role: str) -> None:
        if not self._store:
            return
        try:
            st = self.state[role]
            self._store(f"blackthorn_fleet_{role}",
                        json.dumps({"active": st.active, "reason": st.reason, "since": st.since}))
        except Exception:  # noqa: BLE001 - bookkeeping must never break a failover
            log.warning("fleet persist failed", exc_info=True)

    def restore(self, loader: Optional[Callable[[str], Optional[str]]] = None) -> None:
        """Restore persisted active slots after a controller restart."""
        if not self._store and not loader:
            return
        for role in ROLES:
            try:
                raw = (loader or (lambda k: None))(f"blackthorn_fleet_{role}")
                if raw:
                    data = json.loads(raw)
                    self.state[role].active = str(data.get("active") or "primary")
                    self.state[role].reason = str(data.get("reason") or "restored")
                    self.state[role].since = float(data.get("since") or self._clock())
            except Exception:  # noqa: BLE001
                log.warning("fleet restore failed for %s", role, exc_info=True)


def kernel_slug_for(account: Account) -> str:
    return f"{account.user}/blackthorn-{account.role}"


def notebook_env_for(account: Account, base_env: Dict[str, str]) -> Dict[str, str]:
    """Per-account environment for the notebook: the kernel always speaks with its own
    credentials (state dataset, heartbeats) so pairs can be synchronized and failed over."""
    env = dict(base_env)
    env["KAGGLE_USERNAME"] = account.user
    env["KAGGLE_API_TOKEN"] = account.token
    # Shared model cache across primary+backup so neither redownloads after first publish.
    primary_model = None
    primary_computer = None
    # account objects in this function don't carry fleet; use env or conventional primary names.
    if account.role == "model":
        # Always use the controller-configured cache (primary owner) so primary and
        # backup both hit the same permanent GGUF dataset and never redownload.
        shared_cache = (base_env.get("QWEN38_CACHE_DATASET") or env.get("QWEN38_CACHE_DATASET") or "").strip()
        if not shared_cache:
            shared_cache = "black797/qwen38-27b-uncensored-model-cache"
        env["QWEN38_CACHE_DATASET"] = shared_cache
        env.setdefault("ORNITH_ROLE", "model")
    if account.role == "computer":
        env["ORNITH_ROLE"] = "computer"
        # Primary owns the state dataset; backup restores from the mirrored copy under its own slug.
        env["COMPUTER_STATE_DATASET"] = f"{account.user}/blackthorn-computer-state"
        env.setdefault("COMPUTER_CHECKPOINT_S", "120")
    return env


def mirror_state_dataset(from_account: Account, to_account: Account, *,
                         dataset_slug: str = "blackthorn-computer-state",
                         runner: Optional[Callable[[List[str], Dict[str, str]], int]] = None) -> Dict[str, Any]:
    """Copy the computer state dataset from one account to its pair partner.

    The primary is the source of truth. This runs in the controller (it holds all four
    credentials) right after checkpoints and before failovers, so the backup always has
    the newest published generation to boot from.
    """
    import os
    import json
    import shutil
    import subprocess
    import tempfile
    import zipfile
    from pathlib import Path

    last_cli_error = ""
    def _run(cmd: List[str], env: Dict[str, str]) -> int:
        nonlocal last_cli_error
        if runner is not None:
            return runner(cmd, env)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
        if proc.returncode:
            import re
            # Owner-facing diagnostics must never echo credentials or bearer headers.
            last_cli_error = (proc.stderr or proc.stdout or "CLI failed")[-350:]
            for secret in (from_account.token, to_account.token):
                if secret:
                    last_cli_error = last_cli_error.replace(secret, "[redacted]")
            last_cli_error = re.sub(r"KGAT_[A-Za-z0-9_-]+", "[redacted]", last_cli_error)
            last_cli_error = last_cli_error.replace("\n", " ")[:180]
        return proc.returncode

    src_slug = f"{from_account.user}/{dataset_slug}"
    dst_slug = f"{to_account.user}/{dataset_slug}"
    with tempfile.TemporaryDirectory() as tmp:
        download = Path(tmp) / "download"
        upload = Path(tmp) / "upload"
        download.mkdir()
        upload.mkdir()
        # KGAT tokens require Bearer auth. KAGGLE_KEY is a legacy Basic credential.
        base_env = dict(os.environ)
        env_down = {**base_env, "KAGGLE_USERNAME": from_account.user,
                    "KAGGLE_API_TOKEN": from_account.token}
        env_down.pop("KAGGLE_KEY", None)
        # A freshly published Kaggle dataset version can briefly return 404 on
        # the download endpoint even while datasets/view reports the new version.
        # Retry only transient 404/429/5xx responses, never authentication errors.
        import time as _t
        for attempt in range(5):
            rc = _run(["kaggle", "datasets", "download", "-p", str(download), src_slug], env_down)
            if rc == 0:
                break
            transient = any(code in last_cli_error for code in ("404", "429", "500", "502", "503", "504"))
            if not transient or attempt == 4:
                return {"ok": False, "error": f"download rc={rc}: {last_cli_error}"}
            _t.sleep((4, 8, 12, 20)[attempt])
        archives = list(download.glob("*.zip"))
        if len(archives) != 1:
            return {"ok": False, "error": "expected one downloaded state archive"}
        try:
            with zipfile.ZipFile(archives[0]) as zf:
                # Kaggle may return a flat workspace archive OR an archive with
                # workspace.zip (older CLI). Both are valid snapshots.
                copied = 0
                total = 0
                for info in zf.infolist():
                    if info.is_dir() or info.filename == "dataset-metadata.json":
                        continue
                    path = Path(info.filename)
                    mode = (info.external_attr >> 16) & 0o170000
                    if path.is_absolute() or ".." in path.parts or mode == 0o120000:
                        return {"ok": False, "error": "unsafe path in state archive"}
                    total += info.file_size
                    if total > 25 * 1024**3 or copied >= 10000:
                        return {"ok": False, "error": "state archive exceeds mirror limits"}
                    dest = upload / path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                    copied += 1
                if copied == 0:
                    return {"ok": False, "error": "state archive has no workspace snapshot"}
        except (OSError, KeyError, zipfile.BadZipFile):
            return {"ok": False, "error": "state archive is corrupt"}
        (upload / "dataset-metadata.json").write_text(json.dumps({
            "title": "Blackthorn Computer State", "id": dst_slug, "isPrivate": True,
            "licenses": [{"name": "other"}]
        }))
        env_up = {**base_env, "KAGGLE_USERNAME": to_account.user,
                  "KAGGLE_API_TOKEN": to_account.token}
        env_up.pop("KAGGLE_KEY", None)
        rc = _run(["kaggle", "datasets", "create", "-p", str(upload), "--dir-mode", "skip"], env_up)
        if rc != 0:
            rc = _run(["kaggle", "datasets", "version", "-p", str(upload),
                       "-m", "mirror", "--dir-mode", "skip"], env_up)
        return {"ok": rc == 0, "from": src_slug, "to": dst_slug,
                "error": "" if rc == 0 else f"upload rc={rc}: {last_cli_error}"}


def accounts_from_env(env: Dict[str, str]) -> List[Account]:
    """Build the fleet from environment variables (KAGGLE_FLEET_*)."""

    def parse(spec: str, role: str, slot: str, label: str) -> Optional[Account]:
        spec = (spec or "").strip()
        if not spec:
            return None
        user, _, token = spec.partition(":")
        user, token = user.strip(), token.strip()
        if not (user and token):
            return None
        return Account(user=user, token=token, role=role, slot=slot, label=label)

    out = [a for a in (
        parse(env.get("KAGGLE_FLEET_MODEL_PRIMARY", ""), "model", "primary", "black797"),
        parse(env.get("KAGGLE_FLEET_MODEL_BACKUP", ""), "model", "backup", "temilolo"),
        parse(env.get("KAGGLE_FLEET_COMPUTER_PRIMARY", ""), "computer", "primary", "alagbo"),
        parse(env.get("KAGGLE_FLEET_COMPUTER_BACKUP", ""), "computer", "backup", "teslaarymo"),
    ) if a]
    # legacy single-account fallback: the old model account only
    if not any(a.role == "model" and a.slot == "primary" for a in out):
        user, token = env.get("KAGGLE_USERNAME", ""), env.get("KAGGLE_API_TOKEN", "")
        if user and token:
            out.append(Account(user=user, token=token, role="model", slot="primary", label="Joshbond123"))
    return out
