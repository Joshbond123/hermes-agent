"""PandaStack API-key pool: verification, real usage measurement and automatic rotation.

Rotation rules (no invented data — every number here comes from the provider):

* ``GET /v1/usage`` returns the month-to-date rollup for the key's workspace:
  ``events``, ``cpu_seconds``, ``gb_seconds``, ``cost_usd`` and
  ``monthly_budget_usd`` (0 means the workspace has no spend cap configured).
* When a cap exists, the remaining budget divided by the observed burn rate gives
  a projected exhaustion time. Rotation starts when fewer than
  ``rotation_threshold_s`` (default 300 s = five minutes) remain.
* When no cap is configured the provider only reports an exhausted quota through
  a HTTP 402/429 (or an explicit quota message). Those responses rotate
  immediately instead of waiting for a projection that cannot exist.

Secrets never leave this module: logs and stored state carry a salted-by-nothing
SHA-256 *fingerprint* prefix of the key, never the key itself.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import httpx

API = "https://api.pandastack.ai"
ACTIVE_KEY_SETTING = "pandastack_active_key_fingerprint"
ROTATION_LOG = "pandastack_key_rotations"
DEFAULT_THRESHOLD_S = 300.0          # rotate ~5 minutes before the projected limit


def fingerprint(key: str) -> str:
    """Short, non-reversible identifier used in logs and settings."""
    return hashlib.sha256(("blackthorn:" + (key or "")).encode("utf-8", "replace")).hexdigest()[:12]


@dataclass
class KeyState:
    fingerprint: str
    key: str
    valid: bool = False
    workspace: str = ""
    cost_usd: float = 0.0
    budget_usd: float = 0.0
    cpu_seconds: float = 0.0
    events: int = 0
    window_start: Optional[str] = None
    checked_at: float = 0.0
    error: str = ""

    @property
    def has_cap(self) -> bool:
        return self.budget_usd > 0

    @property
    def remaining_usd(self) -> Optional[float]:
        return max(0.0, self.budget_usd - self.cost_usd) if self.has_cap else None

    def burn_usd_per_hour(self) -> Optional[float]:
        """Average spend rate since the start of the provider's usage window."""
        if not self.window_start or self.cost_usd <= 0:
            return None
        try:
            start = _parse_iso(self.window_start)
        except ValueError:
            return None
        elapsed_h = (time.time() - start) / 3600.0
        if elapsed_h <= 0:
            return None
        return self.cost_usd / elapsed_h

    def seconds_to_limit(self) -> Optional[float]:
        """Projected seconds until the configured monthly cap is reached."""
        if not self.has_cap:
            return None
        remaining = self.remaining_usd or 0.0
        rate = self.burn_usd_per_hour()
        if rate is None or rate <= 0:
            return None
        return (remaining / rate) * 3600.0

    def public(self) -> Dict[str, Any]:
        limit = self.seconds_to_limit()
        return {
            "fingerprint": self.fingerprint, "valid": self.valid, "workspace": self.workspace,
            "cost_usd": round(self.cost_usd, 6), "budget_usd": self.budget_usd,
            "remaining_usd": None if self.remaining_usd is None else round(self.remaining_usd, 6),
            "cpu_seconds": round(self.cpu_seconds, 1), "events": self.events,
            "projected_seconds_to_limit": None if limit is None else round(limit, 1),
            "error": self.error, "checked_at": self.checked_at,
        }


def _parse_iso(value: str) -> float:
    text = str(value or "").strip().replace("Z", "+00:00")
    # ``datetime.fromisoformat`` in 3.11+ handles the offsets PandaStack returns.
    from datetime import datetime
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.timestamp()
    return parsed.timestamp()


def parse_keys(raw: str) -> List[str]:
    out: List[str] = []
    for chunk in (raw or "").replace("\n", ",").replace(";", ",").split(","):
        key = chunk.strip()
        if key and key not in out:
            out.append(key)
    return out


class PandaKeyPool:
    """Ordered pool of authorized PandaStack keys with health + quota tracking."""

    def __init__(self, keys: Sequence[str], http: httpx.AsyncClient, store: Any = None, *,
                 threshold_s: float = DEFAULT_THRESHOLD_S, api: str = API):
        self.keys = [k for k in keys if k]
        if not self.keys:
            raise ValueError("no PandaStack API keys configured")
        self.http = http
        self.store = store
        self.threshold_s = float(threshold_s)
        self.api = api.rstrip("/")
        self.states: Dict[str, KeyState] = {fingerprint(k): KeyState(fingerprint(k), k) for k in self.keys}
        self._active = fingerprint(self.keys[0])
        self.rotations: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_env(cls, http: httpx.AsyncClient, store: Any = None, environ: Optional[Dict[str, str]] = None) -> Optional["PandaKeyPool"]:
        env = environ if environ is not None else os.environ
        raw = [env.get("PANDASTACK_API_KEY", "")]
        raw += [env.get(f"PANDASTACK_API_KEY_{i}", "") for i in range(2, 10)]
        raw += parse_keys(env.get("PANDASTACK_API_KEYS", ""))
        keys = parse_keys(",".join(r for r in raw if r))
        if not keys:
            return None
        threshold = env.get("PANDASTACK_ROTATE_BEFORE_SECONDS") or DEFAULT_THRESHOLD_S
        try:
            threshold_s = float(threshold)
        except ValueError:
            threshold_s = DEFAULT_THRESHOLD_S
        return cls(keys, http, store, threshold_s=threshold_s)

    # ------------------------------------------------------------------ state
    def key_for(self, fp: str) -> Optional[KeyState]:
        return self.states.get(fp)

    @property
    def active(self) -> KeyState:
        state = self.states.get(self._active)
        if state is None:                     # defensive: never leave the pool without a key
            state = next(iter(self.states.values()))
            self._active = state.fingerprint
        return state

    def active_key(self) -> str:
        return self.active.key

    def snapshot(self) -> List[Dict[str, Any]]:
        return [{**s.public(), "active": s.fingerprint == self._active} for s in self.states.values()]

    async def restore_active(self) -> None:
        """Honour a rotation performed by another worker (fingerprint only, never a key)."""
        if self.store is None or not hasattr(self.store, "get_setting"):
            return
        try:
            stored = await self.store.get_setting(ACTIVE_KEY_SETTING)
        except Exception:
            return
        if stored and stored in self.states:
            self._active = stored

    # ------------------------------------------------------------------ provider calls
    async def probe(self, key: str, *, timeout: float = 20.0) -> KeyState:
        state = self.states.get(fingerprint(key)) or KeyState(fingerprint(key), key)
        self.states[state.fingerprint] = state
        headers = {"Authorization": "Bearer " + key}
        try:
            me = await self.http.get(self.api + "/v1/me", headers=headers, timeout=timeout)
            usage = await self.http.get(self.api + "/v1/usage", headers=headers, timeout=timeout)
        except httpx.HTTPError as exc:
            state.valid, state.error, state.checked_at = False, f"network: {type(exc).__name__}", time.time()
            return state
        if me.status_code in (401, 403):
            state.valid, state.error, state.checked_at = False, f"auth HTTP {me.status_code}", time.time()
            return state
        if me.status_code != 200:
            state.valid, state.error, state.checked_at = False, f"HTTP {me.status_code}", time.time()
            return state
        state.valid = True
        try:
            body = me.json()
            state.workspace = str(body.get("user_id") or (body.get("current_org") or {}).get("id") or "")
        except ValueError:
            state.workspace = ""
        if usage.status_code == 200:
            try:
                data = usage.json()
            except ValueError:
                data = {}
            rollup = data.get("rollup") or {}
            state.cost_usd = float(rollup.get("cost_usd") or 0.0)
            state.cpu_seconds = float(rollup.get("cpu_seconds") or 0.0)
            state.events = int(rollup.get("events") or 0)
            # The provider reports 0 for "no cap configured"; an operator may still
            # declare the applicable monthly limit for projection purposes.
            budget = float(data.get("monthly_budget_usd") or 0.0)
            override = os.environ.get("PANDASTACK_MONTHLY_BUDGET_USD", "").strip()
            if budget <= 0 and override:
                try:
                    budget = float(override)
                except ValueError:
                    budget = 0.0
            state.budget_usd = budget
            state.window_start = data.get("from")
        elif usage.status_code == 429:
            state.error = "rate limited"
        state.checked_at = time.time()
        return state

    async def probe_all(self) -> List[KeyState]:
        import asyncio
        return list(await asyncio.gather(*(self.probe(s.key) for s in list(self.states.values()))))

    # ------------------------------------------------------------------ selection
    def _candidates(self) -> List[KeyState]:
        """Keys other than the active one (unprobed ones included, they get probed on use)."""
        others = [s for s in self.states.values()
                  if s.fingerprint != self._active and (s.valid or not s.checked_at)]
        def rank(state: KeyState):
            limit = state.seconds_to_limit()
            return (-(state.remaining_usd or 0.0), -(limit if limit is not None else 1e18))
        return sorted(others, key=rank)

    async def ensure_active(self) -> KeyState:
        """Verify the active key; rotate if it is invalid or exhausted."""
        await self.restore_active()
        state = self.active
        if not state.checked_at:
            await self.probe(state.key)
        if not state.valid:
            rotated = await self.rotate(reason=f"active key invalid ({state.error or 'unverified'})")
            return rotated or state
        if state.has_cap:
            limit = state.seconds_to_limit()
            if limit is not None and limit <= self.threshold_s:
                rotated = await self.rotate(reason=f"projected limit in {int(limit)}s", seconds_left=limit)
                if rotated:
                    return rotated
        return state

    async def rotate(self, *, reason: str, seconds_left: Optional[float] = None) -> Optional[KeyState]:
        """Switch new requests to the best verified key. Returns None when none is usable."""
        previous = self.active
        for candidate in self._candidates():
            if not candidate.checked_at:
                await self.probe(candidate.key)
            if not candidate.valid:
                continue
            limit = candidate.seconds_to_limit()
            if candidate.has_cap and limit is not None and limit <= self.threshold_s:
                continue      # this key is about to be exhausted too
            self._active = candidate.fingerprint
            event = {
                "at": time.time(), "reason": reason,
                "from_fingerprint": previous.fingerprint, "to_fingerprint": candidate.fingerprint,
                "seconds_to_limit": None if seconds_left is None else round(seconds_left, 1),
                "workspace": candidate.workspace,
            }
            self.rotations.append(event)
            await self._persist(event)
            return candidate
        return None

    async def _persist(self, event: Dict[str, Any]) -> None:
        if self.store is None or not hasattr(self.store, "set_setting"):
            return
        try:
            await self.store.set_setting(ACTIVE_KEY_SETTING, self._active)
            import json
            history = []
            try:
                raw = await self.store.get_setting(ROTATION_LOG)
                history = json.loads(raw) if raw else []
            except Exception:
                history = []
            history = (history + [event])[-50:]
            await self.store.set_setting(ROTATION_LOG, json.dumps(history))
        except Exception:
            # Rotation must never fail a task because bookkeeping failed.
            pass

    async def note_failure(self, key: str, status_code: int, message: str = "") -> Optional[KeyState]:
        """Called by the computer client when the provider rejects a key."""
        state = self.states.get(fingerprint(key))
        if state is not None and status_code in (401, 403, 402, 429):
            state.valid = status_code not in (401, 403)
            state.error = message or f"HTTP {status_code}"
        if status_code in (401, 402, 403, 429):
            return await self.rotate(reason=f"provider HTTP {status_code}")
        return None
