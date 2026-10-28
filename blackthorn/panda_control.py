"""Durable PandaStack computer lifecycle, separate from the Kaggle model GPU.

Only a named production sandbox is managed; validation sandboxes are never touched.
The persistent intention and sandbox ID live in the existing D1 settings store.
"""
from __future__ import annotations

import os
import re
import asyncio
import time
import uuid

import httpx

INTENT = "pandastack_computer_intent"
SANDBOX = "pandastack_computer_sandbox_id"
PURPOSE = "blackthorn-production-computer"
# A handover replacement is created while the old sandbox is still alive (overlap),
# tagged so reconciliation can tell the two apart until the swap is verified.
ROLE_PRIMARY = "primary"
ROLE_REPLACEMENT = "replacement"
RECOVERY_LOCK = "pandastack_computer_recovery_lock"
# Which production computer is live, and why the normal one could not be used.
ROLE_SETTING = "pandastack_computer_live_role"
LAST_ERROR = "pandastack_computer_last_error"
UUID = re.compile(r"^[0-9a-fA-F-]{32,36}$")
BASE = "https://api.pandastack.ai/v1/sandboxes"


class PandaControl:
    def __init__(self, store, http: httpx.AsyncClient, key=None):
        self.store = store
        self.http = http
        # ``key`` may be a string, a callable, or a PandaKeyPool (rotation-aware).
        # With no explicit key we use the whole configured pool, not just
        # PANDASTACK_API_KEY: one rate-limited account must not take the computer down.
        self._key_source = key
        self._pool = key if hasattr(key, "ensure_active") else None
        self.attempts: list[str] = []

    async def _keys(self):
        """Every usable key, active first. Yields (key, fingerprint)."""
        if self._pool is None and self._key_source is None:
            from .panda_keys import PandaKeyPool, fingerprint
            pool = PandaKeyPool.from_env(self.http, self.store)
            if pool is None:
                single = (os.environ.get("PANDASTACK_API_KEY") or "").strip()
                return [(single, fingerprint(single))] if single else []
            self._pool = pool
        if self._pool is not None:
            try:
                await self._pool.restore_active()
                active = await self._pool.ensure_active()
            except Exception:
                active = self._pool.active
            out = [(active.key, active.fingerprint)]
            for state in self._pool.states.values():
                if state.fingerprint != active.fingerprint:
                    out.append((state.key, state.fingerprint))
            return out
        source = self._key_source
        value = str(source() if callable(source) else (source or "")).strip()
        return [(value, value[-4:])] if value else []

    async def _rotate(self, reason: str) -> bool:
        if self._pool is None:
            return False
        try:
            return bool(await self._pool.rotate(reason=reason))
        except Exception:
            return False

    @property
    def key(self) -> str:
        source = self._key_source
        if callable(source):
            return str(source())
        if hasattr(source, "active_key"):
            return str(source.active_key())
        return str(source or "")

    @key.setter
    def key(self, value) -> None:
        self._key_source = value

    def _headers(self):
        if not self.key:
            raise RuntimeError("PandaStack is not configured")
        return {"Authorization": "Bearer " + self.key}

    async def _request(self, method: str, path: str = "", **kw):
        response = await self.http.request(method, BASE + path, headers=self._headers(), timeout=35, **kw)
        if response.status_code in (401, 403):
            raise RuntimeError("PandaStack denied access")
        return response

    async def _owned(self):
        response = await self._request("GET")
        if response.status_code != 200 or not isinstance(response.json(), list):
            raise RuntimeError(f"PandaStack list returned HTTP {response.status_code}")
        return [s for s in response.json() if (s.get("metadata") or {}).get("purpose") == PURPOSE]

    # ------------------------------------------------------------------ record helpers
    @staticmethod
    def role(item: dict) -> str:
        return str(((item or {}).get("metadata") or {}).get("role") or ROLE_PRIMARY)

    async def current(self, owned: list | None = None):
        """The sandbox this deployment owns: the stored ID if it is still listed.

        A second, intentionally over-lapping sandbox can exist during a handover, so
        "more than one row" is no longer by itself an error — only an ambiguous set
        without the stored ID is.
        """
        owned = await self._owned() if owned is None else owned
        sid = await self.store.get_setting(SANDBOX) if self.store is not None else ""
        if sid:
            match = next((x for x in owned if x.get("id") == sid), None)
            if match is not None:
                return match
        if not owned:
            return None
        primaries = [x for x in owned if self.role(x) == ROLE_PRIMARY]
        if len(primaries) == 1:
            return primaries[0]
        if len(owned) == 1:
            return owned[0]
        return None

    async def reconcile_extras(self, owned: list | None = None) -> list:
        """Delete duplicate managed sandboxes that are not the stored one.

        A worker that crashed during creation can leak a billable VM. The stored ID is
        authoritative; anything else carrying our purpose marker is released.
        """
        owned = await self._owned() if owned is None else owned
        stored = await self.store.get_setting(SANDBOX) if self.store is not None else ""
        # Never release a sandbox an in-flight handover depends on: during an overlap both
        # the source and its verified replacement are legitimate, and deleting the wrong
        # one destroys a workspace that was just restored.
        protected = set()
        if self.store is not None and hasattr(self.store, "get_setting"):
            try:
                import json as _json
                state = _json.loads(await self.store.get_setting("pandastack_lifecycle_state") or "{}")
                protected = {str(state.get(key)) for key in ("source", "replacement") if state.get(key)}
            except Exception:
                protected = set()
        if stored:
            protected.add(stored)
        keep = next((x for x in owned if x.get("id") == stored), None)
        if keep is None and protected:
            # No stored ID, but a handover names a live sandbox: adopt that one.
            keep = next((x for x in owned if x.get("id") in protected), None)
        if keep is None:
            # Nothing authoritative: keep the newest and release the rest.
            ordered = sorted(owned, key=lambda x: str(x.get("created_at") or ""))
            keep = ordered[-1] if ordered else None
        if keep is not None and self.store is not None:
            await self.store.set_setting(SANDBOX, keep["id"])
        released = []
        for item in owned:
            if keep is not None and item.get("id") == keep.get("id"):
                continue
            if item.get("id") in protected:
                continue
            try:
                await self.delete(item["id"])
                released.append(item.get("id"))
            except Exception:
                continue
        return released

    async def describe(self, sid: str) -> dict:
        response = await self._request("GET", "/" + sid)
        if response.status_code == 404:
            return {}
        if response.status_code != 200:
            raise RuntimeError(f"PandaStack describe returned HTTP {response.status_code}")
        body = response.json()
        return body if isinstance(body, dict) else {}

    async def lifetime(self, sid: str) -> dict:
        response = await self._request("GET", "/" + sid + "/lifecycle")
        if response.status_code != 200:
            return {}
        body = response.json()
        return body if isinstance(body, dict) else {}

    CREATE_TEMPLATES = ("code-interpreter", "base")

    async def create(self, role: str = ROLE_PRIMARY, metadata: dict | None = None,
                     templates: tuple = None) -> str:
        """Create a computer, rotating keys and templates instead of failing on one 429.

        Every attempt is recorded in ``self.attempts`` so a real failure reports the
        provider's status per account rather than a single opaque error.
        """
        self.attempts = []
        templates = templates or self.CREATE_TEMPLATES
        keys = await self._keys()
        if not keys:
            raise RuntimeError("PandaStack is not configured (no API key)")
        tried: set[str] = set()
        while keys:
            key, fp = keys[0]
            if fp in tried:
                keys = keys[1:]
                continue
            tried.add(fp)
            key_level_failure = ""
            for template in templates:
                payload = {"template": template, "ttl_seconds": 3600,
                           "metadata": {"purpose": PURPOSE, "role": role, **(metadata or {})}}
                try:
                    response = await self.http.post(
                        BASE, json=payload, headers={"Authorization": "Bearer " + key}, timeout=60)
                except httpx.HTTPError as exc:
                    self.attempts.append(f"{fp}/{template}: network {type(exc).__name__}")
                    key_level_failure = key_level_failure or f"network {type(exc).__name__}"
                    continue
                if response.status_code == 201:
                    sid = str(response.json().get("id") or "")
                    if not UUID.fullmatch(sid):
                        raise RuntimeError("PandaStack create returned no valid sandbox ID; reconcile before retry")
                    await self._ensure_lifetime(sid)
                    return sid
                detail = (response.text or "").strip().replace("\n", " ")[:160]
                self.attempts.append(f"{fp}/{template}: HTTP {response.status_code}{(' ' + detail) if detail else ''}")
                if response.status_code in (401, 402, 403, 429):
                    key_level_failure = f"provider HTTP {response.status_code}"
                    break                      # this account is unusable right now
            if key_level_failure:
                await self._rotate(reason="computer create: " + key_level_failure)
                keys = await self._keys()      # re-read: the pool may have a new active key
                continue
            keys = keys[1:]                    # provider-side problem, not the account: next key
        raise RuntimeError("PandaStack could not create a computer: " + "; ".join(self.attempts))

    async def delete(self, sid: str) -> bool:
        response = await self._request("DELETE", "/" + sid)
        if response.status_code not in (200, 204, 404):
            raise RuntimeError(f"PandaStack delete returned HTTP {response.status_code}")
        return True

    async def probe(self, sid: str) -> bool:
        """Live verification that a sandbox can execute work."""
        response = await self._request("POST", "/" + sid + "/exec",
                                       json={"cmd": "python3 -c 'print(19*23)'", "timeout_seconds": 15})
        if response.status_code != 200:
            return False
        try:
            body = response.json()
        except ValueError:
            return False
        return body.get("exit_code") == 0 and str(body.get("stdout", "")).strip() == "437"

    async def status(self):
        if not self.key:
            return {"status": "unconfigured", "online": False, "host": "PandaStack", "storage": "private GitHub Computer-drive"}
        intent = await self.store.get_setting(INTENT)
        sid = await self.store.get_setting(SANDBOX)
        if intent != "on":
            # Hibernation stops the VM without deleting unsaved workspace state.
            owned = await self._owned()
            if not owned or all(str(item.get("status") or "").lower() == "hibernated" for item in owned):
                return {"status": "off", "online": False, "host": "PandaStack", "storage": "private GitHub Computer-drive",
                        "display_status": "Off · VM hibernated until provider TTL" if owned else "Off · VM expired; named drive files retained"}
            return {"status": "stop_unverified", "online": False, "host": "PandaStack", "storage": "private GitHub Computer-drive",
                    "error": "Shutdown requested but a PandaStack VM is still active"}
        owned = await self._owned()
        item = await self.current(owned)
        if item is None:
            return {"status": "missing" if not owned else "duplicate", "online": False,
                    "host": "PandaStack", "storage": "private GitHub Computer-drive",
                    "error": ("Provider VM expired while idle; the next computer tool call will create a replacement and restore the GitHub workspace."
                              if not owned else "Expected exactly one production computer")}
        if sid != item["id"]:
            await self.store.set_setting(SANDBOX, item["id"])
        state = str(item.get("status") or "unknown").lower()
        # This account cannot enable provider persistence (403). The supported idle
        # TTL is one hour; named GitHub-backed writes outlive sandbox expiry.
        role = ""
        last_error = ""
        try:
            role = str(await self.store.get_setting(ROLE_SETTING) or "")
            last_error = str(await self.store.get_setting(LAST_ERROR) or "")
        except Exception:
            pass
        return {"status": state, "online": state in ("running", "hibernated"),
                "role": role or ROLE_PRIMARY, "fallback": role == "lab-fallback",
                "last_error": last_error,
                "host": "PandaStack", "storage": "private GitHub Computer-drive",
                "environment": "Firecracker CPU microVM · 1-hour idle TTL", "transport": "PandaStack API",
                "display_status": "Ready · 1-hour idle TTL" if state == "running" else ("Hibernated until provider TTL — wakes on task" if state == "hibernated" else state)}

    async def _ensure_lifetime(self, sid: str):
        life = await self._request("GET", "/" + sid + "/lifecycle")
        if life.status_code != 200:
            raise RuntimeError("PandaStack sandbox lifetime could not be verified")
        if life.json().get("persistent") is not True and int(life.json().get("ttl_seconds") or 0) < 3600:
            change = await self._request("PATCH", "/" + sid + "/lifecycle", json={"ttl_seconds": 3600})
            if change.status_code != 200:
                raise RuntimeError("PandaStack one-hour idle TTL was refused")
            check = await self._request("GET", "/" + sid + "/lifecycle")
            if check.status_code != 200 or int(check.json().get("ttl_seconds") or 0) < 3600:
                raise RuntimeError("PandaStack sandbox lifetime was not confirmed")

    async def ensure_for_task(self):
        """Recreate only for a real task after provider expiry, never as an idle quota loop.

        A D1 compare-and-insert lease prevents two Render workers creating two
        billable sandboxes. It expires if the worker crashes during creation.
        """
        if await self.store.get_setting(INTENT) != "on":
            raise RuntimeError("PandaStack computer is powered off")
        owned = await self._owned()
        item = await self.current(owned)
        if item is None and owned:
            # Two sandboxes without a matching stored ID: release the extras and adopt
            # the survivor instead of stalling the whole task on reconciliation.
            await self.reconcile_extras(owned)
            owned = await self._owned()
            item = await self.current(owned)
        if item is not None:
            sid = item["id"]
            if await self.store.get_setting(SANDBOX) != sid:
                await self.store.set_setting(SANDBOX, sid)
            if str(item.get("status") or "").lower() == "hibernated":
                return await self.start()
            return await self.status()
        if not hasattr(self.store, "db"):
            return await self.start()
        token = f"{time.time()+100:.3f}:{uuid.uuid4().hex}"
        for _ in range(50):
            if await self.store.get_setting(INTENT) != "on":
                raise RuntimeError("Computer was turned off during task recovery")
            await self.store.db.query("DELETE FROM state_meta WHERE key = ? AND CAST(value AS REAL) < ?",
                                      [RECOVERY_LOCK, time.time()])
            claim = await self.store.db.query(
                "INSERT INTO state_meta (key,value) SELECT ?,? WHERE NOT EXISTS "
                "(SELECT 1 FROM state_meta WHERE key = ?)", [RECOVERY_LOCK, token, RECOVERY_LOCK])
            if claim.changes:
                try:
                    # Recheck after acquiring the lease: an earlier owner may have
                    # finished between the initial list and this claim.
                    if await self.store.get_setting(INTENT) != "on":
                        raise RuntimeError("Computer was turned off during task recovery")
                    return await self.start()
                finally:
                    await self.store.db.query("DELETE FROM state_meta WHERE key = ? AND value = ?", [RECOVERY_LOCK, token])
            await asyncio.sleep(0.5)
            owned = await self._owned()
            if await self.current(owned) is not None:
                return await self.status()
            if owned:
                await self.reconcile_extras(owned)
                owned = await self._owned()
                if await self.current(owned) is not None:
                    return await self.status()
        raise RuntimeError("Timed out waiting for computer recovery lease")

    async def start(self):
        # Reconcile before create because a lost response could have already
        # created a billable sandbox. Never blindly retry a non-idempotent create.
        await self.store.set_setting(INTENT, "on")
        owned = await self._owned()
        item = await self.current(owned)
        if item is None and owned:
            await self.reconcile_extras(owned)
            owned = await self._owned()
            item = await self.current(owned)
        if item is not None:
            sid = item["id"]
            await self.store.set_setting(SANDBOX, sid)
            await self._ensure_lifetime(sid)
            if str(item.get("status") or "").lower() == "hibernated":
                woke = await self._request("POST", "/" + sid + "/wake", json={})
                if woke.status_code not in (200, 201, 204):
                    raise RuntimeError(f"PandaStack wake returned HTTP {woke.status_code}")
            probe = await self._request("POST", "/" + sid + "/exec",
                json={"cmd": "python3 -c 'print(19*23)'", "timeout_seconds": 15})
            if probe.status_code != 200 or probe.json().get("exit_code") != 0 or str(probe.json().get("stdout", "")).strip() != "437":
                raise RuntimeError("Existing PandaStack computer did not pass live exec check")
            return await self.status()
        # The current PandaStack tier rejects persistent=true. Request the
        # supported one-hour idle TTL explicitly; never pretend the VM is durable.
        try:
            # The production computer is a code-interpreter VM. If that cannot be
            # created, start_with_fallback() retries with the lab-style base VM and
            # reports which one is running, so the failure is never hidden.
            sid = await self.create(templates=("code-interpreter",))
        except RuntimeError:
            await self.store.set_setting(INTENT, "off")
            raise
        await self.store.set_setting(SANDBOX, sid)
        # A successful create is not enough: execute harmless work on the live VM.
        probe = await self._request("POST", "/" + sid + "/exec",
                                    json={"cmd": "python3 -c 'print(19*23)'", "timeout_seconds": 15})
        if probe.status_code != 200 or probe.json().get("exit_code") != 0 or str(probe.json().get("stdout", "")).strip() != "437":
            raise RuntimeError("PandaStack computer was created but its live exec probe failed")
        return await self.status()

    async def start_with_fallback(self, *, allow_lab_fallback: bool = True):
        """Start the production computer; rotate keys, then fall back to a lab-style VM.

        Returns the real status dict. ``fallback`` is true only when the normal
        computer could not be created and the lab VM stood in for it, and
        ``fallback_reason`` always carries the provider's actual error.
        """
        try:
            status = await self.start()
            try:
                await self.store.set_setting(ROLE_SETTING, ROLE_PRIMARY)
                await self.store.set_setting(LAST_ERROR, "")
            except Exception:
                pass
            status.update({"role": ROLE_PRIMARY, "fallback": False, "last_error": ""})
            return status
        except Exception as exc:
            primary_error = str(exc) or type(exc).__name__
            try:
                await self.store.set_setting(LAST_ERROR, primary_error[:500])
            except Exception:
                pass
            if not allow_lab_fallback:
                raise RuntimeError(primary_error) from exc
            # Lab-style stand-in: same provider, base template, internet, but tagged as the computer.
            sid = ""
            try:
                sid = await self.create(role="lab-fallback", templates=("base",),
                                        metadata={"fallback": "true", "primary_error": primary_error[:180]})
            except Exception as lab_exc:
                raise RuntimeError(
                    "No computer could be started. Primary: " + primary_error +
                    ". Lab fallback: " + (str(lab_exc) or type(lab_exc).__name__)) from lab_exc
            await self.store.set_setting(SANDBOX, sid)
            await self.store.set_setting(ROLE_SETTING, "lab-fallback")
            if not await self.probe(sid):
                await self.delete(sid)
                raise RuntimeError("A lab-style VM was created but could not execute work; it was deleted. Primary error: " + primary_error)
            status = await self.status()
            status.update({"role": "lab-fallback", "fallback": True, "fallback_reason": primary_error,
                           "last_error": primary_error})
            return status

    async def stop(self):
        """Power off CPU/network; unsaved VM state survives only until provider TTL."""
        await self.store.set_setting(INTENT, "off")
        owned = await self._owned()
        for item in owned:
            if str(item.get("status") or "").lower() == "hibernated":
                continue
            response = await self._request("POST", "/" + item["id"] + "/hibernate", json={})
            if response.status_code not in (200, 201, 204):
                raise RuntimeError(f"PandaStack hibernate returned HTTP {response.status_code}")
        result = await self.status()
        if result["status"] != "off":
            raise RuntimeError("PandaStack shutdown was not verified")
        return result

    async def destroy(self):
        """Delete only our production sandboxes after an external backup (test cleanup)."""
        await self.store.set_setting(INTENT, "off")
        for item in await self._owned():
            response = await self._request("DELETE", "/" + item["id"])
            if response.status_code not in (204, 404):
                raise RuntimeError(f"PandaStack deletion returned HTTP {response.status_code}")
        if await self._owned():
            raise RuntimeError("Production sandbox still exists")
        await self.store.set_setting(SANDBOX, "")
        return await self.status()
