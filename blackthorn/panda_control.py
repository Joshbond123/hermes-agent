"""Durable PandaStack computer lifecycle, separate from the Kaggle model GPU.

Only a named production sandbox is managed; validation sandboxes are never touched.
The persistent intention and sandbox ID live in the existing D1 settings store.
"""
from __future__ import annotations

import os
import re

import httpx

INTENT = "pandastack_computer_intent"
SANDBOX = "pandastack_computer_sandbox_id"
PURPOSE = "blackthorn-production-computer"
UUID = re.compile(r"^[0-9a-fA-F-]{32,36}$")
BASE = "https://api.pandastack.ai/v1/sandboxes"


class PandaControl:
    def __init__(self, store, http: httpx.AsyncClient, key: str | None = None):
        self.store = store
        self.http = http
        self.key = key if key is not None else os.environ.get("PANDASTACK_API_KEY", "")

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
                        "display_status": "Off · workspace hibernated" if owned else "Off"}
            return {"status": "stop_unverified", "online": False, "host": "PandaStack", "storage": "private GitHub Computer-drive",
                    "error": "Shutdown requested but a PandaStack VM is still active"}
        owned = await self._owned()
        if len(owned) != 1:
            return {"status": "missing" if not owned else "duplicate", "online": False,
                    "host": "PandaStack", "storage": "private GitHub Computer-drive",
                    "error": "Expected exactly one production computer"}
        item = owned[0]
        if sid != item["id"]:
            await self.store.set_setting(SANDBOX, item["id"])
        state = str(item.get("status") or "unknown").lower()
        # Do not auto-wake on a read-only status query: the VM may be hibernated.
        return {"status": state, "online": state in ("running", "hibernated"),
                "host": "PandaStack", "storage": "private GitHub Computer-drive",
                "environment": "Firecracker CPU microVM", "transport": "PandaStack API",
                "display_status": "Ready" if state == "running" else ("Hibernated — wakes on task" if state == "hibernated" else state)}

    async def start(self):
        # Reconcile before create because a lost response could have already
        # created a billable sandbox. Never blindly retry a non-idempotent create.
        await self.store.set_setting(INTENT, "on")
        owned = await self._owned()
        if len(owned) > 1:
            raise RuntimeError("Duplicate PandaStack computers require manual reconciliation")
        if owned:
            sid = owned[0]["id"]
            await self.store.set_setting(SANDBOX, sid)
            if str(owned[0].get("status") or "").lower() == "hibernated":
                woke = await self._request("POST", "/" + sid + "/wake", json={})
                if woke.status_code not in (200, 201, 204):
                    raise RuntimeError(f"PandaStack wake returned HTTP {woke.status_code}")
            probe = await self._request("POST", "/" + sid + "/exec",
                json={"cmd": "python3 -c 'print(19*23)'", "timeout_seconds": 15})
            if probe.status_code != 200 or probe.json().get("exit_code") != 0 or str(probe.json().get("stdout", "")).strip() != "437":
                raise RuntimeError("Existing PandaStack computer did not pass live exec check")
            return await self.status()
        response = await self._request("POST", json={"template": "code-interpreter", "persistent": True,
            "metadata": {"purpose": PURPOSE}})
        if response.status_code != 201:
            await self.store.set_setting(INTENT, "off")
            raise RuntimeError(f"PandaStack create returned HTTP {response.status_code}")
        item = response.json()
        sid = item.get("id", "")
        if not UUID.fullmatch(sid):
            raise RuntimeError("PandaStack create returned no valid sandbox ID; reconcile before retry")
        await self.store.set_setting(SANDBOX, sid)
        # A successful create is not enough: execute harmless work on the live VM.
        probe = await self._request("POST", "/" + sid + "/exec",
                                    json={"cmd": "python3 -c 'print(19*23)'", "timeout_seconds": 15})
        if probe.status_code != 200 or probe.json().get("exit_code") != 0 or str(probe.json().get("stdout", "")).strip() != "437":
            raise RuntimeError("PandaStack computer was created but its live exec probe failed")
        return await self.status()

    async def stop(self):
        """Power off CPU/network, retaining a snapshot of unsaved workspace state."""
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
