"""GPU control facade: honest, secret-free status + the turn-on / turn-off / auto-off actions.

The heavy lifting (Kaggle API, tunnel health, watchdog, quota) stays in ``cloudflare_d1_client``; this module
adds what the UI needs to tell the truth: the real boot stage with elapsed time, real byte progress when the
notebook reports it, and which actions are currently sensible.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Optional

# Ordered boot stages the notebook really reports (status → stage number). The UI shows "Step n of N" and the
# stage label — never a made-up percentage.
STAGE_OF = {
    "BOOTING_KAGGLE_GPU": 1, "CHECKING_ENVIRONMENT": 2, "STARTING_INSTALL": 3, "INSTALLING_DEPS": 3, "CHECKING_CACHE": 4,
    "CACHE_MISS": 5, "DOWNLOADING_MODEL": 5, "CACHE_HIT": 6, "MODEL_DOWNLOADED": 6, "VERIFYING_MODEL": 6,
    "INSTALLING_OLLAMA": 7, "STARTING_OLLAMA": 8, "LOADING_MODEL": 9, "STARTING_GATEWAY": 10, "TUNNEL_ONLINE": 11,
    "WARMING_GPU": 12,
}
TOTAL_STAGES = 12
READY = {"ONLINE", "MODEL_READY_AND_WARMED", "HEARTBEAT_ONLINE", "MODEL_READY", "MODEL_READY_COLD"}
BOOTING = {"BOOTING_KAGGLE_GPU", "CHECKING_ENVIRONMENT", "CHECKING_CACHE", "CACHE_HIT", "CACHE_MISS", "DOWNLOADING_MODEL",
           "MODEL_DOWNLOADED", "VERIFYING_MODEL", "INSTALLING_DEPS", "STARTING_INSTALL", "INSTALLING_OLLAMA",
           "STARTING_OLLAMA", "LOADING_MODEL", "STARTING_GATEWAY", "TUNNEL_ONLINE", "WARMING_GPU"}
BOOT_META_KEY = "blackthorn_gpu_boot_started"


def _parse_detail(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


class GpuService:
    def __init__(self, store: Any = None, d1_module: Any = None):
        self._store = store
        self._d1 = d1_module

    def _module(self) -> Any:
        if self._d1 is None:
            import cloudflare_d1_client as module  # imported lazily: heavy, and absent in some dev setups
            self._d1 = module
        return self._d1

    async def _boot_started(self) -> Optional[float]:
        if self._store is None:
            return None
        try:
            value = await self._store.get_setting(BOOT_META_KEY)
            return float(value) if value else None
        except Exception:
            return None

    async def _augment(self, state: Dict[str, Any]) -> Dict[str, Any]:
        d1 = self._module()
        pub = d1.public_gpu_status(state)
        status = str(state.get("status") or "").upper()
        booting = bool(state.get("booting"))
        pub.pop("progress_pct", None)  # a label→percentage table is not progress; stages + bytes below are
        stage = STAGE_OF.get(status)
        if booting and stage:
            pub.update(progress_stage=stage, progress_total_stages=TOTAL_STAGES, progress_kind="stage")
        detail = _parse_detail(state.get("detail"))
        done, total = detail.get("bytes_done"), detail.get("bytes_total")
        if booting and isinstance(done, (int, float)) and isinstance(total, (int, float)) and total > 0:
            pub.update(progress_bytes_done=int(done), progress_bytes_total=int(total), progress_kind="bytes")
            if detail.get("label"):
                pub["progress_label"] = str(detail["label"])[:80]
        if booting and detail.get("stalled"):
            pub["progress_stalled"] = True
        if booting:
            started = await self._boot_started()
            if started:
                pub["elapsed_seconds"] = max(0, int(time.time() - started))
        if status in READY and state.get("model_loaded") is False:
            pub["display_status"] = "Loading Model"
        info = str(state.get("gpu_info") or "")
        if status in ("BOOT_FAILED", "GPU_UNAVAILABLE", "INTERNET_UNAVAILABLE", "MODEL_CORRUPT", "TUNNEL_ERROR") or info.startswith("FAILED"):
            pub["error"] = info[:240] if info.startswith("FAILED") else str(state.get("progress_step") or status).replace("_", " ").lower()
        online = bool(pub.get("online"))
        pub["can_turn_on"] = not online and not booting and status != "STOPPING_KAGGLE_GPU"
        pub["can_turn_off"] = online or booting
        pub["server_time"] = time.time()
        return pub

    # -- activity: tells the controller real work is happening, so inactivity auto-off never fires mid-conversation -------
    def run_started(self) -> None:
        try:
            d1 = self._module()
            d1.task_started("chat")
        except Exception:
            pass

    def run_touched(self, reason: str = "tool") -> None:
        try:
            self._module().mark_activity(reason)
        except Exception:
            pass

    def run_finished(self) -> None:
        try:
            d1 = self._module()
            d1.task_finished("chat")
            d1.mark_activity("chat-done")
        except Exception:
            pass

    async def status(self, refresh: bool = False) -> Dict[str, Any]:
        d1 = self._module()
        model_pub = await self._augment(await asyncio.to_thread(d1.get_kaggle_gpu_status, refresh))
        # Attach computer-role health + fleet quotas so the UI can show both systems.
        try:
            crow = await self._store.computer_row() if self._store is not None else {}
            if crow:
                c_status = str(crow.get("status") or "").upper()
                c_url = str(crow.get("tunnel_url") or "").strip()
                c_online = c_status in READY and bool(c_url)
                c_booting = c_status in BOOTING or bool(crow.get("booting"))
                computer_pub = {
                    "status": c_status,
                    "online": c_online,
                    "booting": c_booting,
                    "has_endpoint": bool(c_url),
                    "display_status": (
                        "Computer Ready" if c_online else
                        ("Computer Starting" if c_booting else "Computer Offline")
                    ),
                    "gpu_info": crow.get("gpu_info") or "",
                    "model": crow.get("model") or "",
                }
                if c_status and c_status not in ("", "OFF", "OFFLINE"):
                    model_online = bool(model_pub.get("online"))
                    if model_online and not c_online and not c_booting:
                        model_pub["display_status"] = "Model ready — computer offline"
                        model_pub["computer_ready"] = False
                    elif model_online and c_booting:
                        model_pub["display_status"] = "Waiting for computer GPU"
                        model_pub["computer_ready"] = False
                        model_pub["online"] = False
                        model_pub["can_turn_on"] = False
                    elif model_online and c_online:
                        model_pub["display_status"] = "Model + Computer Ready"
                        model_pub["computer_ready"] = True
                    model_pub["computer"] = computer_pub
        except Exception:
            pass
        try:
            import os
            from .fleet import accounts_from_env, FleetManager
            accounts = accounts_from_env(os.environ)
            if accounts:
                fleet = FleetManager(accounts)
                quotas = {}
                for role in ("model", "computer"):
                    acc = next((a for a in accounts if a.role == role and getattr(a, "slot", "primary") in ("primary", None, "")), None)
                    if acc is None:
                        acc = next((a for a in accounts if a.role == role), None)
                    if acc is None:
                        continue
                    try:
                        q = fleet.quota(acc)
                        quotas[role] = {
                            "account": acc.user,
                            "used_hours": round(float(q.get("used_seconds") or 0) / 3600.0, 2),
                            "total_hours": round(float(q.get("total_seconds") or 108000) / 3600.0, 2),
                            "remaining_hours": round(max(0.0, float(q.get("total_seconds") or 108000) - float(q.get("used_seconds") or 0)) / 3600.0, 2),
                            "used_pct": round(100.0 * float(q.get("used_seconds") or 0) / max(1.0, float(q.get("total_seconds") or 108000)), 1),
                            "refresh_time": q.get("refresh_time") or "",
                        }
                    except Exception:
                        continue
                if quotas:
                    model_pub["quotas"] = quotas
        except Exception:
            pass
        return model_pub

    async def turn_on(self) -> Dict[str, Any]:
        """Power on the model and computer GPUs from one action. Returns the real status of both roles."""
        d1 = self._module()
        import os
        from .fleet import accounts_from_env
        if hasattr(d1, "turn_on_fleet") and accounts_from_env(os.environ):
            # Errors propagate: a failed fleet start must not silently fall back to a different code path.
            out = await asyncio.to_thread(d1.turn_on_fleet)
            st = await self.status(refresh=True)
            st["fleet"] = out
            return st
        return await self._augment(await asyncio.to_thread(d1.turn_on_kaggle_gpu))

    async def turn_off(self) -> Dict[str, Any]:
        """Power off both GPUs. ``turn_off_ok`` is true only when every role was verified stopped."""
        d1 = self._module()
        import os
        from .fleet import accounts_from_env
        if hasattr(d1, "turn_off_fleet") and accounts_from_env(os.environ):
            out = await asyncio.to_thread(d1.turn_off_fleet)
            st = await self.status(refresh=True)
            failed = {role: v for role, v in out.items() if isinstance(v, dict) and v.get("status") == "stop_failed"}
            st["fleet"] = out
            st["turn_off_ok"] = not failed
            if failed:
                reasons = "; ".join(f"{role}: {v.get('error') or 'stop refused'}" for role, v in failed.items())
                st["error"] = f"Turn-off did not complete. Kaggle still reports the session running ({reasons}). Stop it on kaggle.com or give the token kernels.delete."
            return st
        st = await self._augment(await asyncio.to_thread(d1.turn_off_kaggle_gpu))
        st["turn_off_ok"] = True
        return st

    async def activity(self) -> Dict[str, Any]:
        d1 = self._module()
        snap = await asyncio.to_thread(d1.activity_snapshot)
        decision = await asyncio.to_thread(d1.auto_off_decision)
        minutes = await asyncio.to_thread(d1.get_auto_off_minutes)
        return {"activity": snap, "auto_off": decision, "auto_off_minutes": minutes, "choices": [0, 5, 10, 15, 30, 60]}

    async def set_auto_off(self, minutes: int) -> Dict[str, Any]:
        d1 = self._module()
        applied = await asyncio.to_thread(d1.set_auto_off_minutes, int(minutes))
        out = await self.activity()
        out.update(ok=True, applied=applied)
        return out

    async def logs(self, limit: int = 120) -> Dict[str, Any]:
        import httpx
        d1 = self._module()
        state = await asyncio.to_thread(d1.get_kaggle_gpu_status, False)
        url = str(state.get("tunnel_url") or "").rstrip("/")
        if not url:
            return {"lines": [], "error": "The GPU is offline"}
        key = await asyncio.to_thread(d1._gateway_api_key)
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"{url}/logs", headers={"Authorization": f"Bearer {key}"})
            lines = list(resp.json().get("lines") or [])
        except Exception as exc:
            return {"lines": [], "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
        from .tools.base import redact
        return {"lines": [redact(str(x))[:300] for x in lines[-max(1, min(int(limit), 300)):]]}
