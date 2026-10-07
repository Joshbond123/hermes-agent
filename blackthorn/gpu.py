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
        return await self._augment(await asyncio.to_thread(d1.get_kaggle_gpu_status, refresh))

    async def turn_on(self) -> Dict[str, Any]:
        d1 = self._module()
        return await self._augment(await asyncio.to_thread(d1.turn_on_kaggle_gpu))

    async def turn_off(self) -> Dict[str, Any]:
        d1 = self._module()
        return await self._augment(await asyncio.to_thread(d1.turn_off_kaggle_gpu))

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
