"""Where the model lives right now: the tunnel URL + bearer key the Kaggle notebook publishes to D1."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional

OFFLINE = {"", "OFF", "OFFLINE", "GPU_STOPPED_SAVING_QUOTA", "STOPPED", "STOPPING_KAGGLE_GPU", "ERROR", "FAILED"}
FAILED = {"BOOT_FAILED", "TUNNEL_ERROR", "GPU_UNAVAILABLE", "INTERNET_UNAVAILABLE", "MODEL_CORRUPT"}
BOOTING = {"BOOTING_KAGGLE_GPU", "CHECKING_ENVIRONMENT", "CHECKING_CACHE", "CACHE_HIT", "CACHE_MISS", "DOWNLOADING_MODEL",
           "MODEL_DOWNLOADED", "VERIFYING_MODEL", "INSTALLING_DEPS", "STARTING_INSTALL", "INSTALLING_OLLAMA",
           "STARTING_OLLAMA", "LOADING_MODEL", "STARTING_GATEWAY", "TUNNEL_ONLINE", "WARMING_GPU"}


@dataclass(frozen=True)
class Route:
    url: str
    api_key: str
    model: str
    status: str


class RouteError(RuntimeError):
    """The model endpoint is not usable; ``code`` tells the UI what to offer."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class RouteResolver:
    def __init__(self, store: Any, *, ttl: float = 8.0, default_model: str = "Qwen3.8-27B-Uncensored"):
        self._store = store
        self._ttl = ttl
        self._default_model = default_model
        self._cached: Optional[tuple[float, Route]] = None
        self._lock = asyncio.Lock()

    def invalidate(self) -> None:
        self._cached = None

    async def get(self, *, force: bool = False) -> Route:
        now = time.monotonic()
        if not force and self._cached and now - self._cached[0] < self._ttl:
            return self._cached[1]
        async with self._lock:
            now = time.monotonic()
            if not force and self._cached and now - self._cached[0] < self._ttl:
                return self._cached[1]
            row = await self._store.gpu_row()
            status = str(row.get("status") or "").upper()
            url = str(row.get("tunnel_url") or "").rstrip("/")
            if url.endswith("/v1"):
                url = url[:-3]
            key = str(row.get("api_key") or "")
            if status in BOOTING and not url:
                raise RouteError("gpu_booting", "The GPU is still starting up. Wait for the GPU button to show Ready, then send again.")
            if status in FAILED:
                raise RouteError("gpu_error", f"The GPU is not usable ({status.replace('_', ' ').lower()}). Open the GPU panel and turn it on again.")
            if status in OFFLINE or not url:
                raise RouteError("gpu_off", "The GPU is off. Turn it on from the GPU button in the header, wait until it is Ready, then send again.")
            if not key:
                raise RouteError("gpu_error", "The GPU endpoint has no access key yet. Wait for the GPU to finish starting.")
            route = Route(url=url, api_key=key, model=str(row.get("model") or self._default_model), status=status)
            self._cached = (now, route)
            return route
