"""Where the model lives: NVIDIA's hosted Nemotron endpoint, configured by environment.

The model used to be reached through a per-boot tunnel that a GPU notebook published into D1;
that whole indirection is gone. NVIDIA serves an OpenAI-compatible API at a fixed base URL, so a
route is simply the base URL, the API key and the model id — there is nothing to poll, nothing to
publish, and no tunnel to die.

Kept deliberately small and dependency-free: the agent, the relay-free tool layer and the UI all
read the same ``Route`` object, and every caller already treats it as immutable.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger("blackthorn.route")

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
# Nemotron 3 Ultra is documented at a 1M-token window; the agent's budget is deliberately far
# below it so a long tool-heavy run stays cheap, and overflow is still handled gracefully.
DEFAULT_CONTEXT_TOKENS = 131_072


@dataclass(frozen=True)
class Route:
    url: str
    api_key: str
    model: str
    status: str = "ONLINE"
    provider: str = "nvidia"
    label: str = ""          # what the UI calls this model
    key: str = ""            # the catalogue id the user picked (defaults to the wire model id)


class RouteError(RuntimeError):
    """The model endpoint is not usable; ``code`` tells the UI what to show."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def context_tokens(default: int = DEFAULT_CONTEXT_TOKENS) -> int:
    """The context budget the agent plans with (the provider enforces its own limit)."""
    for name in ("BLACKTHORN_CONTEXT_TOKENS", "BLACKTHORN_MODEL_CONTEXT"):
        raw = _env(name)
        if raw:
            try:
                value = int(float(raw))
            except ValueError:
                continue
            if value > 1024:
                return value
    return default


# ==================================================================================================
# The models a conversation may use
# --------------------------------------------------------------------------------------------------
# One entry per model this deployment can actually serve. An entry whose credentials are absent is
# still listed — the picker then shows *why* it cannot be used instead of offering a button that
# fails. Nothing here is invented: each option maps to a real endpoint configured by environment.
# ==================================================================================================
NVIDIA_MODEL_ID = "nvidia/nemotron-3-ultra-550b-a55b"
QWEN_MODEL_ID = "orcarouter/Qwen3.8-27B-Uncensored"
QWEN_LABEL = "Qwen3.8 27B Uncensored"


@dataclass(frozen=True)
class ModelOption:
    key: str                 # what the session stores and the UI sends back
    label: str
    provider: str
    model: str               # wire model id
    base_url: str
    api_key: str
    detail: str = ""

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key)

    @property
    def reason(self) -> str:
        if not self.base_url:
            return "no endpoint URL is configured for this model on the server"
        if not self.api_key:
            return "no API key is configured for this provider on the server"
        return ""

    def as_dict(self, *, active: bool = False) -> dict:
        return {"id": self.key, "label": self.label, "provider": self.provider, "model": self.model,
                "available": self.available, "reason": self.reason, "detail": self.detail,
                "active": active}


def _env_or(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def model_options(default_model: str = DEFAULT_MODEL) -> List[ModelOption]:
    """Every model this deployment knows about, in the order the picker shows them."""
    nvidia = ModelOption(
        key=_env_or("BLACKTHORN_MODEL", default_model) or default_model,
        label="Nemotron 3 Ultra",
        provider="nvidia",
        model=_env_or("BLACKTHORN_MODEL", default_model) or default_model,
        base_url=(_env_or("NVIDIA_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).rstrip("/"),
        api_key=_env_or("NVIDIA_API_KEY"),
        detail="NVIDIA hosted API · long context · tool calling",
    )
    # A self-hosted Qwen is served by llama.cpp over an OpenAI-compatible API; it needs a base URL
    # (typically the notebook's tunnel) and, for llamacpp, no API key is required but one is accepted.
    qwen_url = (_env_or("QWEN_BASE_URL") or _env_or("QWEN38_BASE_URL") or "").rstrip("/")
    qwen_key = _env_or("QWEN_API_KEY") or _env_or("QWEN38_API_KEY") or "sk-local"
    qwen = ModelOption(
        key=_env_or("QWEN_MODEL", QWEN_MODEL_ID) or QWEN_MODEL_ID,
        label=QWEN_LABEL,
        provider="llamacpp" if qwen_url else "openai",
        model=_env_or("QWEN_MODEL", QWEN_MODEL_ID) or QWEN_MODEL_ID,
        base_url=qwen_url,
        api_key=qwen_key if qwen_url else "",
        detail="Q4_K_M · served by llama.cpp · tool calling",
    )
    return [nvidia, qwen]


class RouteResolver:
    """The NVIDIA route, or an honest error that says exactly what is missing."""

    def __init__(self, store: object = None, *, ttl: float = 8.0, default_model: str = DEFAULT_MODEL):
        # ``store`` is accepted for backwards compatibility with the D1-backed resolver; the
        # NVIDIA route needs no database row.
        self._store = store
        self._ttl = ttl
        self._default_model = default_model
        self._cached: Optional[tuple[float, Route]] = None

    # -- provider facts -------------------------------------------------------
    @property
    def provider(self) -> str:
        return "nvidia"

    def base_url(self) -> str:
        return (_env("NVIDIA_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).rstrip("/")

    def model_id(self) -> str:
        return _env("BLACKTHORN_MODEL", self._default_model) or self._default_model

    def api_key(self) -> str:
        return _env("NVIDIA_API_KEY")

    def invalidate(self) -> None:
        self._cached = None

    def describe(self) -> dict:
        """Non-secret description used by the health endpoint and the UI."""
        return {"provider": "nvidia", "model": self.model_id(), "base_url": self.base_url(),
                "configured": bool(self.api_key()), "context_tokens": context_tokens()}

    def options(self) -> List[ModelOption]:
        return model_options(self._default_model)

    def find(self, key: str = "") -> Optional[ModelOption]:
        """The catalogue entry for ``key`` (falls back to the configured default model)."""
        wanted = (key or "").strip()
        options = self.options()
        if not wanted:
            return options[0]
        for option in options:
            if wanted in (option.key, option.model, option.label):
                return option
        return None

    def describe_all(self, *, active: str = "") -> dict:
        options = self.options()
        chosen = (active or "").strip() or options[0].key
        return {"models": [o.as_dict(active=o.key == chosen or o.model == chosen) for o in options],
                "active": chosen,
                "default": options[0].key}

    async def get(self, *, force: bool = False, model: str = "") -> Route:
        wanted = (model or "").strip()
        if wanted:
            option = self.find(wanted)
            if option is None:
                # An unknown id is a client error, not a reason to silently serve a different model.
                raise RouteError("model_unknown", f"The model “{wanted}” is not configured on this server.")
            if not option.available:
                raise RouteError("model_unconfigured",
                                 f"{option.label} is not available: {option.reason}.")
            return Route(url=option.base_url, api_key=option.api_key, model=option.model,
                         status="ONLINE", provider=option.provider, label=option.label, key=option.key)
        now = time.monotonic()
        if not force and self._cached and (now - self._cached[0]) < self._ttl:
            return self._cached[1]
        key = self.api_key()
        if not key:
            raise RouteError("model_unconfigured",
                             "No NVIDIA API key is configured on the server, so the model cannot answer.")
        route = Route(url=self.base_url(), api_key=key, model=self.model_id(), status="ONLINE",
                      provider="nvidia", label="Nemotron 3 Ultra", key=self.model_id())
        self._cached = (now, route)
        return route

    async def get_computer(self, *, force: bool = False) -> Route:
        """Deprecated: the computer is no longer served by the model host.

        Kept so an older caller gets a clear error instead of an attribute crash.
        """
        raise RouteError("computer_moved",
                         "The computer is a GitHub Codespace now; it is not reached through the model route.")


async def probe(client, route: Route, timeout: float = 15.0) -> dict:
    """One authenticated round-trip that proves the route works right now."""
    started = time.monotonic()
    try:
        resp = await client.get(f"{route.url}/v1/models",
                                headers={"Authorization": f"Bearer {route.api_key}"},
                                timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                "latency_ms": int((time.monotonic() - started) * 1000)}
    latency = int((time.monotonic() - started) * 1000)
    if resp.status_code != 200:
        return {"ok": False, "status": resp.status_code,
                "error": str(resp.text)[:200], "latency_ms": latency}
    try:
        ids = [m.get("id") for m in (resp.json().get("data") or [])]
    except ValueError:
        ids = []
    return {"ok": True, "status": 200, "latency_ms": latency, "models": len(ids),
            "model_available": (route.model in ids) if ids else None}
