"""Where the model lives right now (set by the GPU supervisor, read by the agent)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional


@dataclass(frozen=True)
class Route:
    """Live inference endpoint.  ``url`` has no trailing slash and no ``/v1`` suffix."""

    url: str
    api_key: str
    model: str
    status: str = "ready"


class TunnelDown(RuntimeError):
    """The Cloudflare tunnel / Kaggle gateway is not reachable (HTTP 5xx/530, DNS, refused)."""

    def __init__(self, message: str, *, http_status: int = 0) -> None:
        super().__init__(message)
        self.http_status = http_status


RouteProvider = Callable[[], Awaitable[Optional[Route]]]

#: Statuses at which the gateway answers HTTP but the tunnel/process is gone.
DEAD_HTTP_STATUS = frozenset({404, 410, 502, 503, 521, 522, 523, 524, 530})
