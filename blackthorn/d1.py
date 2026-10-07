"""Cloudflare D1 access (REST API) — async + sync, pooled, with batching.

D1 is served from one region and Render from another, so every round trip costs
hundreds of milliseconds.  Callers should therefore

* use :meth:`D1Client.abatch` to fold several statements into **one** request, and
* never block the first byte of a streaming response on a D1 call.

All SQL uses bound parameters; never format user text into SQL.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import httpx

from blackthorn import config

log = logging.getLogger("blackthorn.d1")

Statement = Tuple[str, Optional[Sequence[Any]]]

_RETRY_STATUS = {429, 500, 502, 503, 504}


class D1Error(RuntimeError):
    """A D1 request failed (transport, HTTP status or SQL error)."""

    def __init__(self, message: str, *, status: int = 0, errors: Optional[list] = None) -> None:
        super().__init__(message)
        self.status = status
        self.errors = errors or []


class D1Result(NamedTuple):
    rows: List[Dict[str, Any]]
    meta: Dict[str, Any]

    @property
    def last_row_id(self) -> Optional[int]:
        value = self.meta.get("last_row_id")
        return int(value) if value is not None else None

    @property
    def changes(self) -> int:
        return int(self.meta.get("changes") or 0)


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {config.cloudflare_token()}",
        "Content-Type": "application/json",
        "User-Agent": "blackthorn/1.0",
    }


def _payload(statements: Sequence[Statement]) -> Dict[str, Any]:
    if len(statements) == 1:
        sql, params = statements[0]
        body: Dict[str, Any] = {"sql": sql}
        if params:
            body["params"] = list(params)
        return body
    return {
        "batch": [
            ({"sql": sql, "params": list(params)} if params else {"sql": sql})
            for sql, params in statements
        ]
    }


def _parse(data: Any, expected: int) -> List[D1Result]:
    if not isinstance(data, dict):
        raise D1Error("Cloudflare D1 returned a non-object response")
    if not data.get("success", False):
        raise D1Error(f"Cloudflare D1 error: {data.get('errors')}", errors=data.get("errors"))
    results = data.get("result") or []
    out: List[D1Result] = []
    for item in results:
        if isinstance(item, dict):
            if item.get("success") is False:
                raise D1Error(f"Cloudflare D1 statement failed: {item.get('error') or item}")
            out.append(D1Result(item.get("results") or [], item.get("meta") or {}))
    while len(out) < expected:
        out.append(D1Result([], {}))
    return out


class D1Client:
    """Thin wrapper; instantiate once (module level :data:`d1`)."""

    def __init__(self) -> None:
        self._async: Optional[httpx.AsyncClient] = None
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._sync: Optional[httpx.Client] = None
        self._lock = threading.Lock()

    # -- clients -----------------------------------------------------------
    def _aclient(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._async is None or self._async.is_closed or self._async_loop is not loop:
            self._async = httpx.AsyncClient(
                timeout=httpx.Timeout(25.0, connect=8.0),
                limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=120.0),
            )
            self._async_loop = loop
        return self._async

    def _sclient(self) -> httpx.Client:
        with self._lock:
            if self._sync is None or self._sync.is_closed:
                self._sync = httpx.Client(
                    timeout=httpx.Timeout(25.0, connect=8.0),
                    limits=httpx.Limits(max_keepalive_connections=6, keepalive_expiry=120.0),
                )
            return self._sync

    async def aclose(self) -> None:
        if self._async is not None and not self._async.is_closed:
            await self._async.aclose()

    # -- async API ---------------------------------------------------------
    async def abatch(
        self, statements: Sequence[Statement], *, timeout: float = 25.0, retries: int = 2
    ) -> List[D1Result]:
        body = _payload(statements)
        url = config.d1_query_url()
        last: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = await self._aclient().post(url, json=body, headers=_headers(), timeout=timeout)
                if resp.status_code in _RETRY_STATUS and attempt < retries:
                    await asyncio.sleep(0.3 * (attempt + 1))
                    continue
                if resp.status_code >= 400:
                    raise D1Error(
                        f"Cloudflare D1 HTTP {resp.status_code}: {resp.text[:200]}", status=resp.status_code
                    )
                return _parse(resp.json(), len(statements))
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
                if attempt < retries:
                    await asyncio.sleep(0.3 * (attempt + 1))
                    continue
        raise D1Error(f"Cloudflare D1 unreachable: {type(last).__name__}: {last}")

    async def aexec(self, sql: str, params: Optional[Sequence[Any]] = None, **kw: Any) -> D1Result:
        return (await self.abatch([(sql, params)], **kw))[0]

    async def aquery(self, sql: str, params: Optional[Sequence[Any]] = None, **kw: Any) -> List[Dict[str, Any]]:
        return (await self.aexec(sql, params, **kw)).rows

    # -- sync API (worker threads) ----------------------------------------
    def batch(self, statements: Sequence[Statement], *, timeout: float = 25.0, retries: int = 2) -> List[D1Result]:
        body = _payload(statements)
        url = config.d1_query_url()
        last: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = self._sclient().post(url, json=body, headers=_headers(), timeout=timeout)
                if resp.status_code in _RETRY_STATUS and attempt < retries:
                    time.sleep(0.3 * (attempt + 1))
                    continue
                if resp.status_code >= 400:
                    raise D1Error(
                        f"Cloudflare D1 HTTP {resp.status_code}: {resp.text[:200]}", status=resp.status_code
                    )
                return _parse(resp.json(), len(statements))
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
                if attempt < retries:
                    time.sleep(0.3 * (attempt + 1))
                    continue
        raise D1Error(f"Cloudflare D1 unreachable: {type(last).__name__}: {last}")

    def exec(self, sql: str, params: Optional[Sequence[Any]] = None, **kw: Any) -> D1Result:
        return self.batch([(sql, params)], **kw)[0]

    def query(self, sql: str, params: Optional[Sequence[Any]] = None, **kw: Any) -> List[Dict[str, Any]]:
        return self.exec(sql, params, **kw).rows


#: process-wide client
d1 = D1Client()


# --------------------------------------------------------------------------- #
# key/value helpers on ``state_meta`` (used for the system prompt, settings, keys)
# --------------------------------------------------------------------------- #
_UPSERT_META = (
    "INSERT INTO state_meta (key, value) VALUES (?, ?) "
    "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
)


async def aget_meta(key: str, default: str = "") -> str:
    rows = await d1.aquery("SELECT value FROM state_meta WHERE key = ? LIMIT 1", [key])
    return str(rows[0].get("value") or "") if rows else default


async def aset_meta(key: str, value: str) -> None:
    await d1.aexec(_UPSERT_META, [key, value])


def get_meta(key: str, default: str = "") -> str:
    rows = d1.query("SELECT value FROM state_meta WHERE key = ? LIMIT 1", [key])
    return str(rows[0].get("value") or "") if rows else default


def set_meta(key: str, value: str) -> None:
    d1.exec(_UPSERT_META, [key, value])
