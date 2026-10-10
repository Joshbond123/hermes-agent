"""Tiny async SQL layer with two interchangeable executors.

* ``D1Executor``   — Cloudflare D1 over its REST API (production).
* ``SqliteExecutor`` — stdlib sqlite3 in a worker thread (tests / local development).

Both speak the same dialect subset, so the store's SQL is written once.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

log = logging.getLogger("blackthorn.sql")

Statement = Tuple[str, Sequence[Any]]


@dataclass
class Result:
    rows: List[Dict[str, Any]] = field(default_factory=list)
    last_row_id: int = 0
    changes: int = 0


class SqlError(RuntimeError):
    pass


class D1Executor:
    """Cloudflare D1 REST executor with a pooled keep-alive client and bounded retries."""

    def __init__(self, account_id: str, database_id: str, token: str, *, timeout: float = 25.0,
                 client: Optional[httpx.AsyncClient] = None, base_url: str = "https://api.cloudflare.com/client/v4"):
        if not (account_id and database_id and token):
            raise SqlError("Cloudflare D1 credentials are not configured (CLOUDFLARE_* environment variables)")
        self._url = f"{base_url}/accounts/{account_id}/d1/database/{database_id}/query"
        self._token = token
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=8, keepalive_expiry=30.0),
            )
            self._owns_client = True
        return self._client

    async def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        last: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = await self._http().post(
                    self._url, json=body,
                    headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise SqlError(f"D1 HTTP {resp.status_code}")
                if resp.status_code >= 400:
                    detail = resp.text[:300]
                    raise SqlError(f"D1 HTTP {resp.status_code}: {detail}")
                data = resp.json()
                if not data.get("success"):
                    raise SqlError(f"D1 error: {data.get('errors')}")
                return data
            except (httpx.TransportError, SqlError) as exc:
                last = exc
                transient = isinstance(exc, httpx.TransportError) or "HTTP 4" not in str(exc) or "HTTP 429" in str(exc)
                if attempt == 2 or not transient:
                    break
                await asyncio.sleep(0.4 * (attempt + 1))
        raise SqlError(str(last) if last else "D1 request failed")

    @staticmethod
    def _result(entry: Dict[str, Any]) -> Result:
        meta = entry.get("meta") or {}
        return Result(rows=list(entry.get("results") or []), last_row_id=int(meta.get("last_row_id") or 0),
                      changes=int(meta.get("changes") or 0))

    async def query(self, sql: str, params: Sequence[Any] = ()) -> Result:
        data = await self._post({"sql": sql, "params": list(params)})
        results = data.get("result") or []
        return self._result(results[0]) if results else Result()

    async def batch(self, statements: Sequence[Statement]) -> List[Result]:
        if not statements:
            return []
        body = {"batch": [{"sql": s, "params": list(p)} for s, p in statements]}
        data = await self._post(body)
        return [self._result(r) for r in (data.get("result") or [])]

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client and not self._client.is_closed:
            await self._client.aclose()


class SqliteExecutor:
    """sqlite3 behind a lock, executed in a thread so the event loop never blocks."""

    def __init__(self, path: str = ":memory:"):
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()

    def _run(self, statements: Sequence[Statement], transactional: bool) -> List[Result]:
        out: List[Result] = []
        with self._lock:
            try:
                if transactional:
                    self._conn.execute("BEGIN")
                for sql, params in statements:
                    cur = self._conn.execute(sql, list(params))
                    rows = [dict(r) for r in cur.fetchall()] if cur.description else []
                    out.append(Result(rows=rows, last_row_id=cur.lastrowid or 0, changes=max(cur.rowcount, 0)))
                if transactional:
                    self._conn.execute("COMMIT")
            except Exception as exc:
                if transactional:
                    try:
                        self._conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                raise SqlError(str(exc)) from exc
        return out

    async def query(self, sql: str, params: Sequence[Any] = ()) -> Result:
        return (await asyncio.to_thread(self._run, [(sql, params)], False))[0]

    async def batch(self, statements: Sequence[Statement]) -> List[Result]:
        if not statements:
            return []
        return await asyncio.to_thread(self._run, list(statements), True)

    async def aclose(self) -> None:
        with self._lock:
            self._conn.close()


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
