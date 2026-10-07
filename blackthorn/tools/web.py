"""Live web search through Tavily, with key rotation and failover (keys never leave the server)."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .base import ToolContext, ToolError, ToolResult, ToolSpec, clip

QUOTA_STATUSES = (401, 403, 429, 432, 433)


def _parse_keys(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = data.get("keys") or []
        if isinstance(data, list):
            return [str(k).strip() for k in data if str(k).strip()]
    except ValueError:
        pass
    return [k.strip() for k in raw.replace("\n", ",").split(",") if k.strip()]


class TavilyKeys:
    """Round-robin over the configured keys; a key that reports quota/auth trouble is benched for a while."""

    def __init__(self, store: Any = None, env: Optional[Dict[str, str]] = None, ttl: float = 300.0):
        self._store = store
        self._env = env if env is not None else os.environ
        self._ttl = ttl
        self._keys: List[str] = []
        self._loaded_at = 0.0
        self._cursor = 0
        self._bench: Dict[str, float] = {}

    async def _load(self) -> List[str]:
        now = time.monotonic()
        if self._keys and now - self._loaded_at < self._ttl:
            return self._keys
        keys: List[str] = []
        for name in ("TAVILY_API_KEYS", "TAVILY_API_KEY", "TAVILY_KEYS"):
            keys += _parse_keys(self._env.get(name, ""))
        if self._store is not None:
            try:
                keys += _parse_keys(await self._store.get_setting("tavily_api_keys"))
            except Exception:
                pass
        seen, unique = set(), []
        for k in keys:
            if k not in seen:
                seen.add(k)
                unique.append(k)
        self._keys, self._loaded_at = unique, now
        return unique

    async def order(self) -> List[str]:
        keys = await self._load()
        if not keys:
            return []
        now = time.monotonic()
        start = self._cursor % len(keys)
        self._cursor += 1
        rotated = keys[start:] + keys[:start]
        healthy = [k for k in rotated if self._bench.get(k, 0) <= now]
        return healthy or rotated  # everything benched: try anyway rather than fail blind

    def bench(self, key: str, seconds: float = 600.0) -> None:
        self._bench[key] = time.monotonic() + seconds


def _domain(url: str) -> str:
    return (urlparse(url).netloc or url).removeprefix("www.")


async def web_search(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = args["query"].strip()
    order = await ctx.tavily.order()
    if not order:
        raise ToolError("config", "Web search is not configured (no Tavily API key).")
    body = {"query": query, "max_results": int(args.get("max_results") or 5), "search_depth": "basic",
            "topic": args.get("topic") or "general", "include_answer": False}
    last = "no response"
    for key in order:
        try:
            resp = await ctx.http.post("https://api.tavily.com/search", json=body, timeout=22.0,
                                       headers={"Authorization": f"Bearer {key}"})
        except Exception as exc:
            last = f"{type(exc).__name__}"
            continue
        if resp.status_code in QUOTA_STATUSES:
            ctx.tavily.bench(key)
            last = f"HTTP {resp.status_code} (key limit reached)"
            continue
        if resp.status_code >= 400:
            last = f"HTTP {resp.status_code}"
            continue
        try:
            results = (resp.json() or {}).get("results") or []
        except ValueError:
            last = "invalid response"
            continue
        if not results:
            return ToolResult(ok=True, content=f'No results for "{query}".', summary="no results", data={"sources": []})
        blocks, sources = [], []
        for i, r in enumerate(results[: body["max_results"]], 1):
            title = (r.get("title") or "Untitled").strip()
            url = (r.get("url") or "").strip()
            snippet, _ = clip(" ".join(str(r.get("content") or "").split()), 330)
            blocks.append(f"[{i}] {title} — {_domain(url)}\n{url}\n{snippet}")
            sources.append({"title": title[:120], "url": url, "domain": _domain(url)})
        text, cut = clip(f'Search results for "{query}":\n\n' + "\n\n".join(blocks), ctx.settings.tool_result_chars)
        names = ", ".join(dict.fromkeys(s["domain"] for s in sources[:3]))
        return ToolResult(ok=True, content=text, summary=f"{len(sources)} results · {names}", truncated=cut,
                          data={"sources": sources})
    raise ToolError("search_failed", f"Web search failed on every configured key ({last}).",
                    hint="Tell the user the search is unavailable right now instead of guessing.")


SCHEMA = {"type": "object", "properties": {
    "query": {"type": "string", "minLength": 2, "maxLength": 300, "description": "query"},
    "max_results": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5},
    "topic": {"type": "string", "enum": ["general", "news"], "default": "general"}}, "required": ["query"]}


def specs() -> List[ToolSpec]:
    return [ToolSpec(
        "web_search",
        "Search the live web for current events, recent facts, prices, weather — anything newer than your training. "
        "Not for general knowledge.",
        SCHEMA, web_search, timeout=75.0, kind="net", describe=lambda a: str(a.get("query") or ""))]
