import asyncio
import json

import httpx
import pytest

from blackthorn.tools import ToolContext, ToolRegistry, ToolResult, ToolSpec, clip, default_registry, loads_args, redact, validate_args
from blackthorn.tools.web import TavilyKeys, _parse_keys, web_search
from blackthorn.config import Settings

SCHEMA = {"type": "object", "properties": {
    "s": {"type": "string", "minLength": 1, "maxLength": 5}, "n": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
    "b": {"type": "boolean"}, "e": {"type": "string", "enum": ["a", "b"]}}, "required": ["s"]}


def test_validate_coerces_clamps_and_defaults():
    assert validate_args(SCHEMA, {"s": "ok", "n": "7", "b": "true"}) == {"s": "ok", "n": 7, "b": True}
    assert validate_args(SCHEMA, {"s": "ok", "n": 99})["n"] == 10          # clamped to the schema maximum
    assert validate_args(SCHEMA, {"s": "ok"})["n"] == 3                    # default applied
    assert validate_args(SCHEMA, {"s": 12})["s"] == "12"
    assert validate_args(SCHEMA, {"s": "ok", "extra": 1}) == {"s": "ok", "n": 3}   # unknown keys ignored


@pytest.mark.parametrize("args,msg", [
    ({}, "missing required argument 's'"), ({"s": ""}, "must not be empty"), ({"s": "toolong"}, "too long"),
    ({"s": "x", "n": "abc"}, "must be a number"), ({"s": "x", "n": 1.5}, "whole number"), ({"s": "x", "e": "z"}, "must be one of"),
    ({"s": "x", "b": "maybe"}, "true or false"), ([], "JSON object"),
])
def test_validate_rejects(args, msg):
    with pytest.raises(ValueError, match=msg):
        validate_args(SCHEMA, args)


def test_loads_args():
    assert loads_args("") == {} and loads_args('{"a": 1}') == {"a": 1}
    for bad in ('{"a": ', "[1]", "nope"):
        with pytest.raises(ValueError):
            loads_args(bad)


@pytest.mark.parametrize("raw,leak", [
    ("Authorization: Bearer abcdefghijklmnop1234", "abcdefghijklmnop1234"),
    ("token=supersecretvalue99", "supersecretvalue99"),
    ('{"api_key": "sk-abcdefghijklmnopqrstuv"}', "abcdefghijklmnopqrstuv"),
    ("cf" + "ut_" + "abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnopqrstuvwxyz"),       # built at runtime: no token-shaped
    ("tv" + "ly-" + "abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnopqrstuvwxyz"),       # literal ever exists in source
    ("gh" + "p_" + "abcdefghijklmnopqrstuvwxyz0123", "abcdefghijklmnopqrstuvwxyz0123"),
    ("bt-" + "A" * 43, "A" * 43),
])
def test_redaction(raw, leak):
    assert leak not in redact(raw)


def test_clip_modes():
    text = "0123456789" * 100
    head, cut = clip(text, 100, "head")
    assert cut and head.startswith("0123") and "omitted" in head and len(head) < 140
    tail, _ = clip(text, 100, "tail")
    assert tail.endswith("6789")
    both, _ = clip(text, 100, "ends")
    assert both.startswith("0123") and both.endswith("6789") and "omitted" in both
    assert clip("short", 100) == ("short", False)


async def test_registry_unknown_invalid_timeout_and_crash():
    reg = ToolRegistry()

    async def slow(a, c): await asyncio.sleep(5)
    async def boom(a, c): raise RuntimeError("kaput")
    async def fine(a, c): return ToolResult(ok=True, content="done")
    schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
    reg.register(ToolSpec("slow", "d", schema, slow, timeout=0.1))
    reg.register(ToolSpec("boom", "d", schema, boom))
    reg.register(ToolSpec("fine", "d", schema, fine))
    ctx = ToolContext(settings=None, store=None, computer=None, http=None, tavily=None)
    assert "unknown tool" in (await reg.run("nope", {}, ctx)).error
    assert "missing required argument 'x'" in (await reg.run("fine", {}, ctx)).error
    assert "timed out" in (await reg.run("slow", {"x": "1"}, ctx)).error
    crashed = await reg.run("boom", {"x": "1"}, ctx)
    assert not crashed.ok and "kaput" in crashed.error
    assert (await reg.run("fine", {"x": "1"}, ctx)).ok


def test_default_registry_has_the_expected_native_tools():
    reg = default_registry()
    assert set(reg.names()) == {"web_search", "run_command", "list_files", "read_file", "write_file", "fetch_url", "computer_info", "remember"}
    for tool in reg.schemas():
        assert tool["type"] == "function" and tool["function"]["description"] and tool["function"]["parameters"]["type"] == "object"


def test_tavily_key_parsing():
    assert _parse_keys('{"keys": ["a","b"], "index": 1}') == ["a", "b"]
    assert _parse_keys("a, b\nc") == ["a", "b", "c"] and _parse_keys("") == [] and _parse_keys('["x"]') == ["x"]


class FakeHttp:
    def __init__(self, handler): self.handler, self.calls = handler, []
    async def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(headers["Authorization"])
        return self.handler(headers["Authorization"])


def resp(status, payload=None):
    return httpx.Response(status, json=payload or {})


async def test_web_search_rotates_and_fails_over_between_keys():
    keys = TavilyKeys(store=None, env={"TAVILY_API_KEYS": "k1,k2,k3"})
    results = {"results": [{"title": "Lagos Weather", "url": "https://www.example.com/w", "content": "Sunny 31C " * 50}]}
    http = FakeHttp(lambda auth: resp(429) if auth.endswith("k1") else resp(200, results))
    ctx = ToolContext(settings=Settings(), store=None, computer=None, http=http, tavily=keys)
    first = await web_search({"query": "lagos weather", "max_results": 5, "topic": "general"}, ctx)
    assert first.ok and "Lagos Weather" in first.content and "example.com" in first.summary
    assert first.data["sources"][0]["domain"] == "example.com"
    assert http.calls == ["Bearer k1", "Bearer k2"]                        # k1 hit its limit → failover to k2
    again = await web_search({"query": "q2", "max_results": 5, "topic": "general"}, ctx)
    assert again.ok and "Bearer k1" not in http.calls[2:]                  # benched key is skipped next time


async def test_web_search_reports_failure_honestly():
    from blackthorn.tools.base import ToolError
    keys = TavilyKeys(store=None, env={"TAVILY_API_KEYS": "k1"})
    ctx = ToolContext(settings=Settings(), store=None, computer=None, http=FakeHttp(lambda a: resp(500)), tavily=keys)
    with pytest.raises(ToolError) as err:
        await web_search({"query": "anything", "max_results": 5, "topic": "general"}, ctx)
    assert "failed on every configured key" in err.value.message
    empty = ToolContext(settings=Settings(), store=None, computer=None, http=FakeHttp(lambda a: resp(200)), tavily=TavilyKeys(env={}))
    with pytest.raises(ToolError, match="not configured"):
        await web_search({"query": "anything", "max_results": 5, "topic": "general"}, empty)
