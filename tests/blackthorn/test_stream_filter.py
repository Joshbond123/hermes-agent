"""The stream filter must never leak reasoning or raw tool-call tags — at *any* chunk split."""

from __future__ import annotations

import json

import pytest

from blackthorn.agent.model_client import ToolCallAssembler, ToolCallDelta, parse_chunk
from blackthorn.agent.stream_filter import ContentFilter, parse_text_tool_calls


def run(chunks):
    f = ContentFilter()
    out = "".join(f.feed(c) for c in chunks) + f.flush()
    return out, f


def all_splits(text):
    """Every way to cut ``text`` into two chunks, plus char-by-char."""
    for i in range(len(text) + 1):
        yield [text[:i], text[i:]]
    yield list(text)


CASES = [
    ("plain text stays", "Hello there, world!", "Hello there, world!"),
    ("inline think removed", "<think>secret plan</think>The answer is 4.", "The answer is 4."),
    ("think in the middle", "Start <think>hidden</think> end", "Start  end"),
    ("two think blocks", "<think>a</think>X<think>b</think>Y", "XY"),
    ("case-insensitive tags", "<THINK>hidden</Think>shown", "shown"),
    ("stray closing think dropped", "reasoning leftovers</think>Visible", "reasoning leftoversVisible"),
    ("angle brackets that are not tags", "if a < b and c > d then 1<2", "if a < b and c > d then 1<2"),
    ("almost-tag text", "use the <thing> element", "use the <thing> element"),
    ("tool call hidden", 'Hi <tool_call>{"name":"x","arguments":{}}</tool_call>bye', "Hi bye"),
]


@pytest.mark.parametrize("label,raw,expected", CASES, ids=[c[0] for c in CASES])
def test_filter_output_is_independent_of_chunking(label, raw, expected):
    for chunks in all_splits(raw):
        out, _ = run(chunks)
        assert out == expected, f"{label}: split {chunks!r} -> {out!r}"


def test_secret_reasoning_text_never_appears_for_any_split():
    raw = "<think>my private chain of thought</think>Final."
    for chunks in all_splits(raw):
        out, _ = run(chunks)
        assert "private" not in out and "chain" not in out and out == "Final."


def test_text_tool_call_is_captured_and_parsed():
    raw = 'Let me look. <tool_call>{"name": "list_files", "arguments": {"path": "."}}</tool_call>'
    for chunks in all_splits(raw):
        out, f = run(chunks)
        assert out == "Let me look. "
        calls = parse_text_tool_calls(f.tool_blocks)
        assert calls and calls[0]["name"] == "list_files"
        assert json.loads(calls[0]["arguments"]) == {"path": "."}


def test_unterminated_think_is_discarded_and_unterminated_tool_is_captured():
    out, f = run(["answer <think>never closed"])
    assert out == "answer "
    out, f = run(['<tool_call>{"name":"a","arguments":{}}'])
    assert out == "" and parse_text_tool_calls(f.tool_blocks)[0]["name"] == "a"


def test_held_back_text_is_released_on_flush():
    out, _ = run(["value is <thi"])
    assert out == "value is <thi"


def test_parse_text_tool_calls_ignores_garbage_and_accepts_fences():
    assert parse_text_tool_calls(["not json at all"]) == []
    calls = parse_text_tool_calls(['```json\n{"name":"t","arguments":{"a":1}}\n```'])
    assert calls[0]["name"] == "t"


# ---- native tool-call assembly + chunk parsing -----------------------------------------
def test_assembler_joins_fragments_by_index_and_generates_ids():
    asm = ToolCallAssembler()
    asm.add(ToolCallDelta(0, "call_a", "terminal", '{"comm'))
    asm.add(ToolCallDelta(0, None, None, 'and": "ls"}'))
    asm.add(ToolCallDelta(1, None, "list_files", "{}"))
    calls = asm.calls()
    assert [c.name for c in calls] == ["terminal", "list_files"]
    assert calls[0].id == "call_a" and calls[0].arguments == '{"command": "ls"}'
    assert calls[1].id.startswith("call_")
    assert bool(asm)


def test_parse_chunk_separates_reasoning_content_and_tools():
    events = parse_chunk({"choices": [{"delta": {"reasoning": "thinking...", "content": "Hi"}}]})
    assert [type(e).__name__ for e in events] == ["ReasoningDelta", "ContentDelta"]
    assert events[0].chars == len("thinking...")  # only the size is kept, never the text
    assert not hasattr(events[0], "text")
    events = parse_chunk({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "c1", "function": {"name": "web_search", "arguments": {"query": "x"}}}]}}]})
    assert events[0].name == "web_search" and json.loads(events[0].arguments) == {"query": "x"}
    events = parse_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": 3}})
    assert events[-1].reason == "stop" and events[-1].usage == {"completion_tokens": 3}


def test_parse_chunk_raises_on_error_object():
    from blackthorn.agent.model_client import ModelError

    with pytest.raises(ModelError):
        parse_chunk({"error": {"message": "out of memory"}})
