"""Qwen3.8's actual upstream chat template is the wire-format oracle.

Fixture: https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/chat_template.jinja
The synthetic prompts contain no customer data, credentials, or remote computer output.
"""
from pathlib import Path

from jinja2 import Environment

from blackthorn import prompts


TEMPLATE = Environment().from_string((Path(__file__).parent / "fixtures/qwen38_chat_template.jinja").read_text())


def render(messages):
    return TEMPLATE.render(messages=messages, tools=None, enable_thinking=False,
                           add_generation_prompt=True, raise_exception=lambda message: (_ for _ in ()).throw(ValueError(message)))


def test_compaction_keeps_real_user_for_qwen_template():
    # The system state grows with the task ledger; several preceding turns are compacted.
    msgs = [{"role": "system", "content": "S" * 10000}]
    for _ in range(3):
        msgs.extend([{"role": "user", "content": "Previous request"},
                     {"role": "assistant", "content": "previous answer"}])
    current = {"role": "user", "content": "REAL REQUEST: complete the outstanding task"}
    msgs.extend([current, {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call_{i}", "type": "function", "function": {"name": "run_command", "arguments": {}}}
        for i in range(4)]}])
    msgs.extend({"role": "tool", "tool_call_id": f"call_{i}", "content": "result" * 66} for i in range(4))
    assert prompts.shrink_tool_messages(msgs, 3669, protect=[0, 7])
    assert any(m is current for m in msgs), "real user request was silently discarded"
    assert "REAL REQUEST" in render(msgs)


def test_tool_pairs_are_unique_and_fit_after_sanitization():
    current = {"role": "user", "content": "Find file"}
    msgs = [{"role": "system", "content": "system"}, current, {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call_{i}", "type": "function", "function": {"name": "read_file", "arguments": {}}}
        for i in range(4)]}]
    msgs.extend({"role": "tool", "tool_call_id": f"call_{i}", "content": "result" * 20} for i in range(4))
    assert prompts.shrink_tool_messages(msgs, 300, protect=[0, 1])
    assert [m['tool_call_id'] for m in msgs if m['role'] == 'tool'] == [f'call_{i}' for i in range(4)]
    assert prompts.estimate_messages(msgs) <= 300
    render(msgs)


def test_protected_user_cannot_fit_reports_failure_instead_of_sending_invalid_history():
    current = {"role": "user", "content": "MUST NOT DELETE: " + "x" * 2100}
    msgs = [{"role": "system", "content": "s" * 100}, current,
            {"role": "assistant", "content": "unprotected"}]
    assert not prompts.shrink_tool_messages(msgs, 250, protect=[0, 1])
    assert current in msgs


def test_many_tool_steps_keep_exactly_one_result_per_call_and_the_real_request():
    msgs = [{"role": "system", "content": "s" * 3500},
            {"role": "user", "content": "DO NOT FORGET: migrate service safely"}]
    current = msgs[1]
    for step in range(24):
        calls = [{"id": f"c{step}_{i}", "type": "function", "function": {"name": "run_command", "arguments": {}}}
                 for i in range(3)]
        msgs.append({"role": "assistant", "content": "inspecting", "tool_calls": calls})
        msgs.extend({"role": "tool", "tool_call_id": call["id"], "content": "observed data " * 180} for call in calls)
        assert prompts.shrink_tool_messages(msgs, 3600, protect=[0, msgs.index(current)])
        assert any(m is current for m in msgs)
        calls_kept = [c['id'] for m in msgs for c in m.get('tool_calls', [])]
        results = [m['tool_call_id'] for m in msgs if m.get('role') == 'tool']
        assert sorted(calls_kept) == sorted(results)
        assert prompts.estimate_messages(msgs) <= 3600
        render(msgs)


async def test_upstream_qwen_stream_error_recovers_once_without_rerunning_tools(stack):
    from .fakes import call, finish, say
    stack.backend.queue(
        [call("run_command", {"command": "echo done"}, cid="once"), finish("tool_calls")],
        [{"stream_error": "Jinja Exception: No user query found in messages."}],
        [*say("Task finished after a single safe retry."), finish("stop")],
    )
    out = await stack.stream({"message": "RUN-QUERY-ANCHOR: run one harmless check"})
    assert out.end["status"] == "stop"
    assert len(stack.backend.computer_calls) == 1, "retry must never replay side-effecting tools"
    assert len(stack.backend.requests) == 3
    assert "RUN-QUERY-ANCHOR" in str(stack.backend.requests[-1]["messages"])
    assert any(ev.get("stage") == "qwen_query_retry" for ev in out.of("internal.compaction"))
    assert "Task finished" in out.text

async def test_common_computer_task_prefill_has_less_schema_without_losing_core_tools(stack):
    from .fakes import finish, say
    stack.backend.queue([*say("Ready."), finish("stop")])
    out = await stack.stream({"message": "Please read file notes.txt on the computer"})
    assert out.end["status"] == "stop"
    from blackthorn.tools import default_registry
    wire = stack.backend.requests[0]["tools"]
    names = {t["function"]["name"] for t in wire}
    assert {"read_file", "write_file", "run_command", "list_files", "web_search", "todo_list", "job_status"} <= names
    # Capability discovery + the lab are always disclosed so the agent can choose them
    # itself; the schemas are compacted so that stays affordable on an 8k context.
    assert {"skill_search", "tool_search", "cyber_run", "lab_detonate", "lab_exec"} <= names
    full = default_registry().schemas()
    assert prompts.schema_tokens(wire) < prompts.schema_tokens(full)
    assert prompts.schema_tokens(wire) <= 3200


async def test_deferred_capability_can_be_invoked_without_loading_all_schemas(stack):
    from blackthorn.tools import ToolContext
    reg = stack.services.registry
    ctx = ToolContext(settings=stack.services.settings, store=stack.services.store,
                      computer=stack.services.computer, http=stack.services.http,
                      tavily=stack.services.tavily)
    result = await reg.run("optional_tool", {"name": "session_search", "arguments": {}}, ctx)
    assert not result.ok and "query" in result.error  # original schema still validates
    denied = await reg.run("optional_tool", {"name": "malware_analyzer_submit", "arguments": {}}, ctx)
    assert not denied.ok and "invalid arguments" in denied.error
