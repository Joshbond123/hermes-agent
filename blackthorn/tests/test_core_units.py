import itertools

import pytest

from blackthorn import prompts
from blackthorn.agent import ThinkFilter
from blackthorn.codespaces import UsageLedger
from blackthorn.route import DEFAULT_MODEL, RouteResolver, context_tokens


# ---- ThinkFilter: never eats normal text, whatever the chunking -----------------------------------------------
def run_filter(chunks):
    f = ThinkFilter()
    vis, hid = [], []
    for c in chunks:
        v, h = f.feed(c)
        vis.append(v), hid.append(h)
    v, h = f.flush()
    return "".join(vis) + v, "".join(hid) + h


def test_think_block_is_hidden_at_every_possible_split_point():
    text = "<think>secret plan</think>Hello <b>world</b> a < b > c"
    for i, j in itertools.combinations(range(len(text) + 1), 2):
        vis, hid = run_filter([text[:i], text[i:j], text[j:]])
        assert vis == "Hello <b>world</b> a < b > c" and hid == "secret plan", (i, j)


def test_plain_text_passes_through_untouched():
    for t in ["Hello", "x < y", "a<th b", "<t", "1 <", "<<<think>"]:
        for k in range(1, len(t) + 1):
            vis, hid = run_filter([t[i:i + k] for i in range(0, len(t), k)])
            assert (vis + hid) == t.replace("<think>", "") and (hid == "" or "<think>" in t)


def test_unclosed_think_is_never_shown():
    vis, hid = run_filter(["Answer? <think>never closed"])
    assert vis == "Answer? " and hid == "never closed"


# ---- prompts / context budget -----------------------------------------------------------------------------------
def test_fit_history_keeps_newest_and_starts_with_user():
    hist = [{"role": "user", "content": "u1 " * 200}, {"role": "assistant", "content": "a1 " * 200},
            {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]
    kept = prompts.fit_history(hist, budget_tokens=60)
    assert [m["content"] for m in kept] == ["u2", "a2"]
    assert prompts.fit_history(hist, budget_tokens=10_000) == hist
    assert prompts.fit_history([], 100) == []


def test_fit_history_clips_a_single_oversized_recent_turn():
    kept = prompts.fit_history([{"role": "user", "content": "z" * 5000}], budget_tokens=200)
    assert len(kept) == 1 and len(kept[0]["content"]) < 900 and kept[0]["content"].startswith("[…earlier")


def test_shrink_tool_messages_stubs_old_outputs_first():
    def pair(cid, text):
        return [{"role": "assistant", "content": "", "tool_calls": [
                    {"id": cid, "type": "function", "function": {"name": "run_command", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": cid, "content": text}]
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
            *pair("1", "A" * 3000), *pair("2", "B" * 3000)]
    assert prompts.shrink_tool_messages(msgs, budget_tokens=1200)
    assert msgs[3]["content"] == prompts.STUB and msgs[5]["content"].startswith("B")
    assert prompts.shrink_tool_messages(msgs, budget_tokens=300)
    assert prompts.estimate_messages(msgs) <= 300
    assert any(m.get("role") == "user" and m["content"] == "q" for m in msgs)


def test_system_prompt_composition_is_small_and_contains_context():
    text = prompts.system_prompt("Be brief.", ["likes tea", "lives in Ibadan"])
    assert "Be brief." in text and "likes tea" in text and "Today is" in text
    # The Superpowers bootstrap is part of the prompt (there is no plugin loader here), so the budget
    # allows for it — but it must stay a small fraction of the context window, not a wall of text.
    assert len(text) < 2600, len(text)
    assert "skill_load" in text and "systematic-debugging" in text and "verification-before-completion" in text
    assert "NEVER" not in prompts.BASE_PROMPT


def test_safe_filename():
    assert prompts.safe_filename("../../etc/pass wd.txt") == "pass_wd.txt"
    assert prompts.safe_filename("") == "file" and prompts.safe_filename("a" * 300).startswith("a" * 80)


# ---- the model route is NVIDIA, configured by environment -------------------------------------------------------------
def test_the_route_is_nvidia_and_reads_its_settings_from_the_environment(monkeypatch):
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)      # the default base URL, not a test double
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")
    monkeypatch.setenv("BLACKTHORN_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    import asyncio
    route = asyncio.run(RouteResolver().get())
    assert route.provider == "nvidia" and route.url.startswith("https://integrate.api.nvidia.com")
    assert route.model == "nvidia/nemotron-3-ultra-550b-a55b" and route.api_key == "nvapi-test-key"


def test_without_a_key_the_route_fails_loudly_instead_of_inventing_one(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    import asyncio
    from blackthorn.route import RouteError
    with pytest.raises(RouteError) as exc:
        asyncio.run(RouteResolver().get())
    assert exc.value.code == "model_unconfigured"


def test_the_default_model_is_nemotron_3_ultra():
    assert DEFAULT_MODEL == "nvidia/nemotron-3-ultra-550b-a55b"


def test_context_budget_is_a_planning_budget_not_the_provider_ceiling(monkeypatch):
    monkeypatch.setenv("BLACKTHORN_CONTEXT_TOKENS", "65536")
    assert context_tokens() == 65536
    monkeypatch.setenv("BLACKTHORN_CONTEXT_TOKENS", "12")          # nonsense is ignored, not obeyed
    assert context_tokens() == 131072


# ---- usage hours are measured, never invented --------------------------------------------------------------------------
class _SettingsStore:
    def __init__(self): self.values = {}
    async def get_setting(self, key): return self.values.get(key, "")
    async def set_setting(self, key, value): self.values[key] = value


async def test_usage_ledger_counts_only_observed_running_intervals():
    import time
    ledger = UsageLedger(_SettingsStore())
    await ledger.open_interval("cs-1")
    assert (await ledger.seconds_this_month()) >= 0.0
    await ledger.close_interval()
    total = await ledger.seconds_this_month()
    assert total >= 0.0
    # an interval from another month is not counted in this one
    store = _SettingsStore()
    import json
    store.values[UsageLedger.KEY] = json.dumps([{"name": "cs-1", "start": 0.0, "end": 3600.0}])
    assert await UsageLedger(store).seconds_this_month(now=time.time()) < 1.0
