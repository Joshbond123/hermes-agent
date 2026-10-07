import itertools

import pytest

from blackthorn import prompts
from blackthorn.agent import ThinkFilter
from blackthorn.gpu import GpuService
from blackthorn.kaggle_bundle import BundleError, REQUIRED_ENV, build_notebook_text


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
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
            {"role": "tool", "tool_call_id": "1", "content": "A" * 3000}, {"role": "tool", "tool_call_id": "2", "content": "B" * 3000}]
    assert prompts.shrink_tool_messages(msgs, budget_tokens=1200)
    assert msgs[2]["content"] == prompts.STUB and msgs[3]["content"].startswith("B")
    assert prompts.shrink_tool_messages(msgs, budget_tokens=300) and len(msgs[3]["content"]) < 700


def test_system_prompt_composition_is_small_and_contains_context():
    text = prompts.system_prompt("Be brief.", ["likes tea", "lives in Ibadan"])
    assert "Be brief." in text and "likes tea" in text and "Today is" in text and len(text) < 1500
    assert "NEVER" not in prompts.BASE_PROMPT


def test_safe_filename():
    assert prompts.safe_filename("../../etc/pass wd.txt") == "pass_wd.txt"
    assert prompts.safe_filename("") == "file" and prompts.safe_filename("a" * 300).startswith("a" * 80)


# ---- Kaggle notebook is generated from source with injected credentials -----------------------------------------------
def test_notebook_is_built_from_the_committed_source_with_injected_env():
    import base64, json, re
    env = {k: f"value-of-{k}" for k in REQUIRED_ENV}
    nb = json.loads(build_notebook_text("print('hello gpu')\n", env))
    src = "".join(nb["cells"][0]["source"])
    parts = re.findall(r"parts\.append\('([A-Za-z0-9+/=]+)'\)", src)
    assert base64.b64decode("".join(parts)).decode() == "print('hello gpu')\n"
    injected = json.loads(base64.b64decode(re.search(r"b64decode\('([A-Za-z0-9+/=]+)'\)\)\)", src).group(1)))
    assert injected["CLOUDFLARE_API_TOKEN"] == "value-of-CLOUDFLARE_API_TOKEN"
    compile(src, "cell", "exec")                                    # the cell itself is valid Python


def test_notebook_refuses_to_build_without_credentials():
    with pytest.raises(BundleError, match="CLOUDFLARE_API_TOKEN"):
        build_notebook_text("x", {})


def test_real_server_source_compiles_and_has_no_literal_secrets():
    from blackthorn.kaggle_bundle import SERVER_SOURCE
    src = SERVER_SOURCE.read_text()
    compile(src, str(SERVER_SOURCE), "exec")
    import re
    assert not re.search(r"cfut_[A-Za-z0-9]{10,}|KGAT_[a-f0-9]{10,}|sk-qwen38-[A-Za-z0-9-]{6,}", src)


# ---- GPU public status never leaks the tunnel or keys ----------------------------------------------------------------
class FakeD1:
    def __init__(self, state): self.state = state
    def get_kaggle_gpu_status(self, refresh=False): return dict(self.state)
    def public_gpu_status(self, state):
        import cloudflare_d1_client as real
        return real.public_gpu_status(state)


class Store:
    def __init__(self, started): self.started = started
    async def get_setting(self, key): return str(self.started) if self.started else ""


async def test_gpu_status_has_no_secrets_and_reports_real_stage_and_elapsed():
    import time
    state = {"active": False, "booting": True, "status": "DOWNLOADING_MODEL", "tunnel_url": "https://secret.trycloudflare.com",
             "api_key": "bt-" + "S" * 43, "progress_pct": 30, "progress_step": "Downloading", "model": "m", "gpu_info": "T4",
             "cloudflare_d1": {"account_id": "acct", "database_id": "db"}, "detail": '{"bytes_done": 500, "bytes_total": 1000}',
             "quota": {"used_hours": 1}}
    svc = GpuService(Store(time.time() - 125), FakeD1(state))
    out = await svc._augment(state)
    blob = str(out)
    assert "secret.trycloudflare" not in blob and "S" * 43 not in blob and "acct" not in blob
    assert out["progress_stage"] == 5 and out["progress_total_stages"] == 12 and out["progress_kind"] == "bytes"
    assert out["progress_bytes_done"] == 500 and out["progress_bytes_total"] == 1000
    assert 120 <= out["elapsed_seconds"] <= 135 and "progress_pct" not in out
    assert out["can_turn_on"] is False and out["can_turn_off"] is True


async def test_gpu_ready_and_off_states_offer_the_right_actions():
    ready = {"active": True, "booting": False, "status": "HEARTBEAT_ONLINE", "tunnel_url": "https://x", "model_loaded": True}
    off = {"active": False, "booting": False, "status": "GPU_STOPPED_SAVING_QUOTA", "tunnel_url": ""}
    svc = GpuService(Store(None), FakeD1(ready))
    r = await svc._augment(ready)
    o = await svc._augment(off)
    assert r["online"] and r["can_turn_off"] and not r["can_turn_on"]
    assert not o["online"] and o["can_turn_on"] and not o["can_turn_off"]


async def test_gpu_status_carries_the_notebook_label_and_stalled_flag():
    import time
    state = {"active": False, "booting": True, "status": "DOWNLOADING_MODEL", "tunnel_url": "", "model": "m", "gpu_info": "T4",
             "detail": '{"label": "Downloading the model", "bytes_done": 10, "bytes_total": 100, "stalled": true}'}
    out = await GpuService(Store(time.time() - 5), FakeD1(state))._augment(state)
    assert out["progress_label"] == "Downloading the model" and out["progress_stalled"] is True and out["progress_kind"] == "bytes"
