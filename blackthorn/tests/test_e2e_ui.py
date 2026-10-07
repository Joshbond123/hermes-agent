"""Real-browser verification of the production UI build: streaming, markdown/code, tools, stop, refresh, history actions,
sidebar, header controls, GPU states, errors and responsive layout — at desktop and mobile sizes.
"""

import asyncio
import json
import re
import time

import httpx
import pytest

from .fakes import call, finish, pause, say, think, usage

pytestmark = pytest.mark.asyncio

DONE = "[data-role=assistant][data-status=stop]"


def words(n: int, delay: float = 0.12):
    out = []
    for i in range(n):
        out += [{"content": f"w{i:02d} "}, pause(delay)]
    return out


async def send(page, text: str, *, mobile: bool = False):
    box = page.get_by_test_id("composer-input")
    await box.fill(text)
    if mobile:
        await page.get_by_test_id("send").click()
    else:
        await box.press("Enter")


async def last_assistant_text(page) -> str:
    return await page.evaluate("() => { const els = document.querySelectorAll('[data-role=assistant] .msg-body'); return els.length ? els[els.length - 1].innerText : '' }")


async def chat(ui_stack, page, text, reply="ok", **kw):
    ui_stack.backend.queue([*say(reply, 2), finish("stop")])
    await send(page, text, **kw)
    await page.wait_for_selector(DONE, timeout=20000)


# ====================================================================================================== page load
async def test_new_ui_loads_with_no_legacy_chrome_and_nothing_external(pages, ui_stack):
    page = await pages.new()
    assert await page.title() == "Blackthorn"
    body = await page.inner_text("body")
    for legacy in ("HERMES", "AGENT", "Nous", "Help improve", "Gateway Status", "DOCUMENTATION"):
        assert legacy not in body, legacy
    assert await page.locator("img").count() == 0                      # no avatars / decorative images
    assert await page.locator(".avatar, [class*=avatar]").count() == 0
    external = [u for u in page.requests if not u.startswith(ui_stack.url) and not u.startswith("data:")]
    assert external == [], external                                    # nothing from a CDN
    assert not [p for p in page.problems], page.problems
    html = await page.content()
    assert "?v=" not in html and "cdnjs" not in html


async def test_static_assets_cache_policy(pages, ui_stack):
    async with httpx.AsyncClient() as c:
        index = await c.get(ui_stack.url + "/")
        assert index.headers["cache-control"] == "no-store"
        asset = re.search(r'/assets/[^"]+\.js', index.text).group(0)
        a = await c.get(ui_stack.url + asset)
        assert a.status_code == 200 and "immutable" in a.headers["cache-control"]
        assert re.search(r"-[A-Za-z0-9_-]{8,}\.js$", asset)            # content hash in the file name, so immutable is safe
        v = (await c.get(ui_stack.url + "/api/version")).json()
        assert v["app"] == "blackthorn" and v["ui"]["files"]


# ====================================================================================================== streaming
async def test_response_streams_progressively_as_one_message(pages, ui_stack):
    page = await pages.new()
    ui_stack.backend.queue([*words(14, 0.2), finish("stop")])
    await send(page, "count to fourteen")
    lengths = set()
    t0 = time.time()
    while time.time() - t0 < 15:
        lengths.add(len(await last_assistant_text(page)))
        if await page.locator(DONE).count():
            break
        await asyncio.sleep(0.12)
    assert len(lengths) >= 6, sorted(lengths)                          # text grew step by step in the DOM
    assert await page.get_by_test_id("message").count() == 2           # one user message + ONE continuous assistant message
    assert (await last_assistant_text(page)).split() == [f"w{i:02d}" for i in range(14)]
    assert re.search(r"/c/studio-[0-9a-f]{12}$", page.url)             # the chat got a real, shareable URL
    assert await page.get_by_test_id("session-item").count() == 1


async def test_markdown_code_block_highlight_language_scroll_and_copy(pages, ui_stack):
    page = await pages.new(clipboard=True)
    long_line = "    return a + b  # " + "z" * 400
    ui_stack.backend.queue([*say("Here is code:\n\n```python\n", 1), pause(0.6), *say("def add(a, b):\n" + long_line + "\n", 1), pause(0.6),
                            *say("```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n- item **bold**\n\nDone.", 1), finish("stop")])
    await send(page, "show code")
    await page.wait_for_selector("[data-testid=code-block]", timeout=10000)            # appears while the fence is still OPEN
    assert await page.locator(DONE).count() == 0
    await page.wait_for_selector(DONE, timeout=15000)
    assert await page.get_by_test_id("code-lang").inner_text() == "python"
    assert await page.locator("pre .hljs-keyword").count() >= 1                         # syntax highlighting is applied
    scroll = await page.evaluate("() => { const p = document.querySelector('.code pre'); return [p.scrollWidth, p.clientWidth, getComputedStyle(p).overflowX] }")
    assert scroll[0] > scroll[1] and scroll[2] == "auto", scroll                        # long lines scroll horizontally
    assert await page.locator(".md table").count() == 1 and await page.locator(".md li strong").count() == 1
    code_text = await page.evaluate("() => document.querySelector('.code code').innerText")
    await page.get_by_test_id("copy-code").click()
    await page.wait_for_function("() => document.querySelector('[data-testid=copy-code]').innerText === 'Copied'", timeout=3000)
    assert await page.evaluate("() => navigator.clipboard.readText()") == code_text      # the copy button really copies the code
    widths = await page.evaluate("""() => { const t = document.querySelector('.thread'), cs = getComputedStyle(t);
        return [document.querySelector('[data-role=assistant]').getBoundingClientRect().width, t.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight)] }""")
    assert widths[0] >= widths[1] - 1, widths                                           # fills the whole content column: no narrow bubble


async def test_greeting_has_no_activity_and_tool_use_shows_real_collapsible_activity(pages, ui_stack):
    page = await pages.new()
    await chat(ui_stack, page, "Hi", "Hello! How can I help?")
    assert await page.get_by_test_id("activity-group").count() == 0
    ui_stack.backend.exec_output = "total 0\nnotes.txt\n"
    ui_stack.backend.exec_delay = 1.5                                                   # keep the tool "running" long enough to observe
    ui_stack.backend.queue([pause(0.4), *say("Let me look. ", 1), call("run_command", {"command": "ls"}, cid="c1"), finish("tool_calls")],
                           [pause(0.8), *say("You have notes.txt."), finish("stop")])
    await send(page, "what files do I have?")
    await page.wait_for_selector("[data-testid=tool-row][data-status=running]", timeout=10000)        # a REAL running tool event
    group = page.get_by_test_id("activity-group")
    assert await group.get_attribute("data-open") == "true"                                            # open while it works
    assert "Terminal" in await group.inner_text() and "Working" in await group.inner_text()
    await page.wait_for_selector("[data-role=assistant][data-status=stop]:last-of-type", timeout=15000)
    assert await group.get_attribute("data-open") == "false"                                           # collapsed once finished
    assert "Used tools" in await group.inner_text() and "1 step" in await group.inner_text()
    await page.get_by_test_id("activity-toggle").click()
    assert await group.get_attribute("data-open") == "true"
    row = page.get_by_test_id("tool-row")
    assert await row.get_attribute("data-status") == "ok"
    assert "$ ls" in await row.inner_text()
    assert [await page.get_by_test_id("message").nth(i).get_attribute("data-role") for i in range(2)] == ["user", "assistant"]
    assert "You have notes.txt." in await last_assistant_text(page)


async def test_tool_failure_is_shown_as_an_error_not_hidden(pages, ui_stack):
    page = await pages.new()
    ui_stack.backend.queue([call("read_file", {"path": "missing.txt"}, cid="e1"), finish("tool_calls")], [*say("I couldn't find that file."), finish("stop")])
    await send(page, "read missing.txt")
    await page.wait_for_selector(DONE, timeout=15000)
    await page.get_by_test_id("activity-toggle").click()
    assert await page.get_by_test_id("tool-row").get_attribute("data-status") == "error"
    assert "missing.txt" in await page.get_by_test_id("tool-row").inner_text()


# ====================================================================================================== stop / refresh / retry
async def test_stop_button_cancels_generation_everywhere(pages, ui_stack):
    page = await pages.new()
    ui_stack.backend.queue([*words(80, 0.1), finish("stop")])
    await send(page, "write a lot")
    await page.wait_for_function("() => (document.querySelector('[data-role=assistant] .msg-body')?.innerText || '').length > 20", timeout=10000)
    t0 = time.time()
    await page.get_by_test_id("stop").click()
    await page.wait_for_selector("[data-role=assistant][data-status=cancelled]", timeout=8000)
    assert time.time() - t0 < 4
    assert "Stopped" in await page.get_by_test_id("stopped-note").inner_text()
    for _ in range(40):
        if ui_stack.backend.cancelled:
            break
        await asyncio.sleep(0.05)
    assert ui_stack.backend.cancelled == 1 and ui_stack.backend.completed == 0         # the model was really told to stop
    kept = (await last_assistant_text(page))
    assert kept.startswith("w00") and "w79" not in kept
    await page.wait_for_selector("[data-testid=send]", timeout=3000)                    # composer usable again
    await chat(ui_stack, page, "still there?", "Yes.")
    assert await page.get_by_test_id("message").count() == 4


async def test_refresh_mid_stream_reattaches_and_finishes_without_duplicates(pages, ui_stack):
    page = await pages.new()
    ui_stack.backend.queue([*words(30, 0.15), finish("stop")])
    await send(page, "long answer")
    await page.wait_for_function("() => (document.querySelector('[data-role=assistant] .msg-body')?.innerText || '').length > 25", timeout=10000)
    await page.reload()
    await page.wait_for_selector("[data-testid=message]", timeout=10000)
    await page.wait_for_selector(DONE, timeout=25000)
    text = await last_assistant_text(page)
    assert text.split() == [f"w{i:02d}" for i in range(30)]                             # complete, in order, nothing doubled
    assert await page.get_by_test_id("message").count() == 2
    await page.reload()                                                                  # and it survives a second refresh as history
    await page.wait_for_selector(DONE, timeout=10000)
    assert (await last_assistant_text(page)).split() == [f"w{i:02d}" for i in range(30)]


async def test_completed_response_survives_refresh_and_reopening_from_history(pages, ui_stack):
    page = await pages.new()
    await chat(ui_stack, page, "remember this", "Here is the saved answer.")
    url = page.url
    await page.reload()
    await page.wait_for_selector(DONE)
    assert "Here is the saved answer." in await last_assistant_text(page)
    await page.get_by_test_id("new-chat").click()
    assert await page.get_by_test_id("empty-state").count() == 1
    await page.get_by_test_id("session-item").first.locator(".s-main").click()
    await page.wait_for_selector(DONE)
    assert page.url == url and "Here is the saved answer." in await last_assistant_text(page)


async def test_failed_response_keeps_partial_text_and_retry_replaces_it(pages, ui_stack):
    page = await pages.new()
    ui_stack.backend.queue([*say("Partial answer so far ", 1), pause(0.05), {"drop": 1}])
    await send(page, "question")
    await page.wait_for_selector("[data-role=assistant][data-status=error]", timeout=15000)
    assert "Partial answer so far" in await last_assistant_text(page)
    assert await page.get_by_test_id("error-note").count() == 1
    ui_stack.backend.queue([*say("The full answer.", 1), finish("stop")])
    await page.get_by_test_id("regenerate").click()
    await page.wait_for_selector(DONE, timeout=15000)
    assert await last_assistant_text(page) == "The full answer."
    assert await page.get_by_test_id("message").count() == 2


# ====================================================================================================== history actions
async def make_chats(ui_stack, page, *titles):
    for i, t in enumerate(titles):
        if i:
            await page.get_by_test_id("new-chat").click()
        await chat(ui_stack, page, t, f"reply to {t}")


async def item(page, title):
    return page.locator("[data-testid=session-item]", has=page.locator(f"[data-testid=session-title]:text-is('{title}')"))


async def open_menu(page, title, action):
    row = await item(page, title)
    await row.hover()
    await row.get_by_test_id("session-menu").click()
    await page.get_by_test_id(action).click()


async def test_rename_persists_in_the_database_and_survives_reload(pages, ui_stack):
    page = await pages.new()
    await make_chats(ui_stack, page, "alpha chat")
    await open_menu(page, "alpha chat", "menu-rename")
    box = page.get_by_test_id("rename-input")
    await box.fill("Renamed chat")
    await box.press("Enter")
    await page.wait_for_selector("[data-testid=session-title]:text-is('Renamed chat')")
    assert await page.get_by_test_id("chat-title").inner_text() == "Renamed chat"
    row = (await ui_stack.executor.query("SELECT title, title_source FROM sessions")).rows[0]
    assert row == {"title": "Renamed chat", "title_source": "user"}
    await page.reload()
    await page.wait_for_selector("[data-testid=session-title]:text-is('Renamed chat')")
    assert await page.get_by_test_id("chat-title").inner_text() == "Renamed chat"


async def test_pin_archive_unarchive_delete_all_change_stored_data(pages, ui_stack):
    page = await pages.new()
    await make_chats(ui_stack, page, "first chat", "second chat")
    # pin the older one: it moves to the Pinned section (stored in sessions.pinned) and survives a reload
    await open_menu(page, "first chat", "menu-pin")
    await page.wait_for_selector("text=Pinned")
    assert (await ui_stack.executor.query("SELECT pinned FROM sessions WHERE title='first chat'")).rows[0]["pinned"] == 1
    await page.reload()
    await page.wait_for_selector("text=Pinned")
    titles = await page.get_by_test_id("session-title").all_inner_texts()
    assert titles[0] == "first chat"
    # archive: leaves the list, shows under Archived, flag stored
    await open_menu(page, "second chat", "menu-archive")
    await page.wait_for_selector("[data-testid=session-title]:text-is('second chat')", state="detached")
    assert (await ui_stack.executor.query("SELECT archived FROM sessions WHERE title='second chat'")).rows[0]["archived"] == 1
    await page.get_by_test_id("archived-toggle").click()
    await page.wait_for_selector("[data-testid=archived-list] [data-testid=session-title]:text-is('second chat')")
    # an archived chat opens normally ...
    await page.locator("[data-testid=archived-list] .s-main").first.click()
    await page.wait_for_selector("[data-role=assistant][data-status=stop]")
    assert "reply to second chat" in await last_assistant_text(page)
    # ... and unarchive brings it back
    await open_menu(page, "second chat", "menu-archive")
    await page.wait_for_selector("[data-testid=archived-list] [data-testid=session-title]:text-is('second chat')", state="detached")
    assert (await ui_stack.executor.query("SELECT archived FROM sessions WHERE title='second chat'")).rows[0]["archived"] == 0
    # delete asks first, then removes the session and every message
    await open_menu(page, "second chat", "menu-delete")
    await page.get_by_test_id("confirm-dialog").wait_for()
    await page.get_by_role("button", name="Cancel").click()
    assert await page.get_by_test_id("confirm-dialog").count() == 0
    assert (await ui_stack.executor.query("SELECT count(*) AS n FROM sessions WHERE title='second chat'")).rows[0]["n"] == 1
    await open_menu(page, "second chat", "menu-delete")
    await page.get_by_test_id("confirm-yes").click()
    await page.wait_for_selector("[data-testid=session-title]:text-is('second chat')", state="detached")
    assert (await ui_stack.executor.query("SELECT count(*) AS n FROM sessions WHERE title='second chat'")).rows[0]["n"] == 0
    assert (await ui_stack.executor.query("SELECT count(*) AS n FROM messages")).rows[0]["n"] == 2       # only "first chat" remains
    from urllib.parse import urlparse
    assert await page.get_by_test_id("empty-state").count() == 1             # the deleted chat was open: back to a new chat
    assert urlparse(page.url).path == "/"


async def test_switching_chats_back_button_and_new_chat(pages, ui_stack):
    page = await pages.new()
    await make_chats(ui_stack, page, "apples", "oranges")
    assert "reply to oranges" in await last_assistant_text(page)
    await (await item(page, "apples")).locator(".s-main").click()
    await page.wait_for_function("() => document.querySelector('[data-testid=chat-title]').innerText === 'apples'")
    assert "reply to apples" in await last_assistant_text(page) and await page.get_by_test_id("message").count() == 2
    await page.go_back()
    await page.wait_for_function("() => document.querySelector('[data-testid=chat-title]').innerText === 'oranges'")
    assert "reply to oranges" in await last_assistant_text(page)
    await page.get_by_test_id("new-chat").click()
    assert await page.get_by_test_id("empty-state").count() == 1 and await page.get_by_test_id("chat-title").inner_text() == "New chat"


async def test_search_filters_the_history(pages, ui_stack):
    page = await pages.new()
    await make_chats(ui_stack, page, "gardening tips", "tax questions")
    await page.get_by_test_id("search").fill("garden")
    await page.wait_for_function("() => document.querySelectorAll('[data-testid=session-item]').length === 1")
    assert await page.get_by_test_id("session-title").inner_text() == "gardening tips"
    await page.get_by_test_id("search").fill("")
    await page.wait_for_function("() => document.querySelectorAll('[data-testid=session-item]').length === 2")


# ====================================================================================================== sidebar & responsive
async def test_sidebar_toggle_on_desktop_persists(pages, ui_stack):
    page = await pages.new()
    sidebar = page.get_by_test_id("sidebar")
    assert await sidebar.is_visible()
    await page.get_by_test_id("sidebar-toggle").click()
    assert not await sidebar.is_visible()
    await page.reload()
    await page.get_by_test_id("composer-input").wait_for()
    assert not await sidebar.is_visible()                                               # remembered across reloads
    await page.get_by_test_id("sidebar-toggle").click()
    assert await sidebar.is_visible()


async def test_mobile_drawer_opens_closes_and_navigates(pages, ui_stack):
    page = await pages.new(mobile=True)
    await chat(ui_stack, page, "mobile chat", "mobile reply", mobile=True)
    sidebar = page.get_by_test_id("sidebar")
    assert not await sidebar.is_visible()                                              # closed by default on phones
    await page.get_by_test_id("sidebar-toggle").tap()
    await page.wait_for_function("() => document.querySelector('[data-testid=sidebar]').getBoundingClientRect().left >= -1")
    assert await sidebar.is_visible() and await page.get_by_test_id("scrim").is_visible()
    await page.get_by_test_id("scrim").tap(position={"x": 370, "y": 400})
    await page.wait_for_function("() => document.querySelector('[data-testid=sidebar]').getAttribute('data-open') === 'false'")
    await page.get_by_test_id("sidebar-toggle").tap()
    await page.keyboard.press("Escape")
    await page.wait_for_function("() => document.querySelector('[data-testid=sidebar]').getAttribute('data-open') === 'false'")
    await page.get_by_test_id("sidebar-toggle").tap()
    await page.get_by_test_id("new-chat").tap()                                         # New chat closes the drawer too
    await page.wait_for_function("() => document.querySelector('[data-testid=sidebar]').getAttribute('data-open') === 'false'")
    assert await page.get_by_test_id("empty-state").count() == 1
    await page.get_by_test_id("sidebar-toggle").tap()
    await page.locator("[data-testid=session-item] .s-main").first.tap()
    await page.wait_for_selector(DONE)
    assert await page.get_by_test_id("sidebar").get_attribute("data-open") == "false"
    assert "mobile reply" in await last_assistant_text(page)


@pytest.mark.parametrize("width,height", [(320, 640), (390, 844), (768, 1024), (1024, 768), (1440, 900)])
async def test_layout_has_no_horizontal_overflow_and_controls_fit(pages, ui_stack, width, height):
    page = await pages.new(width=width, height=height, mobile=width < 800)
    ui_stack.backend.queue([*say("```python\nprint('" + "x" * 250 + "')\n```\n\n| a | b | c | d | d |\n|---|---|---|---|---|\n| " + " | ".join(["long-cell-" * 4] * 5) + " |\n\n" + "word " * 120, 1), finish("stop")])
    await send(page, "layout test " + "long " * 40, mobile=width < 800)
    await page.wait_for_selector(DONE, timeout=15000)
    overflow = await page.evaluate("() => [document.documentElement.scrollWidth, innerWidth, document.body.scrollWidth]")
    assert overflow[0] <= overflow[1] and overflow[2] <= overflow[1], overflow          # the page itself never scrolls sideways
    for tid in ("sidebar-toggle", "prompt-button", "gpu-button", "composer-input", "send"):
        box = await page.get_by_test_id(tid).bounding_box()
        assert box is not None and box["x"] >= -1 and box["x"] + box["width"] <= width + 1, (tid, box)
        assert box["y"] >= 0 and box["y"] + box["height"] <= height + 1, (tid, box)
    box = await page.get_by_test_id("composer-input").bounding_box()
    assert box["height"] < height * 0.4


# ====================================================================================================== header controls
async def test_system_prompt_button_loads_saves_and_applies(pages, ui_stack):
    page = await pages.new()
    await page.get_by_test_id("prompt-button").click()
    dlg = page.get_by_test_id("prompt-dialog")
    await dlg.wait_for()
    box = page.get_by_test_id("prompt-input")
    await box.fill("Always answer like a pirate.")
    await page.get_by_test_id("prompt-save").click()
    await page.get_by_test_id("prompt-saved").wait_for()
    async with httpx.AsyncClient() as c:
        assert (await c.get(ui_stack.url + "/api/system-prompt")).json()["prompt"] == "Always answer like a pirate."
    await page.keyboard.press("Escape")
    assert await page.get_by_test_id("prompt-dialog").count() == 0
    await page.reload()
    await page.get_by_test_id("prompt-button").click()
    await page.wait_for_function("() => document.querySelector('[data-testid=prompt-input]').value.includes('pirate')")
    await page.keyboard.press("Escape")
    await chat(ui_stack, page, "ahoy", "Arr!")
    assert "Always answer like a pirate." in ui_stack.backend.requests[-1]["messages"][0]["content"]


async def test_prompt_and_gpu_buttons_work_on_mobile_too(pages, ui_stack):
    page = await pages.new(mobile=True)
    await page.get_by_test_id("prompt-button").tap()
    await page.get_by_test_id("prompt-dialog").wait_for()
    box = await page.get_by_test_id("prompt-dialog").bounding_box()
    assert box["x"] >= 0 and box["x"] + box["width"] <= 390
    await page.keyboard.press("Escape")
    await page.get_by_test_id("gpu-button").tap()
    await page.get_by_test_id("gpu-panel").wait_for()
    panel = await page.get_by_test_id("gpu-panel").bounding_box()
    assert panel["x"] >= 0 and panel["x"] + panel["width"] <= 390 + 1


async def test_gpu_button_panel_quota_and_no_secrets(pages, ui_stack):
    page = await pages.new()
    btn = page.get_by_test_id("gpu-button")
    await page.wait_for_function("() => document.querySelector('[data-testid=gpu-button]').innerText.includes('GPU ready')")
    assert await btn.get_attribute("data-state") == "ok"
    await btn.click()
    panel = page.get_by_test_id("gpu-panel")
    await panel.wait_for()
    text = await panel.inner_text()
    assert "Kaggle Ready" in text and "Fake-Model" in text and "Tesla T4" in text
    assert "19.8 h of 30 h" in await page.get_by_test_id("gpu-quota").inner_text()
    assert "Blackthorn v" in await page.get_by_test_id("version-line").inner_text()
    sel = page.get_by_test_id("gpu-auto-off")
    await sel.select_option("15")
    await page.wait_for_function("() => document.querySelector('[data-testid=gpu-auto-off]').value === '15'")
    assert ui_stack.d1.auto_off == 15
    async with httpx.AsyncClient() as c:
        raw = (await c.get(ui_stack.url + "/api/kaggle-gpu/status")).text
    assert "must-not-leak" not in raw and "hidden.example" not in raw and "acct" not in raw     # no key, tunnel or ids
    await page.keyboard.press("Escape")
    assert await page.get_by_test_id("gpu-panel").count() == 0
    await btn.click()
    await page.get_by_test_id("gpu-panel").wait_for()
    await page.mouse.click(600, 500)                                                            # click outside closes it
    await page.wait_for_function("() => !document.querySelector('[data-testid=gpu-panel]')")


async def test_gpu_lifecycle_reflects_real_state_progress_and_recovery(pages, ui_stack):
    ui_stack.d1.status = "GPU_STOPPED_SAVING_QUOTA"
    await ui_stack.set_gpu("GPU_STOPPED_SAVING_QUOTA", url="")
    page = await pages.new()
    btn = page.get_by_test_id("gpu-button")
    await page.wait_for_function("() => document.querySelector('[data-testid=gpu-button]').innerText.includes('GPU off')")
    assert await btn.get_attribute("data-state") == "off"
    assert await page.get_by_test_id("gpu-banner").is_visible()
    await btn.click()
    await page.get_by_test_id("gpu-turn-on").click()
    await page.wait_for_function("() => document.querySelector('[data-testid=gpu-button]').innerText.includes('GPU starting')")
    for _ in range(100):                                                                          # the request is in flight: wait for it
        if ui_stack.d1.calls == ["on"]:
            break
        await asyncio.sleep(0.05)
    assert ui_stack.d1.calls == ["on"]
    progress = page.get_by_test_id("gpu-progress")
    await progress.wait_for()
    first = await progress.inner_text()
    assert re.search(r"Step \d+ of 12", first), first
    # a stage that reports real bytes shows them (label + done of total), not a made-up percentage
    await page.wait_for_function("() => (document.querySelector('[data-testid=gpu-progress]')?.innerText || '').includes('Downloading the model')", timeout=30000)
    txt = await page.get_by_test_id("gpu-progress").inner_text()
    assert "4.20 GB of 16.80 GB" in txt.replace("\u00a0", " ") or "GB of" in txt, txt
    assert await page.locator("[data-testid=gpu-progress] [role=progressbar]").get_attribute("aria-valuenow") == "25"      # 4.2 / 16.8
    await page.wait_for_function("() => document.querySelector('[data-testid=gpu-button]').innerText.includes('GPU ready')", timeout=40000)
    steps = ui_stack.d1.stage
    assert steps >= 5                                                                            # it really walked the reported stages
    assert await page.get_by_test_id("gpu-banner").count() == 0
    assert await page.get_by_test_id("gpu-turn-off").is_visible()
    # turning it off is explicit: confirm dialog, then state follows
    await page.get_by_test_id("gpu-turn-off").click()
    await page.get_by_test_id("confirm-dialog").wait_for()
    await page.get_by_test_id("confirm-yes").click()
    await page.wait_for_function("() => document.querySelector('[data-testid=gpu-button]').innerText.includes('GPU off')", timeout=15000)
    assert ui_stack.d1.calls == ["on", "off"]


async def test_sending_while_gpu_is_off_keeps_the_draft_and_explains(pages, ui_stack):
    await ui_stack.set_gpu("GPU_STOPPED_SAVING_QUOTA", url="")
    page = await pages.new()
    await send(page, "my important question")
    await page.get_by_test_id("toast").first.wait_for()
    assert "GPU is off" in await page.get_by_test_id("toast").first.inner_text()
    assert await page.get_by_test_id("composer-input").input_value() == "my important question"   # nothing lost
    assert await page.get_by_test_id("message").count() == 0
    assert (await ui_stack.executor.query("SELECT count(*) AS n FROM messages")).rows[0]["n"] == 0


async def test_gpu_error_state_is_shown_not_hidden(pages, ui_stack):
    ui_stack.d1.status = "BOOT_FAILED"
    page = await pages.new()
    await page.wait_for_function("() => document.querySelector('[data-testid=gpu-button]').innerText.includes('GPU')")
    await page.get_by_test_id("gpu-button").click()
    await page.get_by_test_id("gpu-panel").wait_for()
    await page.get_by_test_id("gpu-turn-on").wait_for(state="visible", timeout=8000)                # recovery action is offered
    assert "GPU error" in await page.get_by_test_id("gpu-button").inner_text()
    assert await page.get_by_test_id("gpu-error").count() == 1                                      # and the reason is shown, not hidden


# ====================================================================================================== composer
async def test_enter_sends_shift_enter_adds_a_line(pages, ui_stack):
    page = await pages.new()
    box = page.get_by_test_id("composer-input")
    await box.fill("line one")
    await box.press("Shift+Enter")
    await box.type("line two")
    assert await box.input_value() == "line one\nline two"
    assert await page.get_by_test_id("message").count() == 0
    ui_stack.backend.queue([*say("got it"), finish("stop")])
    await box.press("Enter")
    await page.wait_for_selector(DONE)
    assert await page.get_by_test_id("composer-input").input_value() == ""
    assert "line one\nline two" in await page.locator("[data-role=user] .msg-body").inner_text()


async def test_text_attachment_is_uploaded_and_listed(pages, ui_stack, tmp_path):
    page = await pages.new()
    f = tmp_path / "notes.txt"
    f.write_text("alpha\nbeta\n")
    await page.get_by_test_id("file-input").set_input_files(str(f))
    assert "notes.txt" in await page.get_by_test_id("attachments").inner_text()
    ui_stack.backend.queue([*say("read it"), finish("stop")])
    await send(page, "summarise the file")
    await page.wait_for_selector(DONE)
    assert "notes.txt" in await page.locator("[data-role=user] .chips").inner_text()
    write = [c for c in ui_stack.backend.computer_calls if c["name"] == "write_file"][0]
    assert write["body"] == {"path": "uploads/notes.txt", "content": "alpha\nbeta\n"}
    assert await page.get_by_test_id("attachments").count() == 0


async def test_theme_selector_switches_and_remembers(pages, ui_stack):
    page = await pages.new(scheme="dark")
    bg_dark = await page.evaluate("() => getComputedStyle(document.body).backgroundColor")
    await page.get_by_test_id("theme-select").select_option("light")
    bg_light = await page.evaluate("() => getComputedStyle(document.body).backgroundColor")
    assert bg_dark != bg_light
    await page.reload()
    await page.get_by_test_id("composer-input").wait_for()
    assert await page.evaluate("() => getComputedStyle(document.body).backgroundColor") == bg_light
    assert await page.evaluate("() => document.documentElement.dataset.theme") == "light"


async def test_copy_message_and_last_message_actions(pages, ui_stack):
    page = await pages.new(clipboard=True)
    await chat(ui_stack, page, "say something", "Copy this exact answer.")
    await page.locator("[data-role=assistant] [data-testid=copy-message]").click()
    assert await page.evaluate("() => navigator.clipboard.readText()") == "Copy this exact answer."
    assert await page.get_by_test_id("regenerate").count() == 1                                  # only on the last assistant message
