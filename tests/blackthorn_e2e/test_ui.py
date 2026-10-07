"""UI behaviour that does not need the model. Every assertion that matters is cross-checked
against the stored data in D1, not just against what the page shows."""

from __future__ import annotations

import re
import time

import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE, Seeder, accessible_name_missing, api, no_horizontal_overflow, open_chat, sidebar_titles

T = 15000  # default expectation timeout (ms)


def row(page: Page, title: str):
    return page.locator(".bt-row", has=page.locator(".bt-row-title", has_text=title)).first


def open_row_menu(page: Page, title: str):
    r = row(page, title)
    r.hover()
    r.get_by_role("button", name=re.compile(r"^Actions for")).click()


def pick(page: Page, name: str):
    page.get_by_role("menuitem", name=name).click()


def persisted(page: Page, action, method: str = "PATCH"):
    """Run ``action`` and wait until the server has answered the write it triggers (UI updates optimistically)."""
    with page.expect_response(lambda r: r.request.method == method and "/api/studio/sessions/" in r.url, timeout=20000) as info:
        action()
    assert info.value.status == 200, info.value.status


def gpu_state(page: Page) -> str:
    return api(page, "GET", "/api/kaggle-gpu/status")["json"].get("state")


# --------------------------------------------------------------------------------------- shell
def test_root_opens_the_chat_with_one_header_and_no_errors(desktop):
    open_chat(desktop, "/")
    expect(desktop).to_have_url(re.compile(r"/chat$"))
    expect(desktop.locator("header:visible")).to_have_count(1)  # the baseline showed two stacked headers
    expect(desktop.get_by_role("heading", name="Blackthorn", level=1)).to_be_visible()
    expect(desktop.get_by_role("textbox", name="Message")).to_be_visible()
    assert no_horizontal_overflow(desktop)
    assert accessible_name_missing(desktop) == []
    desktop.wait_for_timeout(1500)
    assert desktop.errors == []


def test_every_header_control_works_and_none_blocks_the_others(desktop):
    """Baseline bug: opening the GPU panel left an overlay that made Files/Export/New chat unclickable."""
    open_chat(desktop, "/chat")
    toggle = desktop.get_by_role("button", name=re.compile(r"(Close|Open) sidebar"))
    sidebar = desktop.locator("#bt-sidebar")

    # sidebar close / open, remembered across reloads
    expect(toggle).to_have_attribute("aria-expanded", "true")
    w_open = sidebar.bounding_box()["width"]
    toggle.click()
    expect(toggle).to_have_attribute("aria-expanded", "false")
    desktop.wait_for_timeout(450)
    assert desktop.locator(".bt-root").get_attribute("data-sidebar") == "closed"
    assert sidebar.bounding_box() is None or sidebar.bounding_box()["x"] + sidebar.bounding_box()["width"] <= 2
    desktop.reload()
    desktop.wait_for_selector(".bt-chat-root")
    expect(desktop.locator(".bt-root")).to_have_attribute("data-sidebar", "closed")
    desktop.get_by_role("button", name="Open sidebar").click()
    desktop.wait_for_timeout(450)
    assert abs(sidebar.bounding_box()["width"] - w_open) < 2

    # GPU panel opens, shows the truth, closes with Escape — and the other controls still respond
    gpu_btn = desktop.locator("button.bt-gpu-btn")
    gpu_btn.click()
    panel = desktop.get_by_role("dialog", name="GPU controls")
    expect(panel).to_be_visible()
    expect(panel.locator(".bt-gpu-head strong")).to_have_text(re.compile(r"GPU is (off|ready|starting|stopping)|GPU error"))
    desktop.keyboard.press("Escape")
    expect(panel).to_have_count(0)

    prompt_btn = desktop.get_by_role("button", name="System prompt")
    prompt_btn.click()
    dlg = desktop.get_by_role("dialog", name="System prompt")
    expect(dlg).to_be_visible()
    desktop.keyboard.press("Escape")
    expect(dlg).to_have_count(0)

    # clicking outside closes the panel and the click is not swallowed
    gpu_btn.click()
    expect(panel).to_be_visible()
    desktop.get_by_role("button", name="More options").click()
    expect(panel).to_have_count(0)
    menu = desktop.get_by_role("menu", name="More options")
    expect(menu).to_be_visible()
    expect(menu.get_by_role("menuitem", name="Admin dashboard")).to_be_visible()
    expect(menu.get_by_role("menuitem", name="Export chat as Markdown")).to_be_disabled()  # nothing to export yet
    desktop.keyboard.press("Escape")
    expect(menu).to_have_count(0)

    desktop.get_by_role("link", name="New chat").first.click()
    expect(desktop).to_have_url(re.compile(r"/chat$"))
    assert desktop.errors == []


def test_system_prompt_is_saved_to_the_server_and_survives_reload(desktop):
    open_chat(desktop, "/chat")
    original = api(desktop, "GET", "/api/system-prompt")["json"]["prompt"]
    text = f"E2E system prompt {int(time.time())}"
    try:
        desktop.get_by_role("button", name="System prompt").click()
        dlg = desktop.get_by_role("dialog", name="System prompt")
        box = dlg.get_by_role("textbox", name="System prompt")
        expect(box).to_be_visible()
        box.fill(text)
        save = dlg.get_by_role("button", name="Save")
        expect(save).to_be_enabled()
        save.click()
        expect(dlg).to_have_count(0)
        expect(desktop.get_by_text(re.compile("System prompt saved"))).to_be_visible()
        assert api(desktop, "GET", "/api/system-prompt")["json"]["prompt"] == text  # stored server-side
        desktop.reload()
        desktop.wait_for_selector(".bt-chat-root")
        desktop.get_by_role("button", name="System prompt").click()
        expect(desktop.get_by_role("dialog", name="System prompt").get_by_role("textbox")).to_have_value(text)
        # Cancel leaves it untouched
        desktop.get_by_role("dialog").get_by_role("button", name="Cancel").click()
        assert api(desktop, "GET", "/api/system-prompt")["json"]["prompt"] == text
    finally:
        api(desktop, "PUT", "/api/system-prompt", {"prompt": original})
    assert api(desktop, "GET", "/api/system-prompt")["json"]["prompt"] == original


def test_theme_choice_is_remembered(desktop):
    open_chat(desktop, "/chat")
    desktop.get_by_role("button", name="More options").click()
    desktop.get_by_role("menuitem", name="Theme: light").click()
    expect(desktop.locator(".bt-root")).to_have_attribute("data-theme", "light")
    bg = desktop.evaluate("getComputedStyle(document.querySelector('.bt-root')).backgroundColor")
    assert bg in ("rgb(255, 255, 255)", "rgba(255, 255, 255, 1)")
    desktop.reload()
    desktop.wait_for_selector(".bt-chat-root")
    expect(desktop.locator(".bt-root")).to_have_attribute("data-theme", "light")
    desktop.get_by_role("button", name="More options").click()
    desktop.get_by_role("menuitem", name="Theme: dark").click()
    assert desktop.evaluate("getComputedStyle(document.querySelector('.bt-root')).backgroundColor") != bg
    desktop.get_by_role("button", name="More options").click()
    desktop.get_by_role("menuitem", name="Theme: match system").click()


# --------------------------------------------------------------------------------------- history
def two_turns(q="hello", a="hi there"):
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a, "finish_reason": "stop"}]


def test_rename_pin_archive_restore_delete_all_persist(desktop, seeder):
    sid = seeder.chat("E2E alpha chat", two_turns())
    open_chat(desktop, "/chat")
    expect(row(desktop, "E2E alpha chat")).to_be_visible(timeout=T)

    # rename
    open_row_menu(desktop, "E2E alpha chat")
    pick(desktop, "Rename")
    box = desktop.get_by_role("textbox", name="Chat title")
    box.fill("E2E renamed chat")
    persisted(desktop, lambda: box.press("Enter"))
    expect(row(desktop, "E2E renamed chat")).to_be_visible()
    assert Seeder.row(sid)["title"] == "E2E renamed chat"
    # rename can be cancelled and cannot be emptied
    open_row_menu(desktop, "E2E renamed chat")
    pick(desktop, "Rename")
    desktop.get_by_role("textbox", name="Chat title").fill("discard me")
    desktop.keyboard.press("Escape")
    expect(row(desktop, "E2E renamed chat")).to_be_visible()
    assert Seeder.row(sid)["title"] == "E2E renamed chat"

    # pin
    open_row_menu(desktop, "E2E renamed chat")
    persisted(desktop, lambda: pick(desktop, "Pin to top"))
    expect(desktop.get_by_role("heading", name="Pinned")).to_be_visible()
    assert Seeder.row(sid)["pinned"] == 1
    desktop.reload()
    desktop.wait_for_selector(".bt-chat-root")
    expect(desktop.get_by_role("heading", name="Pinned")).to_be_visible(timeout=T)
    open_row_menu(desktop, "E2E renamed chat")
    persisted(desktop, lambda: pick(desktop, "Unpin"))
    expect(desktop.get_by_role("heading", name="Pinned")).to_have_count(0)
    assert Seeder.row(sid)["pinned"] == 0

    # archive -> gone from the list, present under Archived, readable, restorable
    open_row_menu(desktop, "E2E renamed chat")
    persisted(desktop, lambda: pick(desktop, "Archive"))
    expect(row(desktop, "E2E renamed chat")).to_have_count(0)
    assert Seeder.row(sid)["archived"] == 1
    desktop.get_by_role("button", name="Archived").click()
    expect(row(desktop, "E2E renamed chat")).to_be_visible(timeout=T)
    row(desktop, "E2E renamed chat").get_by_role("link").click()
    expect(desktop.locator("article.bt-msg-user")).to_contain_text("hello")
    open_row_menu(desktop, "E2E renamed chat")
    persisted(desktop, lambda: pick(desktop, "Restore"))
    expect(desktop.locator(".bt-history .bt-archived .bt-row")).to_have_count(0)
    assert Seeder.row(sid)["archived"] == 0
    desktop.reload()
    desktop.wait_for_selector(".bt-chat-root")
    expect(row(desktop, "E2E renamed chat")).to_be_visible(timeout=T)

    # delete (with confirmation)
    open_row_menu(desktop, "E2E renamed chat")
    pick(desktop, "Delete")
    dlg = desktop.get_by_role("dialog", name="Delete this chat?")
    expect(dlg).to_be_visible()
    dlg.get_by_role("button", name="Cancel").click()
    assert Seeder.row(sid) is not None
    open_row_menu(desktop, "E2E renamed chat")
    pick(desktop, "Delete")
    persisted(desktop, lambda: desktop.get_by_role("dialog").get_by_role("button", name="Delete").click(), method="DELETE")
    expect(row(desktop, "E2E renamed chat")).to_have_count(0)
    assert Seeder.row(sid) is None and Seeder.message_count(sid) == 0
    expect(desktop).to_have_url(re.compile(r"/chat/?$"))
    assert desktop.errors == []


def test_new_existing_selected_and_archived_chats_behave_consistently(desktop, seeder):
    a = seeder.chat("E2E chat A", two_turns("question A", "answer A"), age_seconds=30)
    b = seeder.chat("E2E chat B", two_turns("question B", "answer B"), age_seconds=20)
    c = seeder.chat("E2E chat C archived", two_turns("question C", "answer C"), archived=True, age_seconds=10)
    open_chat(desktop, "/chat")
    expect(desktop.get_by_role("heading", name="Blackthorn", level=1)).to_be_visible()       # new chat = empty
    expect(desktop.locator("article.bt-msg")).to_have_count(0)

    row(desktop, "E2E chat A").get_by_role("link").click()
    expect(desktop).to_have_url(re.compile(rf"/chat/{a}$"))
    expect(desktop.locator("article.bt-msg-user")).to_contain_text("question A")
    expect(desktop.locator("article.bt-msg-assistant")).to_contain_text("answer A")
    expect(row(desktop, "E2E chat A").get_by_role("link")).to_have_attribute("aria-current", "page")
    expect(desktop.locator("header h1")).to_have_text("E2E chat A")

    row(desktop, "E2E chat B").get_by_role("link").click()
    expect(desktop).to_have_url(re.compile(rf"/chat/{b}$"))
    expect(desktop.locator("article.bt-msg-assistant")).to_contain_text("answer B")
    expect(desktop.locator("article.bt-msg")).to_have_count(2)            # nothing left over from chat A
    expect(desktop.locator("header h1")).to_have_text("E2E chat B")

    desktop.go_back()
    expect(desktop).to_have_url(re.compile(rf"/chat/{a}$"))
    expect(desktop.locator("article.bt-msg-assistant")).to_contain_text("answer A")
    desktop.go_forward()
    expect(desktop.locator("article.bt-msg-assistant")).to_contain_text("answer B")

    desktop.get_by_role("link", name="New chat").first.click()
    expect(desktop).to_have_url(re.compile(r"/chat$"))
    expect(desktop.locator("article.bt-msg")).to_have_count(0)
    expect(desktop.get_by_role("textbox", name="Message")).to_be_focused()

    desktop.get_by_role("button", name="Archived").click()
    row(desktop, "E2E chat C archived").get_by_role("link").click()
    expect(desktop.locator("article.bt-msg-assistant")).to_contain_text("answer C")
    assert Seeder.row(c)["archived"] == 1                                  # opening does not un-archive

    desktop.goto(BASE + f"/chat/{a}", wait_until="domcontentloaded")       # deep link / refresh
    expect(desktop.locator("article.bt-msg-assistant")).to_contain_text("answer A", timeout=T)
    desktop.goto(BASE + "/chat/studio-doesnotexist", wait_until="domcontentloaded")
    expect(desktop.get_by_role("alert")).to_contain_text("no longer exists", timeout=T)
    desktop.get_by_role("button", name="New chat").click()
    expect(desktop).to_have_url(re.compile(r"/chat$"))


def test_sidebar_search_filters_on_the_server(desktop, seeder):
    seeder.chat("E2E kubernetes notes", two_turns("how do pods restart", "they restart per policy"))
    seeder.chat("E2E pasta recipe", two_turns("carbonara?", "eggs, guanciale, pecorino"))
    open_chat(desktop, "/chat")
    search = desktop.get_by_role("searchbox", name="Search chats")
    search.fill("kubernetes")
    expect(row(desktop, "E2E kubernetes notes")).to_be_visible(timeout=T)
    expect(row(desktop, "E2E pasta recipe")).to_have_count(0)
    search.fill("guanciale")                                  # matches message text, not the title
    expect(row(desktop, "E2E pasta recipe")).to_be_visible(timeout=T)
    expect(row(desktop, "E2E kubernetes notes")).to_have_count(0)
    search.fill("zzzzqqqq")
    expect(desktop.get_by_text("No chats match your search.")).to_be_visible(timeout=T)
    desktop.get_by_role("button", name="Clear search").click()
    expect(row(desktop, "E2E kubernetes notes")).to_be_visible(timeout=T)


# --------------------------------------------------------------------------------------- messages
LONG = ('def long_function_name(argument_one, argument_two, argument_three, argument_four, argument_five):\n'
        '    if argument_one:\n'
        '        return argument_two + argument_three + argument_four + argument_five + "a very long string literal that forces horizontal scrolling"\n'
        '    return None')
MARKDOWN = (
    "# Heading One\n\nSome **bold**, *italic* and `inline code` with a [link](https://example.com/page).\n\n"
    "- item one\n- item two\n\n1. first\n2. second\n\n| Name | Qty |\n|:-----|----:|\n| apple | 3 |\n| pear | 10 |\n\n"
    f"```python\n{LONG}\n```\n\nClosing paragraph.")


def test_markdown_tables_and_code_blocks_render_like_a_real_chat(desktop, seeder):
    seeder.chat("E2E markdown", [{"role": "user", "content": "show me everything"},
                                  {"role": "assistant", "content": MARKDOWN, "finish_reason": "stop"}])
    open_chat(desktop, "/chat")
    row(desktop, "E2E markdown").get_by_role("link").click()
    msg = desktop.locator("article.bt-msg-assistant")
    expect(msg.locator("h1")).to_have_text("Heading One")
    expect(msg.locator("strong")).to_have_text("bold")
    expect(msg.locator("em")).to_have_text("italic")
    expect(msg.locator(".bt-inline-code").first).to_have_text("inline code")
    link = msg.locator("a", has_text="link")
    expect(link).to_have_attribute("target", "_blank")
    expect(link).to_have_attribute("rel", re.compile("noopener"))
    expect(msg.locator("ul li")).to_have_count(2)
    expect(msg.locator("ol li")).to_have_count(2)
    expect(msg.locator("table th")).to_have_count(2)
    expect(msg.locator("table td").last).to_have_text("10")

    code = msg.locator(".bt-code")
    expect(code.locator(".bt-code-lang")).to_have_text("python")
    pre = code.locator("pre")
    assert desktop.evaluate("e => getComputedStyle(e).whiteSpace", pre.element_handle()) == "pre"
    assert desktop.evaluate("e => getComputedStyle(e).overflowX", pre.element_handle()) in ("auto", "scroll")
    assert desktop.evaluate("e => e.scrollWidth > e.clientWidth", pre.element_handle()), "long lines scroll horizontally"
    assert desktop.evaluate("document.querySelector('.bt-thread').scrollWidth <= document.querySelector('.bt-thread').clientWidth + 1")
    text = pre.inner_text()
    assert "    if argument_one:\n        return argument_two" in text   # indentation preserved exactly
    assert code.locator(".hljs-keyword").count() > 0                     # syntax highlighting is applied

    copy = code.get_by_role("button", name="Copy code to clipboard")
    copy.click()
    expect(copy).to_have_text("Copied")
    assert desktop.evaluate("navigator.clipboard.readText()") == LONG
    expect(copy).to_have_text("Copy", timeout=5000)

    # no avatars / decorative icons inside messages; messages use the available width
    assert desktop.locator("article.bt-msg img").count() == 0
    thread_w = desktop.locator(".bt-thread").bounding_box()["width"]
    for sel in ("article.bt-msg-user", "article.bt-msg-assistant"):
        assert desktop.locator(sel).bounding_box()["width"] >= 0.9 * (thread_w - 2 * 34), sel
    assert desktop.errors == []


def test_stored_tool_activity_is_collapsible_and_truthful(desktop, seeder):
    parts = [
        {"t": "text", "text": "Let me check the disk."},
        {"t": "tool", "id": "c1", "name": "terminal", "label": "Terminal", "status": "ok", "args": {"command": "df -h /"},
         "output": "[exit 0]\n/dev/root 25G 5.2G 19G 22% /", "duration_ms": 420, "exit_code": 0},
        {"t": "tool", "id": "c2", "name": "web_search", "label": "Web search", "status": "error", "args": {"query": "x"},
         "output": "error: search provider returned HTTP 500", "duration_ms": 900},
        {"t": "text", "text": "There are **19G** free."},
    ]
    seeder.chat("E2E tools", [{"role": "user", "content": "disk?"},
                               {"role": "assistant", "content": "Let me check the disk.\n\nThere are 19G free.",
                                "finish_reason": "stop", "parts": parts}])
    open_chat(desktop, "/chat")
    row(desktop, "E2E tools").get_by_role("link").click()
    rows = desktop.locator(".bt-tool")
    expect(rows).to_have_count(2)
    head = rows.first.locator(".bt-tool-head")
    expect(head).to_have_attribute("aria-expanded", "false")             # collapsed by default
    expect(rows.first.locator(".bt-tool-out")).to_have_count(0)
    expect(head).to_contain_text("Terminal")
    expect(head).to_contain_text("df -h /")
    head.click()
    expect(head).to_have_attribute("aria-expanded", "true")
    expect(rows.first.locator(".bt-tool-out")).to_contain_text("19G")
    expect(rows.nth(1).locator(".bt-tool-state")).to_contain_text("failed")
    # order inside the one continuous message: text, tool, tool, text
    kids = desktop.eval_on_selector(".bt-parts", "e => [...e.children].map(c => c.className.split(' ')[0])")
    assert kids == ["bt-md", "bt-tool", "bt-tool", "bt-md"], kids
    assert desktop.locator("article.bt-msg-assistant").count() == 1


def test_interrupted_and_stopped_messages_say_so(desktop, seeder):
    seeder.chat("E2E interrupted", [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "partial answer", "finish_reason": "interrupted"},
    ])
    open_chat(desktop, "/chat")
    row(desktop, "E2E interrupted").get_by_role("link").click()
    expect(desktop.locator(".bt-finish-note")).to_contain_text("Interrupted")
    expect(desktop.get_by_role("button", name="Regenerate")).to_be_visible()


# --------------------------------------------------------------------------------------- composer
def test_composer_keyboard_behaviour(desktop):
    open_chat(desktop, "/chat")
    box = desktop.get_by_role("textbox", name="Message")
    send = desktop.get_by_role("button", name="Send message")
    expect(send).to_be_disabled()
    box.click()
    box.type("line one")
    expect(send).to_be_enabled()
    h1 = box.bounding_box()["height"]
    desktop.keyboard.press("Shift+Enter")
    desktop.keyboard.type("line two")
    desktop.keyboard.press("Shift+Enter")
    desktop.keyboard.type("line three")
    assert box.input_value() == "line one\nline two\nline three"       # Shift+Enter inserts a newline, does not send
    assert box.bounding_box()["height"] > h1                            # it grows with the text
    expect(desktop.locator("article.bt-msg")).to_have_count(0)
    box.fill("")
    expect(send).to_be_disabled()
    box.fill("   ")
    expect(send).to_be_disabled()                                       # whitespace is not a message
    # suggestion chips fill the box (they do not send)
    desktop.get_by_role("button", name=re.compile("Explain how TCP")).click()
    expect(box).to_have_value(re.compile("TCP congestion"))
    expect(desktop.locator("article.bt-msg")).to_have_count(0)


def test_attachment_upload_chip_lifecycle(desktop):
    open_chat(desktop, "/chat")
    name = f"e2e-note-{int(time.time())}.txt"
    desktop.locator("input[type=file]").set_input_files({"name": name, "mimeType": "text/plain", "buffer": b"hello attachment"})
    chip = desktop.locator(".bt-chip", has_text=name)
    expect(chip).to_be_visible()
    expect(chip).to_have_class(re.compile("bt-chip-ready"), timeout=T)
    expect(desktop.get_by_role("button", name="Send message")).to_be_enabled()  # an attachment alone can be sent
    chip.get_by_role("button", name=re.compile("^Remove")).click()
    expect(chip).to_have_count(0)


# --------------------------------------------------------------------------------------- GPU (no start)
def test_gpu_panel_reports_the_real_state_and_offers_the_right_action(desktop):
    open_chat(desktop, "/chat")
    state = gpu_state(desktop)
    desktop.locator("button.bt-gpu-btn").click()
    panel = desktop.get_by_role("dialog", name="GPU controls")
    expect(panel).to_be_visible()
    expect(desktop.locator("button.bt-gpu-btn .bt-gpu-label")).to_have_text(
        {"off": "GPU off", "ready": "GPU ready", "error": "GPU error"}.get(state, re.compile(".+")))
    if state == "off":
        expect(panel.get_by_role("button", name="Start GPU")).to_be_visible()
        expect(panel.get_by_role("button", name="Stop GPU")).to_have_count(0)      # never both at once (baseline bug)
        expect(panel.get_by_text(re.compile(r"Kaggle GPU quota"))).to_be_visible()
        expect(panel.get_by_role("group", name="Auto-off after inactivity")).to_be_visible()
        assert panel.get_by_role("progressbar").count() >= 1                       # the quota bar (measured)
        expect(panel.get_by_role("progressbar", name=re.compile("duration unknown"))).to_have_count(0)
    elif state == "ready":
        expect(panel.get_by_role("button", name="Stop GPU")).to_be_visible()
        expect(panel.get_by_role("button", name="Start GPU")).to_have_count(0)
    # the status endpoint answers instantly (it was 36–75 s before)
    t0 = time.time()
    assert api(desktop, "GET", "/api/kaggle-gpu/status")["status"] == 200
    assert time.time() - t0 < 2


# --------------------------------------------------------------------------------------- GPU off: sending
def test_sending_while_the_gpu_is_off_keeps_the_message_and_offers_recovery(desktop):
    open_chat(desktop, "/chat")
    if gpu_state(desktop) != "off":
        pytest.skip("this test needs the GPU to be off")
    sid = None
    try:
        box = desktop.get_by_role("textbox", name="Message")
        box.fill("E2E gpu off message")
        box.press("Enter")
        expect(desktop.locator("article.bt-msg-user")).to_contain_text("E2E gpu off message")
        err = desktop.locator("article.bt-msg-assistant .bt-error")
        expect(err).to_be_visible(timeout=T)
        expect(err).to_contain_text("GPU")
        expect(err.get_by_role("button", name="Start GPU")).to_be_visible()
        expect(err.get_by_role("button", name="Retry")).to_be_visible()
        expect(desktop).to_have_url(re.compile(r"/chat/studio-"), timeout=T)
        sid = desktop.url.rsplit("/", 1)[-1]
        expect(row(desktop, "E2E gpu off message")).to_be_visible(timeout=T)
        # Retry must not duplicate the user's message
        err.get_by_role("button", name="Retry").click()
        expect(desktop.locator("article.bt-msg-assistant .bt-error")).to_be_visible(timeout=T)
        expect(desktop.locator("article.bt-msg-user")).to_have_count(1)
        # the failure is stored, so a reload shows the same thing
        desktop.reload()
        desktop.wait_for_selector(".bt-chat-root")
        expect(desktop.locator("article.bt-msg-assistant .bt-error")).to_be_visible(timeout=T)
        expect(desktop.locator("article.bt-msg-user")).to_have_count(1)
        stored = api(desktop, "GET", f"/api/studio/sessions/{sid}/messages")["json"]["messages"]
        assert [m["role"] for m in stored] == ["user", "assistant"] and stored[1]["finish_reason"] == "error"
    finally:
        if sid:
            api(desktop, "DELETE", f"/api/studio/sessions/{sid}")


# --------------------------------------------------------------------------------------- mobile
def test_mobile_layout_drawer_and_header_controls(mobile, seeder):
    seeder.chat("E2E mobile chat", two_turns("mobile q", "mobile a"))
    open_chat(mobile, "/chat")
    assert no_horizontal_overflow(mobile)
    expect(mobile.locator(".bt-root")).to_have_attribute("data-sidebar", "closed")     # drawer starts closed
    expect(mobile.locator("header:visible")).to_have_count(1)
    # the composer is fully on screen
    box = mobile.get_by_role("textbox", name="Message").bounding_box()
    assert 0 <= box["x"] and box["x"] + box["width"] <= 390 and box["y"] + box["height"] <= 844

    mobile.get_by_role("button", name="Open sidebar").click()
    expect(mobile.locator(".bt-root")).to_have_attribute("data-sidebar", "open")
    expect(mobile.locator(".bt-backdrop")).to_be_visible()
    assert mobile.locator("#bt-sidebar").bounding_box()["width"] <= 331
    mobile.locator("#bt-sidebar").get_by_role("button", name="Close sidebar").click()
    expect(mobile.locator(".bt-root")).to_have_attribute("data-sidebar", "closed")

    mobile.get_by_role("button", name="Open sidebar").click()
    row(mobile, "E2E mobile chat").get_by_role("link").click()
    expect(mobile.locator("article.bt-msg-assistant")).to_contain_text("mobile a")
    expect(mobile.locator(".bt-root")).to_have_attribute("data-sidebar", "closed")     # choosing a chat closes the drawer
    mobile.get_by_role("button", name="Open sidebar").click()
    mobile.locator(".bt-backdrop").click(position={"x": 370, "y": 400})                # tapping outside closes it too
    expect(mobile.locator(".bt-root")).to_have_attribute("data-sidebar", "closed")

    # every header control is reachable and works on a phone
    for name, opener in (("System prompt", lambda: mobile.get_by_role("button", name="System prompt").click()),):
        opener()
        expect(mobile.get_by_role("dialog", name=name)).to_be_visible()
        mobile.keyboard.press("Escape")
        expect(mobile.get_by_role("dialog", name=name)).to_have_count(0)
    mobile.locator("button.bt-gpu-btn").click()
    sheet = mobile.locator(".bt-popover.bt-sheet")
    expect(sheet).to_be_visible()
    sb = sheet.bounding_box()
    assert abs((sb["y"] + sb["height"]) - 844) < 3 and sb["width"] >= 388              # a bottom sheet
    mobile.keyboard.press("Escape")
    expect(sheet).to_have_count(0)
    mobile.get_by_role("button", name="More options").click()
    expect(mobile.get_by_role("menu")).to_be_visible()
    mobile.keyboard.press("Escape")

    # tap targets are comfortably sized
    for sel in ("button.bt-gpu-btn", "button[aria-label='System prompt']", "button[aria-label='More options']",
                "button[aria-label='Open sidebar']", "button[aria-label='Send message']"):
        b = mobile.locator(sel).first.bounding_box()
        assert b["height"] >= 33 and b["width"] >= 33, (sel, b)
    assert no_horizontal_overflow(mobile)
    assert mobile.errors == []


def test_mobile_markdown_does_not_overflow_the_screen(mobile, seeder):
    seeder.chat("E2E mobile code", [{"role": "user", "content": "code please"},
                                     {"role": "assistant", "content": MARKDOWN, "finish_reason": "stop"}])
    open_chat(mobile, "/chat")
    mobile.get_by_role("button", name="Open sidebar").click()
    row(mobile, "E2E mobile code").get_by_role("link").click()
    expect(mobile.locator(".bt-code")).to_be_visible()
    assert no_horizontal_overflow(mobile)                                    # code and table scroll *inside* their blocks
    assert mobile.evaluate("e => e.scrollWidth > e.clientWidth", mobile.locator(".bt-code pre").element_handle())
    assert mobile.evaluate("e => e.scrollWidth >= e.clientWidth", mobile.locator(".bt-table-wrap").element_handle())
