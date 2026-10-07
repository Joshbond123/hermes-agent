"""History actions must change the stored data, and behave the same for new, existing and archived chats."""

import httpx

from .fakes import finish, say


async def send(stack, text, sid=None):
    stack.backend.queue([*say(f"answer to {text}"), finish("stop")])
    payload = {"message": text}
    if sid:
        payload["session_id"] = sid
    out = await stack.stream(payload)
    assert out.status == 200, out.error_body
    return out.session_id


async def listing(stack, **params):
    async with httpx.AsyncClient() as c:
        return (await c.get(f"{stack.url}/api/chat/sessions", params=params)).json()["sessions"]


async def patch(stack, sid, **body):
    async with httpx.AsyncClient() as c:
        return await c.patch(f"{stack.url}/api/chat/sessions/{sid}", json=body)


async def test_new_chat_appears_with_title_from_first_message(stack):
    sid = await send(stack, "What is the boiling point of water at altitude?")
    rows = await listing(stack)
    assert [r["id"] for r in rows] == [sid]
    assert rows[0]["title"].startswith("What is the boiling point") and rows[0]["pinned"] is False


async def test_rename_is_persisted_in_the_database(stack):
    sid = await send(stack, "first")
    r = await patch(stack, sid, title="  My research notes ")
    assert r.status_code == 200 and r.json()["session"]["title"] == "My research notes"
    assert (await listing(stack))[0]["title"] == "My research notes"
    row = (await stack.executor.query("SELECT title, display_name, title_source FROM sessions WHERE id = ?", [sid])).rows[0]
    assert row == {"title": "My research notes", "display_name": "My research notes", "title_source": "user"}
    # a later message must not overwrite a user-chosen title
    await send(stack, "another message", sid)
    assert (await listing(stack))[0]["title"] == "My research notes"


async def test_empty_or_invalid_rename_is_rejected(stack):
    sid = await send(stack, "first")
    assert (await patch(stack, sid, title="   ")).status_code == 422
    assert (await patch(stack, "studio-doesnotexist", title="x")).status_code == 404
    assert (await listing(stack))[0]["title"].startswith("first")


async def test_pin_unpin_orders_and_persists(stack):
    a = await send(stack, "older chat")
    b = await send(stack, "newer chat")
    assert [r["id"] for r in await listing(stack)] == [b, a]
    assert (await patch(stack, a, pinned=True)).json()["session"]["pinned"] is True
    rows = await listing(stack)
    assert [r["id"] for r in rows] == [a, b] and rows[0]["pinned"] is True
    assert (await stack.executor.query("SELECT pinned FROM sessions WHERE id = ?", [a])).rows[0]["pinned"] == 1
    await patch(stack, a, pinned=False)
    assert [r["id"] for r in await listing(stack)] == [b, a]


async def test_archive_hides_from_history_and_lists_in_archive_and_unpins(stack):
    a = await send(stack, "to archive")
    await patch(stack, a, pinned=True)
    await send(stack, "keep me")
    assert (await patch(stack, a, archived=True)).json()["session"]["archived"] is True
    assert a not in [r["id"] for r in await listing(stack)]
    arch = await listing(stack, archived=True)
    assert [r["id"] for r in arch] == [a] and arch[0]["pinned"] is False and arch[0]["archived"] is True
    assert (await stack.executor.query("SELECT archived FROM sessions WHERE id = ?", [a])).rows[0]["archived"] == 1
    # restoring brings it back to the main list
    await patch(stack, a, archived=False)
    assert a in [r["id"] for r in await listing(stack)] and not await listing(stack, archived=True)


async def test_archived_chat_opens_read_write_and_new_message_unarchives(stack):
    a = await send(stack, "archived talk")
    await patch(stack, a, archived=True)
    async with httpx.AsyncClient() as c:
        data = (await c.get(f"{stack.url}/api/chat/sessions/{a}")).json()
    assert data["session"]["archived"] is True and len(data["messages"]) == 2
    await send(stack, "follow-up", a)
    assert a in [r["id"] for r in await listing(stack)] and not await listing(stack, archived=True)


async def test_delete_removes_session_and_all_messages(stack):
    a = await send(stack, "doomed")
    b = await send(stack, "survivor")
    async with httpx.AsyncClient() as c:
        assert (await c.delete(f"{stack.url}/api/chat/sessions/{a}")).status_code == 200
        assert (await c.delete(f"{stack.url}/api/chat/sessions/{a}")).status_code == 404
        assert (await c.get(f"{stack.url}/api/chat/sessions/{a}")).status_code == 404
    assert [r["id"] for r in await listing(stack)] == [b]
    assert (await stack.executor.query("SELECT count(*) AS n FROM messages WHERE session_id = ?", [a])).rows[0]["n"] == 0
    assert (await stack.executor.query("SELECT count(*) AS n FROM messages WHERE session_id = ?", [b])).rows[0]["n"] == 2


async def test_sessions_are_isolated_and_message_order_is_stable(stack):
    a = await send(stack, "alpha one")
    b = await send(stack, "beta one")
    await send(stack, "alpha two", a)
    async with httpx.AsyncClient() as c:
        da = (await c.get(f"{stack.url}/api/chat/sessions/{a}")).json()
        db = (await c.get(f"{stack.url}/api/chat/sessions/{b}")).json()
    assert [m["content"] for m in da["messages"]] == ["alpha one", "answer to alpha one", "alpha two", "answer to alpha two"]
    assert [m["content"] for m in db["messages"]] == ["beta one", "answer to beta one"]
    seqs = [m["seq"] for m in da["messages"]]
    assert seqs == sorted(seqs)


async def test_search_matches_titles_and_message_content(stack):
    a = await send(stack, "gardening tips")
    stack.backend.queue([*say("Tomatoes need full sun."), finish("stop")])
    await stack.stream({"message": "unrelated title", "session_id": None})
    assert [r["id"] for r in await listing(stack, q="gardening")] == [a]
    assert len(await listing(stack, q="Tomatoes")) == 1
    assert await listing(stack, q="zzzz-nothing") == []


async def test_unfinished_response_after_restart_is_marked_interrupted(stack):
    sid = await send(stack, "hello")
    await stack.executor.query("UPDATE messages SET finish_reason = 'streaming' WHERE session_id = ? AND role = 'assistant'", [sid])
    async with httpx.AsyncClient() as c:
        data = (await c.get(f"{stack.url}/api/chat/sessions/{sid}")).json()
    assert data["messages"][-1]["status"] == "interrupted"
    row = (await stack.executor.query("SELECT finish_reason FROM messages WHERE session_id = ? AND role='assistant'", [sid])).rows[0]
    assert row["finish_reason"] == "interrupted"


async def test_system_prompt_roundtrip_and_is_used_next_turn(stack):
    async with httpx.AsyncClient() as c:
        r = (await c.get(f"{stack.url}/api/system-prompt")).json()
        assert "prompt" in r and len(r["prompt"]) > 20  # default BASE_PROMPT when unset
        r = await c.put(f"{stack.url}/api/system-prompt", json={"prompt": "Always answer in French."})
        assert r.status_code == 200
        assert (await c.get(f"{stack.url}/api/system-prompt")).json()["prompt"] == "Always answer in French."
    await send(stack, "bonjour")
    assert "Always answer in French." in stack.backend.requests[-1]["messages"][0]["content"]


async def test_unknown_api_route_is_json_404_and_spa_serves_index(stack):
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{stack.url}/api/nope")
        assert r.status_code == 404 and r.headers["content-type"].startswith("application/json")
        page = await c.get(f"{stack.url}/some/client/route")
        assert page.status_code == 200 and "<title>t</title>" in page.text
        assert page.headers["cache-control"] == "no-store"
        assert "default-src 'self'" in page.headers["content-security-policy"]
        assert page.headers["x-content-type-options"] == "nosniff"


async def test_head_requests_work_for_uptime_monitors(stack):
    async with httpx.AsyncClient() as c:
        for path in ("/", "/api/health", "/some/spa/route"):
            r = await c.head(f"{stack.url}{path}")
            assert r.status_code == 200, (path, r.status_code)
        assert (await c.head(f"{stack.url}/api/health")).headers["cache-control"] == "no-store"
