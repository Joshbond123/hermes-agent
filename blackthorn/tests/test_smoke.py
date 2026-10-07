import httpx

from .fakes import call, finish, say, think, usage


async def test_stack_boots_and_health(stack):
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{stack.url}/api/health")
    assert r.status_code == 200 and r.json()["ok"] is True


async def test_greeting_streams_without_tools(stack):
    stack.backend.queue([think("simple greeting"), *say("Hello there! How can I help?", 5), usage(90, 9), finish("stop")])
    out = await stack.stream({"message": "Hi"})
    assert out.status == 200, out.error_body
    assert out.text == "Hello there! How can I help?"
    assert out.end["status"] == "stop"
    assert not out.of("tool.start")
    # tools were offered to the model, but it chose not to use any
    assert stack.backend.requests[0]["tools"]
    assert stack.backend.requests[0]["tool_choice"] == "auto"
