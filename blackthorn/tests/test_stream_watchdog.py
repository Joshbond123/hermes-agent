"""Stream watchdog: a model that never starts must fail fast, not after the idle window.

Uses a real local HTTP server (no mocks of httpx) so the timeouts are exercised on sockets.
"""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from blackthorn.llm import LLMError, stream_chat, tunnel_headers
from blackthorn.route import Route

SEEN_HEADERS: list = []


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # silence
        pass

    def do_GET(self):  # route health probe: the server is alive, the model may still be silent
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        SEEN_HEADERS.append({k.lower(): v for k, v in self.headers.items()})
        mode = self.path.strip("/").split("/")[0]
        if mode == "hang":            # accept the request, send nothing at all
            time.sleep(30)
            return
        if mode == "midstall":        # headers + one event, then silence
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            data = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()
            time.sleep(30)
            return
        body = (b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                b"data: [DONE]\n\n")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def upstream():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


async def _run(base, mode, first_byte, idle):
    route = Route(url=f"{base}/{mode}", api_key="k", model="m", status="READY")
    async with httpx.AsyncClient() as client:
        events, err, t0 = [], None, time.monotonic()
        try:
            async for ev in stream_chat(route, [{"role": "user", "content": "x"}], client=client,
                                        idle_timeout=idle, first_byte_timeout=first_byte):
                events.append(ev)
        except LLMError as exc:
            err = exc
        return events, err, time.monotonic() - t0


async def test_healthy_stream_still_completes(upstream):
    events, err, _ = await _run(upstream, "ok", first_byte=5, idle=5)
    assert err is None
    assert [e["v"] for e in events if e["t"] == "content"] == ["ok"]


async def test_silent_model_fails_at_first_byte_watchdog(upstream):
    # Production idle window is 120 s; the first-byte watchdog must fire well before it.
    events, err, elapsed = await _run(upstream, "hang", first_byte=2, idle=120)
    assert isinstance(err, LLMError) and err.code == "gpu_stalled" and err.retryable
    assert elapsed < 6, f"stall took {elapsed:.1f}s; first-byte watchdog did not fire"


async def test_dead_route_is_reported_as_unreachable_not_as_a_stall(upstream):
    # nothing listens on this port: the route probe fails and the error says so (retry on a fresh route)
    import socket
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); dead = sock.getsockname()[1]; sock.close()
    events, err, elapsed = await _run(f"http://127.0.0.1:{dead}", "x", first_byte=2, idle=120)
    assert isinstance(err, LLMError) and err.code in ("gpu_unreachable", "gpu_dropped") and err.retryable
    assert elapsed < 6


async def test_stall_after_first_chunk_uses_idle_window(upstream):
    events, err, elapsed = await _run(upstream, "midstall", first_byte=2, idle=3)
    assert [e["v"] for e in events if e["t"] == "content"] == ["hi"]
    assert isinstance(err, LLMError) and err.code == "gpu_stalled"
    assert 2.5 < elapsed < 8, f"idle window not honoured ({elapsed:.1f}s)"


def test_defaults_are_bounded():
    from blackthorn.config import Settings
    s = Settings()
    assert s.llm_first_byte_timeout_s <= 60
    assert s.llm_idle_timeout_s <= 120


async def test_inrok_hosts_get_interstitial_bypass_header_only_there(upstream):
    assert tunnel_headers("https://blackthorn.share.inrok.in") == {"skip_zrok_interstitial": "true"}
    assert tunnel_headers("https://evil.example.com") == {}
    assert tunnel_headers("https://share.inrok.in.evil.com") == {}
    # the header is actually sent on the wire for a stream to a same-host route
    SEEN_HEADERS.clear()
    await _run(upstream, "ok", first_byte=5, idle=5)
    assert SEEN_HEADERS and "skip_zrok_interstitial" not in SEEN_HEADERS[-1]
