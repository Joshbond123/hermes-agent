"""PandaStack key-pool tests: verification, real usage projection and rotation.

All provider data comes from a mock that returns the same JSON shapes the live API
returns (``/v1/me``, ``/v1/usage``), including the real-world case where
``monthly_budget_usd`` is 0 because the workspace has no spend cap.
"""

from __future__ import annotations

import time

import httpx
import pytest

from blackthorn.panda_keys import PandaKeyPool, fingerprint, parse_keys

KEY_A = "pds_" + "a" * 32
KEY_B = "pds_" + "b" * 32
KEY_C = "pds_" + "c" * 32


class Store:
    def __init__(self):
        self.values = {}

    async def get_setting(self, key):
        return self.values.get(key, "")

    async def set_setting(self, key, value):
        self.values[key] = value


def make_transport(*, cost_a=0.0, budget_a=0.0, cost_b=0.0, budget_b=0.0, window_start=None,
                   invalid=(), status_a=None):
    """MockTransport mirroring the documented /v1/me and /v1/usage responses."""
    window_start = window_start or time.strftime("%Y-%m-01T00:00:00Z", time.gmtime())

    def handler(request: httpx.Request):
        key = (request.headers.get("authorization") or "").replace("Bearer ", "")
        if key in invalid:
            return httpx.Response(401, json={"error": "invalid api token"})
        if request.url.path == "/v1/me":
            return httpx.Response(200, json={"auth_method": "token", "user_id": "workspace-" + key[-1],
                                             "current_org": None, "orgs": []})
        if request.url.path == "/v1/usage":
            if key == KEY_A and status_a:
                return httpx.Response(status_a, json={"error": "quota"})
            cost, budget = (cost_a, budget_a) if key == KEY_A else (cost_b, budget_b)
            return httpx.Response(200, json={
                "from": window_start, "to": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "monthly_budget_micros": int(budget * 1_000_000), "monthly_budget_usd": budget,
                "pricing": {"cpu_usd_per_hour": 0.054},
                "rollup": {"workspace": "w", "events": 3, "cpu_seconds": 60.0, "gb_seconds": 10.0,
                           "cost_micros": int(cost * 1_000_000), "cost_usd": cost}})
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_keys_are_verified_and_fingerprinted_never_logged():
    async with httpx.AsyncClient(transport=make_transport()) as http:
        pool = PandaKeyPool([KEY_A, KEY_B], http)
        states = await pool.probe_all()
        assert all(state.valid for state in states)
        public = pool.snapshot()
        assert all("key" not in row for row in public)
        assert KEY_A not in str(public) and KEY_B not in str(public)
        assert public[0]["fingerprint"] == fingerprint(KEY_A)


@pytest.mark.asyncio
async def test_invalid_active_key_rotates_to_a_verified_one():
    store = Store()
    async with httpx.AsyncClient(transport=make_transport(invalid={KEY_A})) as http:
        pool = PandaKeyPool([KEY_A, KEY_B], http, store)
        active = await pool.ensure_active()
        assert active.fingerprint == fingerprint(KEY_B)
        assert await store.get_setting("pandastack_active_key_fingerprint") == fingerprint(KEY_B)


@pytest.mark.asyncio
async def test_projection_rotates_five_minutes_before_the_configured_limit():
    """cost 0.99 of a 1.00 USD budget with a month-to-date burn rate ⇒ rotate now."""
    # The provider reports its usage window start; the burn rate is derived from it.
    window = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 26 * 3600))
    async with httpx.AsyncClient(transport=make_transport(cost_a=0.999, budget_a=1.0, window_start=window)) as http:
        pool = PandaKeyPool([KEY_A, KEY_B], http, Store())
        state = await pool.probe(KEY_A)
        assert state.has_cap and state.remaining_usd is not None
        projected = state.seconds_to_limit()
        assert projected is not None and projected < 300, f"expected <5 min left, got {projected}"
        rotated = await pool.ensure_active()
        assert rotated.fingerprint == fingerprint(KEY_B)


@pytest.mark.asyncio
async def test_no_configured_budget_means_no_invented_deadline():
    async with httpx.AsyncClient(transport=make_transport(cost_a=5.0, budget_a=0.0)) as http:
        pool = PandaKeyPool([KEY_A, KEY_B], http, Store())
        state = await pool.probe(KEY_A)
        assert state.budget_usd == 0.0
        assert state.seconds_to_limit() is None, "an absent cap must not be guessed"
        active = await pool.ensure_active()
        assert active.fingerprint == fingerprint(KEY_A), "no rotation without evidence"


@pytest.mark.asyncio
async def test_provider_quota_rejection_rotates_and_records_without_secrets():
    store = Store()
    async with httpx.AsyncClient(transport=make_transport()) as http:
        pool = PandaKeyPool([KEY_A, KEY_B], http, store)
        await pool.probe_all()
        rotated = await pool.note_failure(KEY_A, 429, "rate limited")
        assert rotated is not None and rotated.fingerprint == fingerprint(KEY_B)
        history = await store.get_setting("pandastack_key_rotations")
        assert "HTTP 429" in history
        assert KEY_A not in history and KEY_B not in history


@pytest.mark.asyncio
async def test_when_no_replacement_key_is_valid_the_pool_says_so():
    async with httpx.AsyncClient(transport=make_transport(invalid={KEY_B, KEY_C})) as http:
        pool = PandaKeyPool([KEY_A, KEY_B, KEY_C], http, Store())
        await pool.probe_all()
        assert await pool.rotate(reason="test") is None
        assert pool.active_key() == KEY_A, "the active key is unchanged when nothing better exists"


def test_parse_keys_accepts_commas_newlines_and_deduplicates():
    assert parse_keys(f"{KEY_A}, {KEY_B}\n{KEY_A};") == [KEY_A, KEY_B]
