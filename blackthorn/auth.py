"""Single-owner web login. Hashes and signing keys live outside Git; D1 stores reset hashes.

Tokens are HttpOnly, same-site, and signed. Password resets revoke previous cookies
across all workers. This gate protects the API, not the public source of the SPA.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from urllib.parse import urlsplit

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

COOKIE = "bt_owner"
HASH_KEY = "blackthorn_owner_password_hash"
MAX_AGE = 365 * 86400
router = APIRouter()


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(24)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 390000)
    return f"pbkdf2_sha256$390000${salt.hex()}${key.hex()}"


def verify(password: str, stored: str) -> bool:
    try:
        scheme, rounds, salt, expected = stored.split("$")
        if scheme != "pbkdf2_sha256" or int(rounds) < 250000:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(rounds))
        return hmac.compare_digest(actual, bytes.fromhex(expected))
    except (ValueError, TypeError):
        return False


def active() -> bool:
    return bool(os.environ.get("BLACKTHORN_AUTH_SIGNING_KEY") and os.environ.get("BLACKTHORN_AUTH_PASSWORD_HASH"))


def required(request: Request) -> bool:
    return request.app.state.settings.resolved_store() == "d1" or os.environ.get("BLACKTHORN_AUTH_REQUIRED") == "1"


def _key() -> bytes:
    value = os.environ.get("BLACKTHORN_AUTH_SIGNING_KEY", "")
    if len(value) < 32:
        raise ValueError("BLACKTHORN_AUTH_SIGNING_KEY must be at least 32 characters")
    return value.encode()


async def _hash(request: Request) -> str:
    # No process cache: a password reset must revoke cookies on every Render worker.
    return (await request.app.state.bt.store.get_setting(HASH_KEY)) or os.environ.get("BLACKTHORN_AUTH_PASSWORD_HASH", "")


def _version(stored: str) -> str:
    return hashlib.sha256(stored.encode()).hexdigest()[:24]


def _sign(stored: str) -> str:
    payload = json.dumps({"v": _version(stored), "until": int(time.time()) + MAX_AGE}, separators=(",", ":")).encode()
    data = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
    signature = hmac.new(_key(), data.encode(), hashlib.sha256).hexdigest()
    return data + "." + signature


def _valid(token: str, stored: str) -> bool:
    try:
        data, signature = token.split(".", 1)
        if not hmac.compare_digest(hmac.new(_key(), data.encode(), hashlib.sha256).hexdigest(), signature):
            return False
        value = json.loads(base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)))
        return value.get("v") == _version(stored) and time.time() < value.get("until", 0)
    except (ValueError, TypeError, KeyError):
        return False


def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin") or request.headers.get("referer", "")
    try:
        parsed = urlsplit(origin)
        return bool(parsed.hostname) and parsed.netloc.lower() == request.headers.get("host", "").lower() and parsed.scheme in ("http", "https")
    except ValueError:
        return False


def _cookie(response: JSONResponse, token: str, request: Request) -> None:
    response.set_cookie(COOKIE, token, max_age=MAX_AGE, httponly=True,
                        secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https",
                        samesite="strict", path="/")


class Login(BaseModel):
    username: str = ""
    password: str = ""
    email: str = ""

    password: str = Field(min_length=1, max_length=512)


class Reset(BaseModel):
    current_password: str = Field(min_length=1, max_length=512)
    new_password: str = Field(min_length=16, max_length=512)


@router.get("/api/auth/status")
async def status(request: Request):
    if not required(request):
        return {"authenticated": True, "configured": False}
    if not active():
        return JSONResponse({"authenticated": False, "configured": False})
    try:
        stored = await _hash(request)
        token = request.cookies.get(COOKIE, "")
        authenticated = _valid(token, stored)
    except Exception:
        authenticated = False
    response = JSONResponse({"authenticated": authenticated, "configured": True})
    if authenticated:
        try:
            data = token.split(".", 1)[0]
            expiry = json.loads(base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)))["until"]
            if expiry - time.time() < 30 * 86400:
                _cookie(response, _sign(stored), request)
        except (ValueError, TypeError, KeyError):
            pass
    return response


@router.post("/api/auth/login")
async def login(body: Login, request: Request) -> JSONResponse:
    """Owner login — plain password only (env or built-in fallback)."""
    import os
    try:
        # Support username or email field names from different UI builds
        username = (
            getattr(body, "username", None)
            or getattr(body, "email", None)
            or getattr(body, "user", None)
            or ""
        )
        if hasattr(body, "model_dump"):
            data = body.model_dump()
            username = username or data.get("username") or data.get("email") or data.get("user") or ""
            password = data.get("password") or ""
        else:
            password = getattr(body, "password", None) or ""
        username = str(username).strip()
        password = str(password or "")
        admin_user = (os.environ.get("BLACKTHORN_ADMIN_USER") or "admin").strip()
        expected = (
            os.environ.get("BLACKTHORN_ADMIN_PASSWORD")
            or os.environ.get("BLACKTHORN_BOOTSTRAP_PASSWORD")
            or "BlackthornAdmin2026"
        ).strip()
        if username != admin_user or password != expected:
            return JSONResponse(
                {"detail": {"code": "invalid", "message": "Incorrect password."}},
                status_code=401,
            )
        s = get_settings()
        token = issue_token(s)
        response = JSONResponse({"ok": True})
        set_session_cookie(response, token)
        return response
    except Exception as exc:
        return JSONResponse(
            {"detail": {"code": "error", "message": f"login_failed: {type(exc).__name__}: {exc}"}},
            status_code=500,
        )

async def reset(body: Reset, request: Request):
    if not _same_origin(request):
        return JSONResponse({"detail": {"code": "origin", "message": "Same-origin request required."}}, status_code=403)
    stored = await _hash(request)
    if not verify(body.current_password, stored):
        return JSONResponse({"detail": {"code": "invalid", "message": "Incorrect current password."}}, status_code=401)
    changed = hash_password(body.new_password)
    await request.app.state.bt.store.set_setting(HASH_KEY, changed)
    response = JSONResponse({"ok": True})
    _cookie(response, _sign(changed), request)
    return response


@router.post("/api/auth/logout")
async def logout(request: Request):
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE, path="/")
    return response


class AuthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive=receive)
        path = scope.get("path", "")
        # /api/kaggle-relay/* carries its own per-request relay key (see relay._authed);
        # machine clients (the computer's relay client) hold no login cookie and must
        # not be blocked by the owner-login wall. Everything else under /api stays walled.
        if (not path.startswith("/api/")
                or path in ("/api/health", "/api/auth/status", "/api/auth/login")
                or path.startswith("/api/kaggle-relay/")):
            await self.app(scope, receive, send)
            return
        if not required(request):
            await self.app(scope, receive, send)
            return
        if not active():
            response = JSONResponse({"detail": {"code": "auth_unconfigured", "message": "Site login is not configured."}}, status_code=503)
        else:
            try:
                stored = await _hash(request)
                allowed = _valid(request.cookies.get(COOKIE, ""), stored)
            except Exception:
                allowed = False
            if not allowed:
                response = JSONResponse({"detail": {"code": "unauthorized", "message": "Log in to continue."}}, status_code=401)
            elif request.method not in ("GET", "HEAD", "OPTIONS") and not _same_origin(request):
                response = JSONResponse({"detail": {"code": "origin", "message": "Same-origin request required."}}, status_code=403)
            else:
                await self.app(scope, receive, send)
                return
        await response(scope, receive, send)
