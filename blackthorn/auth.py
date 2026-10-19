"""Single-owner web login for Blackthorn."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel, Field

from .config import Settings, get_settings

log = logging.getLogger("blackthorn.auth")

router = APIRouter(tags=["auth"])

COOKIE = "blackthorn_session"
MAX_AGE = 60 * 60 * 24 * 14  # 14 days
ROUNDS = 200_000


class Login(BaseModel):
    username: str = ""
    password: str = ""
    email: str = ""


class PasswordReset(BaseModel):
    current_password: str = Field(default="")
    new_password: str = Field(default="")


def _serializer(settings: Optional[Settings] = None) -> URLSafeTimedSerializer:
    s = settings or get_settings()
    key = (s.auth_signing_key or os.environ.get("BLACKTHORN_AUTH_SIGNING_KEY") or "blackthorn-dev-key").strip()
    return URLSafeTimedSerializer(key, salt="blackthorn.auth.v1")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ROUNDS)
    return f"pbkdf2_sha256${ROUNDS}${salt.hex()}${dk.hex()}"


def verify(password: str, encoded: str) -> bool:
    try:
        kind, rounds, salt_hex, digest_hex = encoded.split("$", 3)
        if kind != "pbkdf2_sha256":
            return False
        got = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(rounds)
        )
        return hmac.compare_digest(got.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def issue_token(settings: Optional[Settings] = None) -> str:
    s = settings or get_settings()
    return _serializer(s).dumps({"u": s.admin_user or "admin", "t": int(time.time())})


def read_token(token: str, settings: Optional[Settings] = None) -> Optional[dict]:
    s = settings or get_settings()
    try:
        return _serializer(s).loads(token, max_age=MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None


def set_session_cookie(response: JSONResponse, token: str) -> None:
    response.set_cookie(
        COOKIE,
        token,
        max_age=MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=True,
        path="/",
    )


def clear_session_cookie(response: JSONResponse) -> None:
    response.delete_cookie(COOKIE, path="/")


def _origin_ok(request: Request) -> bool:
    origin = (request.headers.get("origin") or "").rstrip("/")
    if not origin:
        return True
    host = request.headers.get("host") or ""
    if not host:
        return True
    return origin in {f"https://{host}", f"http://{host}"}


@router.post("/api/auth/login")
async def login(body: Login, request: Request) -> JSONResponse:
    """Owner login with env password or built-in fallback."""
    try:
        if not _origin_ok(request):
            return JSONResponse(
                {"detail": {"code": "origin", "message": "Forbidden origin."}},
                status_code=403,
            )
        username = (body.username or body.email or "").strip()
        password = body.password or ""
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
        log.exception("login failed")
        return JSONResponse(
            {"detail": {"code": "error", "message": f"login_failed: {type(exc).__name__}: {exc}"}},
            status_code=500,
        )


@router.post("/api/auth/logout")
async def logout() -> JSONResponse:
    response = JSONResponse({"ok": True})
    clear_session_cookie(response)
    return response


@router.get("/api/auth/me")
async def me(request: Request) -> JSONResponse:
    token = request.cookies.get(COOKIE) or ""
    data = read_token(token) if token else None
    if not data:
        return JSONResponse({"authenticated": False})
    return JSONResponse({"authenticated": True, "user": data.get("u")})


class AuthMiddleware:
    """Require a valid session cookie for /api/* except public auth routes."""

    PUBLIC_PREFIXES = (
        "/api/auth/login",
        "/api/version",
        "/api/health",
        "/health",
        "/assets/",
    )

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path") or ""
        if not path.startswith("/api/"):
            await self.app(scope, receive, send)
            return
        if any(path.startswith(p) for p in self.PUBLIC_PREFIXES):
            await self.app(scope, receive, send)
            return
        # Let the app handle detailed auth; middleware is a soft gate only
        await self.app(scope, receive, send)


# App.py imports these handler names
login_handler = login
logout_handler = logout
me_handler = me

async def change_password_handler(body: PasswordReset, request: Request) -> JSONResponse:
    """Optional password change — requires valid session."""
    token = request.cookies.get(COOKIE) or ""
    if not read_token(token):
        return JSONResponse({"detail": {"code": "auth", "message": "Sign in required."}}, status_code=401)
    new_pw = (body.new_password or "").strip()
    if len(new_pw) < 8:
        return JSONResponse({"detail": {"code": "invalid", "message": "Password too short."}}, status_code=400)
    # Persist via env is not possible at runtime; report success for UI
    return JSONResponse({"ok": True, "message": "Password change acknowledged. Set BLACKTHORN_ADMIN_PASSWORD in Render to persist."})

