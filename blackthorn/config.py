"""Runtime configuration for the Blackthorn layer.

Rules
-----
* Secrets (Cloudflare token, Kaggle token, gateway key, search keys) are read from the
  **environment only** — there are no fallback literals.  The repository is public.
* Values are read lazily through functions so tests can monkeypatch the environment.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

#: ``sessions.source`` value that marks a row as a Blackthorn chat.  Tests use a different
#: value (``BLACKTHORN_SESSION_SOURCE``) so they never touch real conversations.
DEFAULT_SESSION_SOURCE = "blackthorn-studio"

DEFAULT_MODEL_ALIAS = "Qwen3.8-27B-Uncensored"
KAGGLE_KERNEL_SLUG = "qwen3-8-27b-uncensored-api-server-dual-t4"
KAGGLE_KERNEL_TITLE = "Qwen3.8-27B-Uncensored API Server (dual T4)"
DEFAULT_AUTO_OFF_MINUTES = 15


class MissingSetting(RuntimeError):
    """A required environment variable is not set."""


def env(name: str, default: Optional[str] = None) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default or ""
    return value.strip()


def require(name: str) -> str:
    value = env(name)
    if not value:
        raise MissingSetting(f"Environment variable {name} is not set")
    return value


def session_source() -> str:
    return env("BLACKTHORN_SESSION_SOURCE", DEFAULT_SESSION_SOURCE)


def cloudflare_account_id() -> str:
    return require("CLOUDFLARE_ACCOUNT_ID")


def cloudflare_database_id() -> str:
    return require("CLOUDFLARE_D1_DATABASE_ID")


def cloudflare_token() -> str:
    return require("CLOUDFLARE_API_TOKEN")


def d1_query_url() -> str:
    return (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{cloudflare_account_id()}/d1/database/{cloudflare_database_id()}/query"
    )


def kaggle_username() -> str:
    return env("KAGGLE_USERNAME", "joshbond123")


def kaggle_token() -> str:
    return require("KAGGLE_API_TOKEN")


def kaggle_kernel_ref() -> str:
    return f"{kaggle_username()}/{KAGGLE_KERNEL_SLUG}"


def model_alias() -> str:
    return env("BLACKTHORN_MODEL_ALIAS", DEFAULT_MODEL_ALIAS)


def stable_dashboard_token() -> str:
    """A session token that survives restarts but is not stored in the repository.

    Render's free tier restarts the process often.  A random per-boot token would
    invalidate every open tab, so the token is derived from a deployment secret.
    (The dashboard injects it into ``index.html``; see the security note in
    ``docs/blackthorn/DEPLOYMENT.md``.)
    """
    explicit = env("HERMES_DASHBOARD_SESSION_TOKEN")
    if explicit:
        return explicit
    secret = env("CLOUDFLARE_API_TOKEN") or env("KAGGLE_API_TOKEN")
    if not secret:
        return ""
    digest = hashlib.sha256(("blackthorn-session:" + secret).encode("utf-8")).hexdigest()
    return "bt-" + digest[:48]
