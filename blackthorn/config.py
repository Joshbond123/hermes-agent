"""Runtime settings. Every secret comes from the environment (Render) or from D1 — never from source."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # ---- storage -------------------------------------------------------------------------
    store: str = field(default_factory=lambda: os.environ.get("BLACKTHORN_STORE", "").strip().lower())
    sqlite_path: str = field(default_factory=lambda: os.environ.get("BLACKTHORN_SQLITE_PATH", "blackthorn-dev.sqlite3"))
    session_source: str = "blackthorn-studio"  # value of sessions.source for chats created here

    # ---- model context budget (MUST match the Kaggle engine window: QWEN38_NUM_CTX, default 8192) ----
    # When these disagree the engine rejects long steps (ContextOverflow) and the run
    # thrashes on trimming — one of the real causes of mid-task interruptions.
    context_tokens: int = field(default_factory=lambda: _int("BLACKTHORN_CONTEXT_TOKENS", 8192))
    completion_reserve_tokens: int = field(default_factory=lambda: _int("BLACKTHORN_COMPLETION_RESERVE", 2048))
    chars_per_token: float = 3.2  # conservative for English + code

    # ---- agent loop limits ----------------------------------------------------------------
    # Defaults are sized for multi-hour autonomous work, not a short chat turn.
    # Env vars still override. Per-step cap exists only as a safety valve against
    # a single model response exploding into dozens of calls.
    max_steps: int = field(default_factory=lambda: _int("BLACKTHORN_MAX_STEPS", 160))
    max_tool_calls: int = field(default_factory=lambda: _int("BLACKTHORN_MAX_TOOL_CALLS", 400))
    max_calls_per_step: int = field(default_factory=lambda: _int("BLACKTHORN_MAX_CALLS_PER_STEP", 16))  # concurrency
    max_repeat_calls: int = 2          # identical (tool, args) executions allowed per run
    max_malformed_calls: int = 8
    run_timeout_s: float = field(default_factory=lambda: _float("BLACKTHORN_RUN_TIMEOUT", 14400.0))

    # ---- network timeouts -----------------------------------------------------------------
    llm_connect_timeout_s: float = 20.0
    llm_idle_timeout_s: float = field(default_factory=lambda: _float("BLACKTHORN_LLM_IDLE_TIMEOUT", 240.0))
    tool_timeout_s: float = 180.0
    heartbeat_s: float = 12.0

    # ---- tool output budgets (characters handed back to the model) ---------------------------
    tool_result_chars: int = 3200
    ui_result_chars: int = 1200

    # ---- attachments ----------------------------------------------------------------------
    max_attachments: int = 5
    max_attachment_chars: int = 200_000
    attachment_preview_chars: int = 1500

    # ---- history --------------------------------------------------------------------------
    history_messages: int = 48
    run_ttl_s: float = 900.0           # finished runs stay resumable this long
    max_active_runs: int = 4

    def resolved_store(self) -> str:
        if self.store in ("d1", "sqlite"):
            return self.store
        has_d1 = all(os.environ.get(k) for k in ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_D1_DATABASE_ID"))
        return "d1" if has_d1 else "sqlite"


def get_settings() -> Settings:
    return Settings()
