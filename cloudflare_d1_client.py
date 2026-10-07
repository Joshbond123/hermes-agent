"""
Cloudflare D1 Exclusive Database & Memory Engine + Kaggle GPU Controller for Hermes Agent.
Connects Hermes Agent exclusively to Cloudflare D1 Database:
  - Account / database IDs come from CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_D1_DATABASE_ID (Render env).
  - No credential is stored in this file: secrets come from the environment or from D1.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("cloudflare_d1")

CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_D1_DATABASE_ID = os.environ.get("CLOUDFLARE_D1_DATABASE_ID", "")

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
KAGGLE_API_TOKEN = os.environ.get("KAGGLE_API_TOKEN", "")
CACHE_DATASET_NAME = "qwen38-27b-uncensored-model-cache"
KAGGLE_KERNEL_SLUG = "qwen3-8-27b-uncensored-api-server-dual-t4"
KAGGLE_KERNEL_ID = f"{KAGGLE_USERNAME}/{KAGGLE_KERNEL_SLUG}"
# Optional override only. The live gateway key is published to D1 by the Kaggle notebook and read at
# runtime through _gateway_api_key(); it is never a constant in source and never written back here.
CYBER_ORNITH_API_KEY = os.environ.get("CYBER_ORNITH_API_KEY", "")
CYBER_ORNITH_MODEL = "Qwen3.8-27B-Uncensored"
TELEMETRY_TOPIC = "qwen38_kaggle_blackthorn_8492"

D1_API_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}"
    f"/d1/database/{CLOUDFLARE_D1_DATABASE_ID}/query"
)


_GATEWAY_KEY_CACHE: Tuple[float, str] = (0.0, "")


def _gateway_api_key(max_age: float = 20.0) -> str:
    """Bearer key of the live Kaggle gateway, read from the D1 row the notebook publishes."""
    global _GATEWAY_KEY_CACHE
    now = time.monotonic()
    ts, cached = _GATEWAY_KEY_CACHE
    if cached and (now - ts) < max_age:
        return cached
    key = ""
    try:
        rows = d1_query("SELECT api_key FROM kaggle_gpu_state WHERE id = 'primary' LIMIT 1;", timeout=8.0)
        if rows:
            key = str(rows[0].get("api_key") or "").strip()
    except Exception as exc:  # network dependent
        logger.debug("gateway key lookup note: %s", exc)
    key = key or CYBER_ORNITH_API_KEY
    if key:
        _GATEWAY_KEY_CACHE = (now, key)
    return key


# Fields that are safe to show any visitor of the web UI. Everything else (tunnel URL, keys,
# account / database identifiers) stays server-side.
_PUBLIC_STATUS_FIELDS = (
    "active", "booting", "status", "display_status", "engine_state", "busy", "model", "model_loaded",
    "gpu_info", "progress_pct", "progress_step", "progress_stage", "progress_total_stages", "progress_kind",
    "progress_bytes_done", "progress_bytes_total", "progress_label", "progress_stalled", "elapsed_seconds", "worker_status", "quota",
    "auto_off", "boot_seconds", "cache_source", "download_skipped", "error", "kaggle_kernel",
)


def public_gpu_status(state: Dict[str, Any]) -> Dict[str, Any]:
    """Allow-listed, secret-free view of the GPU state for HTTP responses."""
    out = {k: state[k] for k in _PUBLIC_STATUS_FIELDS if state.get(k) is not None}
    out["online"] = bool(state.get("active")) and not bool(state.get("booting"))
    out["has_endpoint"] = bool(state.get("tunnel_url"))
    return out

_SCHEMA_INITIALIZED = False
_SCHEMA_LOCK = threading.Lock()


def d1_query(sql: str, params: Optional[List[Any]] = None, timeout: float = 15.0) -> List[Dict[str, Any]]:
    """Execute a parameterized SQL query directly against Cloudflare D1 and return rows."""
    payload: Dict[str, Any] = {"sql": sql}
    if params is not None:
        clean_params = []
        for p in params:
            if isinstance(p, (bytes, bytearray)):
                clean_params.append(p.decode("utf-8", errors="replace"))
            else:
                clean_params.append(p)
        payload["params"] = clean_params

    req = urllib.request.Request(
        D1_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "HermesAgent-CloudflareD1/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not data.get("success"):
        raise RuntimeError(f"Cloudflare D1 error: {data.get('errors')}")
    results = data.get("result") or []
    if results and isinstance(results[0], dict):
        return results[0].get("results") or []
    return []


def ensure_d1_schema() -> None:
    """Ensure all required Hermes Agent tables exist in Cloudflare D1."""
    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_INITIALIZED:
            return
        try:
            d1_query(
                """CREATE TABLE IF NOT EXISTS hermes_memories (
                    id TEXT PRIMARY KEY,
                    target TEXT NOT NULL DEFAULT 'memory',
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'factual',
                    importance REAL NOT NULL DEFAULT 0.8,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    session_id TEXT NOT NULL DEFAULT '',
                    profile TEXT NOT NULL DEFAULT 'default',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );"""
            )
            d1_query(
                """CREATE TABLE IF NOT EXISTS kaggle_gpu_state (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    tunnel_url TEXT NOT NULL DEFAULT '',
                    api_key TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT 'Qwen3.8-27B-Uncensored',
                    gpu_info TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL
                );"""
            )
            _SCHEMA_INITIALIZED = True
        except Exception as exc:
            logger.warning("Cloudflare D1 schema check warning: %s", exc)


# ==============================================================================
# 1. CLOUDFLARE D1 MEMORY STORE OPERATIONS
# ==============================================================================

def d1_get_memory_entries(target: str = "memory", profile: str = "default") -> List[str]:
    """Read ordered memory entries for target ('memory' or 'user') from Cloudflare D1."""
    ensure_d1_schema()
    rows = d1_query(
        "SELECT content FROM hermes_memories WHERE target = ? AND profile = ? ORDER BY created_at ASC, rowid ASC;",
        [target, profile],
    )
    return [str(r["content"]).strip() for r in rows if str(r.get("content") or "").strip()]


def d1_set_memory_entries(target: str, entries: List[str], profile: str = "default") -> None:
    """Replace memory entries for target in Cloudflare D1 atomically."""
    ensure_d1_schema()
    now = time.time()
    d1_query(
        "DELETE FROM hermes_memories WHERE target = ? AND profile = ?;",
        [target, profile],
    )
    for idx, content in enumerate(entries):
        text = str(content or "").strip()
        if not text:
            continue
        mem_id = hashlib.sha256(f"{profile}:{target}:{idx}:{text}".encode("utf-8")).hexdigest()[:24]
        d1_query(
            """INSERT OR REPLACE INTO hermes_memories
               (id, target, content, memory_type, importance, user_id, session_id, profile, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'default', '', ?, ?, ?);""",
            [
                mem_id,
                target,
                text,
                "preference" if target == "user" else "factual",
                0.8,
                profile,
                now + idx * 0.001,
                now,
            ],
        )


def d1_add_memory(
    content: str,
    target: str = "memory",
    memory_type: str = "factual",
    importance: float = 0.8,
    user_id: str = "default",
    session_id: str = "",
    profile: str = "default",
) -> Dict[str, Any]:
    """Add a single memory item directly to Cloudflare D1."""
    ensure_d1_schema()
    text = str(content or "").strip()
    if not text:
        return {"success": False, "error": "Memory content cannot be empty."}
    now = time.time()
    mem_id = hashlib.sha256(f"{profile}:{target}:{text}".encode("utf-8")).hexdigest()[:24]
    d1_query(
        """INSERT OR REPLACE INTO hermes_memories
           (id, target, content, memory_type, importance, user_id, session_id, profile, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
        [mem_id, target, text, memory_type, float(importance), user_id, session_id, profile, now, now],
    )
    return {
        "success": True,
        "id": mem_id,
        "target": target,
        "content": text,
        "memory_type": memory_type,
        "database_id": CLOUDFLARE_D1_DATABASE_ID,
    }


def d1_search_memories(query: str, top_k: int = 10, profile: str = "default") -> List[Dict[str, Any]]:
    """Search memories stored in Cloudflare D1."""
    ensure_d1_schema()
    q = str(query or "").strip()
    if not q:
        return d1_query(
            "SELECT id, target, content, memory_type, importance, created_at, updated_at "
            "FROM hermes_memories WHERE profile = ? ORDER BY updated_at DESC LIMIT ?;",
            [profile, int(top_k)],
        )
    terms = [t.strip() for t in q.split() if len(t.strip()) >= 2][:6]
    if not terms:
        terms = [q]
    like_clauses = " OR ".join(["LOWER(content) LIKE ?" for _ in terms])
    params: List[Any] = [profile] + [f"%{t.lower()}%" for t in terms] + [int(top_k)]
    rows = d1_query(
        f"SELECT id, target, content, memory_type, importance, created_at, updated_at "
        f"FROM hermes_memories WHERE profile = ? AND ({like_clauses}) "
        f"ORDER BY importance DESC, updated_at DESC LIMIT ?;",
        params,
    )
    if not rows:
        rows = d1_query(
            "SELECT id, target, content, memory_type, importance, created_at, updated_at "
            "FROM hermes_memories WHERE profile = ? ORDER BY updated_at DESC LIMIT ?;",
            [profile, int(top_k)],
        )
    return rows


def d1_delete_memory(memory_id: str) -> Dict[str, Any]:
    """Delete a memory entry by ID or exact content from Cloudflare D1."""
    ensure_d1_schema()
    d1_query("DELETE FROM hermes_memories WHERE id = ? OR content = ?;", [memory_id, memory_id])
    return {"success": True, "deleted": memory_id}


# ==============================================================================
# 2. CLOUDFLARE D1 SESSION & MESSAGE DATABASE SYNC
# ==============================================================================

def d1_sync_session(session_row: Dict[str, Any]) -> None:
    """Upsert a session record into Cloudflare D1's `sessions` table."""
    def _worker():
        try:
            d1_query(
                """INSERT OR REPLACE INTO sessions (
                    id, source, user_id, model, title, started_at, ended_at, end_reason,
                    message_count, tool_call_count, input_tokens, output_tokens, last_activity_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                [
                    session_row.get("id"),
                    session_row.get("source", "cli"),
                    session_row.get("user_id"),
                    session_row.get("model", CYBER_ORNITH_MODEL),
                    session_row.get("title"),
                    session_row.get("started_at", time.time()),
                    session_row.get("ended_at"),
                    session_row.get("end_reason"),
                    session_row.get("message_count", 0),
                    session_row.get("tool_call_count", 0),
                    session_row.get("input_tokens", 0),
                    session_row.get("output_tokens", 0),
                    session_row.get("last_activity_at", time.time()),
                ],
            )
        except Exception as exc:
            logger.debug("Cloudflare D1 session sync note: %s", exc)

    _worker()


def d1_sync_message(session_id_or_row: Any, msg_row: Optional[Dict[str, Any]] = None) -> None:
    """Insert a message record into Cloudflare D1's `messages` table and update session counters."""
    if isinstance(session_id_or_row, dict) and msg_row is None:
        row = dict(session_id_or_row)
    else:
        row = dict(msg_row or {})
        row["session_id"] = str(session_id_or_row)

    session_id = str(row.get("session_id") or "")
    tool_calls_val = row.get("tool_calls")
    if isinstance(tool_calls_val, (list, dict)):
        tool_calls_val = json.dumps(tool_calls_val)

    def _worker():
        try:
            d1_query(
                """INSERT INTO messages (
                    session_id, role, content, tool_call_id, tool_calls, tool_name,
                    timestamp, token_count, finish_reason, reasoning, message_uid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                [
                    session_id,
                    row.get("role", "user"),
                    row.get("content"),
                    row.get("tool_call_id"),
                    tool_calls_val,
                    row.get("tool_name"),
                    row.get("timestamp") or time.time(),
                    row.get("token_count"),
                    row.get("finish_reason"),
                    row.get("reasoning"),
                    row.get("message_uid") or uuid.uuid4().hex,
                ],
            )
            if session_id:
                d1_query(
                    "UPDATE sessions SET message_count = COALESCE(message_count, 0) + 1, last_activity_at = ? WHERE id = ?;",
                    [time.time(), session_id],
                )
        except Exception as exc:
            logger.debug("Cloudflare D1 message sync note: %s", exc)

    _worker()


def hydrate_local_sqlite_from_d1(db_path: Path) -> None:
    """Hydrate local SQLite cache from Cloudflare D1 on startup so Cloudflare D1 is the source of truth."""
    try:
        ensure_d1_schema()
        sessions = d1_query("SELECT * FROM sessions ORDER BY started_at DESC LIMIT 200;")
        messages = d1_query("SELECT * FROM messages ORDER BY timestamp ASC LIMIT 2000;")
        if not sessions and not messages:
            return
        conn = sqlite3.connect(str(db_path))
        with conn:
            for s in sessions:
                conn.execute(
                    """INSERT OR IGNORE INTO sessions (
                        id, source, user_id, model, title, started_at, ended_at, end_reason,
                        message_count, tool_call_count, input_tokens, output_tokens, last_activity_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                    (
                        s.get("id"),
                        s.get("source") or "cli",
                        s.get("user_id"),
                        s.get("model"),
                        s.get("title"),
                        s.get("started_at") or time.time(),
                        s.get("ended_at"),
                        s.get("end_reason"),
                        s.get("message_count") or 0,
                        s.get("tool_call_count") or 0,
                        s.get("input_tokens") or 0,
                        s.get("output_tokens") or 0,
                        s.get("last_activity_at") or s.get("started_at") or time.time(),
                    ),
                )
            for m in messages:
                conn.execute(
                    """INSERT OR IGNORE INTO messages (
                        id, session_id, role, content, tool_call_id, tool_calls, tool_name,
                        timestamp, token_count, finish_reason, reasoning, message_uid
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                    (
                        m.get("id"),
                        m.get("session_id"),
                        m.get("role") or "user",
                        m.get("content"),
                        m.get("tool_call_id"),
                        m.get("tool_calls"),
                        m.get("tool_name"),
                        m.get("timestamp") or time.time(),
                        m.get("token_count"),
                        m.get("finish_reason"),
                        m.get("reasoning"),
                        m.get("message_uid"),
                    ),
                )
        conn.close()
    except Exception as exc:
        logger.debug("Hydrate from Cloudflare D1 note: %s", exc)


# ==============================================================================
# 3. KAGGLE GPU CONTROLLER & TELEMETRY SYNC
# ==============================================================================

def _kaggle_rpc(method: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Call Kaggle KernelsApiService directly via pure Python stdlib urllib."""
    req = urllib.request.Request(
        f"https://api.kaggle.com/v1/kernels.KernelsApiService/{method}",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {KAGGLE_API_TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "kaggle-api/v1.7.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=18) as resp:
        return json.loads(resp.read().decode("utf-8")) or {}


# ── GPU engine state model ────────────────────────────────────────────────
# Status codes written by the Kaggle notebook / turn-on path, mapped to the
# user-facing step text and one of the canonical UI states.
_STEP_MAP: Dict[str, tuple] = {
    "BOOTING_KAGGLE_GPU": (8, "Allocating 2× NVIDIA Tesla T4 GPUs on Kaggle…", "starting"),
    "CHECKING_ENVIRONMENT": (14, "Verifying GPU + internet inside the Kaggle container…", "starting"),
    "CHECKING_CACHE": (22, "Looking for the persistent model copy…", "starting"),
    "CACHE_HIT": (36, "Persistent model found — reusing it, no download", "starting"),
    "DOWNLOADING_MODEL": (30, "Downloading the model (first run only)…", "starting"),
    "MODEL_DOWNLOADED": (38, "Model downloaded and verified", "starting"),
    "VERIFYING_MODEL": (40, "Verifying model integrity (sha256)…", "starting"),
    "CACHE_DATASET_PUBLISHED": (100, "Kaggle Ready — model archived for instant future starts", "ready"),
    "CACHE_DATASET_READY": (100, "Kaggle Ready — persistent model cache is complete", "ready"),
    "CACHE_MISS": (24, "No persistent copy found — downloading once", "starting"),
    "STARTING_INSTALL": (46, "Installing only the missing dependencies…", "starting"),
    "INSTALLING_DEPS": (46, "Installing only the missing dependencies…", "starting"),
    "INSTALLING_OLLAMA": (54, "Installing the inference engine…", "starting"),
    "STARTING_OLLAMA": (56, "Starting the CUDA inference engine…", "starting"),
    "LOADING_MODEL": (68, "Registering the model in the local store…", "starting"),
    "STARTING_GATEWAY": (76, "Starting the model API gateway…", "starting"),
    "TUNNEL_ONLINE": (86, "Tunnel online — warming weights into GPU VRAM…", "starting"),
    "WARMING_GPU": (92, "Loading weights into GPU VRAM…", "starting"),
    "MODEL_READY_COLD": (97, "Model Ready — loading weights on first request", "ready"),
    "MODEL_READY_AND_WARMED": (100, "Kaggle Ready", "ready"),
    "HEARTBEAT_ONLINE": (100, "Kaggle Ready", "ready"),
    "ONLINE": (100, "Kaggle Ready", "ready"),
    "STOPPING_KAGGLE_GPU": (40, "Stopping Kaggle GPU…", "stopping"),
    "GPU_STOPPED_SAVING_QUOTA": (0, "Kaggle Stopped — GPU runtime saved", "stopped"),
    "OFF": (0, "Kaggle Offline", "off"),
    "GPU_UNAVAILABLE": (100, "Model Unavailable — no GPU attached to the Kaggle session", "error"),
    "INTERNET_UNAVAILABLE": (100, "Model Unavailable — Kaggle internet is disabled", "error"),
    "MODEL_CORRUPT": (100, "Model Unavailable — download was corrupted, retry to recover", "error"),
    "BOOT_FAILED": (0, "Kaggle boot failed — see the reason below", "error"),
    "TUNNEL_ERROR": (100, "Connection Lost — the tunnel could not be created", "error"),
}
_TERMINAL_ERROR_STATUSES = ("GPU_UNAVAILABLE", "INTERNET_UNAVAILABLE", "MODEL_CORRUPT", "BOOT_FAILED", "TUNNEL_ERROR")

_ACTIVITY_LOCK = threading.Lock()
_LAST_ACTIVITY_TS = time.time()
_ACTIVE_TASKS = 0
AUTO_OFF_DEFAULT_MINUTES = 0  # permanent GPU: never auto-off by default
AUTO_OFF_CHOICES = (0, 5, 10, 15, 30, 60)  # 0 == never
_AUTO_OFF_KEY = "blackthorn_auto_off_minutes"
_AUTO_OFF_LAST_KEY = "blackthorn_auto_off_last_reason"


def _engine_state_from_status(status: str, active: bool, booting: bool) -> str:
    mapped = _STEP_MAP.get(status)
    if mapped:
        return mapped[2]
    if active:
        return "ready"
    if booting:
        return "starting"
    return "off"


# ── Activity tracking (drives the inactivity auto-off timer) ──────────────
def mark_activity(_reason: str = "") -> None:
    """Record genuine Blackthorn activity: model turns, tool runs, file work."""
    global _LAST_ACTIVITY_TS
    with _ACTIVITY_LOCK:
        _LAST_ACTIVITY_TS = time.time()


def task_started(_reason: str = "") -> None:
    """A unit of work began — the GPU must not be auto-stopped while any is open."""
    global _ACTIVE_TASKS, _LAST_ACTIVITY_TS
    with _ACTIVITY_LOCK:
        _ACTIVE_TASKS += 1
        _LAST_ACTIVITY_TS = time.time()


def task_finished(_reason: str = "") -> None:
    global _ACTIVE_TASKS, _LAST_ACTIVITY_TS
    with _ACTIVITY_LOCK:
        _ACTIVE_TASKS = max(0, _ACTIVE_TASKS - 1)
        _LAST_ACTIVITY_TS = time.time()


def activity_snapshot() -> Dict[str, Any]:
    with _ACTIVITY_LOCK:
        idle = max(0.0, time.time() - _LAST_ACTIVITY_TS)
        return {
            "idle_seconds": round(idle, 1),
            "active_tasks": _ACTIVE_TASKS,
            "busy": _ACTIVE_TASKS > 0,
            "last_activity_at": _LAST_ACTIVITY_TS,
        }


def get_auto_off_minutes() -> int:
    """Configured inactivity window in minutes (0 == never auto-off)."""
    try:
        rows = d1_query("SELECT value FROM state_meta WHERE key = ? LIMIT 1;", [_AUTO_OFF_KEY])
        if rows:
            value = int(float(str(rows[0].get("value") or "").strip()))
            return value if value in AUTO_OFF_CHOICES else AUTO_OFF_DEFAULT_MINUTES
    except Exception as exc:
        logger.debug("auto-off read note: %s", exc)
    return AUTO_OFF_DEFAULT_MINUTES


def set_auto_off_minutes(minutes: int) -> int:
    value = int(minutes)
    if value not in AUTO_OFF_CHOICES:
        raise ValueError(f"minutes must be one of {AUTO_OFF_CHOICES}")
    ensure_d1_schema()
    d1_query(
        "INSERT OR REPLACE INTO state_meta (key, value) VALUES (?, ?);",
        [_AUTO_OFF_KEY, str(value)],
    )
    logger.info("Blackthorn auto-off set to %s minutes", "never" if value == 0 else value)
    return value


def _record_auto_off(reason: str) -> None:
    try:
        d1_query(
            "INSERT OR REPLACE INTO state_meta (key, value) VALUES (?, ?);",
            [_AUTO_OFF_LAST_KEY, reason],
        )
    except Exception:
        pass


# Only these published engine statuses mean "the GPU is up and idle-able".
# Anything else (booting, stopping, offline, unknown) must never be auto-stopped,
# which is what previously let a stale idle clock kill a GPU mid-boot.
READY_STATUSES = {
    "ONLINE", "MODEL_READY", "MODEL_READY_AND_WARMED", "MODEL_READY_COLD",
    "HEARTBEAT_ONLINE", "GPU_ONLINE", "CACHE_DATASET_READY", "CACHE_DATASET_PUBLISHED",
}


def _published_status() -> str:
    try:
        rows = d1_query("SELECT status FROM kaggle_gpu_state WHERE id='primary' LIMIT 1;")
        return str((rows[0] if rows else {}).get("status") or "").upper()
    except Exception:
        return ""


def auto_off_decision() -> Dict[str, Any]:
    """Pure decision helper: should the GPU be stopped right now?"""
    minutes = get_auto_off_minutes()
    snap = activity_snapshot()
    decided = {
        "enabled": minutes > 0,
        "minutes": minutes,
        "idle_seconds": snap["idle_seconds"],
        "active_tasks": snap["active_tasks"],
        "should_stop": False,
        "reason": "",
    }
    if minutes <= 0:
        decided["reason"] = "auto-off disabled (Never)"
        return decided
    cached = _STATUS_CACHE or {}
    if cached.get("booting") or cached.get("engine_state") == "starting":
        decided["reason"] = "GPU is still starting — the timer only runs while it is idle and ready"
        return decided
    published = _published_status()
    if published not in READY_STATUSES:
        decided["reason"] = (f"GPU is not ready yet ({published or 'unknown'}) — "
                             "no shutdown while it boots or stops")
        return decided
    if snap["active_tasks"] > 0:
        decided["reason"] = f"{snap['active_tasks']} task(s) still running"
        return decided
    remaining = minutes * 60 - snap["idle_seconds"]
    if remaining <= 0:
        decided["should_stop"] = True
        decided["reason"] = f"idle for {snap['idle_seconds'] / 60:.1f} min (limit {minutes} min)"
    else:
        decided["reason"] = f"{remaining / 60:.1f} min of inactivity left"
    return decided


_LAST_SAVED_TUNNEL_URL: str = ""
_TUNNEL_CHANGE_HOOKS: List[Any] = []


def add_tunnel_change_hook(fn: Any) -> None:
    """Register a callback fired whenever the GPU tunnel URL changes.

    The chat routers cache the GPU route in-process; without this they would keep
    talking to a tunnel that a kernel restart already killed (HTTP 404).
    """
    if callable(fn) and fn not in _TUNNEL_CHANGE_HOOKS:
        _TUNNEL_CHANGE_HOOKS.append(fn)


def _notify_tunnel_change(url: str) -> None:
    for hook in list(_TUNNEL_CHANGE_HOOKS):
        try:
            hook(url)
        except Exception as exc:
            logger.debug("tunnel change hook note: %s", exc)
_STATUS_CACHE: Dict[str, Any] = {}
_STATUS_CACHE_TS: float = 0.0
_STATUS_CACHE_LOCK = threading.Lock()
_STATUS_REFRESH_IN_FLIGHT = False
_STATUS_REFRESH_TS = 0.0
_QUOTA_CACHE: Optional[Dict[str, Any]] = None
_QUOTA_CACHE_TS: float = 0.0


def _ensure_kaggle_credentials() -> None:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    token_file = kaggle_dir / "access_token"
    if not token_file.exists() or token_file.read_text(encoding="utf-8").strip() != KAGGLE_API_TOKEN:
        token_file.write_text(KAGGLE_API_TOKEN, encoding="utf-8")
        try:
            os.chmod(token_file, 0o600)
        except OSError:
            pass
    os.environ["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME


def _normalize_v1_url(tunnel_url: str) -> str:
    url = str(tunnel_url or "").strip().rstrip("/")
    if not url:
        return ""
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def sync_agent_to_active_kaggle_tunnel(agent: Any, session: Optional[Dict[str, Any]] = None, sid: str = "") -> bool:
    """Ensure a live AIAgent instance is connected to the latest Kaggle GPU tunnel URL."""
    if agent is None:
        return False
    base_url = _normalize_v1_url(os.environ.get("OPENAI_BASE_URL") or _LAST_SAVED_TUNNEL_URL)
    if not base_url or "pending-kaggle-gpu" in base_url:
        try:
            rows = d1_query("SELECT tunnel_url, status FROM kaggle_gpu_state WHERE id = 'primary' LIMIT 1;")
            if rows and rows[0].get("tunnel_url") and rows[0].get("status") != "GPU_STOPPED_SAVING_QUOTA":
                base_url = _normalize_v1_url(rows[0]["tunnel_url"])
        except Exception:
            pass
    if not base_url:
        return False

    cur_base = _normalize_v1_url(getattr(agent, "base_url", "") or getattr(agent, "_base_url", "") or "")
    cur_model = str(getattr(agent, "model", "") or "")
    cur_provider = str(getattr(agent, "provider", "") or "")

    if isinstance(session, dict):
        override = session.get("model_override")
        if isinstance(override, dict) and override.get("model") == "z-ai/glm-5.2":
            session["model_override"] = None

    if cur_base == base_url and cur_model == CYBER_ORNITH_MODEL and cur_provider == "custom":
        return False

    try:
        if hasattr(agent, "switch_model"):
            agent.switch_model(
                CYBER_ORNITH_MODEL,
                "custom",
                api_key=_gateway_api_key(),
                base_url=base_url,
                api_mode="chat_completions",
            )
        else:
            agent.model = CYBER_ORNITH_MODEL
            agent.provider = "custom"
            agent.base_url = base_url
            agent.api_key = _gateway_api_key()
        if isinstance(session, dict):
            session["config_model_seen"] = (CYBER_ORNITH_MODEL, "custom")
        if sid:
            import sys
            srv = sys.modules.get("tui_gateway.server")
            if srv is not None and hasattr(srv, "_emit_session_info_for_session") and isinstance(session, dict):
                srv._emit_session_info_for_session(sid, session)
        return True
    except Exception as exc:
        logger.debug("Live agent tunnel sync note: %s", exc)
        return False


def sync_all_live_gateway_sessions(base_url: str) -> None:
    """Switch all currently active TUI/Web chat sessions in-place to the active Kaggle GPU tunnel."""
    import sys
    srv = sys.modules.get("tui_gateway.server")
    if srv is None:
        return
    try:
        with srv._sessions_lock:
            sessions = list(srv._sessions.items())
        for sid, sess in sessions:
            if not isinstance(sess, dict):
                continue
            override = sess.get("model_override")
            if isinstance(override, dict) and override.get("model") == "z-ai/glm-5.2":
                sess["model_override"] = None
            agent = sess.get("agent")
            if agent is not None:
                sync_agent_to_active_kaggle_tunnel(agent, sess, sid=sid)
            else:
                srv._emit_session_info_for_session(sid, sess)
    except Exception as exc:
        logger.debug("Broadcast live gateway session sync note: %s", exc)


def update_hermes_model_endpoint(tunnel_url: str = "", force_save: bool = False) -> None:
    """Automatically configure Hermes Agent to call the active Kaggle GPU Cloudflare Tunnel."""
    global _LAST_SAVED_TUNNEL_URL
    effective_url = tunnel_url.strip() if tunnel_url else ""
    base_url = _normalize_v1_url(effective_url or "https://pending-kaggle-gpu.trycloudflare.com")

    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = _gateway_api_key() or os.environ.get("OPENAI_API_KEY", "")
    os.environ["HERMES_MODEL"] = CYBER_ORNITH_MODEL
    os.environ["HERMES_TUI_PROVIDER"] = "custom"

    if not force_save and _LAST_SAVED_TUNNEL_URL == base_url:
        if effective_url:
            sync_all_live_gateway_sessions(base_url)
        return

    try:
        from hermes_cli.config import load_config, save_config
        cfg = load_config()
        if not isinstance(cfg.get("model"), dict):
            cfg["model"] = {}
        needs_save = (
            force_save
            or cfg["model"].get("provider") != "custom"
            or cfg["model"].get("default") != CYBER_ORNITH_MODEL
            or cfg["model"].get("base_url") != base_url
            or (bool(_gateway_api_key()) and cfg["model"].get("api_key") != _gateway_api_key())
            or not isinstance(cfg.get("memory"), dict)
            or cfg["memory"].get("provider") != "cloudflare_d1"
        )
        if needs_save:
            cfg["model"]["provider"] = "custom"
            cfg["model"]["default"] = CYBER_ORNITH_MODEL
            cfg["model"]["base_url"] = base_url
            cfg["model"]["api_key"] = _gateway_api_key() or cfg["model"].get("api_key", "")
            if not isinstance(cfg.get("memory"), dict):
                cfg["memory"] = {}
            cfg["memory"]["provider"] = "cloudflare_d1"
            cfg["memory"]["memory_enabled"] = True
            cfg["memory"]["user_profile_enabled"] = True
            save_config(cfg)
        _LAST_SAVED_TUNNEL_URL = base_url
        if effective_url:
            sync_all_live_gateway_sessions(base_url)
    except Exception as exc:
        logger.debug("Could not update config.yaml with tunnel URL: %s", exc)


_LAST_OBSERVED_STATUS: str = ""


def _note_status_transition(status: str) -> None:
    """Reset the inactivity clock when the engine becomes (or stops being) ready."""
    global _LAST_OBSERVED_STATUS
    status = str(status or "").upper()
    if status == _LAST_OBSERVED_STATUS:
        return
    previous, _LAST_OBSERVED_STATUS = _LAST_OBSERVED_STATUS, status
    if status in READY_STATUSES and previous not in READY_STATUSES:
        try:
            mark_activity("gpu-ready")
            logger.info("Kaggle GPU reached %s — inactivity timer restarted", status)
        except Exception:
            pass


def _refresh_quota(force_refresh: bool = False) -> Optional[Dict[str, Any]]:
    """Weekly GPU quota from Kaggle (cached 25 s). Returns the last known value, or None if it was never fetched.

    Never invents numbers: a process that has not yet reached Kaggle reports *no* quota instead of "0 h used".
    """
    global _QUOTA_CACHE, _QUOTA_CACHE_TS
    now_mono = time.monotonic()
    if force_refresh or _QUOTA_CACHE is None or (now_mono - _QUOTA_CACHE_TS) >= 25.0:
        try:
            q_dict = _kaggle_rpc("GetAcceleratorQuotaStatistics", {})
            gpu_q = q_dict.get("gpuQuota") or {}
            raw_used = str(gpu_q.get("timeUsed") or "0").replace("s", "")
            parts = raw_used.split(".")
            used_sec = float(parts[0] + ("." + parts[1] if len(parts) > 1 else ""))
            total_sec = float(str(gpu_q.get("totalTimeAllowed") or "108000").replace("s", "").split(".")[0])
            display_total_sec = 108000.0 if total_sec <= 21600 else total_sec
            reserved_sec = float(str(gpu_q.get("timeReserved") or "0").replace("s", "").split(".")[0])
            _QUOTA_CACHE = {
                "used_seconds": round(used_sec, 1),
                "used_hours": round(used_sec / 3600.0, 2),
                "total_seconds": display_total_sec,
                "total_hours": round(display_total_sec / 3600.0, 1),
                "remaining_hours": max(0.0, round((display_total_sec - used_sec) / 3600.0, 2)),
                "used_pct": min(100.0, round((used_sec / display_total_sec) * 100.0, 1)) if display_total_sec > 0 else 0.0,
                "refresh_time": str(q_dict.get("quotaRefreshTime") or ""),
                "reserved_seconds": reserved_sec,
            }
            _QUOTA_CACHE_TS = now_mono
        except Exception as q_err:
            logger.debug("Quota fetch note: %s", q_err)
    return _QUOTA_CACHE


def get_kaggle_gpu_status(force_refresh: bool = False) -> Dict[str, Any]:
    """Return comprehensive Kaggle GPU usage, quota, kernel state, and live tunnel status."""
    global _STATUS_CACHE, _STATUS_CACHE_TS, _QUOTA_CACHE, _QUOTA_CACHE_TS
    now_mono = time.monotonic()
    # Keep ONLINE status sticky for longer so page reloads never flash "OFF".
    cache_ttl = 12.0
    if _STATUS_CACHE and str(_STATUS_CACHE.get("status") or "") in (
        "ONLINE", "MODEL_READY_AND_WARMED", "HEARTBEAT_ONLINE", "MODEL_READY", "MODEL_READY_COLD"
    ) and bool(_STATUS_CACHE.get("active")):
        cache_ttl = 45.0
    if not force_refresh and _STATUS_CACHE and (now_mono - _STATUS_CACHE_TS) < cache_ttl:
        return dict(_STATUS_CACHE)

    with _STATUS_CACHE_LOCK:
        global _STATUS_REFRESH_IN_FLIGHT, _STATUS_REFRESH_TS
        now_mono = time.monotonic()
        cache_ttl = 12.0
        if _STATUS_CACHE and str(_STATUS_CACHE.get("status") or "") in (
            "ONLINE", "MODEL_READY_AND_WARMED", "HEARTBEAT_ONLINE", "MODEL_READY", "MODEL_READY_COLD"
        ) and bool(_STATUS_CACHE.get("active")):
            cache_ttl = 45.0
        if not force_refresh and _STATUS_CACHE and (now_mono - _STATUS_CACHE_TS) < cache_ttl:
            return dict(_STATUS_CACHE)
        # If another thread is already building status, never block the UI on it.
        if _STATUS_REFRESH_IN_FLIGHT and _STATUS_CACHE and not force_refresh:
            return dict(_STATUS_CACHE)
        if _STATUS_REFRESH_IN_FLIGHT and (time.time() - _STATUS_REFRESH_TS) < 45 and not force_refresh:
            # No cache yet but refresh running — return a minimal offline snapshot quickly.
            return {
                "active": False, "booting": True, "status": "BOOTING_KAGGLE_GPU",
                "progress_pct": 5, "progress_step": "Refreshing GPU status…",
                "tunnel_url": "", "model": CYBER_ORNITH_MODEL,
                "gpu_info": "2× NVIDIA Tesla T4", "display_status": "Starting Kaggle",
                "engine_state": "starting", "busy": False,
                "quota": _QUOTA_CACHE,
            }
        _STATUS_REFRESH_IN_FLIGHT = True
        _STATUS_REFRESH_TS = time.time()

        _ensure_kaggle_credentials()
        ensure_d1_schema()

        state: Dict[str, Any] = {
            "active": False,
            "booting": False,
            "status": "OFF",
            "progress_pct": 0,
            "progress_step": "GPU is currently turned OFF (0s quota reserved)",
            "tunnel_url": "",
            "model": CYBER_ORNITH_MODEL,
            "gpu_info": "2× NVIDIA Tesla T4 (30 GB VRAM)",
            "kaggle_username": KAGGLE_USERNAME,
            "kaggle_kernel": KAGGLE_KERNEL_ID,
            "worker_status": "OFF",
            "quota": _QUOTA_CACHE,
            "cloudflare_d1": {
                "connected": True,
                "account_id": CLOUDFLARE_ACCOUNT_ID,
                "database_id": CLOUDFLARE_D1_DATABASE_ID,
            },
        }

        if state.get("quota") is None:
            state.pop("quota", None)  # unknown is not "0 h used"

        # 1. Read persisted state from Cloudflare D1
        d1_row: Dict[str, Any] = {}
        try:
            rows = d1_query("SELECT * FROM kaggle_gpu_state WHERE id = 'primary' LIMIT 1;")
            if rows:
                d1_row = rows[0]
                state["status"] = str(d1_row.get("status") or "OFF")
                row_url = str(d1_row.get("tunnel_url") or "").rstrip("/")
                # A status update that carries no URL (e.g. the notebook's cache
                # bookkeeping) must not blank a tunnel we already know about —
                # otherwise the UI reports "Connection Lost" on a healthy GPU.
                state["tunnel_url"] = row_url or _LAST_SAVED_TUNNEL_URL

                # Sticky ONLINE: if D1 already recorded a live tunnel, treat GPU as on
                # immediately so page reloads never show "OFF" while health is re-probed.
                if str(state.get("status") or "").upper() in (
                    "ONLINE", "MODEL_READY_AND_WARMED", "HEARTBEAT_ONLINE",
                    "MODEL_READY", "MODEL_READY_COLD", "TUNNEL_ONLINE", "WARMING_GPU"
                ) and state.get("tunnel_url"):
                    state["active"] = True
                    if str(state.get("status") or "").upper() in (
                        "ONLINE", "MODEL_READY_AND_WARMED", "HEARTBEAT_ONLINE", "MODEL_READY"
                    ):
                        state["booting"] = False
                        state["worker_status"] = "RUNNING"
                        state["progress_pct"] = 100
                        state["progress_step"] = "GPU Permanently Connected & Active"
                        state["display_status"] = "Kaggle Ready"
                if not row_url and _LAST_SAVED_TUNNEL_URL:
                    try:
                        d1_query(
                            "UPDATE kaggle_gpu_state SET tunnel_url = ? WHERE id = 'primary' AND (tunnel_url IS NULL OR tunnel_url = '');",
                            [_LAST_SAVED_TUNNEL_URL],
                        )
                    except Exception:
                        pass
                state["detail"] = d1_row.get("detail") or ""
                if d1_row.get("gpu_info"):
                    state["gpu_info"] = str(d1_row["gpu_info"]).replace("\n", " + ")
        except Exception as exc:
            logger.debug("D1 GPU state read warning: %s", exc)

        
        # FAST PATH: when D1 already has a live ONLINE tunnel, skip slow Kaggle API
        # and long probes so the UI never hangs on "loading". Optional 2s health ping.
        _st_up = str(state.get("status") or "").upper()
        if (
            state.get("active")
            and state.get("tunnel_url")
            and _st_up in ("ONLINE", "MODEL_READY_AND_WARMED", "HEARTBEAT_ONLINE", "MODEL_READY", "MODEL_READY_COLD")
        ):
            quota = _refresh_quota(False) if force_refresh else _QUOTA_CACHE   # HTTP polls use the cache; the watchdog refreshes it
            if quota:
                state["quota"] = quota
            state["booting"] = False
            state["worker_status"] = state.get("worker_status") or "RUNNING"
            state["progress_pct"] = 100
            state["progress_step"] = "GPU Permanently Connected & Active"
            state["display_status"] = "Kaggle Ready"
            state["engine_state"] = "ready"
            # quick health (never block UI more than 2.5s). On confirmed dead tunnel,
            # do NOT report ONLINE — fall through to full probe/clear logic.
            _fast_dead = False
            try:
                h_req = urllib.request.Request(
                    f"{state['tunnel_url'].rstrip('/')}/health",
                    headers={"User-Agent": "HermesAgentGPUFast/1.0"},
                )
                with urllib.request.urlopen(h_req, timeout=2.5) as resp:
                    h_data = json.loads(resp.read().decode("utf-8"))
                    if h_data.get("gpu"):
                        state["gpu_info"] = str(h_data["gpu"]).replace("\n", " + ")
                    state["model_loaded"] = bool(h_data.get("model_loaded"))
                    if h_data.get("status") not in (None, "online", "ready", "ok"):
                        _fast_dead = True
            except Exception as _fast_exc:
                http_code = 0
                try:
                    import urllib.error as _ue
                    if isinstance(_fast_exc, _ue.HTTPError):
                        http_code = int(getattr(_fast_exc, "code", 0) or 0)
                except Exception:
                    pass
                _fast_dead = http_code in (404, 410, 502, 503, 521, 522, 523, 524, 530) or (
                    "530" in str(_fast_exc) or "1033" in str(_fast_exc)
                    or "Name or service not known" in str(_fast_exc)
                    or "nodename nor servname" in str(_fast_exc).lower()
                )
                if not _fast_dead:
                    # Transient timeout — keep ON briefly but mark model_loaded unknown
                    state["model_loaded"] = None
            if _fast_dead:
                logger.info(
                    "Fast-path health failed for %s — clearing stale tunnel (will not report ONLINE)",
                    (state.get("tunnel_url") or "")[:48],
                )
                try:
                    d1_query(
                        "UPDATE kaggle_gpu_state SET status = ?, tunnel_url = '' WHERE id = 'primary';",
                        ["TUNNEL_ERROR"],
                    )
                except Exception:
                    pass
                state["tunnel_url"] = ""
                state["active"] = False
                state["booting"] = False
                state["status"] = "TUNNEL_ERROR"
                state["display_status"] = "Connection Lost"
                state["progress_step"] = "Cloudflare tunnel is down — turn GPU ON to recover"
                state["progress_pct"] = 0
                state["engine_state"] = "error"
                _notify_tunnel_change("")
                # fall through is not needed — return honest offline state
                state["auto_off"] = auto_off_decision()
                state["busy"] = False
                _STATUS_CACHE = dict(state); _STATUS_REFRESH_IN_FLIGHT = False
                _STATUS_CACHE_TS = time.monotonic()
                return dict(state)
            state["auto_off"] = auto_off_decision()
            state["busy"] = bool(state.get("auto_off", {}).get("active_tasks"))
            # Keep Hermes model endpoint in sync with live tunnel on every status read
            try:
                if state.get("active") and state.get("tunnel_url"):
                    update_hermes_model_endpoint(state["tunnel_url"], force_save=False)
            except Exception as _ue:
                logger.debug("status endpoint model sync note: %s", _ue)
            _STATUS_CACHE = dict(state); _STATUS_REFRESH_IN_FLIGHT = False
            _STATUS_CACHE_TS = time.monotonic()
            return dict(state)

        # 2. Check Kaggle API for live quota statistics (cached 25s) & kernel session status
        try:
            quota = _refresh_quota(force_refresh)
            if quota:
                state["quota"] = quota

            try:
                st_dict = _kaggle_rpc(
                    "GetKernelSessionStatus",
                    {"userName": KAGGLE_USERNAME, "kernelSlug": KAGGLE_KERNEL_SLUG},
                )
                state["worker_status"] = str(st_dict.get("status") or "OFF")
            except Exception:
                state["worker_status"] = "OFF"
        except Exception as k_err:
            logger.debug("Kaggle RPC status note: %s", k_err)

        # 2b. Dataset cache + inactivity snapshot travel with every status response.
        try:
            state["cache_dataset"] = _model_cache_dataset_slug()
        except Exception:
            state["cache_dataset"] = ""
        try:
            state["auto_off"] = auto_off_decision()
        except Exception as exc:
            logger.debug("auto-off snapshot note: %s", exc)
            state["auto_off"] = {"enabled": False, "minutes": AUTO_OFF_DEFAULT_MINUTES, "should_stop": False}

        # 3. If not explicitly stopped, also check ntfy.sh telemetry ONLY for messages strictly newer than D1
        d1_already_online = (
            bool(state["tunnel_url"])
            and state["status"] in ("ONLINE", "MODEL_READY_AND_WARMED", "TUNNEL_ONLINE", "HEARTBEAT_ONLINE")
        )
        if state["status"] != "GPU_STOPPED_SAVING_QUOTA" and not d1_already_online:
            try:
                req = urllib.request.urlopen(
                    f"https://ntfy.sh/{TELEMETRY_TOPIC}/json?poll=1&since=10m", timeout=4
                )
                lines = req.read().decode("utf-8", errors="ignore").strip().splitlines()
                for line in lines:
                    if not line.strip():
                        continue
                    msg = json.loads(line)
                    try:
                        payload = json.loads(msg.get("message", ""))
                        if isinstance(payload, dict) and payload.get("status"):
                            msg_ts = float(payload.get("timestamp") or msg.get("time") or 0)
                            d1_ts = float(d1_row.get("updated_at") or 0)
                            if msg_ts > d1_ts + 1.0:
                                state["status"] = payload["status"]
                                if payload.get("tunnel_url"):
                                    state["tunnel_url"] = payload["tunnel_url"].rstrip("/")
                                if payload.get("gpu"):
                                    state["gpu_info"] = str(payload["gpu"]).replace("\n", " + ")
                    except Exception:
                        pass
            except Exception:
                pass

        # 4. Verify live health of the tunnel if present and not stopped by the user
        tunnel_healthy = False
        if state["tunnel_url"] and state["status"] != "GPU_STOPPED_SAVING_QUOTA":
            try:
                h_req = urllib.request.Request(
                    f"{state['tunnel_url']}/health",
                    headers={"User-Agent": "HermesAgentGPUCheck/1.0"},
                )
                with urllib.request.urlopen(h_req, timeout=5) as resp:
                    h_data = json.loads(resp.read().decode("utf-8"))
                    model_loaded = h_data.get("model_loaded")
                    if h_data.get("status") == "online" and model_loaded is False:
                        # Tunnel is up but the weights are not resident yet: stay honest
                        # and keep the boot stages visible instead of faking Ready.
                        tunnel_healthy = True
                        state["active"] = True
                        state["booting"] = True
                        state["engine_state"] = "starting"
                        state["progress_pct"] = max(int(state.get("progress_pct") or 0), 92)
                        state["progress_step"] = "Loading weights into GPU VRAM…"
                        state["display_status"] = "Loading Model"
                    elif h_data.get("status") == "online":
                        tunnel_healthy = True
                        state["active"] = True
                        state["booting"] = False
                        state["status"] = "ONLINE"
                        state["worker_status"] = "RUNNING"
                        state["progress_pct"] = 100
                        state["progress_step"] = "GPU Permanently Connected & Active"
                        if h_data.get("gpu"):
                            state["gpu_info"] = str(h_data["gpu"]).replace("\n", " + ")
                        boot = h_data.get("boot") or {}
                        if boot:
                            state["boot_seconds"] = boot.get("boot_seconds")
                            state["cache_source"] = boot.get("cache_source")
                            state["download_skipped"] = boot.get("download_skipped")
                        update_hermes_model_endpoint(state["tunnel_url"])
                        if d1_row.get("status") != "ONLINE" or d1_row.get("tunnel_url") != state["tunnel_url"]:
                            try:
                                d1_query(
                                    """INSERT INTO kaggle_gpu_state (id, status, tunnel_url, api_key, model, gpu_info, updated_at)
                                       VALUES ('primary', 'ONLINE', ?, ?, ?, ?, ?)
                                       ON CONFLICT(id) DO UPDATE SET status = excluded.status, tunnel_url = excluded.tunnel_url, model = excluded.model, gpu_info = excluded.gpu_info, updated_at = excluded.updated_at;""",
                                    [state["tunnel_url"], CYBER_ORNITH_API_KEY, CYBER_ORNITH_MODEL, state["gpu_info"], time.time()],
                                )
                            except Exception:
                                pass
                            if state["tunnel_url"] != _LAST_SAVED_TUNNEL_URL:
                                _notify_tunnel_change(state["tunnel_url"])
            except Exception as _hexc:
                # Classify Cloudflare/tunnel failures so we never report ONLINE for a dead endpoint.
                http_code = 0
                try:
                    import urllib.error as _ue
                    if isinstance(_hexc, _ue.HTTPError):
                        http_code = int(getattr(_hexc, "code", 0) or 0)
                except Exception:
                    pass
                dead_tunnel = http_code in (404, 410, 502, 503, 521, 522, 523, 524, 530) or (
                    "530" in str(_hexc) or "1033" in str(_hexc) or "tunnel" in str(_hexc).lower()
                )
                # Brief stickiness only while the kernel is mid-generation (probe timeout),
                # never when Cloudflare confirms the tunnel is gone.
                if (
                    not dead_tunnel
                    and d1_already_online
                    and state["worker_status"] not in ("COMPLETE", "ERROR", "CANCELLED", "OFF")
                ):
                    tunnel_healthy = True
                    state["active"] = True
                    state["booting"] = False
                    state["status"] = "HEARTBEAT_ONLINE"
                    state["progress_pct"] = 100
                    state["progress_step"] = "GPU busy — health probe timed out, keeping session"
                    update_hermes_model_endpoint(state["tunnel_url"])
                else:
                    tunnel_healthy = False
                    if dead_tunnel:
                        logger.info(
                            "Tunnel health failed (HTTP %s / %s) — clearing stale endpoint %s",
                            http_code or "?",
                            type(_hexc).__name__,
                            (state.get("tunnel_url") or "")[:48],
                        )
                        state["tunnel_url"] = ""
                        state["active"] = False
                        state["booting"] = False
                        state["status"] = "TUNNEL_ERROR"
                        state["display_status"] = "Connection Lost"
                        state["progress_step"] = "Cloudflare tunnel is down — turn GPU ON to recover"
                        try:
                            d1_query(
                                "UPDATE kaggle_gpu_state SET status = ?, tunnel_url = '' WHERE id = 'primary';",
                                ["TUNNEL_ERROR"],
                            )
                        except Exception:
                            pass
                        _notify_tunnel_change("")

        if not tunnel_healthy:
            is_recently_started = (
                state["status"] != "GPU_STOPPED_SAVING_QUOTA"
                and state["status"] != "OFF"
                and (time.time() - float(d1_row.get("updated_at") or 0)) < 120
            )
            if (state["worker_status"] in ("RUNNING", "QUEUED") or is_recently_started) and state["status"] != "GPU_STOPPED_SAVING_QUOTA":
                state["active"] = False
                state["booting"] = state["status"] not in _TERMINAL_ERROR_STATUSES
                mapped = _STEP_MAP.get(state["status"])
                if mapped:
                    state["progress_pct"], state["progress_step"] = mapped[0], mapped[1]
                else:
                    state["progress_pct"] = 25
                    state["progress_step"] = "Provisioning Kaggle 2× Tesla T4 GPU container…"
            else:
                state["active"] = False
                state["booting"] = False
                state["status"] = "OFF"
                state["tunnel_url"] = ""
                state["progress_pct"] = 0
                state["progress_step"] = "Kaggle GPU is OFF — Click 'Turn ON GPU' to launch"

        # Canonical engine state for the UI (never leaves a stuck "loading").
        _note_status_transition(str(state["status"]))
        state["engine_state"] = _engine_state_from_status(
            str(state["status"]), bool(state["active"]), bool(state["booting"])
        )
        if state.get("auto_off", {}).get("active_tasks"):
            state["engine_state"] = "busy"
        state["busy"] = bool(state.get("auto_off", {}).get("active_tasks"))
        state["display_status"] = {
            "off": "Kaggle Offline",
            "starting": "Starting Kaggle",
            "ready": "Kaggle Ready",
            "busy": "Kaggle Busy",
            "stopping": "Stopping Kaggle",
            "stopped": "Kaggle Stopped",
            "error": "Connection Lost",
        }.get(state["engine_state"], "Kaggle Offline")
        if state["status"] in ("GPU_UNAVAILABLE", "MODEL_CORRUPT", "INTERNET_UNAVAILABLE"):
            state["display_status"] = "Model Unavailable"
        elif state["status"] == "BOOT_FAILED":
            state["display_status"] = "Agent Error"
        elif state["status"] == "MODEL_READY_COLD":
            state["display_status"] = "Model Ready"
        elif state["status"] in ("TUNNEL_ERROR",):
            state["display_status"] = "Connection Lost"

        try:
            if state.get("active") and state.get("tunnel_url"):
                update_hermes_model_endpoint(state["tunnel_url"], force_save=False)
        except Exception as _ue:
            logger.debug("status final model sync note: %s", _ue)
        _STATUS_CACHE = dict(state); _STATUS_REFRESH_IN_FLIGHT = False
        _STATUS_CACHE_TS = time.monotonic()
        return state


_CACHE_DATASET_KEY = "blackthorn_model_cache_dataset"


def _model_cache_dataset_slug() -> str:
    """Slug of the private Kaggle dataset holding the GGUF, or "" when unknown."""
    try:
        rows = d1_query("SELECT value FROM state_meta WHERE key = ? LIMIT 1;", [_CACHE_DATASET_KEY])
        if rows:
            return str(rows[0].get("value") or "").strip()
    except Exception:
        pass
    return ""


def _set_cache_dataset(slug: str) -> None:
    try:
        d1_query(
            "INSERT OR REPLACE INTO state_meta (key, value) VALUES (?, ?);",
            [_CACHE_DATASET_KEY, slug or ""],
        )
    except Exception:
        pass


def refresh_model_cache_dataset(expected_bytes: int = 5629110976) -> str:
    """Confirm the private model-cache dataset exists and is ready, then remember it.

    Uses the dataset VIEW endpoint (owner search does not index private datasets).
    The dataset is published by the Kaggle notebook itself on its first real
    download, and is attached to every later kernel so the 5.6 GB GGUF is never
    downloaded twice.
    """
    slug = f"{KAGGLE_USERNAME}/{CACHE_DATASET_NAME}"
    try:
        req = urllib.request.Request(
            f"https://www.kaggle.com/api/v1/datasets/view/{slug}",
            headers={"Authorization": f"Bearer {KAGGLE_API_TOKEN}", "User-Agent": "HermesAgent/1.0"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8")) or {}
        total = int(data.get("totalBytesNullable") or data.get("totalBytes") or 0)
        if total and (expected_bytes <= 0 or total >= expected_bytes):
            _set_cache_dataset(slug)
            logger.info("Model cache dataset confirmed: %s (%s bytes)", slug, total)
            return slug
        logger.info("Model cache dataset exists but is not ready yet (%s bytes)", total)
    except Exception as exc:
        logger.debug("cache dataset verify note: %s", exc)
    return _model_cache_dataset_slug()


def _stop_stale_kernel_session(wait_seconds: int = 45) -> bool:
    """Stop a running-but-unhealthy kernel so the next push boots the new notebook.

    Returns True when Kaggle is *still* running it after the wait (the push is
    queued behind it and the watchdog will finish the job).
    """
    deadline = time.time() + wait_seconds
    try:
        sess = _kaggle_rpc("GetKernelSessionStatus", {"userName": KAGGLE_USERNAME, "kernelSlug": KAGGLE_KERNEL_SLUG})
        if str(sess.get("status") or "").upper() not in ("RUNNING", "QUEUED", "PENDING"):
            return False
        logger.info("Stale Kaggle session detected (no healthy tunnel) — stopping it before the re-push")
        for attempt in range(3):
            try:
                _kaggle_rpc("DeleteKernel", {"userName": KAGGLE_USERNAME, "kernelSlug": KAGGLE_KERNEL_SLUG})
            except Exception as exc:
                logger.debug("DeleteKernel attempt %s note: %s", attempt + 1, exc)
            while time.time() < deadline:
                time.sleep(4)
                try:
                    again = _kaggle_rpc("GetKernelSessionStatus",
                                        {"userName": KAGGLE_USERNAME, "kernelSlug": KAGGLE_KERNEL_SLUG})
                except Exception:
                    continue
                if str(again.get("status") or "").upper() not in ("RUNNING", "QUEUED", "PENDING"):
                    logger.info("Stale session stopped — pushing the current notebook")
                    return False
            if attempt < 2:
                deadline = time.time() + 25
        logger.warning("Kaggle session refused to stop within %ss; pushing anyway (queued)", wait_seconds)
        return True
    except Exception as exc:
        logger.debug("pre-push session check note: %s", exc)
        return False


def _set_meta(key: str, value: str) -> None:
    """state_meta has no unique key, so replace = delete + insert."""
    try:
        d1_query("DELETE FROM state_meta WHERE key = ?;", [key])
        d1_query("INSERT INTO state_meta (key, value) VALUES (?, ?);", [key, value])
    except Exception as exc:
        logger.debug("state_meta write note (%s): %s", key, exc)


def _turn_on_kaggle_gpu_sync() -> Dict[str, Any]:
    """The slow path: stop any stale session, push the notebook, verify the push.

    Never call this straight from an HTTP handler — the UI must answer instantly,
    so `turn_on_kaggle_gpu()` runs it on a worker thread instead.
    """
    global _STATUS_CACHE_TS
    _ensure_kaggle_credentials()
    ensure_d1_schema()
    # If already online and healthy, return current online state without restarting the kernel
    if _probe_live_tunnel():
        mark_activity("turn-on-reuse")
        return get_kaggle_gpu_status(force_refresh=True)

    # A running-but-unhealthy session would keep executing the notebook version
    # it started with, so stop it first — the next push then boots fresh code.
    still_running = _stop_stale_kernel_session()

    try:
        from blackthorn.kaggle_bundle import build_notebook_text
        nb_text = build_notebook_text()
    except Exception as exc:
        # Never push a notebook that cannot report status; show the real reason instead of sitting on "Starting".
        reason = f"cannot build the Kaggle notebook: {exc}"[:200]
        logger.error(reason)
        d1_query(
            """INSERT INTO kaggle_gpu_state (id, status, tunnel_url, api_key, model, gpu_info, updated_at)
               VALUES ('primary', 'BOOT_FAILED', '', '', ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET status = excluded.status, tunnel_url = excluded.tunnel_url,
               model = excluded.model, gpu_info = excluded.gpu_info, updated_at = excluded.updated_at;""",
            [CYBER_ORNITH_MODEL, "FAILED: " + reason, time.time()],
        )
        _STATUS_CACHE_TS = 0.0
        return get_kaggle_gpu_status(force_refresh=True)

    kernel_payload: Dict[str, Any] = {
        "slug": KAGGLE_KERNEL_ID,
        "newTitle": "Qwen3.8-27B-Uncensored API Server (dual T4)",
        "text": nb_text,
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": True,
        "enableGpu": True,
        "enableTpu": False,
        "enableInternet": True,
        "machineShape": "NvidiaTeslaT4",
    }
    # Attaching the 5.6 GB cache dataset was MEASURED to stall kernel startup:
    # with the mount, runs sat in RUNNING for 30+ minutes without executing a
    # single cell (both GPU and CPU kernels); without it a test kernel finished
    # in ~20 s and the full server booted in ~2.5 min. The notebook therefore
    # pulls the same dataset from Kaggle's own storage *inside* the run instead
    # (fast, resumable, verified) and only falls back to Hugging Face when that
    # copy is missing. Set BLACKTHORN_ATTACH_MODEL_CACHE=1 to force the mount.
    cache_slug = _model_cache_dataset_slug()
    attach_cache = (os.environ.get("BLACKTHORN_ATTACH_MODEL_CACHE", "").strip() == "1")
    if cache_slug and attach_cache:
        kernel_payload["datasetDataSources"] = [cache_slug]
    # Pushing is what actually starts the run. It is the one call that must not be
    # lost: if it fails the GPU would sit on "Starting Kaggle" forever, so it is
    # retried and the outcome is verified before we advertise a boot.
    push: Dict[str, Any] = {}
    last_error = ""
    for attempt in range(1, 4):
        try:
            push = _kaggle_rpc("SaveKernel", kernel_payload) or {}
            invalid = [x for x in (push.get("invalidDatasetSources") or []) if x]
            if invalid and cache_slug:
                logger.warning("Kaggle rejected dataset source %s; pushing without it", invalid)
                _set_cache_dataset("")
                kernel_payload.pop("datasetDataSources", None)
                push = _kaggle_rpc("SaveKernel", kernel_payload) or {}
            if push.get("versionNumber") or push.get("ref"):
                break
            # Some SaveKernel responses omit versionNumber even when the kernel
            # is queued/running. Treat that as success when the session is live.
            try:
                st = (_kaggle_rpc("GetKernelSessionStatus", {"userName": KAGGLE_USERNAME, "kernelSlug": KAGGLE_KERNEL_SLUG}) or {})
                sess = str((st.get("status") or st.get("sessionStatus") or "")).upper()
                if sess in ("RUNNING", "QUEUED", "STARTING", "RUNNING_QUEUED"):
                    logger.info("SaveKernel returned no version but session is %s — treating push as OK", sess)
                    push = push or {"ref": KAGGLE_KERNEL_ID, "versionNumber": "existing"}
                    break
            except Exception as probe_exc:
                logger.debug("session probe after empty version: %s", probe_exc)
            last_error = "Kaggle accepted the push but returned no version"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("SaveKernel attempt %s failed: %s", attempt, last_error)
        if attempt < 3:
            time.sleep(5 * attempt)
    else:
        # Final fallback: if a session is already running, do not hard-fail the UI
        try:
            st = (_kaggle_rpc("GetKernelSessionStatus", {"userName": KAGGLE_USERNAME, "kernelSlug": KAGGLE_KERNEL_SLUG}) or {})
            sess = str((st.get("status") or st.get("sessionStatus") or "")).upper()
            if sess in ("RUNNING", "QUEUED", "STARTING", "RUNNING_QUEUED"):
                logger.warning("Push verification failed (%s) but kernel is %s — continuing boot wait", last_error, sess)
                push = {"ref": KAGGLE_KERNEL_ID, "versionNumber": "existing-running"}
            else:
                raise RuntimeError(last_error or "unknown Kaggle error")
        except Exception as final_exc:
            detail = str(final_exc)[:200]
            logger.error("Kaggle push failed after 3 attempts: %s", detail)
            d1_query(
                """INSERT INTO kaggle_gpu_state (id, status, tunnel_url, api_key, model, gpu_info, updated_at)
                   VALUES ('primary', 'BOOT_FAILED', '', ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET status = excluded.status, tunnel_url = excluded.tunnel_url, model = excluded.model, gpu_info = excluded.gpu_info, updated_at = excluded.updated_at;""",
                [CYBER_ORNITH_API_KEY, CYBER_ORNITH_MODEL,
                 f"Kaggle refused the kernel push — {detail[:120]}", time.time()],
            )
            _STATUS_CACHE_TS = 0.0
            return get_kaggle_gpu_status(force_refresh=True)

    logger.info("Kaggle kernel pushed (version %s)%s", push.get("versionNumber") or "?",
                " — previous session was still running" if still_running else "")

    now = time.time()
    _set_meta("blackthorn_gpu_boot_started", str(now))
    d1_query(
        """INSERT INTO kaggle_gpu_state (id, status, tunnel_url, api_key, model, gpu_info, updated_at)
           VALUES ('primary', 'BOOTING_KAGGLE_GPU', '', ?, ?, 'Allocating 2x NVIDIA Tesla T4...', ?)
           ON CONFLICT(id) DO UPDATE SET status = excluded.status, tunnel_url = excluded.tunnel_url, model = excluded.model, gpu_info = excluded.gpu_info, updated_at = excluded.updated_at;""",
        [CYBER_ORNITH_API_KEY, CYBER_ORNITH_MODEL, now],
    )
    mark_activity("turn-off")
    mark_activity("turn-on")
    _STATUS_CACHE_TS = 0.0
    return get_kaggle_gpu_status(force_refresh=True)


def turn_off_kaggle_gpu() -> Dict[str, Any]:
    """Immediately turn OFF the Kaggle GPU session to save weekly GPU quota."""
    global _STATUS_CACHE_TS
    _ensure_kaggle_credentials()
    ensure_d1_schema()
    # Surface "stopping" immediately so the UI shows real progress.
    try:
        d1_query(
            """INSERT INTO kaggle_gpu_state (id, status, tunnel_url, api_key, model, gpu_info, updated_at)
               VALUES ('primary', 'STOPPING_KAGGLE_GPU', '', ?, ?, 'Stopping Kaggle GPU…', ?)
               ON CONFLICT(id) DO UPDATE SET status = excluded.status, tunnel_url = excluded.tunnel_url, model = excluded.model, gpu_info = excluded.gpu_info, updated_at = excluded.updated_at;""",
            [CYBER_ORNITH_API_KEY, CYBER_ORNITH_MODEL, time.time()],
        )
        _STATUS_CACHE_TS = 0.0
    except Exception:
        pass
    try:
        _kaggle_rpc(
            "DeleteKernel",
            {"userName": KAGGLE_USERNAME, "kernelSlug": KAGGLE_KERNEL_SLUG},
        )
    except Exception as exc:
        logger.debug("DeleteKernel note: %s", exc)

    now = time.time()
    d1_query(
        """INSERT INTO kaggle_gpu_state (id, status, tunnel_url, api_key, model, gpu_info, updated_at)
           VALUES ('primary', 'GPU_STOPPED_SAVING_QUOTA', '', ?, ?, 'OFF (0s Quota Reserved)', ?)
           ON CONFLICT(id) DO UPDATE SET status = excluded.status, tunnel_url = excluded.tunnel_url, model = excluded.model, gpu_info = excluded.gpu_info, updated_at = excluded.updated_at;""",
        [CYBER_ORNITH_API_KEY, CYBER_ORNITH_MODEL, now],
    )
    mark_activity("turn-off")
    _STATUS_CACHE_TS = 0.0
    return get_kaggle_gpu_status(force_refresh=True)


_DAEMON_STARTED = False
_DAEMON_LOCK = threading.Lock()


def start_permanent_gpu_daemon() -> None:
    """Start a background daemon that keeps the Kaggle GPU connection & Render service permanently alive
    until the user explicitly disconnects the GPU via turn_off_kaggle_gpu()."""
    global _DAEMON_STARTED
    with _DAEMON_LOCK:
        if _DAEMON_STARTED:
            return
        _DAEMON_STARTED = True

    def _daemon_loop() -> None:
        consecutive_dead = 0
        last_render_ping = 0.0
        was_ready = False
        last_ready_ts = 0.0
        last_boot_heal_ts = 0.0
        stale_boot_heals = 0
        while True:
            try:
                now = time.monotonic()
                # 1. Ping Render public URL every 60s so Render Free Tier never spins down
                if now - last_render_ping >= 60.0:
                    last_render_ping = now
                    try:
                        _self_url = (os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("BLACKTHORN_PUBLIC_URL") or "").rstrip("/")
                        if _self_url:
                            req = urllib.request.Request(
                                f"{_self_url}/api/health",
                                headers={"User-Agent": "BlackthornPermanentKeepAlive/1.0"},
                            )
                            urllib.request.urlopen(req, timeout=10).read()
                    except Exception:
                        pass

                # 2. Refresh GPU status & keep Cloudflare tunnel + Kaggle GPU warm
                st = get_kaggle_gpu_status(force_refresh=True)

                # 2a. Inactivity auto-off: only when genuinely idle and nothing is running.
                try:
                    d1_state = str((d1_query("SELECT status FROM kaggle_gpu_state WHERE id='primary' LIMIT 1;") or [{}])[0].get("status") or "OFF")
                    ready_now = bool(st.get("active")) and not st.get("booting")
                    if ready_now and not was_ready:
                        # The boot itself just finished — start the idle clock here,
                        # never from process start.
                        mark_activity("gpu-ready")
                        was_ready = True
                        last_ready_ts = time.monotonic()
                        logger.info("Kaggle GPU is ready — inactivity timer restarted")
                    elif not ready_now:
                        was_ready = False
                    # Decide only AFTER the clock bookkeeping above, and never within
                    # the first minute of a fresh boot (a decision computed from the
                    # pre-boot idle time would otherwise kill the GPU instantly).
                    decision = auto_off_decision()
                    if (
                        decision.get("should_stop")
                        and decision.get("active_tasks", 0) == 0
                        and ready_now
                        and not st.get("booting")
                        and str(d1_state).upper() in READY_STATUSES
                        and (time.monotonic() - last_ready_ts) > 60.0
                        and d1_state not in ("GPU_STOPPED_SAVING_QUOTA", "OFF", "STOPPING_KAGGLE_GPU")
                    ):
                        logger.info("Blackthorn auto-off: %s — stopping Kaggle GPU", decision.get("reason"))
                        _record_auto_off(str(decision.get("reason") or "inactivity"))
                        turn_off_kaggle_gpu()
                        consecutive_dead = 0
                        time.sleep(20.0)
                        continue
                except Exception as exc:
                    logger.debug("auto-off tick note: %s", exc)
                rows = d1_query("SELECT status, tunnel_url, updated_at FROM kaggle_gpu_state WHERE id = 'primary' LIMIT 1;")
                d1_status = str(rows[0].get("status") or "OFF") if rows else "OFF"
                d1_updated = float((rows[0] if rows else {}).get("updated_at") or 0)

                # 2b. Boot watchdog: a push that never reaches a stage again would
                # otherwise leave the UI on "Starting Kaggle" forever. If the boot
                # went quiet and the kernel is not actually running, push again.
                BOOTING_STAGES = {
                    "BOOTING_KAGGLE_GPU", "CHECKING_ENVIRONMENT", "CHECKING_CACHE", "CACHE_MISS",
                    "DOWNLOADING_MODEL", "MODEL_DOWNLOADED", "VERIFYING_MODEL", "INSTALLING_DEPS",
                    "INSTALLING_OLLAMA", "STARTING_OLLAMA", "LOADING_MODEL", "STARTING_GATEWAY",
                }
                if d1_status in BOOTING_STAGES and not st.get("active"):
                    quiet_for = time.time() - d1_updated if d1_updated else 9999
                    if quiet_for > 240 and (now - last_boot_heal_ts) > 300 and stale_boot_heals < 3:
                        kernel_status = ""
                        try:
                            kernel_status = str(_kaggle_rpc(
                                "GetKernelSessionStatus",
                                {"userName": KAGGLE_USERNAME, "kernelSlug": KAGGLE_KERNEL_SLUG},
                            ).get("status") or "").upper()
                        except Exception as exc:
                            logger.debug("boot watchdog session probe note: %s", exc)
                        logger.warning(
                            "Boot watchdog: stuck on %s for %.0fs (kernel=%s) — re-pushing",
                            d1_status, quiet_for, kernel_status or "unknown",
                        )
                        last_boot_heal_ts = now
                        stale_boot_heals += 1
                        try:
                            turn_on_kaggle_gpu()
                        except Exception as exc:
                            logger.warning("boot watchdog re-push failed: %s", exc)
                elif st.get("active"):
                    stale_boot_heals = 0

                if d1_status in ("BOOT_FAILED", "GPU_UNAVAILABLE", "INTERNET_UNAVAILABLE", "MODEL_CORRUPT"):
                    # Terminal boot errors must not be auto-retried in a loop; the
                    # user sees the reason and can turn the GPU on again.
                    stale_boot_heals = 3

                # Only auto-heal if the user has NOT explicitly turned OFF the GPU
                if d1_status not in ("GPU_STOPPED_SAVING_QUOTA", "OFF"):
                    if st.get("active") or st.get("booting"):
                        consecutive_dead = 0
                    else:
                        consecutive_dead += 1
                        if consecutive_dead >= 3:
                            logger.info("Permanent GPU watchdog: restarting Kaggle GPU kernel to maintain permanent connection")
                            consecutive_dead = 0
                            try:
                                turn_on_kaggle_gpu()
                            except Exception as exc:
                                logger.debug("Watchdog restart note: %s", exc)
                else:
                    consecutive_dead = 0
            except Exception as exc:
                logger.debug("Permanent GPU daemon tick note: %s", exc)
            time.sleep(20.0)

    t = threading.Thread(target=_daemon_loop, daemon=True, name="blackthorn-permanent-gpu-daemon")
    t.start()



def _probe_live_tunnel(timeout: int = 14) -> bool:
    """True when D1 holds a tunnel whose gateway answers /health as online."""
    try:
        rows = d1_query("SELECT status, tunnel_url FROM kaggle_gpu_state WHERE id = 'primary' LIMIT 1;")
        row = rows[0] if rows else {}
        url = str(row.get("tunnel_url") or "").rstrip("/")
        if not url or str(row.get("status") or "").upper() in ("GPU_STOPPED_SAVING_QUOTA", "OFF", "BOOT_FAILED"):
            return False
        req = urllib.request.Request(f"{url}/health", headers={"User-Agent": "HermesAgentGPUCheck/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")).get("status") == "online"
    except Exception as exc:
        logger.debug("live tunnel probe note: %s", exc)
        return False


_TURN_ON_LOCK = threading.Lock()
_TURN_ON_IN_FLIGHT = False
_TURN_ON_TS = 0.0


def turn_on_kaggle_gpu(blocking: bool = False) -> Dict[str, Any]:
    """Turn the Kaggle GPU ON and return *immediately*.

    The Kaggle API needs a few seconds to a couple of minutes (a stale session has
    to be stopped first), and the header button used to sit there until all of it
    finished — which read as "the GPU isn't starting". Now the D1 state flips to
    BOOTING_KAGGLE_GPU right away and the push happens in the background while the
    UI follows the real boot stages.
    """
    global _TURN_ON_IN_FLIGHT, _TURN_ON_TS
    _ensure_kaggle_credentials()
    ensure_d1_schema()

    if blocking:
        return _turn_on_kaggle_gpu_sync()

    # Already healthy? Then this is a no-op — reusing the live session is the
    # fastest possible "turn on" and it must never restart a working kernel.
    health = _probe_live_tunnel()
    if health:
        logger.info("Turn-on: existing Kaggle session is healthy — reusing it")
        mark_activity("turn-on-reuse")
        cached = dict(_STATUS_CACHE or {})
        if cached.get("active") and not cached.get("booting"):
            # Already healthy and already known: answer from memory so the button
            # reacts instantly instead of waiting on another quota/kernel round-trip.
            cached["status"] = cached.get("status") or "ONLINE"
            cached["progress_step"] = cached.get("progress_step") or "GPU Permanently Connected & Active"
            return cached
        return get_kaggle_gpu_status(force_refresh=False)

    with _TURN_ON_LOCK:
        if _TURN_ON_IN_FLIGHT and (time.time() - _TURN_ON_TS) < 600:
            logger.info("Turn-on already in progress — reusing it")
            mark_activity("turn-on-reuse")
            return get_kaggle_gpu_status(force_refresh=True)
        _TURN_ON_IN_FLIGHT = True
        _TURN_ON_TS = time.time()

    # Optimistic, truthful state: the GPU is not up yet, but a boot has begun.
    try:
        d1_query(
            """INSERT INTO kaggle_gpu_state (id, status, tunnel_url, api_key, model, gpu_info, updated_at)
               VALUES ('primary', 'BOOTING_KAGGLE_GPU', '', ?, ?, 'Allocating 2x NVIDIA Tesla T4...', ?)
               ON CONFLICT(id) DO UPDATE SET status = excluded.status, tunnel_url = excluded.tunnel_url, model = excluded.model, gpu_info = excluded.gpu_info, updated_at = excluded.updated_at;""",
            [CYBER_ORNITH_API_KEY, CYBER_ORNITH_MODEL, time.time()],
        )
    except Exception as exc:
        logger.debug("turn-on optimistic state note: %s", exc)
    mark_activity("turn-on")
    _STATUS_CACHE_TS = 0.0

    def _worker() -> None:
        global _TURN_ON_IN_FLIGHT
        try:
            _turn_on_kaggle_gpu_sync()
        except Exception as exc:
            logger.error("Kaggle turn-on failed: %s", exc)
            try:
                d1_query(
                    """INSERT INTO kaggle_gpu_state (id, status, tunnel_url, api_key, model, gpu_info, updated_at)
                       VALUES ('primary', 'BOOT_FAILED', '', ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET status = excluded.status, tunnel_url = excluded.tunnel_url, model = excluded.model, gpu_info = excluded.gpu_info, updated_at = excluded.updated_at;""",
                    [CYBER_ORNITH_API_KEY, CYBER_ORNITH_MODEL,
                     f"Kaggle turn-on failed — {str(exc)[:110]}", time.time()],
                )
            except Exception:
                pass
        finally:
            with _TURN_ON_LOCK:
                _TURN_ON_IN_FLIGHT = False

    threading.Thread(target=_worker, daemon=True, name="blackthorn-kaggle-turn-on").start()
    return get_kaggle_gpu_status(force_refresh=True)
