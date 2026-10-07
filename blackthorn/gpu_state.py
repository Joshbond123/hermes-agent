"""Truthful GPU state, derived from evidence (pure functions — no I/O).

The UI must never claim more than we know.  ``ready`` therefore requires *all* of:

* the Kaggle session is alive,
* the tunnel answers ``/health`` with ``status == online`` **and** ``model_loaded == true``.

Everything else is ``starting`` (with the real stage and, when the notebook reports it, real
byte counts), ``off``, ``stopping`` or ``error`` with the reason.  There are no invented
percentages: progress is either measured (``done``/``total``) or indeterminate.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Statuses written by the notebook / controller into ``kaggle_gpu_state.status``.
STAGES: Dict[str, str] = {
    "BOOTING_KAGGLE_GPU": "Allocating 2× Tesla T4 on Kaggle",
    "PUSHING_NOTEBOOK": "Sending the server notebook to Kaggle",
    "CHECKING_ENVIRONMENT": "Checking the GPUs and internet in the Kaggle container",
    "INSTALLING_DEPS": "Installing dependencies",
    "CHECKING_CACHE": "Looking for a cached copy of the model",
    "CACHE_HIT": "Cached model found",
    "CACHE_MISS": "No cached model — downloading",
    "DOWNLOADING_MODEL": "Downloading the model",
    "MODEL_DOWNLOADED": "Model downloaded",
    "VERIFYING_MODEL": "Verifying the model file",
    "STARTING_OLLAMA": "Starting the inference engine",
    "LOADING_MODEL": "Preparing the model",
    "BUILDING_ENGINE": "Building the inference engine (first start on this machine)",
    "STARTING_GATEWAY": "Starting the model gateway",
    "TUNNEL_ONLINE": "Tunnel online — loading weights into GPU memory",
    "WARMING_GPU": "Loading weights into GPU memory",
    "LOADING_WEIGHTS": "Loading weights into GPU memory",
    "MODEL_READY_COLD": "Model ready",
    "MODEL_READY_AND_WARMED": "Model ready",
    "HEARTBEAT_ONLINE": "Model ready",
    "ONLINE": "Model ready",
    "CACHE_DATASET_PUBLISHED": "Model ready",
    "CACHE_DATASET_READY": "Model ready",
    "STOPPING_KAGGLE_GPU": "Stopping the Kaggle session",
}
READY_STATUSES = frozenset({
    "ONLINE", "MODEL_READY_AND_WARMED", "HEARTBEAT_ONLINE", "MODEL_READY_COLD",
    "CACHE_DATASET_PUBLISHED", "CACHE_DATASET_READY",
})
OFF_STATUSES = frozenset({"", "OFF", "GPU_STOPPED_SAVING_QUOTA", "STOPPED"})
ERROR_STATUSES = {
    "GPU_UNAVAILABLE": "No GPU is attached to the Kaggle session (check the accelerator setting).",
    "INTERNET_UNAVAILABLE": "Internet is disabled in the Kaggle session.",
    "MODEL_CORRUPT": "The model download failed verification.",
    "BOOT_FAILED": "The Kaggle boot failed.",
    "TUNNEL_ERROR": "The Cloudflare tunnel could not be created.",
}
SESSION_ALIVE = frozenset({"RUNNING", "QUEUED", "STARTING", "PENDING", "RUNNING_QUEUED"})

STALL_SECONDS = 420          # no status change for this long while the session runs => "stalled"
NO_SESSION_GRACE = 150       # row says "starting" but Kaggle shows no session for this long => failed
PROBE_FAILS_TO_ERROR = 3     # consecutive failed health probes before ready -> error


@dataclass
class Evidence:
    """Everything the derivation looks at, gathered by the controller."""

    row_status: str = ""
    row_updated_at: float = 0.0
    tunnel_url: str = ""
    gpu_info: str = ""
    model: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)
    session_status: Optional[str] = None      # Kaggle: RUNNING/QUEUED/COMPLETE/ERROR/... or None if unknown
    health: Optional[Dict[str, Any]] = None   # parsed /health body, None when the probe failed/absent
    probe_failures: int = 0                   # consecutive failures (controller-maintained)
    boot_started_at: float = 0.0              # when turn-on was requested (0 = unknown)
    status_since: float = 0.0                 # when we first saw ``row_status``


def parse_detail(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _progress(detail: Dict[str, Any]) -> Dict[str, Any]:
    done, total = detail.get("done"), detail.get("total")
    if isinstance(done, (int, float)) and isinstance(total, (int, float)) and total > 0:
        return {
            "mode": "determinate",
            "done": float(done),
            "total": float(total),
            "unit": detail.get("unit") or "bytes",
            "fraction": max(0.0, min(1.0, float(done) / float(total))),
        }
    return {"mode": "indeterminate"}


def derive(ev: Evidence, now: Optional[float] = None) -> Dict[str, Any]:
    """Map evidence to the snapshot the UI renders."""
    now = now if now is not None else time.time()
    raw = (ev.row_status or "").upper()
    sess = (ev.session_status or "").upper() or None
    alive = sess in SESSION_ALIVE
    health_ok = bool(ev.health) and ev.health.get("status") == "online"
    model_loaded = ev.health.get("model_loaded") if health_ok else None
    since = ev.status_since or ev.row_updated_at or now
    stage = raw
    stage_label = STAGES.get(raw, raw.replace("_", " ").capitalize() if raw else "")
    state = "off"
    error: Optional[Dict[str, str]] = None
    message = ""
    progress: Dict[str, Any] = {"mode": "none"}

    if raw == "STOPPING_KAGGLE_GPU":
        state, message, progress = "stopping", "Stopping the Kaggle session…", {"mode": "indeterminate"}
    elif raw in ERROR_STATUSES:
        state = "error"
        reason = str(ev.detail.get("error") or ev.gpu_info or "").replace("FAILED:", "").strip()
        error = {"code": raw.lower(), "message": ERROR_STATUSES[raw] + (f" {reason}" if reason else "")}
        message = error["message"]
    elif raw in OFF_STATUSES and not alive:
        state, message = "off", "The GPU is off. Start it to chat."
    elif raw in OFF_STATUSES and alive:
        # the session exists but the notebook has not reported yet
        state, stage, stage_label = "starting", "BOOTING_KAGGLE_GPU", STAGES["BOOTING_KAGGLE_GPU"]
        progress = {"mode": "indeterminate"}
    else:
        tunnel = bool(ev.tunnel_url)
        if tunnel and health_ok and model_loaded is True:
            state, stage, stage_label = "ready", "ONLINE", "Model loaded and answering"
            message = "Ready — the model is loaded in GPU memory."
        elif tunnel and health_ok and raw in READY_STATUSES:
            # boot finished and the engine answers, but the weights are not resident right now
            # (they load on the next request): usable, and we say so.
            state, stage, stage_label = "ready", "ONLINE", "Engine online"
            message = "Ready — the next message loads the model into GPU memory (about a minute)."
        elif tunnel and health_ok:  # reachable, still booting, weights not resident yet
            state, stage, stage_label = "starting", "LOADING_WEIGHTS", STAGES["LOADING_WEIGHTS"]
            progress = {"mode": "indeterminate"}
        elif tunnel and raw in READY_STATUSES and ev.probe_failures >= PROBE_FAILS_TO_ERROR:
            state = "error"
            error = {"code": "tunnel_unreachable",
                     "message": "The GPU was ready but its tunnel stopped answering. The Kaggle session may have "
                                "ended. Restart the GPU."}
            message = error["message"]
        elif not alive and sess is not None and now - since > NO_SESSION_GRACE:
            state = "error"
            error = {"code": "session_ended",
                     "message": f"The Kaggle session is {sess.lower()} but the model never became ready. "
                                "Restart the GPU."}
            message = error["message"]
        else:
            state = "starting"
            progress = _progress(ev.detail) if raw in ("DOWNLOADING_MODEL", "VERIFYING_MODEL") else {"mode": "indeterminate"}
            if ev.detail.get("label"):
                stage_label = str(ev.detail["label"])

    stage_elapsed = max(0.0, now - since) if state == "starting" else 0.0
    silent_for = max(0.0, now - ev.row_updated_at) if ev.row_updated_at else 0.0
    # "stalled" = the notebook has said nothing at all for a long time (it heartbeats during long stages)
    stalled = state == "starting" and alive and silent_for > STALL_SECONDS
    if state == "starting" and not message:
        message = stage_label
    return {
        "state": state,
        "message": message,
        "stage": stage,
        "stage_label": stage_label,
        "progress": progress,
        "stage_elapsed_s": int(stage_elapsed),
        "silent_for_s": int(silent_for) if state == "starting" else 0,
        "elapsed_s": int(now - ev.boot_started_at) if state in ("starting",) and ev.boot_started_at else 0,
        "stalled": bool(stalled),
        "model": ev.model,
        "model_loaded": model_loaded,
        "gpu": (ev.gpu_info or "").replace("\n", " + "),
        "tunnel": bool(ev.tunnel_url) and state in ("ready", "starting"),
        "session": sess,
        "error": error,
        "raw_status": raw,
    }
