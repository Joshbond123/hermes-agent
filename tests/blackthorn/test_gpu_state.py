"""The GPU state is derived from evidence; the UI must never claim more than we know."""

from __future__ import annotations

from blackthorn.gpu_state import Evidence, derive

NOW = 10_000.0
HEALTH_LOADED = {"status": "online", "model_loaded": True}
HEALTH_COLD = {"status": "online", "model_loaded": False}


def ev(**kw):
    base = dict(row_status="", row_updated_at=NOW - 5, tunnel_url="", session_status=None, health=None)
    base.update(kw)
    return Evidence(**base)


def test_off_when_nothing_is_running():
    s = derive(ev(row_status="GPU_STOPPED_SAVING_QUOTA", session_status="NONE"), NOW)
    assert s["state"] == "off" and s["progress"]["mode"] == "none" and not s["tunnel"]


def test_empty_row_is_off():
    assert derive(ev(), NOW)["state"] == "off"


def test_running_session_with_empty_row_is_starting_not_off():
    s = derive(ev(row_status="OFF", session_status="RUNNING"), NOW)
    assert s["state"] == "starting" and s["stage"] == "BOOTING_KAGGLE_GPU"


def test_boot_stages_are_starting_with_indeterminate_progress():
    for status in ("BOOTING_KAGGLE_GPU", "CHECKING_ENVIRONMENT", "INSTALLING_DEPS", "LOADING_MODEL", "STARTING_GATEWAY"):
        s = derive(ev(row_status=status, session_status="RUNNING"), NOW)
        assert s["state"] == "starting" and s["progress"] == {"mode": "indeterminate"}, status
        assert s["stage_label"]


def test_download_progress_is_measured_never_invented():
    s = derive(ev(row_status="DOWNLOADING_MODEL", session_status="RUNNING",
                  detail={"done": 4.2e9, "total": 16.8e9, "unit": "bytes"}), NOW)
    assert s["progress"]["mode"] == "determinate" and abs(s["progress"]["fraction"] - 0.25) < 1e-9
    # no numbers reported -> honest "unknown", not a made-up percentage
    s = derive(ev(row_status="DOWNLOADING_MODEL", session_status="RUNNING"), NOW)
    assert s["progress"] == {"mode": "indeterminate"}


def test_ready_requires_a_live_probe_with_the_model_loaded():
    base = dict(row_status="MODEL_READY_AND_WARMED", tunnel_url="https://t", session_status="RUNNING")
    ready = derive(ev(health=HEALTH_LOADED, **base), NOW)
    assert ready["state"] == "ready" and ready["model_loaded"] is True and ready["tunnel"]
    # the row *says* ready but nothing answered yet: not ready
    unknown = derive(ev(health=None, probe_failures=1, **base), NOW)
    assert unknown["state"] != "ready"


def test_ready_row_but_weights_not_resident_is_ready_with_an_honest_note():
    s = derive(ev(row_status="MODEL_READY_COLD", tunnel_url="https://t", session_status="RUNNING", health=HEALTH_COLD), NOW)
    assert s["state"] == "ready" and s["model_loaded"] is False and "next message" in s["message"]


def test_tunnel_up_but_still_booting_with_cold_model_is_starting():
    s = derive(ev(row_status="TUNNEL_ONLINE", tunnel_url="https://t", session_status="RUNNING", health=HEALTH_COLD), NOW)
    assert s["state"] == "starting" and s["stage"] == "LOADING_WEIGHTS"


def test_ready_flips_to_error_only_after_repeated_probe_failures():
    base = dict(row_status="HEARTBEAT_ONLINE", tunnel_url="https://t", session_status="RUNNING", health=None)
    assert derive(ev(probe_failures=1, **base), NOW)["state"] != "error"
    assert derive(ev(probe_failures=2, **base), NOW)["state"] != "error"
    err = derive(ev(probe_failures=3, **base), NOW)
    assert err["state"] == "error" and err["error"]["code"] == "tunnel_unreachable"


def test_session_that_ended_before_ready_is_an_error_after_a_grace_period():
    base = dict(row_status="BOOTING_KAGGLE_GPU", session_status="COMPLETE")
    assert derive(ev(row_updated_at=NOW - 30, status_since=NOW - 30, **base), NOW)["state"] == "starting"
    s = derive(ev(row_updated_at=NOW - 400, status_since=NOW - 400, **base), NOW)
    assert s["state"] == "error" and s["error"]["code"] == "session_ended"


def test_stalled_means_long_silence_from_a_live_session():
    s = derive(ev(row_status="LOADING_MODEL", session_status="RUNNING", row_updated_at=NOW - 500,
                  status_since=NOW - 500), NOW)
    assert s["state"] == "starting" and s["stalled"] is True and s["silent_for_s"] == 500
    fresh = derive(ev(row_status="LOADING_MODEL", session_status="RUNNING", row_updated_at=NOW - 20), NOW)
    assert fresh["stalled"] is False


def test_terminal_error_statuses_carry_their_reason():
    s = derive(ev(row_status="GPU_UNAVAILABLE", gpu_info="FAILED: no cuda"), NOW)
    assert s["state"] == "error" and "No GPU" in s["error"]["message"] and "no cuda" in s["error"]["message"]
    s = derive(ev(row_status="BOOT_FAILED", detail={"error": "Kaggle refused the push"}), NOW)
    assert "Kaggle refused the push" in s["error"]["message"]


def test_stopping_state():
    s = derive(ev(row_status="STOPPING_KAGGLE_GPU", session_status="RUNNING"), NOW)
    assert s["state"] == "stopping"


def test_stage_elapsed_uses_stage_start_when_known():
    s = derive(ev(row_status="DOWNLOADING_MODEL", session_status="RUNNING", status_since=NOW - 125,
                  boot_started_at=NOW - 300), NOW)
    assert s["stage_elapsed_s"] == 125 and s["elapsed_s"] == 300
