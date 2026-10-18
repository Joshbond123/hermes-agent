#!/usr/bin/env python3
"""Watch vllm_sm75 row until terminal state. Baselines updated_at at start.

Usage: python3 monitor_row.py <baseline_updated_at> [timeout_s]
Exit: 0 DONE, 1 FAIL_*, 2 timeout, 3 heartbeat/session death.
"""
import json
import os
import sys
import time
import urllib.request

ENV = os.path.expanduser("~/.secrets/com11.env")


def _g(k):
    with open(ENV) as f:
        for line in f:
            if line.startswith(k + "="):
                return line.strip().split("=", 1)[1]
    return ""


def q(sql, params=()):
    for p in params:
        sql = sql.replace("?", "'%s'" % str(p).replace("'", "''"), 1)
    body = json.dumps({"sql": sql, "params": []}).encode()
    req = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/accounts/%s/d1/database/%s/query"
        % (_g("CF_ACCOUNT"), _g("CF_DB")),
        data=body,
        method="POST",
        headers={"Authorization": "Bearer " + _g("CF_KEY"),
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.load(r)
    if not out.get("success"):
        raise RuntimeError(out.get("errors"))
    return out["result"][0]["results"]


def get(row):
    rows = q("SELECT status,gpu_info,updated_at FROM kaggle_gpu_state WHERE id=?", (row,))
    if not rows:
        return None
    r = rows[0]
    if not isinstance(r, dict):
        r = {"status": r[0], "gpu_info": r[1], "updated_at": r[2]}
    return r


def main():
    baseline = float(sys.argv[1])
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 1500.0
    t_end = time.time() + budget
    last = None
    print("monitor start baseline=%.1f budget=%.0fs" % (baseline, budget), flush=True)
    while time.time() < t_end:
        try:
            row = get("vllm_sm75")
            hb = get("vllm_sm75_hb")
        except Exception as e:  # noqa: BLE001
            print("poll err:", repr(e), flush=True)
            time.sleep(20)
            continue
        if row is None:
            print("ROW GONE", flush=True)
            sys.exit(3)
        st = row.get("status", "?")
        upd = float(row.get("updated_at") or 0)
        if st != last:
            print("[%s] status=%s updated_at=%.1f" % (
                time.strftime("%H:%M:%S"), st, upd), flush=True)
            last = st
        if upd > baseline and st.startswith("DONE"):
            print("TERMINAL DONE", flush=True)
            sys.exit(0)
        if upd > baseline and (st.startswith("FAIL") or st.startswith("LOGCAP")):
            print("TERMINAL", st, flush=True)
            sys.exit(1)
        stale = time.time() - max(upd, float(hb.get("updated_at") or 0) if hb else 0)
        if stale > 300 and upd > baseline and st.startswith("RUNNING"):
            print("SESSION DEATH: no update for %.0fs, last=%s" % (stale, st), flush=True)
            sys.exit(3)
        # pre-baseline rows are from the previous run: the new session may
        # still be queueing; only the overall budget bounds that wait.
        time.sleep(20)
    print("TIMEOUT budget, last=%s" % last, flush=True)
    sys.exit(2)


if __name__ == "__main__":
    main()
