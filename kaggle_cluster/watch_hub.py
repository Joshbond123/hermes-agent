#!/usr/bin/env python3
"""Hub watcher: the moment mn_ala shows RUNNING_* (hub executing),
push a fresh spoke kernel on temilolo (its waves start near-prompt).
Exits after pushing. Stdlib only."""
import json
import os
import re
import subprocess
import sys
import time

ENV = open(os.path.expanduser("~/.secrets/com11.env")).read()


def g(n):
    m = re.search(rf"^{n}=(.*)$", ENV, re.M)
    return m.group(1).strip() if m else ""


def d1(sql, params=()):
    for p in params:
        sql = sql.replace("?", "'%s'" % str(p).replace("'", "''"), 1)
    body = json.dumps({"sql": sql, "params": []}).encode()
    req = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/accounts/%s/d1/database/%s/query"
        % (g("CF_ACCOUNT"), g("CF_DB")),
        data=body, method="POST",
        headers={"Authorization": "Bearer " + g("CF_KEY"),
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.load(r)
    if not out.get("success"):
        raise RuntimeError(out.get("errors"))
    return out["result"][0]["results"]


import urllib.request  # noqa: E402

PUSHED = False
t0 = time.time()
print("watcher2 start", time.strftime("%H:%M:%S"), flush=True)
while time.time() - t0 < 14000:
    fired = False
    for row, floor in (("mn_ala", 1792410871), ("mn_jb", 1792410000)):
        try:
            rows = d1("SELECT status,updated_at FROM kaggle_gpu_state WHERE id=?", [row])
            r = rows[0] if rows else None
            if isinstance(r, dict):
                st, upd = r.get("status"), float(r.get("updated_at") or 0)
            else:
                st, upd = (r[0], float(r[1])) if r else (None, 0)
        except Exception as e:  # noqa: BLE001
            print("poll err", row, repr(e)[:100], flush=True)
            st, upd = None, 0
        if st and st.startswith("RUNNING") and upd > floor:
            print(time.strftime("%H:%M:%S"), "HUB", row, "RUNNING -> pushing spoke", flush=True)
            fired = True
            break
        if st and (st.startswith("FAIL") or st == "DONE") and upd > floor and not st.startswith("FAIL_BORE"):
            print(time.strftime("%H:%M:%S"), "HUB", row, "TERMINAL", st, flush=True)
    if fired:
        print("pushing spoke", flush=True)
        cmd = ["python3", "/home/user/work/cluster/push_mnode.py",
               "KAGGLE_BU2", "KAGGLE_B2", "mn-tem-n1", "tem", "1"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        print(p.stdout[-500:], p.stderr[-200:], flush=True)
        PUSHED = True
        break
    time.sleep(15)
sys.exit(0 if PUSHED else 2)
