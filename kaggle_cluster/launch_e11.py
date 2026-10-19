#!/usr/bin/env python3
"""E11 launcher: 4-node 27B PP4xTP2 over the validated overlay.

Run AFTER the E10 mini proves the transport (mn_result=DONE).
Status-gates every kernel first; never pushes over a running session.
Usage: python3 launch_e11.py [check|push]
"""
import json
import os
import re
import subprocess
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "check"

MODEL = "orcarouter/Qwen3.8-27B-Uncensored"
MASTER = "10.77.0.1"
NNODES = "4"
PP = "4"
DL = "smallonly"
HOLD = "3600"

NODES = [
    # (user_key, token_key, title==slug, tag, rank)
    ("KAGGLE_U3", "KAGGLE_P3", "mn27-ala-n0", "ala", "0"),   # alagbo  hub
    ("KAGGLE_BU2", "KAGGLE_B2", "mn27-tem-n1", "tem", "1"),  # temilolo
    ("KAGGLE_U2", "KAGGLE_P2", "mn27-ts-n2", "ts", "2"),     # teslaarymo
    ("KAGGLE_BU1", "KAGGLE_B1", "mn27-jb-n3", "jb", "3"),    # Joshbond123
]

sys.path.insert(0, "/home/user/work/src")
from blackthorn_fleet import Account, FleetManager, FleetError  # noqa: E402

ENV = open(os.path.expanduser("~/.secrets/com11.env")).read()


def g(n):
    m = re.search(rf"^{n}=(.*)$", ENV, re.M)
    return m.group(1).strip() if m else ""


def status_of(uk, tk, slug):
    a = Account(g(uk), g(tk), role="model", slot="primary")
    f = FleetManager([a])
    try:
        st = f.rpc(a, "GetKernelSessionStatus",
                   {"userName": a.user, "kernelSlug": slug})
        return str(st.get("status") or "{}")
    except FleetError:
        return "rpcerr/absent"


def main():
    print(f"== E11 {MODE} model={MODEL} pp={PP} dl={DL}")
    # always status-gate
    clear = True
    for uk, tk, title, tag, rank in NODES:
        s = status_of(uk, tk, title)
        print(f"  {title:14s} session={s}")
        if s in ("RUNNING", "QUEUED", "PENDING"):
            clear = False
    if MODE == "check":
        return 0 if clear else 2
    if not clear:
        print("NOT CLEAR - refusing to push")
        return 2
    for i, (uk, tk, title, tag, rank) in enumerate(NODES):
        cmd = ["python3", "/home/user/work/cluster/push_mnode.py",
               uk, tk, title, tag, rank, MODEL, MASTER, NNODES, PP, DL, HOLD]
        print("pushing", title, "rank", rank, flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if "pushed FULL" in line or "FAILED" in line:
                print("  ", line[:220])
        if i < len(NODES) - 1:
            time.sleep(25)  # stagger creates
    return 0


if __name__ == "__main__":
    sys.exit(main())
