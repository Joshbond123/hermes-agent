#!/usr/bin/env python3
"""Push the E10 multi-node notebook (always ipynb-wrapped creation) to one account.

Usage: push_mnode.py <user_key> <token_key> <title> <TAG> <rank> [model] [master] [nnodes]
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, "/home/user/work/src")
from blackthorn_fleet import Account, FleetManager, FleetError  # noqa: E402

ENV = open(os.path.expanduser("~/.secrets/com11.env")).read()


def g(n):
    m = re.search(rf"^{n}=(.*)$", ENV, re.M)
    return m.group(1).strip() if m else ""


uk, tk, title, TAG, RANK = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
MODEL = sys.argv[6] if len(sys.argv) > 6 else "Qwen/Qwen3.5-4B"
MASTER = sys.argv[7] if len(sys.argv) > 7 else "10.77.0.1"
NNODES = sys.argv[8] if len(sys.argv) > 8 else "2"
PP = sys.argv[9] if len(sys.argv) > 9 else "2"          # pipeline stages (== nnodes)
DL_MODE = sys.argv[10] if len(sys.argv) > 10 else "all"  # all | smallonly
HOLD = sys.argv[11] if len(sys.argv) > 11 else "600"     # worker hold seconds
ROWPEER = "mn_ready_" + RANK
peers = ",".join("mn_ready_%d" % i for i in range(1, int(NNODES)) if str(i) != RANK)

tpl = open("/home/user/work/cluster/mnode_template.py").read()
code = tpl
repl = {
    "@@CF_ACCOUNT@@": g("CF_ACCOUNT"), "@@CF_DB@@": g("CF_DB"),
    "@@CF_KEY@@": g("CF_KEY"), "@@HF_TOKEN@@": g("HF_TOKEN"),
    "@@TAG@@": TAG, "@@NODE_RANK@@": RANK, "@@NNODES@@": NNODES,
    "@@MASTER@@": MASTER, "@@MODEL@@": MODEL, "@@ROWPEER@@": ROWPEER,
    "@@PP@@": PP, "@@DL_MODE@@": DL_MODE, "@@HOLDSECS@@": HOLD, "@@PEERS@@": peers,
}
for k, v in repl.items():
    code = code.replace(k, v)
left = [k for k in repl if k in code]
if left:
    raise SystemExit(f"unreplaced markers: {left}")

lines = code.splitlines(keepends=True)
if lines and not lines[-1].endswith("\n"):
    lines[-1] += "\n"
notebook = {
    "cells": [{"cell_type": "code", "execution_count": None, "id": "c1",
               "metadata": {}, "outputs": [], "source": lines}],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
acct = Account(g(uk), g(tk), role="model", slot="primary" if RANK == "0" else "backup")
fleet = FleetManager([acct])
payload = {
    "slug": f"{acct.user}/{title}",
    "newTitle": title,           # title == desired slug (slugify on creation)
    "text": json.dumps(notebook),
    "language": "python",
    "kernelType": "notebook",
    "isPrivate": True,
    "enableGpu": True,
    "enableTpu": False,
    "enableInternet": True,
    "machineShape": "NvidiaTeslaT4",
    "kernelExecutionType": "SAVE_AND_RUN_ALL",
}
try:
    out = fleet.rpc(acct, "SaveKernel", payload)
    print("pushed FULL:", json.dumps(out)[:600], flush=True)
except FleetError as e:
    raise SystemExit(f"PUSH FAILED: {e}")

# confirm with the REAL ref slug (slugify(title) can differ)
ref = str(out.get("ref") or f"/{acct.user}/{title}")
real = ref.rsplit("/", 1)[-1]
for i in range(6):
    try:
        st = fleet.rpc(acct, "GetKernelSessionStatus",
                       {"userName": acct.user, "kernelSlug": real})
        print("session:", json.dumps(st)[:250], flush=True)
        break
    except FleetError as e:
        print(f"status try{i}: {str(e)[:80]}", flush=True)
        time.sleep(5)
