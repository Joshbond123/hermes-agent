#!/usr/bin/env python3
"""Push the netprobe notebook to two Kaggle accounts (CPU, internet on) and report session states."""
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


tpl = open("/home/user/work/cluster/netprobe_template.py").read()

NODES = [
    # (tag, user_key, token_key, title, slug)
    ("p1", "KAGGLE_U1", "KAGGLE_P1", "Netprobe P1", "netprobe-p1"),
    ("p2", "KAGGLE_U2", "KAGGLE_P2", "Netprobe P2", "netprobe-p2"),
]

for tag, uk, tk, title, slug in NODES:
    text = (
        tpl.replace("@@TAG@@", tag)
        .replace("@@KAGGLE_USER@@", g(uk))
        .replace("@@KAGGLE_TOKEN@@", g(tk))
        .replace("@@CF_ACCOUNT@@", g("CF_ACCOUNT"))
        .replace("@@CF_DB@@", g("CF_DB"))
        .replace("@@CF_KEY@@", g("CF_KEY"))
    )
    acct = Account(g(uk), g(tk), role="model", slot="primary" if tag == "p1" else "backup")
    fleet = FleetManager([acct])
    payload = {
        "slug": slug,
        "newTitle": title,
        "text": text,
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": True,
        "enableGpu": False,
        "enableTpu": False,
        "enableInternet": True,
    }
    try:
        out = fleet.rpc(acct, "SaveKernel", payload)
        print(f"{tag} ({acct.user}) pushed:", json.dumps({k: out.get(k) for k in ("slug", "ref", "versionNumber", "invalidDatasetSources") if k in out}) or out, flush=True)
    except FleetError as e:
        print(f"{tag} ({acct.user}) PUSH FAILED: {e}", flush=True)
        continue
    # confirm session state
    for i in range(6):
        try:
            st = fleet.rpc(acct, "GetKernelSessionStatus", {"userName": acct.user, "kernelSlug": slug})
            print(f"{tag} session: {json.dumps(st)[:300]}", flush=True)
            break
        except FleetError as e:
            print(f"{tag} status try{i}: {e}", flush=True)
            time.sleep(4)
print("pUSH DONE", flush=True)
