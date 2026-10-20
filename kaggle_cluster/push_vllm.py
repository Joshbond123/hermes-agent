#!/usr/bin/env python3
"""Push the vLLM sm75 notebook (GPU, internet on, SAVE_AND_RUN_ALL) to one account."""
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


# usage: push_vllm.py [user_key] [token_key] [slug] [template] [--raw]
uk = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "KAGGLE_BU3"
tk = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "KAGGLE_B3"
slug = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "vllm-sm75-bf16-test"
tpl_path = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else "/home/user/work/cluster/vllm_sm75_template.py"
RAW = "--raw" in sys.argv
TAG = {"KAGGLE_BU3": "hap", "KAGGLE_BU2": "tem", "KAGGLE_U3": "ala",
       "KAGGLE_U1": "bla", "KAGGLE_U2": "tes", "KAGGLE_BU1": "jos"}.get(uk, "x")
PART = "1"
for a in sys.argv:
    if a.startswith("--part="):
        PART = a.split("=", 1)[1]
KUSER_KEY = {"KAGGLE_BU3": "KAGGLE_B3", "KAGGLE_BU2": "KAGGLE_B2",
             "KAGGLE_U3": "KAGGLE_P3", "KAGGLE_U1": "KAGGLE_P1",
             "KAGGLE_U2": "KAGGLE_P2", "KAGGLE_BU1": "KAGGLE_B1"}.get(uk, tk)
KUSER = g(uk)   # kaggle username of the chosen account (value not printed)
KKEY = g(KUSER_KEY)

tpl = open(tpl_path).read()
code = (
    tpl.replace("@@CF_ACCOUNT@@", g("CF_ACCOUNT"))
    .replace("@@CF_DB@@", g("CF_DB"))
    .replace("@@CF_KEY@@", g("CF_KEY"))
    .replace("@@HF_TOKEN@@", g("HF_TOKEN"))
    .replace("@@TAG@@", TAG)
    .replace("@@PART@@", PART)
    .replace("@@KUSER@@", KUSER)
    .replace("@@KKEY@@", KKEY)
)
# generic: any @@NAME@@ whose NAME exists in the secrets env
import re as _re
_ENVD = {}
for _line in ENV.splitlines():
    if "=" in _line and _line.split("=", 1)[0].isidentifier():
        _ENVD[_line.split("=", 1)[0]] = _line.split("=", 1)[1].strip()
for _m in _re.finditer(r"@@([A-Z0-9_]+)@@", code):
    _name = _m.group(1)
    if _name not in _ENVD or not _ENVD[_name]:
        raise SystemExit(f"unreplaced marker @@{_name}@@ (missing/empty in env)")
    code = code.replace(f"@@{_name}@@", _ENVD[_name])
if _re.search(r"@@[A-Z0-9_]+@@", code):
    raise SystemExit("unreplaced markers remain: " + str(_re.findall(r"@@[A-Z0-9_]+@@", code)))

if RAW:
    text = code  # control: raw .py (scheduled fine as v10; fails at session parse)
else:
    # DEPRECATED path kept for A/B: ipynb JSON saved fine but never scheduled
    # (v11/v12); raw+SAVE_AND_RUN_ALL is the empirically working combo (v13).
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
    text = json.dumps(notebook)
# Default to raw: flip when --wrap given.
if not RAW:
    text = code
if "--wrap" in sys.argv:
    text = json.dumps(notebook)

acct = Account(g(uk), g(tk), role="model", slot="backup")
fleet = FleetManager([acct])
payload = {
    "slug": f"{acct.user}/{slug}",
    "newTitle": slug,  # creation derives slug from title: keep both identical
    "text": text,
    "language": "python",
    "kernelType": "notebook",
    "isPrivate": True,
    "enableGpu": "--cpu" not in sys.argv,
    "enableTpu": False,
    "enableInternet": True,
    "kernelExecutionType": "SAVE_AND_RUN_ALL",
}
if "--cpu" not in sys.argv:
    payload["machineShape"] = "NvidiaTeslaT4"
try:
    out = fleet.rpc(acct, "SaveKernel", payload)
    print("pushed FULL:", json.dumps(out)[:900], flush=True)
except FleetError as e:
    raise SystemExit(f"PUSH FAILED: {e}")

for i in range(8):
    try:
        st = fleet.rpc(acct, "GetKernelSessionStatus",
                       {"userName": acct.user, "kernelSlug": slug})
        print("session:", json.dumps(st)[:300], flush=True)
        break
    except FleetError as e:
        print(f"status try{i}: {e}", flush=True)
        time.sleep(5)
