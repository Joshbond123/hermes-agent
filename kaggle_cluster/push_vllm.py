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


# usage: push_vllm.py [user_key=P] [token_key=P] [slug] [template]
uk = sys.argv[1] if len(sys.argv) > 1 else "KAGGLE_BU3"
tk = sys.argv[2] if len(sys.argv) > 2 else "KAGGLE_B3"
slug = sys.argv[3] if len(sys.argv) > 3 else "vllm-sm75-bf16-test"
tpl_path = sys.argv[4] if len(sys.argv) > 4 else "/home/user/work/cluster/vllm_sm75_template.py"

tpl = open(tpl_path).read()
text = (
    tpl.replace("@@CF_ACCOUNT@@", g("CF_ACCOUNT"))
    .replace("@@CF_DB@@", g("CF_DB"))
    .replace("@@CF_KEY@@", g("CF_KEY"))
    .replace("@@HF_TOKEN@@", g("HF_TOKEN"))
)
for marker in ("@@CF_ACCOUNT@@", "@@CF_DB@@", "@@CF_KEY@@", "@@HF_TOKEN@@"):
    if marker in text:
        raise SystemExit(f"unreplaced marker {marker}")

acct = Account(g(uk), g(tk), role="model", slot="backup")
fleet = FleetManager([acct])
payload = {
    "slug": f"{acct.user}/{slug}",
    "newTitle": "VLLM SM75 BF16 Test",
    "text": text,
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
    print("pushed:", json.dumps({k: out.get(k) for k in
          ("slug", "ref", "versionNumber") if k in out}), flush=True)
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
