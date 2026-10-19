#!/usr/bin/env python3
"""Poll D1 kaggle_gpu_state rows. Stdlib only.

Usage:
  python3 d1poll.py                 # status rows (vllm_sm75*, np_*, gpu_dbg*)
  python3 d1poll.py vllm_sm75       # exact row + full notes
  python3 d1poll.py vllm_sm75_l1    # a log-chunk row (full content)
Rows may come back as dicts or lists - both handled.
"""
import json
import os
import sys
import urllib.request

ENV_PATH = os.path.expanduser("~/.secrets/com11.env")


def _g(k):
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith(k + "="):
                return line.strip().split("=", 1)[1]
    return ""


def q(sql, params=()):
    # D1 REST rejects ?-binding here; inline literals (values are controlled ids).
    for p in params:
        sql = sql.replace("?", "'%s'" % str(p).replace("'", "''"), 1)
    body = json.dumps({"sql": sql, "params": []}).encode()
    req = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/accounts/%s/d1/database/%s/query"
        % (_g("CF_ACCOUNT"), _g("CF_DB")),
        data=body,
        headers={
            "Authorization": "Bearer " + _g("CF_KEY"),
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.load(r)
    if not out.get("success"):
        raise SystemExit("D1 error: %s" % out.get("errors"))
    return out["result"][0]["results"]


def row_get(r, name, idx):
    """Row is dict keyed by column name OR list of values."""
    if isinstance(r, dict):
        return r.get(name)
    return r[idx] if idx < len(r) else None


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg:
        rows = q(
            "SELECT id,status,detail,gpu_info,updated_at FROM kaggle_gpu_state WHERE id=?",
            (arg,),
        )
        if not rows:
            print("%s: NO ROW" % arg)
            return
        for r in rows:
            if isinstance(r, dict):
                print(json.dumps(r, indent=2, default=str))
            else:
                print(r)
        return
    rows = q(
        "SELECT id,status,updated_at FROM kaggle_gpu_state "
        "WHERE id LIKE 'vllm%' OR id LIKE 'np_%' OR id LIKE 'gpu_dbg%' OR id LIKE 'mn%' "
        "ORDER BY updated_at DESC LIMIT 30"
    )
    for r in rows:
        if isinstance(r, dict):
            print("%-18s %-16s %s" % (r.get("id"), r.get("status"), r.get("updated_at")))
        else:
            print(r)


if __name__ == "__main__":
    main()
