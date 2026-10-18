#!/usr/bin/env python3
"""Commit+push files to Joshbond123/hermes-agent via GitHub API (no clone).

Usage:
  python3 gh_push.py "commit message" local_path@remote_path [local_path@remote_path ...]
Reads HERMES_PAT from ~/.secrets/com11.env (never printed).
Safety: refuses computer-artifact paths (com11*, *comtool*).
"""
import base64
import json
import os
import sys
import urllib.request

ENV = os.path.expanduser("~/.secrets/com11.env")
REPO = "Joshbond123/hermes-agent"
API = "https://api.github.com"


def pat():
    with open(ENV) as f:
        for line in f:
            if line.startswith("HERMES_PAT="):
                return line.strip().split("=", 1)[1]
    raise SystemExit("HERMES_PAT missing")


def gh(method, path, payload=None):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method,
        headers={
            "Authorization": "Bearer " + pat(),
            "Accept": "application/vnd.github+json",
            "User-Agent": "agent",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    msg = sys.argv[1]
    pairs = []
    for spec in sys.argv[2:]:
        local, remote = spec.rsplit("@", 1)
        low = remote.lower()
        if "com11" in low or "comtool" in low:
            raise SystemExit("REFUSED: computer-artifact path %s" % remote)
        pairs.append((local, remote))

    ref = gh("GET", "/repos/%s/git/ref/heads/main" % REPO)
    base_sha = ref["object"]["sha"]
    commit = gh("GET", "/repos/%s/git/commits/%s" % (REPO, base_sha))
    base_tree = commit["tree"]["sha"]

    tree = []
    for local, remote in pairs:
        with open(local, "rb") as fh:
            content = base64.b64encode(fh.read()).decode()
        blob = gh(
            "POST",
            "/repos/%s/git/blobs" % REPO,
            {"content": content, "encoding": "base64"},
        )
        tree.append({"path": remote, "mode": "100644", "type": "blob",
                     "sha": blob["sha"]})
        print("blob %s <- %s (%d B)" % (remote, local, os.path.getsize(local)))

    new_tree = gh("POST", "/repos/%s/git/trees" % REPO,
                  {"base_tree": base_tree, "tree": tree})
    new_commit = gh("POST", "/repos/%s/git/commits" % REPO,
                    {"message": msg, "tree": new_tree["sha"],
                     "parents": [base_sha]})
    gh("PATCH", "/repos/%s/git/refs/heads/main" % REPO,
       {"sha": new_commit["sha"]})
    print("PUSHED %s %s" % (new_commit["sha"][:10], msg.split("\n")[0]))


if __name__ == "__main__":
    main()
