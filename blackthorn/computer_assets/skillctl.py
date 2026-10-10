#!/usr/bin/env python3
"""On-demand skill and security-tool catalog for the Blackthorn computer.

Runs ON THE COMPUTER (not in model context). The model never receives the whole catalog:
it searches for a few matching entries, then loads one SKILL.md or one reference file.
Skill scripts are never executed by this program.

Usage:
  skillctl.py ensure                       clone/refresh the skill and catalog sources if missing
  skillctl.py search --q TEXT [--limit N]  rank skills by name/description keywords
  skillctl.py tools --q TEXT [--limit N]   search the hackingtool catalog (183 entries, metadata only)
  skillctl.py load --name SKILL            SKILL.md body (frontmatter removed) + file listing
  skillctl.py read --name SKILL --path REL read one file inside a skill folder (bounded)

Every command prints one JSON object on stdout: {"ok": bool, ...}.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.environ.get("BLACKTHORN_SKILL_ROOT", "/kaggle/working/blackthorn_workspace/security_catalog")
SKILLS_REPO = "https://github.com/mukul975/Anthropic-Cybersecurity-Skills"
TOOLS_REPO = "https://github.com/AKCodez/hackingtool-plugin"
SKILLS_DIR = os.path.join(ROOT, "anthropic-cybersecurity-skills")
TOOLS_DIR = os.path.join(ROOT, "hackingtool-plugin")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,120}$")
MAX_READ = 20000
CLONE_TIMEOUT = 240
MAX_REL_DEPTH = 4


def out(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def fail(message: str, **extra) -> None:
    out({"ok": False, "error": message, **extra})
    sys.exit(0)


def _clone(repo: str, dest: str) -> dict:
    if os.path.isdir(os.path.join(dest, ".git")):
        return {"repo": repo, "state": "present"}
    if os.path.exists(dest):
        return {"repo": repo, "state": "invalid", "error": "path exists without .git"}
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        proc = subprocess.run(["git", "clone", "--depth", "1", "--quiet", repo, dest],
                              capture_output=True, text=True, timeout=CLONE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"repo": repo, "state": "timeout"}
    if proc.returncode != 0:
        return {"repo": repo, "state": "failed", "error": proc.stderr.strip()[-300:]}
    return {"repo": repo, "state": "cloned"}


def cmd_ensure(_args) -> None:
    results = [_clone(SKILLS_REPO, SKILLS_DIR), _clone(TOOLS_REPO, TOOLS_DIR)]
    ok = all(r["state"] in ("present", "cloned") for r in results)
    out({"ok": ok, "sources": results, "skills_root": SKILLS_DIR})


def _index() -> list:
    path = os.path.join(SKILLS_DIR, "index.json")
    if not os.path.isfile(path):
        cmd_ensure(None)
    if not os.path.isfile(path):
        fail("skill index missing after ensure", hint="run ensure")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("skills", [])


def _terms(text: str) -> list:
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(t) > 2]


def _score(query_terms: list, name: str, description: str) -> int:
    name_terms = set(_terms(name))
    desc_terms = set(_terms(description))
    score = 0
    for term in query_terms:
        if term in name_terms:
            score += 5
        elif any(term in n for n in name_terms):
            score += 3
        if term in desc_terms:
            score += 2
    return score


def cmd_search(args) -> None:
    terms = _terms(args.q)
    if not terms:
        fail("query needs at least one word of 3+ letters")
    ranked = []
    for item in _index():
        s = _score(terms, item.get("name", ""), item.get("description", ""))
        if s > 0:
            ranked.append((s, item))
    ranked.sort(key=lambda pair: (-pair[0], pair[1].get("name", "")))
    results = [{"name": i.get("name"), "description": (i.get("description") or "")[:220],
                "score": s} for s, i in ranked[: max(1, min(args.limit, 15))]]
    out({"ok": True, "query": args.q, "matches": len(ranked), "results": results})


def cmd_tools(args) -> None:
    terms = _terms(args.q)
    if not terms:
        fail("query needs at least one word of 3+ letters")
    path = os.path.join(TOOLS_DIR, "plugins", "hackingtool", "data", "tools.json")
    if not os.path.isfile(path):
        cmd_ensure(None)
    if not os.path.isfile(path):
        fail("tool catalog missing after ensure", hint="run ensure")
    with open(path, encoding="utf-8") as fh:
        catalog = json.load(fh)
    entries = catalog.get("tools", []) if isinstance(catalog, dict) else catalog
    ranked = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        s = _score(terms, str(item.get("id", "")) + " " + str(item.get("title", "")),
                   str(item.get("description", "")) + " " + str(item.get("category", "")))
        if s > 0:
            ranked.append((s, item))
    ranked.sort(key=lambda pair: (-pair[0], str(pair[1].get("id", ""))))
    results = [{"id": i.get("id"), "title": i.get("title"), "category": i.get("category"),
                "description": str(i.get("description") or "")[:200],
                "install_commands": (i.get("install_commands") or [])[:3]}
               for _s, i in ranked[: max(1, min(args.limit, 12))]]
    out({"ok": True, "query": args.q, "matches": len(ranked), "results": results,
         "note": "catalog metadata only; nothing is installed or executed by this search"})


def _skill_dir(name: str) -> str:
    if not NAME_RE.match(name or ""):
        fail("invalid skill name")
    base = os.path.realpath(os.path.join(SKILLS_DIR, "skills"))
    path = os.path.realpath(os.path.join(base, name))
    if os.path.dirname(path) != base or not os.path.isdir(path):
        fail("skill not found", name=name, hint="use skill_search to find the exact name")
    return path


def _strip_frontmatter(text: str) -> tuple:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].lstrip("\n"), text[3:end]
    return text, ""


def cmd_load(args) -> None:
    path = _skill_dir(args.name)
    md = os.path.join(path, "SKILL.md")
    if not os.path.isfile(md):
        fail("SKILL.md missing", name=args.name)
    with open(md, encoding="utf-8", errors="replace") as fh:
        body, front = _strip_frontmatter(fh.read())
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        depth = dirpath[len(path):].count(os.sep)
        if depth >= MAX_REL_DEPTH:
            dirnames[:] = []
        for f in filenames:
            files.append(os.path.relpath(os.path.join(dirpath, f), path))
    truncated = len(body) > MAX_READ
    out({"ok": True, "name": args.name, "frontmatter": front.strip()[:1200],
         "body": body[:MAX_READ], "truncated": truncated, "files": sorted(files)[:200],
         "note": "instructions only; scripts inside the skill were NOT executed"})


def cmd_read(args) -> None:
    path = _skill_dir(args.name)
    rel = args.path or ""
    if not rel or os.path.isabs(rel) or ".." in rel.split("/"):
        fail("path must be relative to the skill folder")
    target = os.path.realpath(os.path.join(path, rel))
    if not target.startswith(path + os.sep) or not os.path.isfile(target):
        fail("file not found inside skill", path=rel)
    with open(target, encoding="utf-8", errors="replace") as fh:
        text = fh.read(MAX_READ + 1)
    out({"ok": True, "name": args.name, "path": rel, "content": text[:MAX_READ],
         "truncated": len(text) > MAX_READ})


def main() -> None:
    ap = argparse.ArgumentParser(description="Blackthorn on-demand skill catalog")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ensure")
    s = sub.add_parser("search")
    s.add_argument("--q", required=True)
    s.add_argument("--limit", type=int, default=8)
    t = sub.add_parser("tools")
    t.add_argument("--q", required=True)
    t.add_argument("--limit", type=int, default=6)
    ld = sub.add_parser("load")
    ld.add_argument("--name", required=True)
    rd = sub.add_parser("read")
    rd.add_argument("--name", required=True)
    rd.add_argument("--path", required=True)
    args = ap.parse_args()
    {"ensure": cmd_ensure, "search": cmd_search, "tools": cmd_tools,
     "load": cmd_load, "read": cmd_read}[args.cmd](args)


if __name__ == "__main__":
    main()
