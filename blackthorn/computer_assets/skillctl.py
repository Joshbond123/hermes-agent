#!/usr/bin/env python3
"""Unified cybersecurity capability registry for the Blackthorn computer.

Runs ON THE COMPUTER, never in model context. One registry covers all three
upstream repositories, so the agent discovers and executes capabilities without
a human choosing a skill:

  anthropic   mukul975/Anthropic-Cybersecurity-Skills   818 defensive+offensive SKILL.md playbooks
  claude-red  SnailSploit/Claude-Red                    79 offensive SKILL.md playbooks by category
  hackingtool AKCodez/hackingtool-plugin                183 installable/runnable tools + wrapper scripts

Design rules
------------
* Discovery is metadata-only. ``search`` returns ids and one-line descriptions;
  the model then loads exactly one SKILL.md or runs exactly one tool.
* ``run`` is the only command that installs or executes anything. Install is
  attempted at most once per tool per workspace (cached in installed.json),
  every command has a timeout, and the outcome is reported honestly:
  ``status`` is one of ran | installed_ran | installed_probe_failed |
  install_failed | blocked_interactive | blocked_remote_script | missing.
* Model-supplied text never reaches the shell unquoted; arguments are passed
  through ``shlex.quote`` and validated first.
* Skill scripts are never executed automatically.

Every command prints exactly one JSON object on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.environ.get("BLACKTHORN_SKILL_ROOT",
                      "/workspace/security_catalog" if os.path.isdir("/workspace")
                      else "/kaggle/working/blackthorn_workspace/security_catalog")
SOURCES = {
    "anthropic": {"repo": "https://github.com/mukul975/Anthropic-Cybersecurity-Skills",
                  "dir": "anthropic-cybersecurity-skills", "license": "Apache-2.0"},
    "claude-red": {"repo": "https://github.com/SnailSploit/Claude-Red",
                   "dir": "claude-red", "license": "MIT"},
    "hackingtool": {"repo": "https://github.com/AKCodez/hackingtool-plugin",
                    "dir": "hackingtool-plugin", "license": "see upstream README"},
}
REGISTRY = os.path.join(ROOT, "registry.json")
INSTALLED = os.path.join(ROOT, "installed.json")
CLONE_TIMEOUT = 300
MAX_READ = 20000
MAX_OUTPUT = 20000
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:/+-]{1,160}$")
REL_PATH = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")
# Commands we refuse to pipe into a shell or answer interactively.
BLOCK_INTERACTIVE = re.compile(r"\bsudo\s+su\b|\brustup\b|\bpasswd\b|--interactive|read\s+-p")
BLOCK_REMOTE_SCRIPT = re.compile(r"(curl|wget)[^|;]*\|\s*(sudo\s+)?(sh|bash)\b")


# ---------------------------------------------------------------- plumbing
def out(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def fail(message: str, **extra) -> None:
    out({"ok": False, "error": message, **extra})
    sys.exit(0)


def path_of(source: str, *parts: str) -> str:
    return os.path.join(ROOT, SOURCES[source]["dir"], *parts)


def _run(cmd: str, timeout: int, cwd: str | None = None) -> dict:
    started = time.time()
    try:
        proc = subprocess.run(["/bin/bash", "-c", cmd], capture_output=True, text=True,
                              timeout=timeout, cwd=cwd or ROOT, stdin=subprocess.DEVNULL,
                              env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"})
        code, output = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        code = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        output = (stdout + stderr) + "\n[timeout]"
    except FileNotFoundError as exc:
        return {"cmd": cmd[:400], "exit_code": 127, "seconds": 0.0, "output": str(exc)[:300]}
    return {"cmd": cmd[:400], "exit_code": code, "seconds": round(time.time() - started, 1),
            "output": output[-MAX_OUTPUT:]}


def _load(path: str, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _save(path: str, value) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------- sources
def clone_sources(force: bool = False) -> dict:
    os.makedirs(ROOT, exist_ok=True)
    results = {}
    for name, meta in SOURCES.items():
        target = path_of(name)
        if os.path.isdir(os.path.join(target, ".git")):
            if force:
                step = _run(f"git -C {json.dumps(target)} fetch --depth 1 -q origin && "
                            f"git -C {json.dumps(target)} reset --hard -q FETCH_HEAD", 300)
                results[name] = {"state": "refreshed" if step["exit_code"] == 0 else "refresh_failed"}
            else:
                results[name] = {"state": "present"}
            continue
        if os.path.exists(target):
            results[name] = {"state": "invalid", "error": "path exists without .git"}
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        step = _run(f"git clone --depth 1 --quiet {meta['repo']} {json.dumps(target)}", CLONE_TIMEOUT)
        results[name] = {"state": "cloned" if step["exit_code"] == 0 else "failed",
                         "detail": None if step["exit_code"] == 0 else step["output"][-300:]}
    return results


# ---------------------------------------------------------------- registry build
def _frontmatter(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    meta = {}
    for line in text[3:end].splitlines():
        if ":" not in line or line.startswith(" "):
            continue
        key, _, value = line.partition(":")
        meta[key.strip().lower()] = value.strip().strip("'\"")
    return meta


def _skill_entry(source: str, skill_id: str, name: str, category: str, folder: str) -> dict:
    description = ""
    md = os.path.join(folder, "SKILL.md")
    try:
        with open(md, encoding="utf-8", errors="replace") as fh:
            head = fh.read(4000)
        description = _frontmatter(head).get("description", "")
    except OSError:
        pass
    return {"kind": "skill", "id": skill_id, "name": name, "source": source, "category": category,
            "description": description[:400], "path": os.path.relpath(folder, ROOT)}


def build_registry() -> dict:
    skills: list[dict] = []
    index_path = path_of("anthropic", "index.json")
    index = _load(index_path, {"skills": []})
    for item in index.get("skills", []):
        name = str(item.get("name") or "")
        if not name:
            continue
        skills.append({"kind": "skill", "id": "anthropic:" + name, "name": name, "source": "anthropic",
                       "category": str(item.get("subdomain") or item.get("domain") or "cybersecurity"),
                       "description": str(item.get("description") or "")[:400],
                       # Registry paths are always ROOT-relative, so a skill resolves the same
                       # way whichever repository it came from.
                       "path": os.path.join(SOURCES["anthropic"]["dir"], "skills", name)})
    red_root = path_of("claude-red", "Skills")
    if os.path.isdir(red_root):
        for category in sorted(os.listdir(red_root)):
            cat_dir = os.path.join(red_root, category)
            if not os.path.isdir(cat_dir):
                continue
            for name in sorted(os.listdir(cat_dir)):
                folder = os.path.join(cat_dir, name)
                if os.path.isfile(os.path.join(folder, "SKILL.md")):
                    skills.append(_skill_entry("claude-red", f"claude-red:{name}", name, category, folder))
    tools: list[dict] = []
    catalog_path = path_of("hackingtool", "plugins", "hackingtool", "data", "tools.json")
    catalog = _load(catalog_path, {"tools": []})
    entries = catalog.get("tools", []) if isinstance(catalog, dict) else catalog
    for item in entries:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        tools.append({"kind": "tool", "id": "tool:" + str(item["id"]), "name": str(item["id"]),
                      "title": str(item.get("title") or item["id"]), "source": "hackingtool",
                      "category": str(item.get("category") or "other"),
                      "description": str(item.get("description") or "")[:400],
                      "install_commands": list(item.get("install_commands") or []),
                      "run_commands": list(item.get("run_commands") or []),
                      "project_url": str(item.get("project_url") or ""),
                      "capabilities": item.get("capabilities") or {},
                      "archived": bool(item.get("archived"))})
    registry = {"built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "root": ROOT,
                "sources": {name: {"repo": meta["repo"], "license": meta["license"],
                                   "present": os.path.isdir(path_of(name))}
                            for name, meta in SOURCES.items()},
                "counts": {"skills": len(skills), "tools": len(tools),
                           "by_source": {s: sum(1 for x in skills if x["source"] == s)
                                         for s in ("anthropic", "claude-red")}},
                "skills": skills, "tools": tools}
    _save(REGISTRY, registry)
    return registry


def registry(refresh: bool = False) -> dict:
    if refresh or not os.path.exists(REGISTRY):
        return build_registry()
    data = _load(REGISTRY, None)
    return data if isinstance(data, dict) and "skills" in data else build_registry()


# ---------------------------------------------------------------- search
def _terms(text: str) -> list:
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(t) > 2]


def _score(terms: list, *fields: str) -> int:
    name_terms = set(_terms(fields[0] if fields else ""))
    rest = set()
    for field in fields[1:]:
        rest |= set(_terms(field))
    score = 0
    for term in terms:
        if term in name_terms:
            score += 5
        elif any(term in candidate for candidate in name_terms):
            score += 2
        if term in rest:
            score += 2
    return score


def search(query: str, limit: int, kind: str = "all") -> dict:
    terms = _terms(query)
    if not terms:
        fail("query needs at least one word of 3+ letters")
    data = registry()
    results = []
    if kind in ("all", "skill"):
        for item in data["skills"]:
            score = _score(terms, item["name"], item["description"], item.get("category", ""))
            if score > 0:
                results.append((score, {"kind": "skill", "id": item["id"], "name": item["name"],
                                        "source": item["source"], "category": item.get("category", ""),
                                        "description": item["description"][:220]}))
    if kind in ("all", "tool"):
        for item in data["tools"]:
            score = _score(terms, item["name"] + " " + item["title"], item["description"],
                           item.get("category", ""))
            if score > 0:
                results.append((score, {"kind": "tool", "id": item["id"], "name": item["title"],
                                        "source": "hackingtool", "category": item.get("category", ""),
                                        "description": item["description"][:220],
                                        "runnable": bool(item.get("run_commands"))}))
    results.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    top = [entry for _score_value, entry in results[:max(1, min(limit, 25))]]
    return {"ok": True, "query": query, "kind": kind, "matches": len(results), "results": top,
            "counts": data["counts"],
            "note": "metadata only; nothing was installed or executed by this search"}


# ---------------------------------------------------------------- skill resolution
def resolve_skill(name_or_id: str) -> dict | None:
    data = registry()
    wanted = name_or_id.strip()
    for item in data["skills"]:
        if item["id"] == wanted:
            return item
    matches = [item for item in data["skills"] if item["name"] == wanted]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer the defensive Anthropic catalogue, then claude-red, deterministically.
        matches.sort(key=lambda item: (item["source"] != "anthropic", item["id"]))
        return matches[0]
    return None


def skill_folder(item: dict) -> str:
    return os.path.join(ROOT, item["path"])


def load_skill(name_or_id: str) -> dict:
    item = resolve_skill(name_or_id)
    if item is None:
        fail("skill not found", hint="use search to find an exact id")
    folder = skill_folder(item)
    md = os.path.join(folder, "SKILL.md")
    if not os.path.isfile(md):
        fail("SKILL.md missing", id=item["id"])
    with open(md, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    meta = _frontmatter(text)
    body = text
    if meta:
        end = text.find("\n---", 3)
        body = text[end + 4:].lstrip("\n")
    files = []
    for dirpath, dirnames, filenames in os.walk(folder):
        if dirpath[len(folder):].count(os.sep) >= 4:
            dirnames[:] = []
        for name in filenames:
            files.append(os.path.relpath(os.path.join(dirpath, name), folder))
    return {"ok": True, "id": item["id"], "name": item["name"], "source": item["source"],
            "category": item.get("category"), "frontmatter": meta,
            "body": body[:MAX_READ], "truncated": len(body) > MAX_READ,
            "files": sorted(files)[:200],
            "note": "instructions only; scripts inside the skill were NOT executed"}


def read_skill(name_or_id: str, rel: str) -> dict:
    if not rel or os.path.isabs(rel) or ".." in rel.split("/") or not REL_PATH.match(rel):
        fail("path must be relative to the skill folder")
    item = resolve_skill(name_or_id)
    if item is None:
        fail("skill not found")
    folder = skill_folder(item)
    target = os.path.realpath(os.path.join(folder, rel))
    if not target.startswith(os.path.realpath(folder) + os.sep) or not os.path.isfile(target):
        fail("file not found inside skill", path=rel)
    with open(target, encoding="utf-8", errors="replace") as fh:
        text = fh.read(MAX_READ + 1)
    return {"ok": True, "id": item["id"], "path": rel, "content": text[:MAX_READ],
            "truncated": len(text) > MAX_READ}


# ---------------------------------------------------------------- tool execution
def _tool(name_or_id: str) -> dict:
    data = registry()
    wanted = name_or_id.strip()
    for item in data["tools"]:
        if item["id"] == wanted:
            return item
    matches = [item for item in data["tools"] if item["name"] == wanted or item["title"] == wanted]
    if len(matches) == 1:
        return matches[0]
    fail("tool not found", id=wanted, hint="use search to find an exact tool id")


def _probe_ok(result: dict) -> bool:
    text = (result.get("output") or "").lower()
    return result.get("exit_code") in (0, 1, 2) and "not found" not in text and "no such file" not in text


def preflight() -> dict:
    data = registry()
    binaries = {}
    for item in data["tools"]:
        if not item.get("run_commands"):
            continue
        first = item["run_commands"][0].split()
        if not first:
            continue
        binary = os.path.basename(first[0])
        binaries[binary] = bool(shutil.which(binary))
    present = sum(1 for value in binaries.values() if value)
    return {"ok": True, "root": ROOT, "sources": data["sources"], "counts": data["counts"],
            "binaries_present": present, "binaries_checked": len(binaries),
            "missing": sorted(name for name, ok in binaries.items() if not ok)[:80],
            "installed_registry": len(_load(INSTALLED, {})),
            "python": sys.version.split()[0]}


def run_tool(name_or_id: str, args: str = "", timeout: int = 300, install: bool = True) -> dict:
    item = _tool(name_or_id)
    commands = [str(c) for c in (item.get("run_commands") or [])]
    if not commands:
        return {"ok": False, "id": item["id"], "status": "no_run_command",
                "error": "This catalogue entry has no runnable command; use its skill instead."}
    command = commands[0]
    if args:
        command = command + " " + args
    state = _load(INSTALLED, {})
    record = state.get(item["id"]) or {}
    # Availability is decided by looking for the binary, never by executing a
    # prospective command: a catalogue "run" command can be a real action.
    binary = os.path.basename(command.split()[0]) if command.split() else ""
    available = bool(binary and (shutil.which(binary) or os.path.exists(binary)))
    probe = {"output": "" if available else f"{binary}: not found", "exit_code": 0 if available else 127}
    if not available and install:
        installs = [str(c) for c in (item.get("install_commands") or [])]
        if not installs:
            return {"ok": False, "id": item["id"], "status": "missing",
                    "error": "command is not installed and the catalogue has no install recipe",
                    "probe": probe["output"][-600:]}
        for cmd in installs:
            if BLOCK_INTERACTIVE.search(cmd):
                return {"ok": False, "id": item["id"], "status": "blocked_interactive",
                        "error": "install step needs an interactive shell and was not run", "cmd": cmd[:200]}
            if BLOCK_REMOTE_SCRIPT.search(cmd):
                return {"ok": False, "id": item["id"], "status": "blocked_remote_script",
                        "error": "refusing to pipe a downloaded script into a shell", "cmd": cmd[:200]}
            clean = re.sub(r"\bsudo\s+", "", cmd)               # the guest runs as root
            step = _run(clean, min(timeout, 1500), cwd=ROOT)
            state[item["id"]] = {"status": "install_attempted", "at": time.time(),
                                 "cmd": clean[:200], "exit_code": step["exit_code"]}
            _save(INSTALLED, state)
            if step["exit_code"] != 0:
                record = state[item["id"]]
                record["status"] = "install_failed"
                _save(INSTALLED, state)
                return {"ok": False, "id": item["id"], "status": "install_failed",
                        "error": "install command failed", "install": step}
    result = _run(command, timeout, cwd=ROOT)
    ok = result["exit_code"] == 0
    state[item["id"]] = {"status": "ran" if ok else "ran_nonzero", "at": time.time(),
                         "exit_code": result["exit_code"], "command": command[:200]}
    _save(INSTALLED, state)
    return {"ok": ok, "id": item["id"], "status": "ran" if ok else "ran_nonzero",
            "title": item["title"], "command": command[:400],
            "exit_code": result["exit_code"], "seconds": result["seconds"],
            "output": result["output"]}


# ---------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Blackthorn unified cybersecurity registry")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ensure")
    sub.add_parser("registry")
    sub.add_parser("preflight")
    sub.add_parser("rebuild")
    p_search = sub.add_parser("search")
    p_search.add_argument("--q", required=True)
    p_search.add_argument("--limit", type=int, default=8)
    p_search.add_argument("--kind", default="all", choices=["all", "skill", "tool"])
    p_load = sub.add_parser("load")
    p_load.add_argument("--name", default="", help="skill id, or a unique skill name")
    p_load.add_argument("--id", default="")
    p_read = sub.add_parser("read")
    p_read.add_argument("--name", default="")
    p_read.add_argument("--id", default="")
    p_read.add_argument("--path", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--id", default="")
    p_run.add_argument("--name", default="")
    p_run.add_argument("--args", default="")
    p_run.add_argument("--timeout", type=int, default=300)
    p_run.add_argument("--no-install", action="store_true")
    args = ap.parse_args()

    if args.cmd == "ensure":
        results = clone_sources()
        out({"ok": all(r["state"] in ("present", "cloned", "refreshed") for r in results.values()),
             "sources": results, "root": ROOT})
    elif args.cmd in ("registry", "rebuild"):
        data = build_registry()
        out({"ok": True, "built": data["built"], "counts": data["counts"], "registry": REGISTRY})
    elif args.cmd == "preflight":
        out(preflight())
    elif args.cmd == "search":
        out(search(args.q, args.limit, args.kind))
    elif args.cmd == "load":
        out(load_skill(args.id or args.name))
    elif args.cmd == "read":
        out(read_skill(args.id or args.name, args.path))
    elif args.cmd == "run":
        out(run_tool(args.id or args.name, args.args, args.timeout, install=not args.no_install))


if __name__ == "__main__":
    main()
