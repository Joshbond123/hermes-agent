"""On-demand skills, security tool catalog and Camoufox browsing. All execution happens on the computer.

Design rules:
- Nothing from the skill catalog is injected into model context up front. The model searches,
  then loads one skill or one reference file when it needs it.
- Model text never reaches the shell unquoted: arguments are validated, then passed as base64 JSON
  or as shlex-quoted values to fixed command templates.
- Skill scripts are never executed. Only SKILL.md text and reference files are returned.
"""

from __future__ import annotations

import base64
import json
import re
import shlex
import time
from pathlib import Path
from typing import Any, Dict, List

from .base import ToolContext, ToolResult, ToolSpec, clip

ASSET_DIR = Path(__file__).resolve().parent.parent / "computer_assets"
REMOTE_BIN = "/tmp/blackthorn_skills/bin"
SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{2,120}$")
QUERY = re.compile(r"^[A-Za-z0-9 ._+/-]{2,120}$")
REL_PATH = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")
SCREENSHOT_DIR = "browser_screenshots"


def _asset(name: str) -> str:
    return (ASSET_DIR / name).read_text(encoding="utf-8")


async def _upload(ctx: ToolContext, name: str) -> None:
    """Write a helper script to the computer. Small and idempotent, so it runs before every call."""
    await ctx.computer.call("/computer/write_file",
                            {"path": f"{REMOTE_BIN}/{name}", "content": _asset(name)}, idempotent=True)


async def _run_skillctl(ctx: ToolContext, subcommand: str, flags: str, timeout: int = 300) -> Dict[str, Any]:
    await _upload(ctx, "skillctl.py")
    cmd = f"python3 {REMOTE_BIN}/skillctl.py {subcommand} {flags}"
    data = await ctx.computer.call("/computer/exec", {"command": cmd, "timeout_seconds": timeout, "cwd": "."},
                                   timeout=float(timeout + 30), idempotent=True)
    raw = str(data.get("output") or "").strip().splitlines()
    for line in reversed(raw):
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict) and "ok" in parsed:
            return parsed
    return {"ok": False, "error": "skill catalog returned no JSON", "output": "\n".join(raw)[-400:]}


def _fail(message: str, data: Dict[str, Any] | None = None) -> ToolResult:
    return ToolResult.failure(message, data=data or {"kind": "tool_error"})


def _valid_skill(args: Dict[str, Any]) -> str | None:
    name = str(args.get("name") or "").strip()
    return name if SKILL_NAME.match(name) else None


async def skill_search(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = str(args.get("query") or "").strip()
    if not QUERY.match(query):
        return _fail("query must be 2-120 plain characters (letters, digits, spaces, . _ + / -)")
    limit = max(1, min(15, int(args.get("limit") or 8)))
    data = await _run_skillctl(ctx, "search", f"--q {shlex.quote(query)} --limit {limit}")
    if not data.get("ok"):
        return _fail(str(data.get("error") or "search failed"), {"kind": "catalog"})
    lines = [f"{r['name']}: {r['description']}" for r in data.get("results", [])]
    body = "\n".join(lines) or "no matching skills"
    return ToolResult(ok=True, content=f"{data.get('matches', 0)} matches for '{query}'\n{body}",
                      summary=f"{data.get('matches', 0)} skill matches", data={"matches": data.get("matches", 0)})


async def skill_load(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    name = _valid_skill(args)
    if name is None:
        return _fail("name must be a skill name returned by skill_search")
    data = await _run_skillctl(ctx, "load", f"--name {shlex.quote(name)}")
    if not data.get("ok"):
        return _fail(str(data.get("error") or "load failed"), {"kind": "catalog", "name": name})
    body, cut = clip(str(data.get("body") or ""), ctx.settings.tool_result_chars, "ends")
    files = data.get("files") or []
    listing = ", ".join(files[:40]) + (" ..." if len(files) > 40 else "")
    content = (f"SKILL {name} (instructions only; its scripts were not run)\n"
               f"Other files (load with skill_read): {listing}\n\n{body}")
    return ToolResult(ok=True, content=content, summary=f"loaded {name}", truncated=cut or bool(data.get("truncated")),
                      data={"name": name, "files": len(files)})


async def skill_read(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    name = _valid_skill(args)
    rel = str(args.get("path") or "").strip()
    if name is None or not REL_PATH.match(rel) or rel.startswith("/") or ".." in rel.split("/"):
        return _fail("name and a relative path inside the skill folder are required")
    data = await _run_skillctl(ctx, "read", f"--name {shlex.quote(name)} --path {shlex.quote(rel)}")
    if not data.get("ok"):
        return _fail(str(data.get("error") or "read failed"), {"kind": "catalog", "name": name})
    body, cut = clip(str(data.get("content") or ""), ctx.settings.tool_result_chars, "ends")
    return ToolResult(ok=True, content=f"{name}/{rel}\n{body}", summary=f"read {name}/{rel}", truncated=cut)


async def tool_search(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = str(args.get("query") or "").strip()
    if not QUERY.match(query):
        return _fail("query must be 2-120 plain characters")
    limit = max(1, min(12, int(args.get("limit") or 6)))
    data = await _run_skillctl(ctx, "tools", f"--q {shlex.quote(query)} --limit {limit}")
    if not data.get("ok"):
        return _fail(str(data.get("error") or "tool search failed"), {"kind": "catalog"})
    lines = []
    for r in data.get("results", []):
        lines.append(f"{r['id']} [{r.get('category')}]: {r.get('title')} - {r.get('description', '')[:160]}")
    body = "\n".join(lines) or "no matching tools in the catalog"
    return ToolResult(ok=True, content=f"{data.get('matches', 0)} catalog matches (metadata only; nothing installed)\n{body}",
                      summary=f"{data.get('matches', 0)} tool matches", data={"matches": data.get("matches", 0)})


async def browse(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    url = str(args.get("url") or "").strip()
    if not url.startswith(("http://", "https://")) or len(url) > 2000:
        return _fail("url must start with http:// or https://")
    timeout_ms = max(5000, min(60000, int(args.get("timeout_ms") or 25000)))
    request: Dict[str, Any] = {"url": url, "timeout_ms": timeout_ms, "max_chars": 6000,
                               "storage": "/tmp/blackthorn_browser/storage.json"}
    if args.get("screenshot"):
        request["screenshot"] = f"{SCREENSHOT_DIR}/shot-{time.time_ns()}.png"
    await _upload(ctx, "browse.py")
    payload = base64.b64encode(json.dumps(request).encode()).decode()
    cmd = f"mkdir -p {SCREENSHOT_DIR} && timeout 150 python3 {REMOTE_BIN}/browse.py {payload}"
    data = await ctx.computer.call("/computer/exec", {"command": cmd, "timeout_seconds": 160, "cwd": "."},
                                   timeout=190.0, idempotent=False)
    raw = str(data.get("output") or "").strip().splitlines()
    result = None
    for line in reversed(raw):
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict) and "ok" in parsed:
            result = parsed
            break
    if result is None:
        return _fail("browser produced no result", {"kind": "browser", "output": "\n".join(raw)[-300:]})
    if not result.get("ok"):
        return _fail(f"browse {result.get('error_kind', 'error')}: {result.get('error', '')[:300]}",
                     {"kind": "browser_" + str(result.get("error_kind", "error"))})
    text, cut = clip(str(result.get("text") or ""), ctx.settings.tool_result_chars, "ends")
    links = "\n".join(f"- {l.get('text') or '(no text)'} {l.get('href')}" for l in (result.get("links") or [])[:15])
    notes = "; ".join(result.get("notes") or [])
    shot = f"\nScreenshot saved: {result['screenshot']}" if result.get("screenshot") else ""
    content = (f"{result.get('title')}\nURL: {result.get('final_url')} (HTTP {result.get('status')})"
               f"{' | notes: ' + notes if notes else ''}{shot}\n\n{text}\n\nLinks:\n{links}")
    return ToolResult(ok=True, content=content, summary=f"browsed {result.get('final_url', url)}",
                      truncated=cut or bool(result.get("text_truncated")),
                      data={"final_url": result.get("final_url"), "status": result.get("status"),
                            "screenshot": result.get("screenshot"), "elapsed_s": result.get("elapsed_s")})


def specs() -> List[ToolSpec]:
    return [
        ToolSpec("skill_search",
                 "Find cybersecurity skills (step-by-step playbooks) by keyword. Returns names and one-line descriptions only. "
                 "Use it before a security task, then skill_load the best match.",
                 {"type": "object", "properties": {
                     "query": {"type": "string", "minLength": 2, "maxLength": 120},
                     "limit": {"type": "integer", "minimum": 1, "maximum": 15, "default": 8}},
                  "required": ["query"]},
                 skill_search, timeout=330.0, describe=lambda a: str(a.get("query") or "")),
        ToolSpec("skill_load",
                 "Load one skill's instructions by exact name from skill_search. Scripts in the skill are NOT executed. "
                 "Use only for authorised, defensive or lab work.",
                 {"type": "object", "properties": {"name": {"type": "string", "minLength": 3, "maxLength": 121}},
                  "required": ["name"]},
                 skill_load, timeout=330.0, describe=lambda a: str(a.get("name") or "")),
        ToolSpec("skill_read",
                 "Read one reference file inside a loaded skill folder (for example references/api-reference.md).",
                 {"type": "object", "properties": {
                     "name": {"type": "string", "minLength": 3, "maxLength": 121},
                     "path": {"type": "string", "minLength": 1, "maxLength": 200}},
                  "required": ["name", "path"]},
                 skill_read, timeout=330.0, describe=lambda a: f"{a.get('name', '')}/{a.get('path', '')}"),
        ToolSpec("tool_search",
                 "Search the security tool catalog (183 tools, metadata only: category, description, install commands). "
                 "Nothing is installed or run by this search.",
                 {"type": "object", "properties": {
                     "query": {"type": "string", "minLength": 2, "maxLength": 120},
                     "limit": {"type": "integer", "minimum": 1, "maximum": 12, "default": 6}},
                  "required": ["query"]},
                 tool_search, timeout=330.0, describe=lambda a: str(a.get("query") or "")),
        ToolSpec("browse",
                 "Open a web page in a stealth browser (Camoufox) on the computer. Returns title, final URL, status, "
                 "visible text and links. Use for pages that need JavaScript; use web_search for plain research.",
                 {"type": "object", "properties": {
                     "url": {"type": "string", "minLength": 8, "maxLength": 2000},
                     "timeout_ms": {"type": "integer", "minimum": 5000, "maximum": 60000, "default": 25000},
                     "screenshot": {"type": "boolean", "default": False}},
                  "required": ["url"]},
                 browse, timeout=200.0, kind="net", describe=lambda a: str(a.get("url") or "")),
    ]
