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
    cmd = f"mkdir -p {SCREENSHOT_DIR} && timeout 320 python3 {REMOTE_BIN}/browse.py {payload}"
    data = await ctx.computer.call("/computer/exec", {"command": cmd, "timeout_seconds": 330, "cwd": "."},
                                   timeout=350.0, idempotent=False)
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


SAMPLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


async def sample_triage(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Inspect an uploaded sample without running it; no VM is claimed."""
    sample_id = str(args.get("sample_id") or "")
    if not SAMPLE_ID.fullmatch(sample_id) or sample_id in (".", ".."):
        return _fail("sample_id must be one file name in malware_samples/ (no path separators)")
    await _upload(ctx, "sample_triage.py")
    cmd = f"python3 {REMOTE_BIN}/sample_triage.py --analyze {shlex.quote(sample_id)}"
    data = await ctx.computer.call("/computer/exec", {"command": cmd, "timeout_seconds": 45, "cwd": "."},
                                   timeout=65.0, idempotent=True)
    try:
        result = json.loads(str(data.get("output") or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return _fail("static triage returned no result", {"kind": "triage"})
    if not result.get("ok"):
        return _fail(str(result.get("error") or "static triage failed"), {"kind": "triage"})
    content = (f"Read-only static sample metadata: {sample_id}\n"
               f"Format: {result['format']} ({result['architecture']})\n"
               f"Size: {result['bytes']} bytes; SHA-256: {result['sha256']}\n"
               f"First-64-KiB entropy: {result['entropy_first_64k']}; URL-like count: {result['url_like_count']}\n"
               "No sample was executed. Dynamic Linux/Windows malware analysis requires a separate "
               "verified VM isolation host; this production computer is not one.")
    return ToolResult(ok=True, content=content, summary=f"static triage of {sample_id}",
                      data={"sha256": result["sha256"], "format": result["format"], "dynamic_execution_allowed": False})


async def malware_analyzer_submit(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Explicit, PUBLIC remote submission; only the chosen quarantine basename is read."""
    from .. import malware_analyzer as remote
    sample_id = str(args.get("sample_id") or "")
    if not SAMPLE_ID.fullmatch(sample_id) or sample_id in (".", ".."):
        return _fail("sample_id must be a single file in malware_samples/")
    # Fail before requesting *any* sample data if the VPN has not been established.
    try:
        preflight = await remote.EGRESS.client()
        await preflight.aclose()
    except remote.EgressUnavailable as exc:
        return _fail(str(exc), {"kind": "vpn_unavailable"})
    try:
        data = await ctx.computer.call("/computer/read_sample", {"sample_id": sample_id}, timeout=120, idempotent=True)
        if not data.get("ok"):
            return _fail(str(data.get("error") or "quarantined sample not readable"))
        raw = base64.b64decode(data["fileBase64"], validate=True)
        import hashlib
        if len(raw) != data.get("bytes") or hashlib.sha256(raw).hexdigest() != data.get("sha256"):
            return _fail("sample changed during transfer; no remote upload was sent")
        result = await remote.submit_public(raw)
    except (remote.EgressUnavailable, ValueError, KeyError) as exc:
        return _fail(f"Remote public submission not confirmed ({type(exc).__name__})", {"kind": "remote_unavailable"})
    # The remote service, not this computer, decides whether to detonate or deduplicate.
    body, cut = clip(json.dumps(result, sort_keys=True, ensure_ascii=True), ctx.settings.tool_result_chars, "head")
    return ToolResult(ok=True, content=f"PUBLIC MalwareAnalyzer submission accepted; local sample was never executed.\n{body}",
                      summary=f"public remote submission {result['sha256'][:12]}",
                      data={"sha256": result["sha256"], "public": True}, truncated=cut)


async def malware_analyzer_report(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Public report lookup by SHA-256 via verified VPN; no local file access."""
    from .. import malware_analyzer as remote
    sha256 = str(args.get("sha256") or "")
    if not remote.HASH.fullmatch(sha256):
        return _fail("sha256 must be a 64-character hex digest")
    try:
        result = await remote.request_report(sha256)
    except remote.EgressUnavailable as exc:
        return _fail(str(exc), {"kind": "remote_unavailable"})
    attributes = result.get("data", {}).get("attributes", {}) if isinstance(result.get("data"), dict) else {}
    if not isinstance(attributes, dict):
        attributes = {}
    # Lead with the verdict and degraded status; a huge behavior/strings array must not
    # clip the only information needed to judge whether actual detonation happened.
    view = {key: attributes.get(key) for key in (
        "visibility", "analyzedAt", "verdict", "dynamicDegraded", "dynamicDegradedReason",
        "staticDegraded", "fileType", "sizeBytes", "engineStats", "attack") if key in attributes}
    for key in ("behavior", "iocs"):
        if key in attributes:
            view[key] = json.dumps(attributes[key], ensure_ascii=True)[:650]
    body, cut = clip(json.dumps(view or result, sort_keys=False, ensure_ascii=True), ctx.settings.tool_result_chars, "head")
    note = "Dynamic analysis was degraded; do not claim verified detonation. " if attributes.get("dynamicDegraded") else ""
    return ToolResult(ok=True, content=f"MalwareAnalyzer public report for {sha256}. {note}\n{body}",
                      summary=f"public report {sha256[:12]}", data={"sha256": sha256}, truncated=cut)


def specs() -> List[ToolSpec]:
    return [
        ToolSpec("malware_analyzer_submit",
                 "Submit one selected file in malware_samples/ to MalwareAnalyzer for PUBLIC remote analysis. "
                 "Only call for files authorised for public release. Never runs a sample locally. "
                 "Requires a verified Proton WireGuard exit; fails closed if unavailable. "
                 "A successful submission is not proof of dynamic detonation; fetch the report by hash.",
                 {"type": "object", "properties": {"sample_id": {"type": "string", "minLength": 1, "maxLength": 80}},
                  "required": ["sample_id"]}, malware_analyzer_submit, timeout=190.0, kind="net",
                 describe=lambda a: str(a.get("sample_id") or "")),
        ToolSpec("malware_analyzer_report",
                 "Fetch a PUBLIC MalwareAnalyzer report by SHA-256 over verified Proton VPN. "
                 "Report may still be pending or absent; never claim a dynamic verdict without evidence.",
                 {"type": "object", "properties": {"sha256": {"type": "string", "minLength": 64, "maxLength": 64}},
                  "required": ["sha256"]}, malware_analyzer_report, timeout=100.0, kind="net",
                 describe=lambda a: str(a.get("sha256") or "")[:12]),
        ToolSpec("sample_triage",
                 "Read-only static analysis of an uploaded PE, ELF or script in malware_samples/ on the computer. "
                 "Returns bounded metadata and SHA-256. Never executes samples; there is no VM on the production computer.",
                 {"type": "object", "properties": {
                     "sample_id": {"type": "string", "minLength": 1, "maxLength": 80}},
                  "required": ["sample_id"]},
                 sample_triage, timeout=75.0, describe=lambda a: str(a.get("sample_id") or "")),
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
                 browse, timeout=360.0, kind="net", describe=lambda a: str(a.get("url") or "")),
    ]
