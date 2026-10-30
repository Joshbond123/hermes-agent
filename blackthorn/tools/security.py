"""On-demand skills, the unified security tool registry and Camoufox browsing.

Everything here executes on the computer, never in the web process.

Design rules
- Nothing from the registry is injected into model context up front. The model searches,
  then loads one skill, reads one reference file, or runs one catalogued tool.
- The registry spans three upstream repositories: Anthropic-Cybersecurity-Skills (818
  playbooks), Claude-Red (79 offensive playbooks) and hackingtool-plugin (183 executable
  tools). ``skillctl.py`` builds one index over all three, so discovery never requires a
  human to pick a repository or a skill.
- Model text never reaches the shell unquoted: arguments are validated, then passed as
  base64 JSON or as shlex-quoted values to fixed command templates.
- Skill *documents* are read, never executed. Executable catalog tools run only through
  ``cyber_run``, which installs a missing binary at most once and reports the real exit code.
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
REMOTE_BIN = "/workspace/.blackthorn/bin"
SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{2,120}$")
SKILL_ID = re.compile(r"^(?:anthropic|claude-red):[A-Za-z0-9][A-Za-z0-9._-]{1,150}$")
TOOL_ID = re.compile(r"^tool:[A-Za-z0-9][A-Za-z0-9._-]{1,150}$")
TOOL_ARGS = re.compile(r"^[A-Za-z0-9 ./_:=@+,%^*()\[\]{}\"'$-]{0,500}$")
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
    """Accept a plain skill name or a namespaced id (``anthropic:…`` / ``claude-red:…``)."""
    name = str(args.get("id") or args.get("name") or "").strip()
    if SKILL_ID.match(name):
        return name
    return name if SKILL_NAME.match(name) else None


async def skill_search(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = str(args.get("query") or "").strip()
    if not QUERY.match(query):
        return _fail("query must be 2-120 plain characters (letters, digits, spaces, . _ + / -)")
    limit = max(1, min(15, int(args.get("limit") or 8)))
    kind = str(args.get("kind") or "all").lower()
    if kind not in ("all", "skill", "tool"):
        return _fail("kind must be all, skill or tool")
    data = await _run_skillctl(ctx, "search", f"--q {shlex.quote(query)} --limit {limit} --kind {kind}")
    if not data.get("ok"):
        return _fail(str(data.get("error") or "search failed"), {"kind": "catalog"})
    lines = []
    for r in data.get("results", []):
        tag = r.get("source") or ""
        kind = r.get("kind") or "skill"
        ident = r.get("id") or r.get("name")
        lines.append(f"[{kind} {tag}] {ident}: {r.get('description', '')}")
    body = "\n".join(lines) or "no matching capabilities"
    return ToolResult(ok=True, content=f"{data.get('matches', 0)} matches for '{query}' "
                                       f"(skills {data.get('counts', {}).get('skills', '?')}, tools "
                                       f"{data.get('counts', {}).get('tools', '?')})\n{body}",
                      summary=f"{data.get('matches', 0)} capability matches",
                      data={"matches": data.get("matches", 0),
                            "results": [r.get("id") for r in data.get("results", [])][:10],
                            "runnable": [r.get("id") for r in data.get("results", []) if r.get("runnable")][:10]})


async def skill_load(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    name = _valid_skill(args)
    if name is None:
        return _fail("name must be a skill name returned by skill_search")
    data = await _run_skillctl(ctx, "load", f"--id {shlex.quote(name)}")
    if not data.get("ok"):
        return _fail(str(data.get("error") or "load failed"), {"kind": "catalog", "name": name})
    body, cut = clip(str(data.get("body") or ""), ctx.settings.tool_result_chars, "ends")
    files = data.get("files") or []
    listing = ", ".join(files[:40]) + (" ..." if len(files) > 40 else "")
    content = (f"SKILL {data.get('id') or name} [source: {data.get('source', '?')}, category: {data.get('category', '?')}]"
               f" (instructions only; its scripts were not run)\n"
               f"Other files (load with skill_read): {listing}\n\n{body}")
    return ToolResult(ok=True, content=content, summary=f"loaded {name}", truncated=cut or bool(data.get("truncated")),
                      data={"name": name, "files": len(files)})


async def skill_read(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    name = _valid_skill(args)
    rel = str(args.get("path") or "").strip()
    if name is None or not REL_PATH.match(rel) or rel.startswith("/") or ".." in rel.split("/"):
        return _fail("name and a relative path inside the skill folder are required")
    data = await _run_skillctl(ctx, "read", f"--id {shlex.quote(name)} --path {shlex.quote(rel)}")
    if not data.get("ok"):
        return _fail(str(data.get("error") or "read failed"), {"kind": "catalog", "name": name})
    body, cut = clip(str(data.get("content") or ""), ctx.settings.tool_result_chars, "ends")
    return ToolResult(ok=True, content=f"{name}/{rel}\n{body}", summary=f"read {name}/{rel}", truncated=cut)


async def tool_search(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = str(args.get("query") or "").strip()
    if not QUERY.match(query):
        return _fail("query must be 2-120 plain characters")
    limit = max(1, min(12, int(args.get("limit") or 6)))
    data = await _run_skillctl(ctx, "search", f"--q {shlex.quote(query)} --limit {limit} --kind tool")
    if not data.get("ok"):
        return _fail(str(data.get("error") or "tool search failed"), {"kind": "catalog"})
    lines = []
    for r in data.get("results", []):
        lines.append(f"{r['id']} [{r.get('category')}]: {r.get('title')} - {r.get('description', '')[:160]}")
    body = "\n".join(lines) or "no matching tools in the catalog"
    return ToolResult(ok=True, content=f"{data.get('matches', 0)} catalog matches (metadata only; nothing installed)\n{body}",
                      summary=f"{data.get('matches', 0)} tool matches",
                      data={"matches": data.get("matches", 0),
                            "results": [r.get("id") for r in data.get("results", [])][:10],
                            "runnable": [r.get("id") for r in data.get("results", []) if r.get("runnable")][:10]})


async def browse(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Drive Camoufox on the computer: navigate, click, type, scroll, screenshot, tabs, download."""
    action = str(args.get("action") or "open").lower()
    if action not in ("open", "click", "type", "scroll", "screenshot", "accessibility", "tabs", "download", "close"):
        return _fail("action must be one of open, click, type, scroll, screenshot, accessibility, tabs, download, close")
    url = str(args.get("url") or "").strip()
    if action in ("open", "download"):
        if not url.startswith(("http://", "https://")) or len(url) > 2000:
            return _fail("url must start with http:// or https://")
    elif url and not url.startswith(("http://", "https://")):
        return _fail("url must start with http:// or https://")
    if action in ("click", "type") and not str(args.get("selector") or "").strip():
        return _fail(f"{action} requires a CSS selector")
    timeout_ms = max(3000, min(90000, int(args.get("timeout_ms") or 25000)))
    request: Dict[str, Any] = {
        "action": action, "timeout_ms": timeout_ms, "max_chars": 6000,
        "storage": "/tmp/blackthorn_browser/storage.json",
    }
    for key in ("url", "selector", "text", "scroll_px", "wait_ms", "full_page", "download_path", "close_tab"):
        if args.get(key) not in (None, ""):
            request[key] = args[key]
    if args.get("submit"):
        request["submit"] = True
    if args.get("screenshot"):
        request["screenshot"] = f"{SCREENSHOT_DIR}/shot-{time.time_ns()}.png"
    await _upload(ctx, "browse.py")
    payload = base64.b64encode(json.dumps(request).encode()).decode()
    cmd = f"mkdir -p {SCREENSHOT_DIR} && timeout 330 python3 {REMOTE_BIN}/browse.py {payload}"
    data = await ctx.computer.call("/computer/exec", {"command": cmd, "timeout_seconds": 340, "cwd": "."},
                                   timeout=360.0, idempotent=False)
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
        kind = str(result.get("error_kind", "error"))
        detail = str(result.get("error", ""))[:300]
        notes = "; ".join(result.get("notes") or [])
        if kind in ("dependency", "error") and ("launch" in detail.lower() or "browser process" in detail.lower()):
            # The browser repairs itself (temp roots, system libraries, bundle refetch) and retries
            # before this point, so a launch error here means the repair did not hold. Say what is
            # wrong in terms the next step can act on, instead of pasting a driver traceback.
            kind = "dependency"
            detail = ("the browser could not start on the computer after automatic repair "
                      f"({detail[:180]}). Notes: {notes[:200] or 'none'}")
        return _fail(f"browse {kind}: {detail}",
                     data={"kind": "browser_" + kind, "attempts": result.get("attempts"),
                           "notes": notes[:400]})
    notes = "; ".join(result.get("notes") or [])
    if action in ("tabs", "download", "close"):
        return ToolResult(ok=True, content=json.dumps({k: v for k, v in result.items() if k != "notes"},
                                                      ensure_ascii=False)[:4000],
                          summary=f"browser {action}", data={"action": action, **{k: result.get(k) for k in
                                 ("path", "bytes", "closed") if k in result}})
    text, cut = clip(str(result.get("text") or ""), ctx.settings.tool_result_chars, "ends")
    links = "\n".join(f"- {l.get('text') or '(no text)'} {l.get('href')}" for l in (result.get("links") or [])[:15])
    shot = f"\nScreenshot saved: {result['screenshot']}" if result.get("screenshot") else ""
    captcha = result.get("captcha") or {}
    challenge = (f"\nCAPTCHA/challenge detected: {', '.join(captcha.get('kinds') or [])}. "
                 f"{captcha.get('note', '')}" if captcha.get("present") else "")
    accessibility = (f"\nAccessibility snapshot:\n{json.dumps(result.get('accessibility'), ensure_ascii=False)[:1800]}"
                     if result.get("accessibility") else "")
    content = (f"{result.get('title')}\nURL: {result.get('final_url')} (HTTP {result.get('status')})"
               f"{' | notes: ' + notes if notes else ''}{shot}{challenge}{accessibility}\n\n{text}\n\nLinks:\n{links}")
    return ToolResult(ok=True, content=content, summary=f"browsed {result.get('final_url', url)}",
                      truncated=cut or bool(result.get("text_truncated")),
                      data={"final_url": result.get("final_url"), "status": result.get("status"),
                            "screenshot": result.get("screenshot"), "elapsed_s": result.get("elapsed_s"),
                            "captcha": bool(captcha.get("present")),
                            "text": str(result.get("text") or "")[:2000]})


async def cyber_run(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Install (once) and execute a catalogued security tool on the computer."""
    tool_id = str(args.get("tool_id") or args.get("id") or "").strip()
    if not TOOL_ID.match(tool_id):
        return _fail("tool_id must be a registry id such as tool:web_attack.Nuclei "
                     "(find one with tool_search)", {"kind": "invalid"})
    raw_args = str(args.get("args") or "").strip()
    if raw_args and not TOOL_ARGS.match(raw_args):
        return _fail("args may contain only plain command-line characters (no quotes, semicolons or redirects)",
                     {"kind": "invalid"})
    timeout = max(10, min(1800, int(args.get("timeout_seconds") or 300)))
    flags = f"--id {shlex.quote(tool_id)} --timeout {timeout}"
    if raw_args:
        flags += f" --args {shlex.quote(raw_args)}"
    data = await _run_skillctl(ctx, "run", flags, timeout=timeout + 120)
    status = str(data.get("status") or "")
    if not data.get("ok"):
        detail = data.get("error") or data.get("output") or "tool execution failed"
        return ToolResult.failure(f"cyber_run {tool_id} [{status or 'error'}]: {str(detail)[:600]}",
                                  hint="Search for another tool, or load the matching skill for a manual procedure.",
                                  data={"kind": "tool_failed", "status": status, "tool_id": tool_id,
                                        "exit_code": data.get("exit_code")})
    body, cut = clip(str(data.get("output") or "(no output)"), ctx.settings.tool_result_chars, "ends")
    return ToolResult(ok=True, content=(f"{data.get('title') or tool_id} [{status}] on the computer "
                                        f"(exit {data.get('exit_code')}, {data.get('seconds')}s)\n"
                                        f"$ {data.get('command')}\n{body}"),
                      summary=f"{tool_id} {status} exit={data.get('exit_code')}",
                      truncated=cut, data={"exit_code": data.get("exit_code"), "tool_id": tool_id,
                                           "status": status, "output": body[:2000]})


async def cyber_preflight(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Report which catalogued binaries are already available on this computer."""
    data = await _run_skillctl(ctx, "preflight", "", timeout=300)
    if not data.get("ok"):
        return _fail(str(data.get("error") or "preflight failed"), {"kind": "registry"})
    counts = data.get("counts") or {}
    missing = data.get("missing") or []
    body = (f"registry root: {data.get('root')}\n"
            f"skills: {counts.get('skills')} (anthropic {counts.get('by_source', {}).get('anthropic')}, "
            f"claude-red {counts.get('by_source', {}).get('claude-red')}); tools: {counts.get('tools')}\n"
            f"binaries present: {data.get('binaries_present')}/{data.get('binaries_checked')}\n"
            f"missing (first 40): {', '.join(missing[:40]) or 'none'}")
    return ToolResult(ok=True, content=body, summary=f"{data.get('binaries_present')} binaries ready",
                      data={"counts": counts})


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
                  "required": ["sha256"]}, malware_analyzer_report, timeout=140.0, kind="net",
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
                 "Load one skill's instructions by exact id from skill_search (e.g. anthropic:triaging-security-incident "
                 "or claude-red:offensive-osint). Documents are read, not executed. Authorised work only.",
                 {"type": "object", "properties": {
                     "id": {"type": "string", "minLength": 3, "maxLength": 160},
                     "name": {"type": "string", "minLength": 3, "maxLength": 121}},
                  "required": []},
                 skill_load, timeout=330.0, describe=lambda a: str(a.get("name") or "")),
        ToolSpec("skill_read",
                 "Read one reference file inside a loaded skill folder (for example references/api-reference.md).",
                 {"type": "object", "properties": {
                     "id": {"type": "string", "minLength": 3, "maxLength": 160},
                     "name": {"type": "string", "minLength": 3, "maxLength": 121},
                     "path": {"type": "string", "minLength": 1, "maxLength": 200}},
                  "required": ["path"]},
                 skill_read, timeout=330.0, describe=lambda a: f"{a.get('name', '')}/{a.get('path', '')}"),
        ToolSpec("tool_search",
                 "Search the executable security tool catalog (183 tools across the three integrated repositories; "
                 "metadata only: category, description, runnability). Nothing is installed or run by this search. "
                 "Use cyber_run with the returned id to execute one.",
                 {"type": "object", "properties": {
                     "query": {"type": "string", "minLength": 2, "maxLength": 120},
                     "limit": {"type": "integer", "minimum": 1, "maximum": 12, "default": 6}},
                  "required": ["query"]},
                 tool_search, timeout=330.0, describe=lambda a: str(a.get("query") or "")),
        ToolSpec("cyber_run",
                 "Install (only if needed, once) and execute one catalogued security tool on the computer, e.g. "
                 "tool:web_attack.Nuclei with args '-u https://target.example'. Returns the real exit code and output. "
                 "Only for systems you are authorised to test.",
                 {"type": "object", "properties": {
                     "tool_id": {"type": "string", "minLength": 6, "maxLength": 160},
                     "args": {"type": "string", "maxLength": 500, "default": ""},
                     "timeout_seconds": {"type": "integer", "minimum": 10, "maximum": 1800, "default": 300}},
                  "required": ["tool_id"]},
                 cyber_run, timeout=1900.0, kind="exec",
                 describe=lambda a: str(a.get("tool_id") or "")),
        ToolSpec("cyber_preflight",
                 "Show the cybersecurity registry state on the computer: skill/tool counts per source and which "
                 "binaries are already installed. Call before a long security engagement.",
                 {"type": "object", "properties": {}}, cyber_preflight, timeout=330.0,
                 describe=lambda a: "registry preflight"),
        ToolSpec("browse",
                 "Drive the Camoufox stealth browser on the computer. action=open navigates and returns title, URL, "
                 "status, visible text, links and any bot-challenge it detects; click/type/scroll interact with a "
                 "page (>selector<, >text<); screenshot saves a PNG; accessibility returns a bounded accessibility "
                 "tree; tabs lists/closes tabs; download saves a file under the workspace. Use web_search for plain "
                 "research and browse for JavaScript pages, logins and forms.",
                 {"type": "object", "properties": {
                     "url": {"type": "string", "maxLength": 2000},
                     "action": {"type": "string", "enum": ["open", "click", "type", "scroll", "screenshot",
                                                           "accessibility", "tabs", "download", "close"],
                                "default": "open"},
                     "selector": {"type": "string", "maxLength": 300},
                     "text": {"type": "string", "maxLength": 4000},
                     "submit": {"type": "boolean", "default": False},
                     "scroll_px": {"type": "integer", "minimum": -20000, "maximum": 20000, "default": 800},
                     "wait_ms": {"type": "integer", "minimum": 0, "maximum": 15000},
                     "download_path": {"type": "string", "maxLength": 300},
                     "close_tab": {"type": "integer", "minimum": 0, "maximum": 50},
                     "full_page": {"type": "boolean", "default": False},
                     "timeout_ms": {"type": "integer", "minimum": 3000, "maximum": 90000, "default": 25000},
                     "screenshot": {"type": "boolean", "default": False}},
                  "required": []},
                 browse, timeout=360.0, kind="net", describe=lambda a: str(a.get("url") or "")),
    ]
