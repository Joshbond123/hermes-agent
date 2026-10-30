"""Malware / ransomware / virus execution lab tools.

The lab is a dedicated, network-sealed Firecracker microVM that is **off by
default**. Nothing in this module creates it implicitly except ``lab_detonate``
(which needs a running lab to do its job and so opens one first).

Contract the model is given:
- open it only when a sample genuinely has to be *run* to be understood;
- close it as soon as the analysis is finished (and it also closes itself when
  idle, or when the provider TTL expires, whichever comes first).

Samples never come from the model's imagination: they are bytes the agent read
from its own workspace, and the lab reports the SHA-256 it actually executed.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from typing import Any, Dict, List, Optional

from .base import ToolContext, ToolResult, ToolSpec, clip, redact

MAX_INLINE = 2 * 1024 * 1024
PATH_OK = re.compile(r"^[A-Za-z0-9._/ -]{1,300}$")
REPORT_DIR = "malware_lab/reports"

SCHEMAS: Dict[str, Dict[str, Any]] = {
    "lab_open": {"type": "object", "properties": {
        "reason": {"type": "string", "minLength": 1, "maxLength": 200,
                   "default": "dynamic analysis"}}, "required": []},
    "lab_status": {"type": "object", "properties": {}},
    "lab_detonate": {"type": "object", "properties": {
        "sample_path": {"type": "string", "minLength": 1, "maxLength": 300},
        "sample_text": {"type": "string", "maxLength": MAX_INLINE},
        "sample_b64": {"type": "string", "maxLength": MAX_INLINE},
        "filename": {"type": "string", "minLength": 1, "maxLength": 120, "default": "sample.bin"},
        "timeout_seconds": {"type": "integer", "minimum": 5, "maximum": 600, "default": 60},
        "network": {"type": "string", "enum": ["sealed", "live"], "default": "sealed"},
        "note": {"type": "string", "maxLength": 200, "default": ""}},
        "required": []},
    "lab_close": {"type": "object", "properties": {
        "reason": {"type": "string", "maxLength": 200, "default": "analysis finished"}},
        "required": []},
    "lab_exec": {"type": "object", "properties": {
        "command": {"type": "string", "minLength": 1, "maxLength": 8000},
        "os": {"type": "string", "enum": ["auto", "linux", "windows"], "default": "auto"},
        "timeout_seconds": {"type": "integer", "minimum": 5, "maximum": 900, "default": 120},
        "job": {"type": "string", "maxLength": 64,
                "description": "a job handle returned by an earlier lab_exec; polls it instead of running a command"},
        "note": {"type": "string", "maxLength": 200, "default": ""}},
        "required": ["command"]},
    "lab_os": {"type": "object", "properties": {
        "install_windows": {"type": "boolean", "default": False}}, "required": []},
    "lab_file": {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["write", "read", "list", "stat"], "default": "read"},
        "path": {"type": "string", "maxLength": 300},
        "content": {"type": "string", "maxLength": 1000000},
        "encoding": {"type": "string", "enum": ["text", "base64"], "default": "text"},
        "max_chars": {"type": "integer", "minimum": 200, "maximum": 200000, "default": 20000}},
        "required": []},
}


def _lab(ctx: ToolContext):
    lab = getattr(ctx, "lab", None)
    if lab is None:
        return None
    return lab


def _unconfigured() -> ToolResult:
    return ToolResult.failure(
        "The malware execution lab is not available on this deployment: no PandaStack key is configured.",
        hint="Static analysis still works: sample_triage and the security catalog need no lab.")


async def _read_sample(ctx: ToolContext, path: str) -> bytes:
    """Fetch sample bytes from the agent's own computer (base64 over the exec channel)."""
    clean = path.strip().lstrip("/")
    if clean.startswith("workspace/"):
        clean = clean[len("workspace/"):]
    if not PATH_OK.fullmatch(clean) or ".." in clean.split("/"):
        raise ValueError("sample_path must be a simple path inside the computer workspace")
    size = await ctx.computer.call("/computer/exec", {
        "command": f"stat -c%s {clean!r} 2>/dev/null || echo missing", "timeout_seconds": 45})
    token = str(size.get("output") or "").strip().splitlines()[-1] if size.get("output") else ""
    if token == "missing" or not token.isdigit():
        raise ValueError(f"no such file on the computer workspace: {clean}")
    if int(token) == 0:
        raise ValueError(f"the file on the computer workspace is empty: {clean}")
    if int(token) > MAX_INLINE:
        raise ValueError(f"sample is {token} bytes; the lab accepts at most {MAX_INLINE // (1024 * 1024)} MB in a tool call")
    data = await ctx.computer.call("/computer/exec", {
        "command": f"base64 -w0 {clean!r}", "timeout_seconds": 120}, timeout=180)
    if not data.get("ok"):
        raise ValueError("the sample could not be read from the computer")
    return base64.b64decode(str(data.get("output") or "").strip(), validate=True)


async def _sample(args: Dict[str, Any], ctx: ToolContext) -> tuple[bytes, str]:
    if args.get("sample_b64"):
        try:
            raw = base64.b64decode(args["sample_b64"], validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("sample_b64 is not valid base64") from exc
        return raw, "inline base64"
    if args.get("sample_text"):
        return args["sample_text"].encode("utf-8"), "inline text"
    if args.get("sample_path"):
        return await _read_sample(ctx, args["sample_path"]), "computer workspace"
    raise ValueError("give one of sample_path, sample_text or sample_b64")


async def lab_open(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    lab = _lab(ctx)
    if lab is None or not lab.configured():
        return _unconfigured()
    state = await lab.open(reason=args.get("reason") or "dynamic analysis")
    iso = state.get("probes") or {}
    body = (f"Malware lab is ON.\n"
            f"sandbox: {state.get('sandbox_id')} (separate microVM, not your working computer)\n"
            f"reason: {state.get('reason')}\n"
            f"isolation verified: {iso.get('verified')} — sample runs inside '{iso.get('network_namespace')}', "
            f"as {iso.get('sample_runs_as')}\n"
            f"egress from inside the namespace: {iso.get('egress_inside_namespace')}\n"
            f"lab VM egress: {iso.get('lab_vm_egress')}\n"
            f"auto-close after {lab.idle_s}s idle; provider TTL {lab.ttl_s}s.\n"
            f"Close it with lab_close as soon as the analysis is finished.")
    return ToolResult(ok=True, content=redact(body), summary="lab open · isolated microVM",
                      data={"session": state})


async def lab_status(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    lab = _lab(ctx)
    if lab is None:
        return _unconfigured()
    state = await lab.status()
    if not state.get("open"):
        body = ("Malware lab is OFF." if state.get("configured") else
                "Malware lab is unavailable (no PandaStack key).")
        if state.get("orphans_released"):
            body += f"\nReleased orphaned lab VMs: {', '.join(state['orphans_released'])}"
        return ToolResult(ok=True, content=body, summary="lab off", data=state)
    body = (f"Malware lab is ON · sandbox {state.get('sandbox_id')}\n"
            f"open for {state.get('age_s')}s · provider TTL remaining {state.get('remaining_s')}s\n"
            f"sealed: {state.get('sealed')} · runs this session: {state.get('runs')}\n"
            f"last run: {json.dumps(state.get('last_run') or {}, sort_keys=True)[:400]}")
    return ToolResult(ok=True, content=redact(body), summary="lab open · " + str(state.get("runs")) + " run(s)",
                      data=state)


async def lab_detonate(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    lab = _lab(ctx)
    if lab is None or not lab.configured():
        return _unconfigured()
    try:
        raw, origin = await _sample(args, ctx)
    except ValueError as exc:
        return ToolResult.failure(str(exc))
    note = args.get("note") or ""
    network = "live" if str(args.get("network") or "").lower() == "live" else "sealed"
    result = await lab.detonate(sample=raw, filename=args.get("filename") or "sample.bin",
                                timeout_s=int(args.get("timeout_seconds") or 60), note=note,
                                network=network)
    report = result["report"]
    observed = {
        "exit_code": report.get("exit_code"),
        "duration_ms": report.get("duration_ms"),
        "created_files": (report.get("created_files") or [])[:25],
        "created_file_count": len(report.get("created_files") or []),
        "processes_after": (report.get("processes_after") or [])[:15],
        "stdout": clip(str(report.get("stdout") or ""), 1200)[0],
        "stderr": clip(str(report.get("stderr") or ""), 800)[0],
    }
    # The report is text-only and redacted; the sample bytes and any dropped
    # artifacts deliberately stay inside the lab VM, which is about to be destroyed.
    try:
        await ctx.computer.call("/computer/write_file", {
            "path": f"{REPORT_DIR}/{result['run_id']}.json",
            "content": json.dumps({"record": result["record"], "isolation": result["isolation"],
                                   "report": report}, indent=1)[:120000]})
        saved = f"{REPORT_DIR}/{result['run_id']}.json"
    except Exception:  # noqa: BLE001 - a missing archive must not invalidate the run
        saved = ""
    body = (f"Detonated {args.get('filename') or 'sample.bin'} in the sealed lab.\n"
            f"run: {result['run_id']} · sample came from {origin} · sha256 {result['record']['sample_sha256']}\n"
            f"ran as {result['isolation'].get('sample_runs_as')} with sample network = "
            f"{result['record'].get('network')} (egress inside namespace "
            f"{result['isolation'].get('egress_inside_namespace')}, lab VM internet "
            f"{result['isolation'].get('internet')})\n"
            f"observed:\n{json.dumps(observed, indent=1)[:6000]}\n"
            f"report saved to {saved or '(not archived)'}\n"
            f"The lab is still ON; call lab_close when the analysis is finished.")
    return ToolResult(ok=True, content=redact(body), summary=f"detonated {result['run_id']}",
                      data={"record": result["record"], "isolation": result["isolation"],
                            "report_path": saved})


async def lab_close(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    lab = _lab(ctx)
    if lab is None:
        return _unconfigured()
    state = await lab.close(why=args.get("reason") or "analysis finished")
    if not state.get("closed"):
        return ToolResult(ok=True, content=("Malware lab is OFF (it was already off)." +
                                            (f"\nReleased orphaned lab VMs: {', '.join(state['orphans_released'])}"
                                             if state.get("orphans_released") else "")),
                          summary="lab already off", data=state)
    body = (f"Malware lab is OFF. Sandbox {state.get('sandbox_id')} deleted and verified gone "
            f"by the provider. Runs this session: {state.get('runs')}.")
    return ToolResult(ok=True, content=body, summary="lab closed · VM deleted", data=state)


async def lab_os(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Report the lab guest OS, privileges and whether Windows binaries can run."""
    lab = _lab(ctx)
    if lab is None:
        return _unconfigured()
    if lab.session is None:
        return ToolResult.failure("The malware lab is off. Call lab_open first (or run lab_exec, which opens it).")
    lab.hold()
    try:
        if args.get("install_windows"):
            installed = await lab.install_windows_support()
            info = installed.get("os") or await lab.os_info()
            body = (f"Windows (PE) support installed={installed.get('installed')} "
                    f"verified={installed.get('verified')} wine={installed.get('wine') or 'absent'}")
            return ToolResult(ok=bool(installed.get("verified")), content=body,
                              summary="windows support prepared", data={**installed, **info})
        info = await lab.os_info()
        body = (f"Lab guest: {info.get('kernel')} · uid {info.get('uid')} "
                f"(root={info.get('root')}) · arch {info.get('arch')}\n"
                f"Linux execution: yes · Windows execution: "
                f"{'yes via ' + str(info.get('wine')) if info.get('windows') else 'NOT READY — '
                   + str(info.get('windows_note'))}")
        return ToolResult(ok=True, content=body, summary=f"lab os {info.get('kernel')}", data=info)
    finally:
        lab.release()


async def lab_exec(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Run an administrative or analysis command inside the dedicated lab as root."""
    lab = _lab(ctx)
    if lab is None:
        return _unconfigured()
    command = str(args.get("command") or "").strip()
    job = str(args.get("job") or "").strip()
    if not command and not job:
        return ToolResult.failure("command (or a job handle) is required")
    timeout = int(args.get("timeout_seconds") or 120)
    if job:
        # Polling a long-running lab job (an install, a long dynamic run). It keeps running
        # whether or not this call returns, so the provider edge can never time it out.
        lab.hold()
        try:
            status = await lab.job_status(job)
        finally:
            lab.release()
        return ToolResult(ok=True, content=redact(
            f"job {job}: running={status['running']} exit={status['exit_code']}\n"
            f"{clip(status.get('log_tail') or '', 4000)[0]}"),
            summary=f"lab job {job} running={status['running']}", data=status)
    long_job = timeout > 90
    lab.hold()                       # the idle watchdog must not close the lab mid-command
    try:
        if lab.session is None:
            await lab.open(reason=args.get("note") or "lab command")
        if long_job:
            handle = await lab.start_job(command, name="exec")
            return ToolResult(ok=True, content=(
                f"Started lab job {handle['job']} (pid {handle['pid']}). It keeps running after this call. "
                f"Poll it with lab_exec(job=\"{handle['job']}\"). Long installs and long runs must be done this way."),
                summary=f"lab job {handle['job']} started", data=handle)
        result = await lab.run_os(command, os_name=args.get("os") or "auto", timeout=timeout)
    finally:
        lab.release()
    body = (f"$ {command}\nexit={result.get('exit_code')} ({result.get('duration_ms')} ms)\n"
            f"{clip((result.get('stdout') or '') + (result.get('stderr') or ''), 4000)[0]}")
    return ToolResult(ok=result.get("exit_code") == 0, content=redact(body),
                      summary=f"lab exec exit {result.get('exit_code')}", data=result)


LAB_PATH = re.compile(r"^(?:/lab/[A-Za-z0-9._/-]{0,280}|/tmp/[A-Za-z0-9._/-]{0,280}|/workspace/[A-Za-z0-9._/-]{0,280})$")


def _lab_path(raw: Any) -> str:
    path = str(raw or "").strip()
    if not LAB_PATH.match(path) or ".." in path.split("/"):
        raise ValueError("path must be an absolute path under /lab, /tmp or /workspace")
    return path


async def lab_file(args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
    """File management inside the lab VM: write, read, list or stat any path under /lab, /tmp, /workspace.

    One schema with an op keeps the fast-path tool budget (and therefore prefill latency) unchanged
    while giving the lab a complete filesystem: build scripts, drop payloads, collect artefacts,
    inspect what an installer or a sample left behind.
    """
    lab = _lab(ctx)
    if lab is None:
        return _unconfigured()
    op = str(args.get("op") or "read").strip().lower()
    default_path = "/lab" if op == "list" else ""
    try:
        path = _lab_path(args.get("path") or default_path)
    except ValueError as exc:
        return ToolResult.failure(str(exc))
    limit = max(200, min(200000, int(args.get("max_chars") or 20000)))
    lab.hold()
    try:
        if lab.session is None:
            await lab.open(reason=f"lab file {op}")
        if op == "write":
            raw = str(args.get("content") or "")
            if str(args.get("encoding") or "text") == "base64":
                try:
                    data = base64.b64decode(raw, validate=True)
                except (binascii.Error, ValueError):
                    return ToolResult.failure("content is not valid base64")
            else:
                data = raw.encode("utf-8")
            if len(data) > MAX_INLINE:
                return ToolResult.failure("content is larger than 1 MB; download it inside the lab instead")
            await lab._put(path, data)
            check = await lab.root_exec(f"ls -l {path}; sha256sum {path} | cut -c1-16", timeout=60)
            return ToolResult(ok=check.get("exit_code") == 0,
                              content=redact(f"wrote {len(data)} bytes to {path}\n" + str(check.get("stdout") or "")),
                              summary=f"lab write {path}", data={"path": path, "bytes": len(data)})
        if op == "list":
            result = await lab.root_exec(f"ls -la {path} 2>&1 | head -120", timeout=60)
            return ToolResult(ok=result.get("exit_code") == 0, content=redact(str(result.get("stdout") or "")),
                              summary=f"lab ls {path}", data={"path": path})
        if op == "stat":
            result = await lab.root_exec(f"stat -c '%n %s %a %U %y' {path} 2>&1; file {path} 2>&1", timeout=60)
            return ToolResult(ok=result.get("exit_code") == 0, content=redact(str(result.get("stdout") or "")),
                              summary=f"lab stat {path}", data={"path": path})
        body = await lab._get(path, limit=limit)
    finally:
        lab.release()
    text = body.decode("utf-8", "replace")
    out, cut = clip(redact(text), min(limit, 20000), "ends")
    return ToolResult(ok=True, content=out or "(empty file)", summary=f"lab read {path}",
                      truncated=cut or len(body) >= limit, data={"path": path, "bytes": len(body)})


def specs() -> list[ToolSpec]:
    return [
        ToolSpec("lab_open",
                 "Turn on the isolated malware execution lab: a dedicated PandaStack microVM, separate from your "
                 "working computer, where the sample runs unprivileged inside an empty network namespace with no "
                 "route, DNS or internet. Use it only when a sample must actually be RUN to be understood; it is "
                 "off by default and costs money while it is up.",
                 SCHEMAS["lab_open"], lab_open, timeout=300.0, kind="exec",
                 describe=lambda a: str(a.get("reason") or "dynamic analysis")),
        ToolSpec("lab_status",
                 "Report whether the malware lab is on, how long it has left, and what has run in it. Also releases "
                 "lab VMs orphaned by an earlier crash.",
                 SCHEMAS["lab_status"], lab_status, timeout=120.0, kind="read",
                 describe=lambda a: ""),
        ToolSpec("lab_detonate",
                 "Execute ONE sample in the sealed lab and report what it did: exit code, runtime, files created, "
                 "processes left behind, stdout/stderr. Opens the lab automatically if it is off. Supply the sample "
                 "as sample_path (a file in your workspace), sample_text (a script you are analysing) or sample_b64. "
                 "The sample never runs on your working computer, and its bytes never leave the lab VM.",
                 SCHEMAS["lab_detonate"], lab_detonate, timeout=900.0, kind="exec",
                 describe=lambda a: str(a.get("filename") or a.get("sample_path") or "sample")),
        ToolSpec("lab_os",
                 "Report the lab guest OS, whether commands run as root, and whether Windows binaries can be "
                 "executed. Set install_windows=true to install Wine and verify it by running a real Windows PE.",
                 SCHEMAS["lab_os"], lab_os, timeout=1200.0, kind="exec",
                 describe=lambda a: "install windows" if a.get("install_windows") else "os info"),
        ToolSpec("lab_exec",
                 "Run a command inside the dedicated malware lab VM as root — install tools and antivirus, "
                 "manage processes, inspect files, analyse the OS. os='linux' runs natively; os='windows' runs a "
                 "Windows PE/command through Wine after verifying it is available; os='auto' picks Windows for "
                 ".exe/.bat/.cmd/.ps1. The lab starts automatically if it is off and is left running for you to "
                 "finish the job (call lab_close when done).",
                 SCHEMAS["lab_exec"], lab_exec, timeout=1200.0, kind="exec",
                 describe=lambda a: str(a.get("command") or "")[:80]),
        ToolSpec("lab_file",
                 "Full file management inside the lab VM: op=write saves a file (under /lab, /tmp or /workspace; "
                 "text or base64), op=read returns its contents, op=list shows a directory, op=stat shows "
                 "size/permissions/type. Use it to build scripts, drop payloads, collect reports and inspect "
                 "what an installer or a sample left behind.",
                 SCHEMAS["lab_file"], lab_file, timeout=600.0,
                 kind="write", describe=lambda a: f"{a.get('op') or 'read'} {a.get('path') or '/lab'}"),
        ToolSpec("lab_close",
                 "Destroy the malware lab VM and verify the provider no longer lists it. Call this as soon as the "
                 "analysis is finished so nothing keeps running or billing.",
                 SCHEMAS["lab_close"], lab_close, timeout=180.0, kind="exec",
                 describe=lambda a: str(a.get("reason") or "analysis finished")),
    ]
