#!/usr/bin/env python3
"""Install the Blackthorn security stack on the Kaggle computer. Runs ON THE COMPUTER, as root, Ubuntu 24.04.

Everything persistent goes under WS (/kaggle/working/blackthorn_workspace, a 20 GB volume that survives
runtime restarts). The registry is WS/tools/registry.json: one entry per catalog tool with every step's
exit code and output tail, plus a probe that runs the tool's own run command.

Policy:
- Commands that pipe a remote script into a shell (curl|sh, wget|bash) or need an interactive
  session (sudo su, rustup) are NOT run. They are recorded as status "held_review" for a human to review.
- Every command has a timeout. A failing tool never stops the loop.
- Re-running is safe: tools already marked "installed" are skipped; the same registry is updated.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

WS = Path(os.environ.get("BLACKTHORN_WS", "/kaggle/working/blackthorn_workspace"))
TOOLS = WS / "tools"
HOME = TOOLS / "home"
VENV = TOOLS / "venv"
REGISTRY = TOOLS / "registry.json"
CATALOG_REPO = "https://github.com/AKCodez/hackingtool-plugin"
SKILLS_REPO = "https://github.com/mukul975/Anthropic-Cybersecurity-Skills"
STEP_TIMEOUT = 1500
PROBE_TIMEOUT = 40
HELD = re.compile(r"\|\s*(sudo\s+)?(sh|bash)\b|\|\s*sudo\s+(sh|bash|tar)|sudo\s+su\b|rustup|sh\.rustup|curl[^|]*\|\s*sudo")
APT_BASE = [
    "git", "unzip", "jq", "golang-go", "ruby-full", "python3-venv", "python3-pip", "build-essential", "pkg-config",
    "libgtk-3-0t64", "libdbus-glib-1-2", "libxt6t64", "libasound2t64", "libx11-xcb1", "libxtst6", "libnss3", "libgbm1",
    "nmap", "masscan", "dnsutils", "whois", "netcat-openbsd", "socat", "tcpdump", "binwalk", "foremost", "steghide",
    "exiftool", "file", "binutils", "yara", "sqlmap", "nikto", "hydra", "john", "hashcat", "aircrack-ng", "gobuster",
    "dirb", "sslscan", "smbclient", "proxychains4", "wordlists", "radare2", "strace", "ltrace", "gdb",
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def env() -> dict:
    e = dict(os.environ)
    e["HOME"] = str(HOME)
    e["DEBIAN_FRONTEND"] = "noninteractive"
    e["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
    e["GOPATH"] = str(HOME / "go")
    e["GOBIN"] = str(HOME / "go" / "bin")
    e["PATH"] = ":".join([str(VENV / "bin"), str(HOME / "go" / "bin"), str(HOME / ".local" / "bin"),
                          str(HOME / "bin"), "/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin"])
    return e


def run(cmd: str, timeout: int, cwd: Path | None = None) -> dict:
    started = time.time()
    try:
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout,
                              env=env(), cwd=str(cwd or HOME), stdin=subprocess.DEVNULL)
        rc, out = proc.returncode, (proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired as exc:
        rc = 124
        out = ((exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")) + "\n[timeout]"
    return {"cmd": cmd[:400], "rc": rc, "seconds": round(time.time() - started, 1), "tail": out[-1200:]}


def load() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {"created": now(), "preflight": {}, "base": [], "camoufox": {}, "sources": {}, "tools": []}


def save(reg: dict) -> None:
    reg["updated"] = now()
    tmp = REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=1))
    tmp.replace(REGISTRY)


def preflight(reg: dict) -> None:
    os_release = Path("/etc/os-release").read_text().splitlines()[0] if Path("/etc/os-release").exists() else "?"
    disk = shutil.disk_usage(str(WS))
    reg["preflight"] = {"os": os_release, "user": os.geteuid(), "python": sys.version.split()[0],
                        "ws_free_gb": round(disk.free / 1e9, 2), "ws": str(WS), "checked": now()}


def apt_base(reg: dict) -> None:
    done = {b["pkg"] for b in reg["base"] if b.get("rc") == 0}
    todo = [p for p in APT_BASE if p not in done]
    if not todo:
        return
    upd = run("apt-get update -qq", 900)
    reg["base"].append({"pkg": "apt-get update", **upd})
    for pkg in todo:
        step = run(f"apt-get install -y -qq {pkg}", 900)
        reg["base"].append({"pkg": pkg, **step})
        save(reg)


def camoufox(reg: dict) -> None:
    if reg["camoufox"].get("status") == "installed":
        return
    steps = []
    steps.append(run(f"python3 -m venv {VENV}", 300))
    steps.append(run(f"{VENV}/bin/pip install -q --upgrade pip && {VENV}/bin/pip install -q 'camoufox[geoip]' playwright httpx", 1200))
    steps.append(run(f"TMPDIR={WS}/tmp {VENV}/bin/python -m camoufox fetch", 1500))
    ok = all(s["rc"] == 0 for s in steps)
    probe = run(f"{VENV}/bin/python -c \"import camoufox; print('camoufox', getattr(camoufox,'__version__','ok'))\"", 60)
    reg["camoufox"] = {"status": "installed" if ok and probe["rc"] == 0 else "failed", "steps": steps, "probe": probe}
    save(reg)


def sources(reg: dict) -> None:
    TOOLS.mkdir(parents=True, exist_ok=True)
    HOME.mkdir(parents=True, exist_ok=True)
    for name, repo in (("hackingtool-plugin", CATALOG_REPO), ("anthropic-cybersecurity-skills", SKILLS_REPO)):
        target = WS / "security_catalog" / name
        if (target / ".git").exists():
            reg["sources"][name] = {"state": "present"}
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        step = run(f"git clone --depth 1 -q {repo} {target}", 900)
        reg["sources"][name] = {"state": "cloned" if step["rc"] == 0 else "failed", "step": step}
        save(reg)


def install_tool(tool: dict, reg: dict) -> dict:
    entry = {"id": tool["id"], "title": tool.get("title"), "category": tool.get("category"),
             "archived": bool(tool.get("archived")), "status": "installed", "steps": [], "probe": None}
    for cmd in tool.get("install_commands") or []:
        if HELD.search(cmd):
            entry["steps"].append({"cmd": cmd[:400], "rc": None, "seconds": 0, "tail": "held: remote script or interactive step"})
            entry["status"] = "held_review"
            continue
        if re.search(r"\bsudo\s+su\b", cmd):
            entry["status"] = "held_review"
            continue
        clean = re.sub(r"\bsudo\s+", "", cmd)  # we are root; sudo would prompt
        clean = clean.replace("cd ~/", "cd " + str(HOME) + "/").replace("~/", str(HOME) + "/")
        step = run(clean, STEP_TIMEOUT)
        entry["steps"].append(step)
        if step["rc"] != 0 and entry["status"] == "installed":
            entry["status"] = "failed"
    if not tool.get("install_commands"):
        entry["status"] = "no_install_command"
    probes = tool.get("run_commands") or []
    if entry["status"] in ("installed", "failed") and probes:
        p = run(f"timeout {PROBE_TIMEOUT} {probes[0]}", PROBE_TIMEOUT + 15)
        p["ok"] = p["rc"] in (0, 1, 2) and "not found" not in p["tail"].lower() and "no such file" not in p["tail"].lower()
        entry["probe"] = p
        if entry["status"] == "installed" and not p["ok"]:
            entry["status"] = "installed_probe_failed"
    return entry


def catalog(reg: dict) -> None:
    path = WS / "security_catalog" / "hackingtool-plugin" / "plugins" / "hackingtool" / "data" / "tools.json"
    tools = json.loads(path.read_text())["tools"]
    done = {t["id"]: t for t in reg["tools"] if t["status"] in ("installed", "installed_probe_failed", "no_install_command", "held_review")}
    for tool in tools:
        if tool["id"] in done and done[tool["id"]]["status"] != "failed":
            continue
        started = time.time()
        result = install_tool(tool, reg)
        result["seconds"] = round(time.time() - started, 1)
        reg["tools"] = [t for t in reg["tools"] if t["id"] != tool["id"]] + [result]
        save(reg)
        print(f"[{now()}] {tool['id']}: {result['status']}", flush=True)


def main() -> None:
    TOOLS.mkdir(parents=True, exist_ok=True)
    HOME.mkdir(parents=True, exist_ok=True)
    (WS / "tmp").mkdir(parents=True, exist_ok=True)
    reg = load()
    preflight(reg)
    save(reg)
    apt_base(reg)
    sources(reg)
    camoufox(reg)
    catalog(reg)
    counts: dict = {}
    for t in reg["tools"]:
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    reg["summary"] = counts
    save(reg)
    print("DONE", json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
