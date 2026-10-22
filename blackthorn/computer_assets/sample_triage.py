#!/usr/bin/env python3
"""Read-only PE/ELF/script sample triage on the Blackthorn computer.

This is NOT a virtual machine or a dynamic malware sandbox. Never execute,
import, unpack, or pass a sample to an external parser. The production computer
must not detonate untrusted code without a separately validated isolation host.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import sys
from collections import Counter
from pathlib import Path

MAX_BYTES = 32 * 1024 * 1024
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z", re.ASCII)
MACHINES = {0x14c: "x86", 0x8664: "x86-64", 0xaa64: "ARM64", 0x3: "x86", 0x3e: "x86-64", 0xb7: "ARM64"}


def _entropy(raw: bytes) -> float:
    if not raw:
        return 0.0
    counts = Counter(raw)
    return round(-sum((n / len(raw)) * math.log2(n / len(raw)) for n in counts.values()), 3)


def _format(raw: bytes) -> dict:
    if raw[:2] == b"MZ":
        result = {"format": "MZ (PE header unconfirmed)", "architecture": "unknown"}
        if len(raw) >= 64:
            offset = int.from_bytes(raw[0x3c:0x40], "little")
            if offset <= len(raw) - 26 and raw[offset:offset + 4] == b"PE\x00\x00":
                machine = int.from_bytes(raw[offset + 4:offset + 6], "little")
                result = {"format": "Windows PE", "architecture": MACHINES.get(machine, "unknown")}
        return result
    if raw[:4] == b"\x7fELF":
        order = "big" if len(raw) > 5 and raw[5] == 2 else "little"
        machine = int.from_bytes(raw[18:20], order) if len(raw) >= 20 else 0
        return {"format": "Linux ELF", "architecture": MACHINES.get(machine, "unknown")}
    if raw.startswith(b"#!"):
        return {"format": "script with shebang", "architecture": "interpreted"}
    return {"format": "unrecognized data (never executed)", "architecture": "unknown"}


def _root() -> Path:
    workspace = Path(os.environ.get("BLACKTHORN_WORKSPACE") or "/kaggle/working/blackthorn_workspace")
    root = workspace / "malware_samples"
    # Do not follow a redirected quarantine folder into the host or credentials.
    if root.is_symlink() or workspace.is_symlink():
        raise ValueError("unsafe quarantine symlink")
    return root


def analyze(sample_id: str) -> dict:
    if not NAME.fullmatch(sample_id) or sample_id in (".", ".."):
        raise ValueError("sample_id must be a single safe file name")
    root = _root()
    # No traversal, archive extraction, decompression, subprocess, or execution.
    fd = os.open(root / sample_id, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        meta = os.fstat(fd)
        if not stat.S_ISREG(meta.st_mode) or meta.st_size > MAX_BYTES:
            raise ValueError("sample must be a regular file at most 32 MiB")
        h = hashlib.sha256()
        buf = bytearray()
        while len(buf) <= MAX_BYTES:
            part = os.read(fd, min(1024 * 1024, MAX_BYTES + 1 - len(buf)))
            if not part:
                break
            h.update(part)
            buf.extend(part)
        if len(buf) > MAX_BYTES:
            raise ValueError("sample exceeds 32 MiB")
        if os.fstat(fd).st_size != meta.st_size:
            raise ValueError("sample changed during analysis")
    finally:
        os.close(fd)
    raw = bytes(buf)
    kind = _format(raw)
    # Counts only: never leak embedded URLs, credentials, sample text or payload.
    return {"ok": True, "mode": "read-only-static-analysis", "dynamic_execution_allowed": False,
            "sample_id": sample_id, "bytes": len(raw), "sha256": h.hexdigest(),
            **kind, "entropy_first_64k": _entropy(raw[:65536]),
            "url_like_count": len(re.findall(rb"https?://[A-Za-z0-9.-]{2,}", raw[:MAX_BYTES], re.I)),
            "warning": "Static metadata is not a malware verdict; no sample was run."}


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--probe":
        data = {"ok": True, "mode": "read-only-static-analysis", "dynamic_execution_allowed": False,
                "linux_vm_ready": False, "windows_vm_ready": False,
                "reason": "No validated isolation host; never run samples on the production computer."}
    elif len(argv) == 3 and argv[1] == "--analyze":
        try:
            data = analyze(argv[2])
        except (OSError, ValueError) as exc:
            data = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:120]}",
                    "dynamic_execution_allowed": False}
    else:
        data = {"ok": False, "error": "Only --probe or --analyze SAMPLE_ID is supported; execution is disabled.",
                "dynamic_execution_allowed": False}
    print(json.dumps(data, sort_keys=True))
    return 0 if data.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
