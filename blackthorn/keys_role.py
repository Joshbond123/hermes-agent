"""PandaStack credential roles: general-purpose computers vs the dedicated lab.

The Google Drive credential document defines two distinct pools:

* **computer keys** — the PandaStack Computer accounts used for ordinary work
  (browsing, development, terminal, files). ``PANDASTACK_API_KEY`` and
  ``PANDASTACK_API_KEY_2..9`` / ``PANDASTACK_API_KEYS``.
* **the lab key** — a single account dedicated to the Malware / Ransomware /
  Virus Execution Lab, listed separately underneath the GitHub drive PAT in the
  document. ``PANDASTACK_LAB_API_KEY``.

The two must never mix: a lab operation on a general-purpose computer would run
malware on the agent's working machine, and a general task on the lab account
would bill the lab for unrelated work. This module is the single place that
decides which credential belongs to which role, and it fails loudly on overlap
instead of silently picking "whatever key is available".
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

COMPUTER_SINGLE = "PANDASTACK_API_KEY"
COMPUTER_MULTI = "PANDASTACK_API_KEYS"
COMPUTER_NUMBERED = tuple(f"PANDASTACK_API_KEY_{i}" for i in range(2, 10))
LAB_ENVS = ("PANDASTACK_LAB_API_KEY", "PANDASTACK_LAB_KEY", "PANDASTACK_SANDBOX_LAB_KEY")


class KeyRoleError(RuntimeError):
    """A credential was offered for the wrong role."""


def _parse(raw: str) -> List[str]:
    out: List[str] = []
    for chunk in (raw or "").replace("\n", ",").replace(";", ",").split(","):
        key = chunk.strip()
        if key and key not in out:
            out.append(key)
    return out


def lab_keys(environ: Optional[Dict[str, str]] = None) -> List[str]:
    env = environ if environ is not None else os.environ
    out: List[str] = []
    for name in LAB_ENVS:
        for key in _parse(env.get(name, "")):
            if key not in out:
                out.append(key)
    return out


def lab_key(environ: Optional[Dict[str, str]] = None) -> str:
    """The single dedicated lab credential, or '' when the lab is not configured."""
    keys = lab_keys(environ)
    return keys[0] if keys else ""


def computer_keys(environ: Optional[Dict[str, str]] = None) -> List[str]:
    """Every general-purpose computer credential, with lab keys removed."""
    env = environ if environ is not None else os.environ
    reserved = set(lab_keys(env))
    out: List[str] = []
    for raw in [env.get(COMPUTER_SINGLE, "")] + [env.get(n, "") for n in COMPUTER_NUMBERED] + \
               [env.get(COMPUTER_MULTI, "")]:
        for key in _parse(raw):
            if key in reserved or key in out:
                continue
            out.append(key)
    return out


def overlap(environ: Optional[Dict[str, str]] = None) -> List[str]:
    """Any key configured for both roles (a misconfiguration we must never act on)."""
    env = environ if environ is not None else os.environ
    reserved = set(lab_keys(env))
    return [k for k in computer_keys(env) if k in reserved]


def describe(environ: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """Secret-free description for /api/lab and status output (fingerprints only)."""
    env = environ if environ is not None else os.environ
    computers = computer_keys(env)
    lab = lab_keys(env)
    return {
        "computer_key_count": len(computers),
        # Never expose a key: the last four characters are enough to identify the account.
        "computer_fingerprints": [k[-4:] for k in computers],
        "lab_configured": bool(lab),
        "lab_fingerprint": lab[0][-4:] if lab else "",
        "separated": not overlap(env),
        "lab_env": next((name for name in LAB_ENVS if env.get(name)), ""),
    }


def require_lab_key(environ: Optional[Dict[str, str]] = None) -> str:
    key = lab_key(environ)
    if not key:
        raise KeyRoleError("The malware lab needs its own PANDASTACK_LAB_API_KEY; "
                           "computer keys are never used for lab work.")
    return key


def reject_lab_key_for_computers(key: str, environ: Optional[Dict[str, str]] = None) -> None:
    """Guard used by the computer client: a lab credential must never create a computer."""
    if key and key in set(lab_keys(environ)):
        raise KeyRoleError("Refusing to use the dedicated lab key for a general-purpose computer")


def filter_computer_keys(keys: Sequence[str], environ: Optional[Dict[str, str]] = None) -> List[str]:
    reserved = set(lab_keys(environ))
    return [k for k in keys if k and k not in reserved]
