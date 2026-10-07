"""Build the Kaggle notebook that serves the model, at push time.

The repository only holds ``kaggle_cyber_ornith/cyber_ornith_server.py`` with
``__BT_*__`` placeholders.  Credentials are substituted from the environment of the Render
service when the notebook is pushed, so no secret is ever committed, and there is no
generated ``.ipynb`` copy to drift out of sync.
"""

from __future__ import annotations

import base64
import json
import re
import secrets
from pathlib import Path
from typing import Dict

from blackthorn import config

TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "kaggle_cyber_ornith" / "cyber_ornith_server.py"
_PLACEHOLDER = re.compile(r"__BT_[A-Z0-9_]+__")
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9._~+/=:-]{1,512}$")
CHUNK = 4000


def new_gateway_key() -> str:
    """Per-boot gateway key: the tunnel is protected by a secret that exists nowhere else."""
    return "bt-" + secrets.token_urlsafe(32)


def secret_values(gateway_key: str) -> Dict[str, str]:
    return {
        "__BT_CF_ACCOUNT_ID__": config.cloudflare_account_id(),
        "__BT_CF_DATABASE_ID__": config.cloudflare_database_id(),
        "__BT_CF_API_TOKEN__": config.cloudflare_token(),
        "__BT_GATEWAY_API_KEY__": gateway_key,
        "__BT_KAGGLE_USERNAME__": config.kaggle_username(),
        "__BT_KAGGLE_API_TOKEN__": config.kaggle_token(),
    }


def render_source(template: str, values: Dict[str, str]) -> str:
    for key, value in values.items():
        if not _SAFE_VALUE.match(value or ""):
            raise ValueError(f"refusing to embed {key}: value is empty or contains unsafe characters")
    source = template
    for key, value in values.items():
        source = source.replace(key, value)
    leftover = sorted(set(_PLACEHOLDER.findall(source)))
    if leftover:
        raise ValueError(f"unfilled notebook placeholders: {', '.join(leftover)}")
    return source


_LOADER_HEAD = "import base64, pathlib, runpy\nparts = []\n"
_LOADER_TAIL = (
    "raw = base64.b64decode(''.join(parts))\n"
    "path = pathlib.Path('/kaggle/working/cyber_ornith_server.py')\n"
    "path.write_bytes(raw)\n"
    "path.chmod(0o600)\n"
    "print('Wrote', path, 'bytes', len(raw))\n"
    "runpy.run_path(str(path), run_name='__main__')\n"
)


def build_notebook(source: str) -> str:
    """ipynb JSON with one cell that writes the (base64-chunked) server and runs it."""
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    lines = [f"parts.append({encoded[i:i + CHUNK]!r})\n" for i in range(0, len(encoded), CHUNK)]
    cell = _LOADER_HEAD + "".join(lines) + _LOADER_TAIL
    nb = {
        "cells": [{
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": cell.splitlines(keepends=True),
        }],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb)


def notebook_for_push(gateway_key: str, template: str | None = None) -> str:
    text = template if template is not None else TEMPLATE_PATH.read_text(encoding="utf-8")
    return build_notebook(render_source(text, secret_values(gateway_key)))
