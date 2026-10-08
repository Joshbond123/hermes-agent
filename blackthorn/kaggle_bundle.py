"""Build the Kaggle notebook that is pushed on GPU turn-on — always from the server source in Git.

The previous flow pushed a pre-built, base64-wrapped ``.ipynb`` that had silently drifted from its ``.py`` source
and carried hard-coded secrets. Now the notebook is generated at push time from
``kaggle_cyber_ornith/cyber_ornith_server.py`` and the credentials are injected from the Render environment, so
nothing secret is stored in Git and the pushed code is exactly the committed code.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Mapping, Optional

REQUIRED_ENV = ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_D1_DATABASE_ID", "KAGGLE_USERNAME", "KAGGLE_API_TOKEN")
OPTIONAL_ENV = ("QWEN38_NUM_CTX", "QWEN38_QUANT", "QWEN38_CACHE_DATASET",
                "CLOUDFLARED_TUNNEL_TOKEN", "BLACKTHORN_TUNNEL_HOSTNAME", "PREFER_LLAMA")
SERVER_SOURCE = Path(__file__).resolve().parent.parent / "kaggle_cyber_ornith" / "cyber_ornith_server.py"


class BundleError(RuntimeError):
    pass


def build_notebook_text(server_py: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> str:
    source = server_py if server_py is not None else SERVER_SOURCE.read_text(encoding="utf-8")
    environ = env if env is not None else os.environ
    missing = [k for k in REQUIRED_ENV if not environ.get(k)]
    if missing:
        raise BundleError("cannot build the Kaggle notebook: missing environment variables: " + ", ".join(missing))
    injected = {k: environ[k] for k in (*REQUIRED_ENV, *OPTIONAL_ENV) if environ.get(k)}
    payload = base64.b64encode(source.encode("utf-8")).decode("ascii")
    chunks = [payload[i:i + 4000] for i in range(0, len(payload), 4000)]
    env_b64 = base64.b64encode(json.dumps(injected).encode("utf-8")).decode("ascii")
    cell = [
        "import base64, json, os, pathlib, runpy\n",
        f"os.environ.update(json.loads(base64.b64decode('{env_b64}')))\n",
        "parts = []\n",
        *[f"parts.append('{c}')\n" for c in chunks],
        "raw = base64.b64decode(''.join(parts))\n",
        "path = pathlib.Path('/kaggle/working/cyber_ornith_server.py')\n",
        "path.write_bytes(raw)\n",
        "print('Wrote', path, 'bytes', len(raw))\n",
        "runpy.run_path(str(path), run_name='__main__')\n",
    ]
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 4,
    }
    return json.dumps(notebook)
