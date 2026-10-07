"""The notebook that runs on Kaggle: no committed secrets, correct packaging, sane gateway."""

from __future__ import annotations

import ast
import base64
import json
import re
import types
from pathlib import Path

import pytest

from blackthorn import notebook

TEMPLATE = Path(notebook.TEMPLATE_PATH)
VALUES = {
    "__BT_CF_ACCOUNT_ID__": "acc123", "__BT_CF_DATABASE_ID__": "db-456", "__BT_CF_API_TOKEN__": "tok_ABC123",
    "__BT_GATEWAY_API_KEY__": "bt-gatewaykey", "__BT_KAGGLE_USERNAME__": "someone", "__BT_KAGGLE_API_TOKEN__": "kg_XYZ",
}


def test_template_in_the_repo_contains_only_placeholders_no_credentials():
    text = TEMPLATE.read_text()
    assert sorted(set(re.findall(r"__BT_[A-Z0-9_]+__", text))) == sorted(VALUES)
    for pattern in (r"cfut_[A-Za-z0-9]{10,}", r"KGAT_[A-Za-z0-9]{10,}", r"sk-qwen38[-A-Za-z0-9]+", r"f932764f168d0f77",
                    r"43090570-0d1a", r"ntfy\.sh"):
        assert not re.search(pattern, text), f"committed template must not contain {pattern}"


def test_render_source_fills_every_placeholder():
    out = notebook.render_source(TEMPLATE.read_text(), VALUES)
    assert "__BT_" not in out and 'API_KEY = "bt-gatewaykey"' in out and 'CF_API_TOKEN = "tok_ABC123"' in out
    ast.parse(out)


def test_render_refuses_unsafe_or_missing_values():
    bad = dict(VALUES, __BT_CF_API_TOKEN__='x"; import os; os.system("id") #')
    with pytest.raises(ValueError):
        notebook.render_source(TEMPLATE.read_text(), bad)
    with pytest.raises(ValueError):
        notebook.render_source(TEMPLATE.read_text(), dict(VALUES, __BT_CF_API_TOKEN__=""))
    partial = dict(VALUES)
    partial.pop("__BT_KAGGLE_API_TOKEN__")
    with pytest.raises(ValueError, match="unfilled"):
        notebook.render_source(TEMPLATE.read_text(), partial)


def test_built_notebook_is_valid_ipynb_and_round_trips_the_server_source():
    source = notebook.render_source(TEMPLATE.read_text(), VALUES)
    nb = json.loads(notebook.build_notebook(source))
    assert nb["nbformat"] == 4 and len(nb["cells"]) == 1 and nb["cells"][0]["cell_type"] == "code"
    cell = "".join(nb["cells"][0]["source"])
    parts = re.findall(r"parts\.append\('([^']*)'\)", cell)
    assert base64.b64decode("".join(parts)).decode() == source
    assert "path.chmod(0o600)" in cell and "runpy.run_path" in cell
    ast.parse(cell)
    assert "tok_ABC123" not in cell, "the loader cell must not carry the credentials in the clear"


def test_gateway_key_is_fresh_and_strong():
    keys = {notebook.new_gateway_key() for _ in range(50)}
    assert len(keys) == 50 and all(k.startswith("bt-") and len(k) > 40 for k in keys)


def _load_module():
    src = notebook.render_source(TEMPLATE.read_text(), VALUES)
    mod = types.ModuleType("nb")
    exec(compile(src, "nb", "exec"), mod.__dict__)
    return mod


def test_server_configuration_fixes():
    mod = _load_module()
    assert mod.DEFAULT_NUM_CTX >= 16384, "4096 tokens cannot hold the agent's prompt + tools + a tool result"
    gateway = mod.gateway_code()
    compile(gateway, "gw", "exec")
    assert "start_new_session=True" in gateway and "killpg" in gateway and "is_disconnected" in gateway
    src = TEMPLATE.read_text()
    assert '"--jinja"' in src, "native tool calling needs --jinja on llama.cpp"
    assert '"--retry-all-errors"' not in src  # a 404 must not be retried for minutes


def test_gateway_workspace_paths_do_not_nest():
    mod = _load_module()
    ns = {}
    # execute only the _csafe helper from the generated gateway source
    gw = mod.gateway_code()
    start = gw.index("def _csafe")
    end = gw.index("@app.get(\"/computer/info\")")
    import pathlib as _pathlib
    from fastapi import HTTPException
    ns.update(_pathlib=_pathlib, HTTPException=HTTPException, COMPUTER_ROOT="/tmp/btws")
    exec(gw[start:end], ns)
    root = _pathlib.Path("/tmp/btws").resolve()
    assert ns["_csafe"]("/tmp/btws/fib.py") == root / "fib.py"      # absolute path inside the workspace
    assert ns["_csafe"]("fib.py") == root / "fib.py"
    assert ns["_csafe"]("/tmp/btws") == root
    assert ns["_csafe"](".") == root
    with pytest.raises(HTTPException):
        ns["_csafe"]("../../etc/passwd")


def test_status_reports_go_to_d1_only_and_include_progress_detail():
    mod = _load_module()
    sent = []

    def fake_write(status, tunnel, extra, stage_started):
        sent.append((status, tunnel, extra, stage_started))

    mod._d1_write = fake_write
    mod.notify_workspace("DOWNLOADING_MODEL", extra={"detail": {"label": "x", "done": 5, "total": 10}})
    mod.notify_workspace("DOWNLOADING_MODEL", extra={"detail": {"label": "x", "done": 6, "total": 10}})
    mod.notify_workspace("TUNNEL_ONLINE", tunnel_url="https://t.trycloudflare.com")
    mod.notify_workspace("WARMING_GPU")  # tunnel URL is sticky
    assert [s[0] for s in sent] == ["DOWNLOADING_MODEL", "DOWNLOADING_MODEL", "TUNNEL_ONLINE", "WARMING_GPU"]
    assert sent[0][3] == sent[1][3], "stage_started must not reset while the stage is unchanged"
    assert sent[2][3] >= sent[1][3]
    assert sent[3][1] == "https://t.trycloudflare.com"


def test_measured_progress_is_throttled():
    mod = _load_module()
    sent = []
    mod._d1_write = lambda status, tunnel, extra, started: sent.append(extra["detail"])
    for done in range(0, 100, 10):
        mod.report_progress("DOWNLOADING_MODEL", "dl", done, 100)
    assert len(sent) == 1, "reports are rate-limited (D1 writes cost round trips)"
    mod.report_progress("DOWNLOADING_MODEL", "dl", 100, 100)  # completion is always reported
    assert sent[-1]["done"] == 100.0 and sent[-1]["total"] == 100.0
