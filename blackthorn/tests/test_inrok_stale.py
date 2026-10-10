"""Stale Inrok shares hold tunnel slots (beta limit: 3 per account). A boot must release only its own role's
stale names, never another role's live share and never the current boot's name."""

import ast
from pathlib import Path

SERVER = Path(__file__).resolve().parents[2] / "kaggle_cyber_ornith" / "cyber_ornith_server.py"

STATUS = (
    "NAME                        URL                                                TARGET                 STATE\n"
    "blackthorn-computer-2e182e  https://blackthorn-computer-2e182e.share.inrok.in  http://localhost:8000  ONLINE\n"
    "blackthorn-computer-4606ec  https://blackthorn-computer-4606ec.share.inrok.in  http://localhost:8000  ONLINE\n"
    "blackthorn                  https://blackthorn.share.inrok.in                  http://localhost:8000  ONLINE\n"
)


def _selector():
    import re
    src = SERVER.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stale_inrok_names")
    ns = {"re": re, "_INROK_MAX_STALE_RELEASE": 6}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SERVER), "exec"), ns)
    return ns["stale_inrok_names"]


def test_model_boot_releases_only_model_names_and_not_the_live_computer():
    sel = _selector()
    assert sel(STATUS, "model", "blackthorn-a1b2c3") == ["blackthorn"]


def test_computer_boot_releases_stale_computer_share_but_not_its_own_name():
    sel = _selector()
    assert sel(STATUS, "computer", "blackthorn-computer-2e182e") == ["blackthorn-computer-4606ec"]


def test_probe_and_foreign_names_are_never_selected():
    sel = _selector()
    text = STATUS + "blackthorn-probe-1  https://x  http://localhost:8799  ONLINE\nother-app  https://y  z  ONLINE\n"
    assert "blackthorn-probe-1" not in sel(text, "model", "blackthorn-a1b2c3")
    assert "other-app" not in sel(text, "model", "blackthorn-a1b2c3")


def test_empty_or_header_only_status_selects_nothing():
    sel = _selector()
    assert sel("", "model", "blackthorn-a1b2c3") == []
    assert sel(STATUS.splitlines()[0], "computer", "blackthorn-computer-2e182e") == []


def test_release_is_bounded():
    sel = _selector()
    rows = "NAME X\n" + "".join(f"blackthorn-computer-{i:06x}  u  t  ONLINE\n" for i in range(20))
    assert len(sel(rows, "computer", "blackthorn-computer-ffffff")) == 6
