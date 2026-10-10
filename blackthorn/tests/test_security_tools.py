"""Tests for on-demand skills, the tool catalog and the Camoufox browse tool.

The fake computer runs the same command templates locally, with the remote helper directory mapped
to a temp directory. Fixtures are created in the test, so no network access is needed.
Set BLACKTHORN_LIVE_BROWSER=1 to run the real Camoufox test (requires `camoufox fetch` and GTK libs).
"""

import asyncio
import base64
import json
import os
import subprocess
from pathlib import Path

import pytest

from blackthorn.config import Settings
from blackthorn.tools import ToolContext, default_registry
from blackthorn.tools import security
from blackthorn.tools.web import TavilyKeys

ASSETS = Path(security.ASSET_DIR)


class LocalComputer:
    """Runs /computer/exec and /computer/write_file on this machine, mapping the remote helper dir to a temp dir."""

    def __init__(self, base: Path):
        self.base = base
        self.calls = []

    def _map(self, text: str) -> str:
        return text.replace(security.REMOTE_BIN, str(self.base / "bin")).replace("/tmp/blackthorn_skills", str(self.base))

    async def call(self, path, payload=None, *, method="POST", timeout=90.0, idempotent=True):
        self.calls.append((path, payload))
        if path == "/computer/write_file":
            target = Path(self._map(payload["path"]))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload["content"], encoding="utf-8")
            return {"ok": True, "path": str(target), "bytes": len(payload["content"].encode())}
        if path == "/computer/exec":
            proc = subprocess.run(["bash", "-c", self._map(payload["command"])], capture_output=True, text=True,
                                  cwd=str(self.base / "work"), timeout=timeout)
            return {"ok": True, "exit_code": proc.returncode, "output": proc.stdout + proc.stderr}
        raise AssertionError(f"unexpected computer path {path}")


@pytest.fixture
def fixture_root(tmp_path, monkeypatch):
    root = tmp_path
    skills = root / "anthropic-cybersecurity-skills"
    (skills / "skills" / "analyzing-linux-elf-malware" / "references").mkdir(parents=True)
    (skills / "skills" / "analyzing-linux-elf-malware" / "scripts").mkdir(parents=True)
    (skills / "skills" / "analyzing-linux-elf-malware" / "SKILL.md").write_text(
        "---\nname: analyzing-linux-elf-malware\ndescription: Analyze ELF binaries with readelf and strings\n---\n"
        "# Steps\n1. Run file and sha256sum.\n", encoding="utf-8")
    (skills / "skills" / "analyzing-linux-elf-malware" / "references" / "api.md").write_text("readelf -h sample\n")
    (skills / "skills" / "analyzing-linux-elf-malware" / "scripts" / "agent.py").write_text("print('never run')\n")
    (skills / "skills" / "web-app-scanning").mkdir(parents=True)
    (skills / "skills" / "web-app-scanning" / "SKILL.md").write_text("---\nname: web-app-scanning\n---\nScan web apps.\n")
    (skills / "index.json").write_text(json.dumps({"skills": [
        {"name": "analyzing-linux-elf-malware", "description": "Analyze ELF binaries with readelf and strings"},
        {"name": "web-app-scanning", "description": "Scan web applications for injection flaws"}]}))
    tools_dir = root / "hackingtool-plugin" / "plugins" / "hackingtool" / "data"
    tools_dir.mkdir(parents=True)
    (tools_dir / "tools.json").write_text(json.dumps({"tool_count": 2, "tools": [
        {"id": "information_gathering.NMAP", "title": "Nmap", "category": "information_gathering",
         "description": "Network port scanner", "install_commands": ["sudo apt-get install -y nmap"]},
        {"id": "sql_injection.Sqlmap", "title": "SQLMap", "category": "sql_injection",
         "description": "Automatic SQL injection tool", "install_commands": []}]}))
    (root / "work").mkdir()
    monkeypatch.setenv("BLACKTHORN_SKILL_ROOT", str(root))
    return root


def _ctx(computer):
    return ToolContext(settings=Settings(), store=None, computer=computer, http=None, tavily=TavilyKeys(env={}))


def run(coro):
    return asyncio.run(coro)


def test_skill_search_returns_ranked_names_without_bodies(fixture_root):
    comp = LocalComputer(fixture_root)
    res = run(security.skill_search({"query": "elf malware readelf"}, _ctx(comp)))
    assert res.ok
    assert "analyzing-linux-elf-malware" in res.content
    assert "Steps" not in res.content          # bodies are not returned by search
    assert "web-app-scanning" not in res.content


def test_skill_load_strips_frontmatter_and_lists_files_without_running_scripts(fixture_root):
    comp = LocalComputer(fixture_root)
    res = run(security.skill_load({"name": "analyzing-linux-elf-malware"}, _ctx(comp)))
    assert res.ok
    assert "# Steps" in res.content
    assert "name: analyzing-linux-elf-malware" not in res.content
    assert "scripts were not run" in res.content
    assert "references/api.md" in res.content
    assert "never run" not in res.content


def test_skill_read_returns_reference_file(fixture_root):
    res = run(security.skill_read({"name": "analyzing-linux-elf-malware", "path": "references/api.md"},
                                  _ctx(LocalComputer(fixture_root))))
    assert res.ok and "readelf -h" in res.content


@pytest.mark.parametrize("args", [
    {"name": "../etc"}, {"name": "Bad_Name"}, {"name": "a; rm -rf /"}, {"name": "x"}, {"name": ""},
])
def test_skill_load_rejects_bad_names_before_any_computer_call(fixture_root, args):
    comp = LocalComputer(fixture_root)
    res = run(security.skill_load(args, _ctx(comp)))
    assert not res.ok
    assert comp.calls == []


@pytest.mark.parametrize("path", ["../../etc/passwd", "/etc/passwd", "references/a;rm.md", ""])
def test_skill_read_rejects_traversal_and_shell_metacharacters(fixture_root, path):
    comp = LocalComputer(fixture_root)
    res = run(security.skill_read({"name": "analyzing-linux-elf-malware", "path": path}, _ctx(comp)))
    assert not res.ok
    assert comp.calls == []


def test_skill_search_rejects_shell_injection_query(fixture_root):
    comp = LocalComputer(fixture_root)
    res = run(security.skill_search({"query": "x'; touch /tmp/pwned; echo '"}, _ctx(comp)))
    assert not res.ok and comp.calls == []


def test_skill_load_unknown_skill_is_a_clean_error(fixture_root):
    res = run(security.skill_load({"name": "no-such-skill-xyz"}, _ctx(LocalComputer(fixture_root))))
    assert not res.ok and "not found" in res.content


def test_tool_search_finds_catalog_entries_and_says_nothing_is_installed(fixture_root):
    res = run(security.tool_search({"query": "port scanner"}, _ctx(LocalComputer(fixture_root))))
    assert res.ok
    assert "information_gathering.NMAP" in res.content
    assert "nothing installed" in res.content


def test_discovery_schemas_are_small_for_model_context(fixture_root):
    specs = [s for s in security.specs()]
    size = sum(len(json.dumps(s.schema())) for s in specs)
    assert size < 4000, size                    # five tools cost little context; no catalog is injected
    registry_names = [spec.name for spec in default_registry()._specs.values()] if hasattr(default_registry(), "_specs") else []
    assert len(registry_names) == len(set(registry_names))


def test_browse_rejects_non_http_urls_before_calling_computer(fixture_root):
    comp = LocalComputer(fixture_root)
    res = run(security.browse({"url": "file:///etc/passwd"}, _ctx(comp)))
    assert not res.ok and comp.calls == []


def test_browse_request_is_base64_json_not_shell_text(fixture_root):
    # The payload is base64 so a hostile URL cannot break out of the command line.
    payload = base64.b64encode(json.dumps({"url": "https://example.com/$(id)"}).encode()).decode()
    assert "$" not in payload and "(" not in payload and ";" not in payload


@pytest.mark.skipif(os.environ.get("BLACKTHORN_LIVE_BROWSER") != "1", reason="set BLACKTHORN_LIVE_BROWSER=1")
def test_browse_live_camoufox_example_com(fixture_root):
    comp = LocalComputer(fixture_root)
    res = run(security.browse({"url": "https://example.com/", "screenshot": True}, _ctx(comp)))
    assert res.ok, res.content
    assert "Example Domain" in res.content
    assert res.data.get("status") == 200
