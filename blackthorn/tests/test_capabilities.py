"""The drive-hosted capability catalog: build, publish, restore, verify, survive rotation."""

import asyncio
import io
import json
import zipfile

import pytest

from blackthorn import capabilities as caps


def _repo_zip(top: str, files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, content in files.items():
            archive.writestr(f"{top}/{name}", content)
    return buf.getvalue()


FIXTURES = {
    "mukul975/Anthropic-Cybersecurity-Skills": _repo_zip("Anthropic-Cybersecurity-Skills-main", {
        "index.json": json.dumps({"skills": [{"name": "analyzing-linux-elf-malware"}]}),
        "analyzing-linux-elf-malware/SKILL.md": "---\nname: analyzing-linux-elf-malware\n"
                                                "description: Triage an ELF sample\n---\nbody",
        "assets/huge.png": "x" * 5000,          # excluded: assets are never needed at run time
        "README.md": "# skills",
    }),
    "SnailSploit/Claude-Red": _repo_zip("Claude-Red-main", {
        "Skills/kerberoasting/SKILL.md": "---\nname: kerberoasting\ndescription: Kerberos attacks\n---\nbody",
    }),
    "AKCodez/hackingtool-plugin": _repo_zip("hackingtool-plugin-main", {
        "plugins/hackingtool/data/tools.json": json.dumps({"tools": [{"name": "nmap", "run": "nmap"}]}),
    }),
}


def fetch(repo: str) -> bytes:
    return FIXTURES[repo]


class FakeDrive:
    """In-memory stand-in for GitHubComputerDrive (release assets)."""

    store: dict = {}

    def __init__(self, repo, token=None):
        self.repo = repo

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def upload(self, source, destination):
        FakeDrive.store[destination] = open(source, "rb").read()
        return {"destination": destination}

    def download(self, path, target):
        data = FakeDrive.store[path]
        with open(target, "wb") as fh:
            fh.write(data)
        return {"path": path}


class FakeVM:
    """A VM that keeps files and shells out to a tiny command router."""

    def __init__(self, *, version: str = "", catalog_ok: bool = True):
        self.extract_commands = []
        self.files = {}
        self.commands = []
        self.version = version
        self.catalog_ok = catalog_ok
        self.kind = "vm"

    async def put(self, path, data):
        self.files[path] = data

    async def run(self, command, timeout=300):
        self.commands.append(command)
        if "tar xzf" in command:
            self.extract_commands.append(command)
        if command.startswith("cat ") and "VERSION" in command:
            return {"stdout": self.version + "\n" if self.version else "", "stderr": "", "exit_code": 0}
        if "tar xzf" in command:
            return {"stdout": "restored\n", "stderr": "", "exit_code": 0, "ok": True}
        if "skillctl.py registry" in command:
            payload = {"counts": {"skills": 897, "tools": 183}} if self.catalog_ok else {"counts": {}}
            return {"stdout": json.dumps(payload) + "\n", "stderr": "", "exit_code": 0}
        if "skillctl.py search" in command:
            results = [{"id": "anthropic:analyzing-linux-elf-malware"}] if self.catalog_ok else []
            return {"stdout": json.dumps({"results": results}) + "\n", "stderr": "", "exit_code": 0}
        return {"stdout": "", "stderr": "", "exit_code": 0}


@pytest.fixture(autouse=True)
def _drive(monkeypatch):
    FakeDrive.store = {}
    monkeypatch.setattr(caps.CapabilityManager, "_drive", lambda self: FakeDrive(self.repo, self.token))
    yield


def _manager(tmp_path, **kw):
    return caps.CapabilityManager("jadeniran21/Computer-drive", "token",
                                  cache_dir=str(tmp_path), fetch_repo=fetch, **kw)


def test_bundle_contains_the_three_repositories_and_skips_assets(tmp_path):
    bundle = caps.build_bundle(fetch)
    import tarfile
    with tarfile.open(fileobj=io.BytesIO(bundle)) as tar:
        names = tar.getnames()
    assert "anthropic-cybersecurity-skills/analyzing-linux-elf-malware/SKILL.md" in names
    assert "claude-red/Skills/kerberoasting/SKILL.md" in names
    assert "hackingtool-plugin/plugins/hackingtool/data/tools.json" in names
    assert not any("assets/" in n for n in names), "binary assets must not bloat the catalog"
    assert not any(n.endswith(".png") for n in names)


def test_publish_uploads_the_bundle_and_metadata_and_is_idempotent(tmp_path):
    manager = _manager(tmp_path)
    first = manager.publish()
    assert first["changed"] is True
    assert caps.DRIVE_BUNDLE in FakeDrive.store and caps.DRIVE_META in FakeDrive.store
    second = manager.publish()
    assert second["changed"] is False, "an unchanged upstream must not re-upload gigabytes"
    assert second["version"] == first["version"]


def test_ensure_restores_the_catalog_onto_a_fresh_vm(tmp_path):
    manager = _manager(tmp_path)
    manager.publish()
    vm = FakeVM()
    result = asyncio.run(manager.ensure(vm))
    assert result["restored"] is True
    assert result["ok"] is True and result["skills"] == 897 and result["tools"] == 183
    assert caps.BUNDLE_PATH in vm.files, "the bundle must be pushed to the VM"
    assert caps.BIN_DIR + "/skillctl.py" in vm.files, "the catalog reader is restored too"
    assert any("tar xzf" in c for c in vm.commands)


def test_ensure_is_a_fast_no_op_when_the_vm_already_has_the_version(tmp_path):
    manager = _manager(tmp_path)
    meta = manager.publish()
    vm = FakeVM(version=str(meta["version"]))
    result = asyncio.run(manager.ensure(vm))
    assert result["restored"] is False
    assert not any("tar xzf" in c for c in vm.commands), "no re-extraction when already current"


def test_verification_fails_loudly_when_the_catalog_cannot_index(tmp_path):
    manager = _manager(tmp_path)
    manager.publish()
    result = asyncio.run(manager.ensure(FakeVM(catalog_ok=False)))
    assert result["ok"] is False
    assert result.get("error")


def test_a_replacement_vm_gets_the_same_catalog_after_rotation(tmp_path):
    """Requirement 1/8: the drive — not the VM — is what makes the catalog durable."""
    manager = _manager(tmp_path)
    manager.publish()
    original = FakeVM()
    asyncio.run(manager.ensure(original))
    version = manager._version

    replacement = FakeVM()                    # a brand new VM with no files at all
    result = asyncio.run(manager.ensure(replacement))
    assert result["restored"] is True
    assert result["version"] == version
    assert replacement.files[caps.BUNDLE_PATH] == original.files[caps.BUNDLE_PATH]


def test_lab_vm_target_uses_the_lab_transport(tmp_path):
    manager = _manager(tmp_path)
    manager.publish()

    class FakeLab:
        def __init__(self):
            self.puts, self.execs = {}, []

        async def _put(self, path, data, sid=""):
            self.puts[path] = data

        async def _exec(self, command, *, timeout=120, sid=""):
            self.execs.append(command)
            if "tar xzf" in command:
                return {"stdout": "restored", "exit_code": 0}
            if "cat /opt/blackthorn/catalog/VERSION" in command:
                return {"stdout": "", "exit_code": 0}
            if "registry" in command:
                return {"stdout": json.dumps({"counts": {"skills": 5, "tools": 6}}), "exit_code": 0}
            if "search" in command:
                return {"stdout": json.dumps({"results": [{"id": "claude-red:x"}]}), "exit_code": 0}
            return {"stdout": "restored", "exit_code": 0}

    lab = FakeLab()
    result = asyncio.run(manager.ensure((lab, "sandbox-1"), kind="lab"))
    assert result["ok"] is True and result["kind"] == "lab"
    assert caps.BUNDLE_PATH in lab.puts
    assert all(c for c in lab.execs)


def test_the_bundle_is_extracted_into_the_catalog_root_not_beside_it(tmp_path):
    """A live restore failed once because the members landed in /opt/blackthorn/<repo>.

    skillctl resolves skills from the catalog root, so the extraction target is part of
    the contract: extract with -C <catalog>, then write VERSION inside it.
    """
    manager = _manager(tmp_path)
    manager.publish()
    vm = FakeVM()
    asyncio.run(manager.ensure(vm))
    joined = " ".join(vm.extract_commands)
    assert "tar xzf" in joined and f"-C {caps.CATALOG_DIR}" in joined, joined
    assert f"rm -rf {caps.CATALOG_DIR}" in joined, "a stale catalog must be replaced, not merged"
    assert caps.VERSION_FILE in joined
