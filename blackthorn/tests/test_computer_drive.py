"""Local invariants only. Actual GitHub upload/download is verified separately."""

import pytest

import json

from blackthorn.computer_drive import GitHubComputerDrive, MARKER, _safe_path


@pytest.mark.parametrize("path", ["", "/absolute", "../parent", "a/../b", "a//b", "a\\b", "./a"])
def test_rejects_unsafe_logical_paths(path):
    with pytest.raises(ValueError):
        _safe_path(path)


def test_allows_normal_relative_paths():
    assert _safe_path("skills/tools/setup.sh") == "skills/tools/setup.sh"


def test_rejects_bad_repo_and_chunk_size():
    with pytest.raises(ValueError):
        GitHubComputerDrive("not-a-repo", token="not-a-real-token")
    with pytest.raises(ValueError):
        GitHubComputerDrive("owner/repo", token="not-a-real-token", chunk_bytes=2 * 1024**3)


def test_newest_published_release_wins_even_if_list_order_is_stale(monkeypatch):
    def release(rid, published, size, path="_system/blackthorn-workspace.tar.gz"):
        return {"id": rid, "created_at": "2026-10-26T19:42:51Z", "published_at": published,
                "draft": False, "body": json.dumps({"format": MARKER, "path": path,
                                            "size": size, "sha256": str(size), "chunks": []})}

    # Real GitHub returned the 09:48 snapshot *before* the 09:52 version.
    older = release(20, "2026-10-27T09:48:52Z", 238935)
    newer = release(22, "2026-10-27T09:52:20Z", 239036)
    middle = release(21, "2026-10-27T09:49:06Z", 238935)
    other = release(24, "2026-10-27T09:50:00Z", 43, "validation/proof.txt")
    same_second_newer = release(23, "2026-10-27T09:52:20Z", 239037)
    drive = object.__new__(GitHubComputerDrive)
    monkeypatch.setattr(drive, "_releases", lambda: iter([older, newer, middle, other, same_second_newer]))
    found, manifest = drive._find("_system/blackthorn-workspace.tar.gz")
    assert found["id"] == 23 and manifest["size"] == 239037
    files = drive.list_files()
    assert files["_system/blackthorn-workspace.tar.gz"]["release_id"] == 23
    assert files["validation/proof.txt"]["size"] == 43
    with pytest.raises(FileNotFoundError):
        drive._find("missing/file.txt")


def test_absolute_workspace_paths_are_the_same_file_as_logical_ones():
    """The agent writes /home/user/work/x.py; the checkpoint publishes /workspace.

    ``_prepare_home`` makes the two names one directory, so write_file may accept either and the
    file is still in the durable checkpoint. Rejecting it (\"unsafe logical path\") only taught the
    model to work around the tool with a heredoc and left the file outside the mirror.
    """
    from blackthorn.panda_computer import PandaComputerClient as C
    for given, expected in (("/home/user/work/audit2/x.py", "work/audit2/x.py"),
                            ("/workspace/work/audit2/x.py", "work/audit2/x.py"),
                            ("work/audit2/x.py", "work/audit2/x.py")):
        assert C._relative(given) == expected
    import pytest
    for bad in ("/etc/passwd", "../../escape", "", "/workspace", "/home/user"):
        with pytest.raises(ValueError):
            C._relative(bad)


@pytest.mark.asyncio
async def test_checkpoint_prepares_the_durable_name_before_publishing():
    """A checkpoint publishes /workspace; files in a plain /home/user must be inside it first."""
    from blackthorn.panda_computer import PandaComputerClient as C

    computer = object.__new__(C)
    order = []

    async def prepare():
        order.append("prepare")

    class Mirror:
        async def sync(self, force=False):
            order.append("sync")
            return True

    computer.mirror = Mirror()
    computer._prepare_home = prepare
    import contextlib
    computer.store = None
    computer.sandbox_id = "abc"
    result = await C._call(computer, "/computer/checkpoint", {}, method="POST", timeout=60)
    assert result["published"] is True
    assert order == ["prepare", "sync"], order
