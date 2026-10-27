"""Local invariants only. Actual GitHub upload/download is verified separately."""

import pytest

from blackthorn.computer_drive import GitHubComputerDrive, _safe_path


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
