"""Fetching PRJ-2048: only the needed files, under the approved size, checked against pinned checksums.

DesignSafe is replaced by a dictionary, so these tests use no network.
"""

from pathlib import Path

import pytest

from earthquake_screening import prj2048
from earthquake_screening.simulation.esm_records import read_checksums, sha256, write_checksums

FAKE_PROJECT = {
    "": [("ReadMe.txt", 3), ("DesignSummary.csv", 3), ("SomethingElse.csv", 3)],
    "StructuralResponses/EDPsUnder240GMs/PeakStoryDrift": [("StructuralResponses/x/Building_1_PSDR.csv", 3)],
    "GroundMotions/MirandaGroundMotions/Histories": [("GroundMotions/x/1.txt", 3)],
    "GroundMotions/MirandaGroundMotions/GroundMotionInfo": [],
}


@pytest.fixture
def fake_designsafe(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    downloads: list[str] = []

    def download(path: str) -> bytes:
        downloads.append(path)
        return b"abc"

    def list_folder(folder: str) -> list[tuple[str, int]]:
        return FAKE_PROJECT[folder]

    monkeypatch.setattr(prj2048, "list_folder", list_folder)
    monkeypatch.setattr(prj2048, "download_file", download)
    return downloads


def test_only_the_needed_files_are_fetched_and_pinned(tmp_path: Path, fake_designsafe: list[str]) -> None:
    prj2048.fetch(tmp_path / "prj2048", tmp_path / "prj2048.sha256")
    assert sorted(fake_designsafe) == sorted(
        ["ReadMe.txt", "DesignSummary.csv", "StructuralResponses/x/Building_1_PSDR.csv", "GroundMotions/x/1.txt"]
    )
    assert set(read_checksums(tmp_path / "prj2048.sha256").values()) == {sha256(b"abc")}


def test_files_already_here_are_not_fetched_again(tmp_path: Path, fake_designsafe: list[str]) -> None:
    prj2048.fetch(tmp_path / "prj2048", tmp_path / "prj2048.sha256")
    fake_designsafe.clear()
    assert prj2048.fetch(tmp_path / "prj2048", tmp_path / "prj2048.sha256") == 0
    assert fake_designsafe == []


@pytest.mark.usefixtures("fake_designsafe")
def test_a_file_that_changed_since_it_was_pinned_is_refused(tmp_path: Path) -> None:
    write_checksums({"ReadMe.txt": "0" * 64}, tmp_path / "prj2048.sha256")
    with pytest.raises(RuntimeError, match="differ from their pinned checksums"):
        prj2048.fetch(tmp_path / "prj2048", tmp_path / "prj2048.sha256")


def test_more_than_the_approved_size_is_refused_before_downloading(
    tmp_path: Path, fake_designsafe: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prj2048, "MAX_DOWNLOAD_BYTES", 5)
    with pytest.raises(RuntimeError, match="past the approved cap"):
        prj2048.fetch(tmp_path / "prj2048", tmp_path / "prj2048.sha256")
    assert fake_designsafe == []


def test_a_link_that_is_not_https_is_refused() -> None:
    with pytest.raises(ValueError, match="non-https"):
        prj2048._get("file:///etc/passwd")  # pyright: ignore[reportPrivateUsage]
