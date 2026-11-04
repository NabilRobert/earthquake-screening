"""Reading ESM files, pinning their checksums, and crediting their networks, without the network."""

import zipfile
from pathlib import Path

import numpy as np
import pytest

from earthquake_screening.simulation.esm import SelectedComponent
from earthquake_screening.simulation.esm_records import (
    ChecksumMismatchError,
    attribution_markdown,
    download_recording,
    parse_dyna_ascii,
    read_checksums,
    read_component,
    write_checksums,
)

SELECTED = SelectedComponent("EV-1", "2009-04-06T01:32:39", 6.1, 4.6, "IT", "AQK", "", "HNE", "BEST")


def _dyna_file(samples: list[float], units: str = "cm/s^2", declared: int | None = None) -> str:
    header = [
        "EVENT_ID: EV-1",
        "NETWORK: IT",
        "SAMPLING_INTERVAL_S: 0.005000",
        f"NDATA: {len(samples) if declared is None else declared}",
        f"UNITS: {units}",
        "DATA_LICENSE: CC-BY4_0 (http://creativecommons.org/licenses/by/4.0/)",
        "DATA_CITATION: ESM 3.0, https://doi.org/10.13127/esm.3",
        "ORIGINAL_DATA_CREATOR_CITATION: Italian Strong Motion Network",
        "USER1: ",
    ]
    return "\n".join(header + [f"{sample:.6f}" for sample in samples]) + "\n"


def _write_recording(records_dir: Path, east: str) -> None:
    with zipfile.ZipFile(records_dir / "EV-1.IT.AQK..zip", "w") as archive:
        archive.writestr("IT.AQK..HNE.D.EV-1.ACC.MP.ASC", east)
        archive.writestr("IT.AQK..HNN.D.EV-1.ACC.MP.ASC", _dyna_file([0.0]))


def test_header_and_samples_are_split_at_the_first_line_without_a_colon() -> None:
    header, samples = parse_dyna_ascii(_dyna_file([0.0, 1.5, -2.25]))
    assert header["UNITS"] == "cm/s^2" and header["USER1"] == ""
    assert samples.tolist() == [0.0, 1.5, -2.25]


def test_a_component_is_read_from_its_zip_in_metres_per_second_squared(tmp_path: Path) -> None:
    _write_recording(tmp_path, _dyna_file([0.0, 150.0, -50.0]))
    component = read_component(SELECTED, tmp_path)
    assert np.allclose(component.acceleration, [0.0, 1.5, -0.5])
    assert component.time_step == 0.005


def test_a_file_in_other_units_is_refused(tmp_path: Path) -> None:
    _write_recording(tmp_path, _dyna_file([0.0, 1.0], units="g"))
    with pytest.raises(ValueError, match="expected cm/s"):
        read_component(SELECTED, tmp_path)


def test_a_truncated_file_is_refused(tmp_path: Path) -> None:
    _write_recording(tmp_path, _dyna_file([0.0, 1.0], declared=3))
    with pytest.raises(ValueError, match="header says 3 samples"):
        read_component(SELECTED, tmp_path)


def test_a_file_that_differs_from_its_pinned_checksum_is_refused(tmp_path: Path) -> None:
    _write_recording(tmp_path, _dyna_file([0.0, 1.0]))
    with pytest.raises(ChecksumMismatchError, match="reprocessed"):
        download_recording(SELECTED, tmp_path, pinned={"EV-1.IT.AQK..zip": "0" * 64})


def test_checksums_round_trip_through_their_file(tmp_path: Path) -> None:
    checksums = {"b.zip": "2" * 64, "a.zip": "1" * 64}
    write_checksums(checksums, tmp_path / "records.sha256")
    assert read_checksums(tmp_path / "records.sha256") == checksums


def test_attribution_credits_each_network_and_counts_the_stated_licences(tmp_path: Path) -> None:
    _write_recording(tmp_path, _dyna_file([0.0, 1.0]))
    attribution = attribution_markdown([read_component(SELECTED, tmp_path)] * 2)
    assert "- IT: Italian Strong Motion Network" in attribution
    assert "CC-BY4_0 (http://creativecommons.org/licenses/by/4.0/) (2 components)" in attribution
