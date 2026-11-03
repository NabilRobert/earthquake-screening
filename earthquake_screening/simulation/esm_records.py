"""Download the selected ESM recordings, pin their checksums, and read their components.

- Each recording (three components) comes from ESM's event-data service as a zip of
  DYNA 1.2 ASCII files, in cm/s^2. The zips are kept as downloaded in data/records/,
  which git ignores; the repo carries only the list of what to fetch
  (data/esm_selection.csv) and their checksums (data/esm_records.sha256).
- ESM as a database is CC BY 4.0, but each file's header gives the license of its own
  waveform ("U", unknown, for the ones checked) and the network that recorded it. So
  the files are used here and never redistributed, and every network is credited in
  data/ESM_ATTRIBUTION.md.
"""

import hashlib
import io
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from earthquake_screening.simulation.esm import SelectedComponent
from earthquake_screening.simulation.ground_motion import FloatArray

EVENT_DATA_URL = "https://esm-db.eu/esmws/eventdata/1/query"
CENTIMETRES_PER_METRE = 100.0
CITATION_FIELD = "ORIGINAL_DATA_CREATOR_CITATION"
LICENSE_FIELD = "DATA_LICENSE"


class ChecksumMismatchError(RuntimeError):
    """A downloaded file differs from the pinned one: ESM may have reprocessed it."""


@dataclass(frozen=True)
class Component:
    """One horizontal component, in SI units."""

    record_id: str
    acceleration: FloatArray  # m/s^2
    time_step: float  # s
    header: dict[str, str]

    @property
    def duration_s(self) -> float:
        return self.acceleration.size * self.time_step


def recording_id(component: SelectedComponent) -> str:
    """The recording a component belongs to: the record id without the channel."""
    return component.record_id.rsplit(".", 1)[0]


def recording_url(component: SelectedComponent) -> str:
    query = {
        "eventid": component.event_id,
        "network": component.network,
        "station": component.station,
        "format": "ascii",
        "processing-type": "MP",
        "data-type": "ACC",
    }
    return f"{EVENT_DATA_URL}?{urllib.parse.urlencode(query)}"


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_checksums(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    pairs = (line.split("  ", 1) for line in path.read_text().splitlines() if line.strip())
    return {name: digest for digest, name in pairs}


def write_checksums(checksums: dict[str, str], path: Path) -> None:
    path.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())))


def download_recording(component: SelectedComponent, records_dir: Path, pinned: dict[str, str]) -> tuple[str, int]:
    """Fetch one recording's zip unless it's already here; returns (file name, bytes fetched)."""
    name = f"{recording_id(component)}.zip"
    target = records_dir / name
    if target.exists():
        content, fetched = target.read_bytes(), 0
    else:
        # The URL is built from EVENT_DATA_URL, a fixed https address, so no other scheme can reach here.
        with urllib.request.urlopen(recording_url(component), timeout=300) as response:  # noqa: S310
            content = response.read()
        fetched = len(content)
    if name in pinned and pinned[name] != sha256(content):
        raise ChecksumMismatchError(
            f"{name} doesn't match its pinned checksum: ESM may have reprocessed it. "
            "Compare the files, then update data/esm_records.sha256 on purpose."
        )
    if fetched:
        target.write_bytes(content)
    return name, fetched


def parse_dyna_ascii(text: str) -> tuple[dict[str, str], FloatArray]:
    """Header lines are 'KEY: value'; the samples follow, one per line."""
    header: dict[str, str] = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        key, separator, value = line.partition(":")
        if not separator:
            samples = np.array([float(sample) for sample in lines[index:] if sample.strip()])
            return header, samples
        header[key.strip()] = value.strip()
    raise ValueError("no samples after the header: the file is empty or truncated")


def read_component(component: SelectedComponent, records_dir: Path) -> Component:
    """One horizontal component from its recording's zip, converted to m/s^2."""
    with zipfile.ZipFile(records_dir / f"{recording_id(component)}.zip") as archive:
        matches = [name for name in archive.namelist() if f".{component.channel}." in name]
        if len(matches) != 1:
            raise ValueError(f"expected one file for {component.record_id} in its zip, found {matches}")
        header, samples = parse_dyna_ascii(io.TextIOWrapper(archive.open(matches[0]), "utf-8").read())
    if header.get("UNITS") != "cm/s^2":
        raise ValueError(f"{component.record_id} is in {header.get('UNITS')!r}, expected cm/s^2")
    if int(header["NDATA"]) != samples.size:
        raise ValueError(f"{component.record_id}: header says {header['NDATA']} samples, file has {samples.size}")
    time_step = float(header["SAMPLING_INTERVAL_S"])
    return Component(component.record_id, samples / CENTIMETRES_PER_METRE, time_step, header)


# Arthur approved about 46 MB on 2026-11-03 and asked to stop and ask well past that.
MAX_DOWNLOAD_BYTES = 55_000_000


class DownloadCapError(RuntimeError):
    """The download went past what Arthur approved."""


def fetch_recordings(selected: list[SelectedComponent], records_dir: Path, checksum_path: Path) -> int:
    """Fetch every selected recording once, check or pin its checksum; returns bytes fetched."""
    records_dir.mkdir(parents=True, exist_ok=True)
    pinned = read_checksums(checksum_path)
    checksums = dict(pinned)
    total_fetched = 0
    for recording in sorted({recording_id(component): component for component in selected}.values(), key=recording_id):
        name, fetched = download_recording(recording, records_dir, pinned)
        checksums[name] = sha256((records_dir / name).read_bytes())
        total_fetched += fetched
        if total_fetched > MAX_DOWNLOAD_BYTES:
            write_checksums(checksums, checksum_path)
            raise DownloadCapError(
                f"fetched {total_fetched / 1e6:.1f} MB, past the approved cap of {MAX_DOWNLOAD_BYTES / 1e6:.0f} MB: "
                "ask Arthur before fetching the rest"
            )
    write_checksums(checksums, checksum_path)
    return total_fetched


def attribution_markdown(components: list[Component]) -> str:
    """Credit ESM and every network whose waveforms were used, from the files' own headers."""
    networks: dict[str, str] = {}
    licences: dict[str, int] = {}
    for component in components:
        networks[component.header["NETWORK"]] = component.header.get(CITATION_FIELD) or "not given in the file"
        licence = component.header.get(LICENSE_FIELD) or "not given"
        licences[licence] = licences.get(licence, 0) + 1
    lines = [
        "# ESM records: attribution",
        "",
        "Generated by `earthquake-screening fit-records` from the downloaded files' headers.",
        "The waveforms are used for research here and are not redistributed.",
        "",
        "**Database (CC BY 4.0):** " + components[0].header.get("DATA_CITATION", "ESM, https://doi.org/10.13127/esm.3"),
        "",
        "**Licence stated in each file:** "
        + ", ".join(f"{licence} ({count} components)" for licence, count in sorted(licences.items())),
        "",
        "**Networks that recorded them:**",
        "",
        *(f"- {network}: {citation}" for network, citation in sorted(networks.items())),
    ]
    return "\n".join(lines) + "\n"
