"""Fetch the parts of DesignSafe PRJ-2048 that slice 2 needs, with pinned checksums.

PRJ-2048 (Guan, Burton and Shokrabadi, 2020; Open Data Commons Attribution) holds 621
steel moment frames and their responses to 240 ground motions. Slice 2 needs the
design summary, the peak storey drifts and the ground motions (about 37 MB). The
designs and OpenSees models aren't fetched.

DesignSafe serves each public file through a one-time link from its preview endpoint.
Files go to data/prj2048/ (ignored by git); data/prj2048.sha256 pins them.
"""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from earthquake_screening.simulation.esm_records import read_checksums, sha256, write_checksums

API = "https://www.designsafe-ci.org/api/datafiles/tapis/public"
SYSTEM = "designsafe.storage.published"
PROJECT = "PRJ-2048"
NEEDED_FILES = ("AllBuildingID.csv", "DesignSummary.csv", "ReadMe.txt", "SubgroupBuildingIDs.csv")
NEEDED_FOLDERS = (
    "StructuralResponses/EDPsUnder240GMs/PeakStoryDrift",
    "GroundMotions/MirandaGroundMotions/Histories",
    "GroundMotions/MirandaGroundMotions/GroundMotionInfo",
)
# Arthur approved about 37 MB on 2026-11-03; stop well before anything unexpected.
MAX_DOWNLOAD_BYTES = 45_000_000
# DesignSafe dropped connections at 8 at a time (2026-11-03); 4, with retries, is gentler.
PARALLEL_DOWNLOADS = 4
ATTRIBUTION = """# DesignSafe PRJ-2048: attribution

Guan, X., Burton, H., Shokrabadi, M. (2020). A Database of Seismic Designs, Nonlinear
Models, and Seismic Responses for Steel Moment Resisting Frame Buildings. DesignSafe-CI.
https://doi.org/10.17603/ds2-8yc7-1285

Licence: Open Data Commons Attribution (ODC-By). Only the design summary, peak storey
drifts and ground motions are fetched; they are used here and not redistributed. The
ground-motion histories may carry their original sources' rights.
"""


@retry(
    retry=retry_if_exception_type(OSError),  # dropped connections and timeouts, not bad links
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, max=60),
    reraise=True,
)
def _get(url: str) -> bytes:
    # Only https URLs built from API, or the one-time links DesignSafe returns for them.
    if not url.startswith("https://"):
        raise ValueError(f"refusing a non-https download link: {url}")
    with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310
        return response.read()


def list_folder(folder: str) -> list[tuple[str, int]]:
    """(path inside the project, size in bytes) for each file directly in a folder."""
    listing = json.loads(_get(f"{API}/listing/{SYSTEM}/{PROJECT}/{folder}/?offset=0&limit=1000"))["listing"]
    prefix = f"/{PROJECT}/"
    return [(entry["path"].removeprefix(prefix), int(entry["length"])) for entry in listing if entry["type"] == "file"]


def download_file(relative_path: str) -> bytes:
    link = json.loads(_get(f"{API}/preview/{SYSTEM}/{PROJECT}/{relative_path}/"))["href"]
    return _get(link)


def planned_files() -> list[tuple[str, int]]:
    files = [(path, size) for path, size in list_folder("") if path in NEEDED_FILES]
    for folder in NEEDED_FOLDERS:
        files += list_folder(folder)
    return files


def fetch(target_dir: Path, checksum_path: Path) -> int:
    """Download what's missing, check everything against the pinned checksums; returns bytes fetched."""
    files = planned_files()
    planned_bytes = sum(size for _, size in files)
    if planned_bytes > MAX_DOWNLOAD_BYTES:
        raise RuntimeError(f"PRJ-2048 needs {planned_bytes / 1e6:.1f} MB, past the approved cap: ask Arthur first")
    pinned = read_checksums(checksum_path)
    missing = [path for path, _ in files if not (target_dir / path).exists()]
    with ThreadPoolExecutor(PARALLEL_DOWNLOADS) as executor:
        for path, content in zip(missing, executor.map(download_file, missing), strict=True):
            (target_dir / path).parent.mkdir(parents=True, exist_ok=True)
            (target_dir / path).write_bytes(content)
    checksums = {path: sha256((target_dir / path).read_bytes()) for path, _ in files}
    changed = [path for path, digest in checksums.items() if path in pinned and pinned[path] != digest]
    if changed:
        raise RuntimeError(f"{len(changed)} files differ from their pinned checksums, e.g. {changed[0]}")
    write_checksums(checksums, checksum_path)
    return sum(size for path, size in files if path in missing)
