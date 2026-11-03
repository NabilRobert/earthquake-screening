"""Choose the real records from the Engineering Strong Motion database (ESM, CC BY 4.0).

The paper fitted 97 ESM records with 5.5 < M < 6.5 and distance under 20 km, in 2018,
without listing them. ESM has grown and been reprocessed since, so we apply the same
filter to ESM 3.0's metadata and pin the result in data/esm_selection.csv:
- moment magnitude strictly between 5.5 and 6.5, epicentral distance under 20 km;
- events before the paper's submission (2018-09-25), so the set is what it could have used;
- ESM quality BEST or GOOD, since the model fit needs clean processing;
- both horizontal components of each recording, each fitted as its own record.
The deviations are listed in docs/reproduction/sainct2020.md.

Only metadata is fetched here (about 1 MB). Waveforms wait for Arthur's OK.
"""

import csv
import io
import urllib.request
from dataclasses import asdict, dataclass, fields

# ESM's flatfile web service; the server-side filter is loose, the exact filter is ours.
FLATFILE_URL = "https://esm-db.eu/esmws/flatfile/1/query?min-magnitude=5.5&max-magnitude=6.5&max-epicentral-distance=20"
MIN_MAGNITUDE = 5.5
MAX_MAGNITUDE = 6.5
MAX_EPICENTRAL_DISTANCE_KM = 20.0
LATEST_EVENT = "2018-09-25"  # the paper's arXiv submission date
ACCEPTED_QUALITY = ("BEST", "GOOD")
HORIZONTAL_COMPONENTS = ("u", "v")  # ESM names the horizontals U and V, the vertical W


@dataclass(frozen=True)
class SelectedComponent:
    event_id: str
    event_time: str
    magnitude: float
    epicentral_distance_km: float
    network: str
    station: str
    location: str
    channel: str  # e.g. HNE: the component's own channel code
    quality: str

    @property
    def record_id(self) -> str:
        return f"{self.event_id}.{self.network}.{self.station}.{self.location}.{self.channel}"


def fetch_flatfile() -> str:
    with urllib.request.urlopen(FLATFILE_URL, timeout=120) as response:
        return response.read().decode("utf-8")


def _is_selected(row: dict[str, str]) -> bool:
    if not row["mw"] or not row["epi_dist"]:
        return False
    return (
        MIN_MAGNITUDE < float(row["mw"]) < MAX_MAGNITUDE
        and float(row["epi_dist"]) < MAX_EPICENTRAL_DISTANCE_KM
        and row["event_time"] < LATEST_EVENT
        and row["quality_class"] in ACCEPTED_QUALITY
    )


def select_components(flatfile_csv: str) -> list[SelectedComponent]:
    """Both horizontal components of every recording that passes the paper's filter."""
    selected: list[SelectedComponent] = []
    for row in csv.DictReader(io.StringIO(flatfile_csv), delimiter=";"):
        if not _is_selected(row):
            continue
        for component in HORIZONTAL_COMPONENTS:
            selected.append(
                SelectedComponent(
                    event_id=row["esm_event_id"],
                    event_time=row["event_time"],
                    magnitude=float(row["mw"]),
                    epicentral_distance_km=float(row["epi_dist"]),
                    network=row["network_code"],
                    station=row["station_code"],
                    location=row["location_code"],
                    channel=row["channel_code"] + row[f"{component}_channel_code"],
                    quality=row["quality_class"],
                )
            )
    return sorted(selected, key=lambda component: component.record_id)


def write_selection(components: list[SelectedComponent], path: str) -> None:
    with open(path, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=[field.name for field in fields(SelectedComponent)])
        writer.writeheader()
        writer.writerows(asdict(component) for component in components)
