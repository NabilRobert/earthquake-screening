"""Choosing ESM records with the paper's filter."""

from earthquake_screening.simulation.esm import select_components

HEADER = (
    "esm_event_id;event_time;mw;epi_dist;network_code;station_code;location_code;"
    "channel_code;u_channel_code;v_channel_code;quality_class"
)


def _flatfile(*rows: str) -> str:
    return "\n".join([HEADER, *rows]) + "\n"


def test_a_recording_that_passes_gives_both_horizontal_components() -> None:
    flatfile = _flatfile("IT-2009-0009;2009-04-06T01:32:39;6.1;4.6;IT;AQK;;HN;E;N;BEST")
    channels = [component.channel for component in select_components(flatfile)]
    assert channels == ["HNE", "HNN"]


def test_magnitude_distance_date_and_quality_each_exclude() -> None:
    flatfile = _flatfile(
        "too-small;2009-04-06T01:32:39;5.5;4.6;IT;A;;HN;E;N;BEST",  # magnitude must be strictly above 5.5
        "too-far;2009-04-06T01:32:39;6.1;20.0;IT;B;;HN;E;N;BEST",  # distance must be under 20 km
        "too-late;2019-01-01T00:00:00;6.1;4.6;IT;C;;HN;E;N;BEST",  # after the paper's submission
        "poor;2009-04-06T01:32:39;6.1;4.6;IT;D;;HN;E;N;BAD",  # ESM quality BAD
        "no-magnitude;2009-04-06T01:32:39;;4.6;IT;E;;HN;E;N;BEST",
    )
    assert select_components(flatfile) == []
