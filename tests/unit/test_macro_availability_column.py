"""Every macro source must carry a point-in-time availability stamp.

The `prepare` run failed in Stage 2 with

    Macro data contains missing or invalid point-in-time values in
    realtime_start.

CollectionStage concatenates fred_data, economic_calendar and news_patterns
into a single macro_data frame, and pd.concat fills a column a source lacks
with NaN. fred_data carries realtime_start; economic_calendar carries none,
so its rows arrived null and the whole frame was rejected.

The check itself is pre-existing (commit a35a9d05) and correct: using a
macro figure before it was published is look-ahead. It only became
reachable once the collection repairs earlier in this audit took
economic_calendar from 0 rows to 71 -- until then the frame was fred alone.
Verified against the live database: all 50,513 fred_data rows have a valid
realtime_start, so the null values were entirely from the other source.

Every source is normalised onto ONE column, `available_at`. Splitting them
would only move the problem: the downstream check takes the first of
(available_at, released_at, realtime_start) present anywhere in the frame,
so whichever it picks is null for the other source's rows -- which is what
the first version of this fix did, and what the test below now pins.

Only tables whose own timestamp genuinely IS the publication moment are
filled: an economic-calendar entry becomes public when the event happens, a
news pattern exists from when the news appeared. A source with a real
release lag -- the annual World Bank series dated '1960' in macro_sdmx_data,
say -- must not be handed an invented one, and is left to fail loudly.
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.pipeline.stages.collection.orchestrator import CollectionStage


@pytest.fixture()
def stage():
    instance = object.__new__(CollectionStage)
    instance.logger = logging.getLogger("collection-macro-test")
    return instance


def _fred(rows=3):
    return pd.DataFrame({
        "realtime_start": ["2026-06-04"] * rows,
        "date": ["2026-06-03"] * rows,
        "series_id": ["DGS10"] * rows,
        "value": [4.1] * rows,
    })


def _calendar(rows=3):
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-08-03", periods=rows, freq="D", tz="UTC"),
        "country": ["JPY"] * rows,
        "event": ["Final Manufacturing PMI"] * rows,
        "actual": [""] * rows,
    })


def test_a_source_with_realtime_start_is_mapped_onto_available_at(stage):
    result = stage._ensure_macro_availability(_fred(), "fred_data")

    assert "available_at" in result.columns
    assert result["available_at"].notna().all()


def test_the_calendar_gets_its_event_time(stage):
    result = stage._ensure_macro_availability(_calendar(), "economic_calendar")

    assert result["available_at"].notna().all()
    assert result["available_at"].iloc[0] == pd.Timestamp("2026-08-03", tz="UTC")


def test_the_concatenated_frame_has_no_gaps(stage):
    """The actual failure: mixing the two sources left nulls behind."""
    merged = pd.concat(
        [
            stage._ensure_macro_availability(_fred(), "fred_data"),
            stage._ensure_macro_availability(_calendar(), "economic_calendar"),
        ],
        ignore_index=True,
    )

    assert merged["available_at"].notna().all()


def test_the_downstream_check_would_now_pass(stage):
    """Reproduces Stage 2's own logic rather than trusting that it matches."""
    merged = pd.concat(
        [
            stage._ensure_macro_availability(_fred(), "fred_data"),
            stage._ensure_macro_availability(_calendar(), "economic_calendar"),
        ],
        ignore_index=True,
    )

    column = next(
        c for c in ("available_at", "released_at", "realtime_start")
        if c in merged.columns
    )
    invalid = pd.to_datetime(merged[column], errors="coerce", utc=True).isna()

    assert not invalid.any()


def test_normalising_only_one_side_would_not_have_been_enough(stage):
    """The first attempt at this fix: leaving fred on realtime_start and
    giving only the calendar available_at. The check picks available_at,
    which is then null for every fred row."""
    half_fixed = pd.concat(
        [_fred(), stage._ensure_macro_availability(_calendar(), "economic_calendar")],
        ignore_index=True,
    )

    column = next(
        c for c in ("available_at", "released_at", "realtime_start")
        if c in half_fixed.columns
    )
    assert column == "available_at"
    assert half_fixed[column].isna().any(), "this is the trap the fix avoids"


def test_a_source_with_no_honest_release_time_is_not_invented(stage, caplog):
    """macro_sdmx_data is dated '1960' with no publication date. Filling it
    would claim a 1960 figure was tradable in 1960."""
    frame = pd.DataFrame({
        "date": ["1960", "1961"],
        "indicator": ["FP_CPI_TOTL_ZG"] * 2,
        "value": [1.45, 1.07],
    })

    with caplog.at_level(logging.WARNING):
        result = stage._ensure_macro_availability(frame, "macro_sdmx_data")

    assert "available_at" not in result.columns
    assert any("no availability column" in r.getMessage() for r in caplog.records)


def test_an_empty_frame_is_returned_untouched(stage):
    empty = pd.DataFrame()

    assert stage._ensure_macro_availability(empty, "fred_data").empty


def test_a_frame_that_already_uses_available_at_is_left_alone(stage):
    frame = _calendar()
    frame["available_at"] = pd.Timestamp("2026-01-01", tz="UTC")

    result = stage._ensure_macro_availability(frame, "economic_calendar")

    assert (result["available_at"] == pd.Timestamp("2026-01-01", tz="UTC")).all()


def test_the_live_fred_table_is_not_the_source_of_the_nulls():
    """Checked rather than assumed when diagnosing: every stored fred row
    has a parseable realtime_start, so the nulls came from the other source."""
    import duckdb

    connection = duckdb.connect("data/trading_data.duckdb", read_only=True)
    frame = connection.execute("SELECT realtime_start FROM fred_data").fetchdf()
    connection.close()

    invalid = pd.to_datetime(frame["realtime_start"], errors="coerce", utc=True).isna()

    assert not invalid.any()
