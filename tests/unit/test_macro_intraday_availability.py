"""A date-only release time let intraday models read macro data early.

`fred_data.realtime_start` is a DATE -- '2026-06-04' -- which parses to
midnight UTC. Taken literally that claims a figure published at 08:30 ET was
knowable at 00:00 the same day. On daily bars the error is invisible; on the
60m and 15m series this project also trains, it is a straight look-ahead of
several hours.

This was the one real gap MacroReleaseTimingGuard pointed at before it was
archived. Its answer was a table of official release times per indicator,
which needs every FRED series mapped to an indicator, drifts as schedules
change, and fails toward being too EARLY. Deferring a date-only stamp to the
end of its date can only ever be late, and only by a day, on series that
move monthly or quarterly.
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.pipeline.stages.collection.orchestrator import CollectionStage


@pytest.fixture()
def stage():
    instance = object.__new__(CollectionStage)
    instance.logger = logging.getLogger("macro-availability-test")
    return instance


def _frame(*stamps):
    return pd.DataFrame({
        "value": list(range(len(stamps))),
        # format="mixed": these fixtures deliberately mix a date-only stamp
        # with a full timestamp, which is exactly the shape the concatenated
        # macro frame has and which pandas will not infer from the first
        # element alone.
        "available_at": pd.to_datetime(list(stamps), utc=True, format="mixed"),
    })


def test_a_date_only_stamp_is_deferred_to_the_end_of_that_date(stage):
    result = stage._defer_date_only_availability(_frame("2026-06-04"), "fred_data")

    assert result["available_at"].iloc[0] == pd.Timestamp(
        "2026-06-04 23:59:59", tz="UTC"
    )


def test_an_intraday_bar_can_no_longer_read_it_before_publication(stage):
    """The defect, stated as the thing that actually goes wrong."""
    result = stage._defer_date_only_availability(_frame("2026-06-04"), "fred_data")
    available = result["available_at"].iloc[0]

    morning_bar = pd.Timestamp("2026-06-04 09:00:00", tz="UTC")
    assert morning_bar < available, "a 09:00 bar must not see this figure"


def test_a_real_publication_time_is_left_alone(stage):
    """The economic calendar publishes a genuine moment; that beats anything
    inferred here."""
    stamps = ["2026-08-03 03:30:00+03:00", "2026-08-03 04:45:00+03:00"]
    frame = _frame(*stamps)

    result = stage._defer_date_only_availability(frame, "economic_calendar")

    assert list(result["available_at"]) == list(frame["available_at"])


def test_a_mixed_frame_defers_only_the_date_only_rows(stage):
    frame = _frame("2026-06-04", "2026-06-04 14:30:00+00:00")

    result = stage._defer_date_only_availability(frame, "macro_mixed")

    assert result["available_at"].iloc[0] == pd.Timestamp(
        "2026-06-04 23:59:59", tz="UTC"
    )
    assert result["available_at"].iloc[1] == pd.Timestamp(
        "2026-06-04 14:30:00", tz="UTC"
    )


def test_the_deferral_is_announced(stage, caplog):
    with caplog.at_level(logging.INFO):
        stage._defer_date_only_availability(_frame("2026-06-04"), "fred_data")

    assert any("date-only" in record.message for record in caplog.records)


def test_a_frame_without_the_column_is_returned_unchanged(stage):
    frame = pd.DataFrame({"value": [1, 2]})

    assert stage._defer_date_only_availability(frame, "whatever") is frame


def test_nulls_survive_untouched(stage):
    frame = _frame("2026-06-04", None)

    result = stage._defer_date_only_availability(frame, "fred_data")

    assert pd.isna(result["available_at"].iloc[1])


def test_the_deferral_never_moves_a_stamp_earlier(stage):
    """The whole point: this correction can only ever be conservative."""
    frame = _frame("2026-06-04", "2026-06-05 08:30:00+00:00", "2026-06-06")

    result = stage._defer_date_only_availability(frame, "fred_data")

    assert (result["available_at"] >= frame["available_at"]).all()
