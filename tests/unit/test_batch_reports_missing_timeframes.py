"""A requested timeframe that produces nothing must not pass in silence.

batch_metadata.json for the 2026-08-04 run recorded

    "timeframes": ["15m", "1d", "1h"]

while features.parquet held only 1d (7,162 rows) and 60m (5,757), and
targets.parquet carried no 15m target column at all -- although targets.yaml
declares target_intraday_up_15m and target_intraday_return_15m, and
collectors.yaml requests 15m with a 60-day period.

A third of the requested scope vanished between collection and export.
Nothing said so. Every downstream stage then reported success on what
remained: 506 champions, 396 daily and 110 hourly, zero fifteen-minute.

The metadata recorded what was ASKED FOR and called it the batch's
timeframes. Asked-for and delivered are different questions.
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.pipeline.hybrid.colab_manager import ColabManager


def _features(*timeframes):
    return pd.DataFrame({
        "ticker": ["AAPL"] * len(timeframes),
        "interval": list(timeframes),
        # _create_batch_metadata builds a lineage block further down that
        # needs the identity columns; without them the fixture fails before
        # reaching what these tests are about.
        "datetime": pd.date_range("2026-08-01", periods=len(timeframes), freq="D"),
        "close": [1.0] * len(timeframes),
    })


def test_a_requested_timeframe_with_no_rows_is_reported():
    delivered = ColabManager._delivered_timeframes(_features("1d", "60m"))

    assert ColabManager._missing_timeframes(["15m", "1d", "1h"], delivered) == ["15m"]


def test_one_hour_and_sixty_minutes_are_the_same_timeframe():
    """The request says 1h, the data says 60m. A raw set difference reports a
    phantom gap and hides the real one."""
    delivered = ColabManager._delivered_timeframes(_features("1d", "60m"))

    assert "1h" not in ColabManager._missing_timeframes(["1d", "1h"], delivered)


def test_a_fully_delivered_batch_reports_nothing_missing():
    delivered = ColabManager._delivered_timeframes(_features("15m", "1d", "60m"))

    assert ColabManager._missing_timeframes(["15m", "1d", "1h"], delivered) == []


def test_a_frame_without_a_timeframe_column_delivers_nothing():
    frame = pd.DataFrame({"ticker": ["AAPL"], "close": [1.0]})

    assert ColabManager._delivered_timeframes(frame) == set()


def test_the_metadata_records_both_questions():
    """Requested and delivered, side by side."""
    manager = object.__new__(ColabManager)

    class _Config:
        tickers = ["AAPL"]
        timeframes = ["15m", "1d", "1h"]
        accumulate = True

    features = _features("1d", "60m")
    targets = pd.DataFrame({
        "ticker": ["AAPL", "AAPL"],
        "datetime": pd.date_range("2026-08-01", periods=2, freq="D"),
        "target_up_1d": [1, 0],
    })
    manager._is_test_mode = lambda config: False
    manager._sha256 = lambda path: "sha"

    from pathlib import Path

    metadata = ColabManager._create_batch_metadata(
        manager, "main_database", "20260804", _Config(),
        features, targets, Path("f"), Path("t"), None,
    )

    assert metadata["timeframes"] == ["15m", "1d", "1h"]
    assert metadata["timeframes_delivered"] == ["1d", "60m"]
    assert metadata["timeframes_missing"] == ["15m"]


def test_the_gap_is_logged_at_error(caplog):
    manager = object.__new__(ColabManager)

    class _Config:
        tickers = ["AAPL"]
        timeframes = ["15m", "1d", "1h"]
        accumulate = True

    manager._is_test_mode = lambda config: False
    manager._sha256 = lambda path: "sha"

    from pathlib import Path

    with caplog.at_level(logging.ERROR):
        ColabManager._create_batch_metadata(
            manager, "main_database", "20260804", _Config(),
            _features("1d", "60m"),
            pd.DataFrame({
                "ticker": ["AAPL"],
                "datetime": pd.date_range("2026-08-01", periods=1, freq="D"),
                "target_up_1d": [1],
            }),
            Path("f"), Path("t"), None,
        )

    assert any("produced NONE" in record.message for record in caplog.records)
    assert any("15m" in str(record.args) for record in caplog.records)
