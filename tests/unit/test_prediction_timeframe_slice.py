"""Prediction sliced the last 50 rows of a ticker without asking which
timeframe they were.

features.parquet stacks every timeframe in one frame -- AAPL is 325 rows of
1d and 372 of 60m -- and each timeframe's features carry a suffix, so
SMA_5_1d is NaN on every 60m row by construction. `.tail(50)` therefore
handed a 1d model fifty 60m rows in which all 120 of its selected features
were null.

Measured on the 2026-08-04 run: 310 contexts reported "no data after
dropping incomplete rows" (tabnet 240, mlp 24, 33 others), and Stage 5
finished with 0 predictions out of 330 resolved models -- while the pipeline
reported success.

Verified against the real export: the last 50 AAPL rows are all 60m and
yield 0 usable rows for AAPL_target_return_1d_tabnet; the last 50 AAPL 1d
rows yield 35.
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.pipeline.stages.prediction.data_preparation_service import (
    DataPreparationService,
)


@pytest.fixture()
def service():
    instance = object.__new__(DataPreparationService)
    instance.logger = logging.getLogger("prediction-slice-test")
    return instance


def _frame():
    """Two timeframes stacked, 60m last -- the real export's shape."""
    daily = pd.DataFrame({
        "ticker": ["AAPL"] * 5,
        "interval": ["1d"] * 5,
        "SMA_5_1d": [1.0, 2.0, 3.0, 4.0, 5.0],
        "SMA_5_60m": [None] * 5,
    })
    hourly = pd.DataFrame({
        "ticker": ["AAPL"] * 6,
        "interval": ["60m"] * 6,
        "SMA_5_1d": [None] * 6,
        "SMA_5_60m": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    return pd.concat([daily, hourly], ignore_index=True)


def test_a_daily_model_gets_daily_rows(service):
    rows = service._rows_for_timeframe(_frame(), "AAPL", "1d")

    assert len(rows) == 5
    assert set(rows["interval"]) == {"1d"}
    assert rows["SMA_5_1d"].notna().all()


def test_an_hourly_model_gets_hourly_rows(service):
    rows = service._rows_for_timeframe(_frame(), "AAPL", "60m")

    assert set(rows["interval"]) == {"60m"}


def test_the_defect_without_the_filter(service):
    """The tail of the stacked frame is 60m, so a 1d model saw only nulls."""
    tail = _frame().tail(5)

    assert set(tail["interval"]) == {"60m"}
    assert tail["SMA_5_1d"].isna().all()


def test_the_timeframe_is_read_from_declared_metadata(service):
    meta = {"timeframe": "1h", "selected_features": ["SMA_5_1d"]}

    # 1h normalises to 60m, and declared metadata outranks inference.
    assert service._model_timeframe(meta, "ctx") == "60m"


def test_the_timeframe_is_inferred_from_feature_suffixes(service):
    """Colab writes selected_features_*.json with no timeframe field."""
    meta = {"selected_features": ["SMA_5_1d", "EMA_20_1d", "RSI_14_1d"]}

    assert service._model_timeframe(meta, "ctx") == "1d"


def test_inference_uses_the_majority_not_the_first(service):
    """One stray name must not flip the answer."""
    meta = {"selected_features": ["odd_60m", "SMA_5_1d", "EMA_20_1d", "RSI_14_1d"]}

    assert service._model_timeframe(meta, "ctx") == "1d"


def test_a_mixed_timeframe_model_is_reported(service, caplog):
    meta = {"selected_features": ["a_1d", "b_1d", "c_60m"]}

    with caplog.at_level(logging.WARNING):
        service._model_timeframe(meta, "ctx")

    assert any("mixes feature timeframes" in r.message for r in caplog.records)


def test_unsuffixed_features_mean_no_filter(service):
    """A single-timeframe frame needs none, and guessing would be worse."""
    meta = {"selected_features": ["close", "volume"]}

    assert service._model_timeframe(meta, "ctx") is None


def test_a_missing_timeframe_yields_nothing_rather_than_another_timeframe(
    service, caplog
):
    """Falling through to the unfiltered frame would reproduce the bug."""
    with caplog.at_level(logging.ERROR):
        rows = service._rows_for_timeframe(_frame(), "AAPL", "15m")

    assert rows.empty
    assert any("skipping rather than predicting" in r.message for r in caplog.records)


def test_a_frame_without_a_timeframe_column_is_left_alone(service):
    frame = pd.DataFrame({"ticker": ["AAPL"] * 3, "close": [1.0, 2.0, 3.0]})

    assert len(service._rows_for_timeframe(frame, "AAPL", "1d")) == 3
