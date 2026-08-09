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


# --------------------------------------------------------------------------
# When dropping empties the frame, say WHICH columns did it.
#
# On the 2026-08-09 run 313 of 660 contexts ended at "has no data after
# dropping incomplete rows" -- 127 of 154 on 15m, 84 of 110 on 60m -- and
# the message named not a single feature. The cause could not be
# established from the artifacts afterwards, only guessed at. A row is
# dropped BY a column, and the column is the part worth knowing.
# --------------------------------------------------------------------------


def _service_with_capture(caplog):
    import logging

    from src.pipeline.stages.prediction.data_preparation_service import (
        DataPreparationService,
    )

    service = DataPreparationService.__new__(DataPreparationService)
    service.logger = logging.getLogger("drop-diagnostic-test")
    caplog.set_level(logging.ERROR, logger="drop-diagnostic-test")
    return service


def test_the_column_that_emptied_the_frame_is_named(caplog):
    import numpy as np

    service = _service_with_capture(caplog)
    frame = pd.DataFrame({
        "good_feature": [1.0, 2.0, 3.0],
        "ctx_60m_dead": [np.nan, np.nan, np.nan],
    })

    result = service._drop_incomplete_model_rows(
        frame, ["good_feature", "ctx_60m_dead"], "XLF_60m_target_x"
    )

    assert result is None
    message = caplog.text
    assert "ctx_60m_dead" in message, message
    assert "good_feature" not in message, (
        "the column that was fine is named as if it were a culprit"
    )


def test_the_message_counts_how_many_features_are_wholly_null(caplog):
    import numpy as np

    service = _service_with_capture(caplog)
    frame = pd.DataFrame({
        "a": [np.nan, np.nan],
        "b": [np.nan, np.nan],
        "c": [1.0, 2.0],
    })

    service._drop_incomplete_model_rows(frame, ["a", "b", "c"], "ctx")

    assert "2 of 3" in caplog.text, caplog.text


def test_a_partially_null_feature_is_reported_separately(caplog):
    import numpy as np

    service = _service_with_capture(caplog)
    frame = pd.DataFrame({
        "always_null": [np.nan, np.nan],
        "sometimes_null": [1.0, np.nan],
    })

    service._drop_incomplete_model_rows(frame, ["always_null", "sometimes_null"], "ctx")

    text = caplog.text
    assert "always_null" in text
    assert "sometimes_null" in text
    assert "1 more" in text or "null in some" in text, text


def test_a_frame_that_survives_is_left_alone(caplog):
    """The diagnostic must not fire on the working path."""
    service = _service_with_capture(caplog)
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    result = service._drop_incomplete_model_rows(frame, ["a", "b"], "ctx")

    assert result is not None
    assert len(result) == 2
    assert caplog.text == ""
