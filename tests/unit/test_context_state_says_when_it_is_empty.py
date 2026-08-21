"""A context state that could not be computed used to read as "no change".

`_process_numeric_column` discretises a column into -1/0/1 by comparing its
pct_change against a rolling-std threshold, starting from an array of zeros
and filling in only the rows where both sides exist. In that encoding 0 means
flat, so every row it could not compute came out claiming the market had not
moved.

For a base column that is zero on most bars this is every row. Insider net
value is zero on 62% of them: 0 -> 0 is 0/0, 0 -> x is infinite, and what
reaches the frame is an unbroken run of "no change". Measured on the 18.08
batch, 17 of 615 state_ columns carry a single value.

Most of the other flat ones are honest -- `state_FRED_GDP_15m` does not move
between 15-minute bars because a quarterly figure does not -- so the encoding
is left alone. What changed is that a column which separates nothing is not
emitted, and says which base column produced it.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.context_map_enricher import ContextMapEnricher


@pytest.fixture
def enricher():
    return ContextMapEnricher()


def _frame(values):
    rows = len(values)
    return pd.DataFrame({
        "ticker": ["AAPL"] * rows,
        "datetime": pd.date_range("2026-01-01", periods=rows, freq="D"),
        "base": values,
    })


def test_a_mostly_zero_column_does_not_become_a_run_of_no_change(enricher, caplog):
    """The insider shape: zero on most bars, so pct_change is 0/0."""
    values = [0.0] * 120
    frame = _frame(values)
    state_cols: list[str] = []

    with caplog.at_level(logging.INFO):
        enricher._process_numeric_column(frame, "base", "state_base", state_cols)

    assert "state_base" not in frame.columns
    assert state_cols == []
    assert any("came out constant" in record.message for record in caplog.records)


def test_the_log_names_the_base_column_that_produced_nothing(enricher, caplog):
    frame = _frame([0.0] * 120)
    with caplog.at_level(logging.INFO):
        enricher._process_numeric_column(frame, "base", "state_base", [])
    said = " ".join(record.getMessage() for record in caplog.records)
    assert "state_base" in said and "'base'" in said


def test_a_column_that_actually_moves_is_still_emitted(enricher):
    rng = np.random.default_rng(0)
    values = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 200)))
    frame = _frame(values)
    state_cols: list[str] = []

    enricher._process_numeric_column(frame, "base", "state_base", state_cols)

    assert "state_base" in frame.columns
    assert state_cols == ["state_base"]
    assert frame["state_base"].nunique() > 1
    assert set(frame["state_base"].unique()) <= {-1, 0, 1}


def test_a_constant_but_nonzero_column_is_also_refused(enricher):
    """Flat is flat, whatever the level it is flat at."""
    frame = _frame([42.0] * 120)
    state_cols: list[str] = []
    enricher._process_numeric_column(frame, "base", "state_base", state_cols)
    assert "state_base" not in frame.columns


def test_a_column_that_moves_only_once_still_counts_as_moving(enricher):
    values = [10.0] * 100 + [14.0] * 100
    frame = _frame(values)
    state_cols: list[str] = []
    enricher._process_numeric_column(frame, "base", "state_base", state_cols)
    assert "state_base" in frame.columns
    assert frame["state_base"].nunique() > 1
