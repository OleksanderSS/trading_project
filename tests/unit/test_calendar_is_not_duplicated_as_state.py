"""`state_hour` was a second copy of `hour`, competing with it for selection.

ContextMapEnricher discretises each context column into -1/0/1. Calendar
columns took a different branch:

    if col in self.temporal_features:
        res_df[state_col_name] = res_df[col]      # verbatim, no encoding

so `state_hour_15m` held exactly what `hour_15m` held. Measured on the
2026-08-15 export, 24 such duplicates existed -- 8 calendar names across three
timeframes -- each entering the feature pool as an independently selectable
copy of a column already in it.

They were not harmless duplicates. Ranked by entropy, calendar columns beat
every market column outright: day_of_year scores H=5.06 against 1.58 for the
best price state. Having many distinct values is what wins a correlation or
entropy ranking, so they were selected for a property that says nothing about
the market. In this run's feature importances, state_day_of_month_1d,
state_hour_15m and ctx_60m_hour_cos_60m sat among the most frequently chosen
context columns. The identical pull had already been measured distorting the
context fingerprint, where a date-keyed print matches only coincidence.

The information is not lost -- hour_15m and its siblings remain, once each.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.context_map_enricher import ContextMapEnricher


@pytest.fixture
def enricher():
    return ContextMapEnricher()


def _frame():
    n = 60
    idx = pd.date_range("2026-07-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "ticker": ["AAPL"] * n,
        "datetime": idx,
        "hour": idx.hour,
        "day_of_week": idx.dayofweek,
        "close": np.linspace(100, 130, n),
    })


def test_a_calendar_column_gets_no_state_twin(enricher):
    df = _frame()

    state_cols, temporal_cols = enricher._process_context_columns(
        df, ["hour", "day_of_week", "close"]
    )

    assert "state_hour" not in df.columns
    assert "state_day_of_week" not in df.columns
    assert temporal_cols == []


def test_the_raw_calendar_columns_are_untouched(enricher):
    """Dropping the duplicate must not drop the information."""
    df = _frame()
    before = df["hour"].tolist()

    enricher._process_context_columns(df, ["hour", "close"])

    assert df["hour"].tolist() == before


def test_a_market_column_is_still_discretised(enricher):
    df = _frame()

    state_cols, _ = enricher._process_context_columns(df, ["close"])

    assert state_cols == ["state_close"]
    assert set(pd.unique(df["state_close"].dropna())) <= {-1, 0, 1}, (
        "a market state is a discretisation, which is what made the calendar "
        "branch's verbatim copy wrong"
    )


def test_the_skip_is_announced(enricher, caplog):
    import logging

    with caplog.at_level(logging.INFO):
        enricher._process_context_columns(_frame(), ["hour", "day_of_week", "close"])

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "hour" in logged and "day_of_week" in logged
