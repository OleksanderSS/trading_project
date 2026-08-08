"""sentiment_available was the constant 1.0 on every row of every timeframe.

It was computed from the forward-filled series:

    carried = values.groupby(ticker).ffill()
    df['sentiment_available'] = carried.notna().astype(int)

so one reading anywhere in a ticker's history made every later row
"available" for good. Measured on the 2026-08-06 export: constant 1.0 on
15m, 60m and 1d alike -- including 5,757 daily rows from before any
sentiment source existed in the database.

Two costs. A constant column carries no information, so 3 of the 144
news/sentiment features were dead weight. And it is not merely empty, it
is wrong: it asserts a reading for rows holding a value carried forward
from days earlier, which is exactly the distinction the flag exists to
make.

It also blocked a decision. Whether hourly history can be extended past
the news that explains it depends on being able to tell "no news happened"
from "no news collected" -- and this flag was the thing that was supposed
to say so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.sentiment_features_enricher import (
    SentimentFeaturesEnricher,
)


def _prepare(frame):
    return SentimentFeaturesEnricher._prepare_dataframe(
        SentimentFeaturesEnricher.__new__(SentimentFeaturesEnricher),
        frame,
        "sentiment",
    )


def _frame(values, ticker="AAPL"):
    return pd.DataFrame({
        "ticker": [ticker] * len(values),
        "datetime": pd.date_range("2026-01-01", periods=len(values), freq="D", tz="UTC"),
        "sentiment": values,
    })


def test_a_carried_value_is_not_reported_as_available():
    """One reading, five rows -- the flag used to claim five."""
    out = _prepare(_frame([np.nan, 0.4, np.nan, np.nan, np.nan, np.nan]))

    assert out["sentiment_available"].tolist() == [0, 1, 0, 0, 0, 0]


def test_the_value_is_still_carried_forward():
    """The fix is to the FLAG, not to the fill. A model still sees the last
    known sentiment; it can now also see that it is not fresh."""
    out = _prepare(_frame([np.nan, 0.4, np.nan, np.nan]))

    assert out["sentiment"].tolist() == [0.0, 0.4, 0.4, 0.4]


def test_a_ticker_with_no_sentiment_at_all_reads_zero():
    out = _prepare(_frame([np.nan] * 4))

    assert out["sentiment_available"].tolist() == [0, 0, 0, 0]
    assert out["sentiment"].tolist() == [0.0] * 4


def test_every_row_having_a_reading_still_reads_one():
    out = _prepare(_frame([0.1, -0.2, 0.3]))

    assert out["sentiment_available"].tolist() == [1, 1, 1]


def test_the_flag_is_not_constant_on_realistic_input():
    """The property that was violated, stated directly: a column with one
    distinct value cannot inform anything."""
    out = _prepare(_frame([np.nan, 0.4, np.nan, 0.2, np.nan, np.nan]))

    assert out["sentiment_available"].nunique() > 1


def test_tickers_do_not_lend_each_other_readings():
    frame = pd.concat([
        _frame([0.5, np.nan], ticker="AAPL"),
        _frame([np.nan, np.nan], ticker="MSFT"),
    ], ignore_index=True)

    out = _prepare(frame).sort_values(["ticker", "datetime"])

    assert out[out.ticker == "AAPL"]["sentiment_available"].tolist() == [1, 0]
    assert out[out.ticker == "MSFT"]["sentiment_available"].tolist() == [0, 0]


def test_the_flag_comes_from_the_unfilled_series():
    """Pinned against the exact regression: reading it off the ffilled
    series is what made it constant."""
    import inspect

    source = inspect.getsource(SentimentFeaturesEnricher._prepare_dataframe)
    line = next(
        l for l in source.splitlines()
        if "sentiment_available" in l and "=" in l and not l.strip().startswith("#")
    )

    assert "carried_sentiment" not in line, line


def test_the_exported_batch_shows_the_defect_this_fixes():
    """The measurement the fix rests on. Will keep passing until a batch is
    rebuilt with the fix, at which point it is the fix that has to be
    re-verified -- so it skips rather than fails when the flag varies.
    """
    from pathlib import Path

    path = Path("data/colab/accumulated/main_database/features.parquet")
    if not path.exists():
        pytest.skip("no prepared batch on disk")

    frame = pd.read_parquet(path, columns=["interval", "sentiment_available_1d"])
    if frame["sentiment_available_1d"].nunique() > 1:
        pytest.skip("batch was rebuilt with the fix; nothing to demonstrate")

    assert frame["sentiment_available_1d"].nunique() == 1, (
        "this test documents the pre-fix state of the batch on disk"
    )
