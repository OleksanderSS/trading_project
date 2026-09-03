"""Stage 5 must be able to prepare data for a pooled context.

On 2026-09-01 stages 5-7 ran for the first time in the project's history and
Stage 5 produced `0 predictions` from seven champions: every pooled context
logged "No data for ticker __POOLED__", because the preparation service
filtered a real ticker column by a synthetic name. The pipeline then reported
success (REGISTER #210, #211).

These tests exercise the wiring -- the service as Stage 5 calls it -- not the
predicate on its own. A green `is_pooled` unit test would have passed on
2026-08-31 too, while the pipeline was producing nothing.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.pipeline.modeling_context import POOLED_TICKER, is_pooled, rows_for_ticker
from src.pipeline.stages.prediction.data_preparation_service import (
    DataPreparationService,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """Two tickers, two timeframes, 60 bars each -- more than the tail of 50."""
    rows = []
    for ticker in ("AAPL", "MSFT"):
        for interval in ("15m", "60m"):
            for step in range(60):
                rows.append({
                    "ticker": ticker,
                    "interval": interval,
                    "datetime": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(minutes=15 * step),
                    "CCI_15m": float(step),
                    "Stoch_K_60m": float(step) / 2,
                })
    return pd.DataFrame(rows)


def test_pooled_name_selects_every_ticker(frame):
    assert len(rows_for_ticker(frame, POOLED_TICKER)) == len(frame)
    assert set(rows_for_ticker(frame, "AAPL")["ticker"]) == {"AAPL"}


def test_pooled_name_survives_recasing():
    # The sentinel travels through file names, JSON keys and context ids.
    assert is_pooled("__pooled__")
    assert is_pooled(" __POOLED__ ")
    assert not is_pooled("AAPL")


def test_prepare_ticker_data_returns_rows_for_a_pooled_context(frame):
    prepared = DataPreparationService().prepare_ticker_data(
        frame, POOLED_TICKER, timeframe="15m"
    )
    assert prepared is not None, (
        "a pooled context returned no data -- this is the exact failure that "
        "produced 0 predictions from 7 champions on 2026-09-01"
    )
    assert not prepared.empty


def test_the_pooled_window_is_taken_per_instrument(frame):
    """`.tail(50)` on the pooled frame would describe one ticker as the market.

    Both tickers must be represented, and neither may contribute more than the
    window: 50 rows each out of the 60 bars per (ticker, timeframe).
    """
    rows = rows_for_ticker(frame, POOLED_TICKER)
    rows = rows[rows["interval"] == "15m"]
    windowed = rows.groupby("ticker", sort=False).tail(50)
    counts = windowed["ticker"].value_counts().to_dict()
    assert counts == {"AAPL": 50, "MSFT": 50}
    assert len(rows.tail(50)["ticker"].unique()) == 1, (
        "the blind tail really does collapse to one ticker -- that is why the "
        "per-instrument window exists"
    )
