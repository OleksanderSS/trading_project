"""The trailing slope in closed form must equal the polyfit it replaces.

`_rolling_slope` was `rolling(window).apply(np.polyfit(...), raw=True)`, which
calls into Python once per window. Measured on 2026-08-28 over 156,372 rows:
21.18 seconds against 0.11 for the closed form, and it runs twice per frame
(windows 5 and 20), so about forty seconds of every rebuild.

The slope against evenly spaced x is sum((x - x̄)(y - ȳ)) / sum((x - x̄)²), and
for a fixed window the denominator is constant. That is arithmetic, not an
approximation -- so these tests hold the previous implementation and require
equality rather than closeness in spirit.

The partial windows at the start are the part worth watching: the old version
had min_periods=2 and produced 18 values in the first 19 rows. Dropping them
would be a silent change to where every trend feature begins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.market_context_enricher import MarketContextEnricher


def _old_rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """The previous implementation, kept verbatim as the reference."""
    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any() or len(values) < 2:
            return np.nan
        return float(np.polyfit(np.arange(len(values)), values, 1)[0])

    return series.rolling(window, min_periods=2).apply(slope, raw=True)


def _walk(rows: int = 500, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 1, rows).cumsum())


@pytest.mark.parametrize("window", [5, 20, 50])
def test_full_windows_match_the_polyfit(window):
    series = _walk()
    new = MarketContextEnricher._rolling_slope(series, window)
    old = _old_rolling_slope(series, window)

    tail = slice(window - 1, None)
    assert np.allclose(
        new.to_numpy()[tail], old.to_numpy()[tail], rtol=1e-9, equal_nan=True
    )


def test_the_opening_rows_keep_their_values():
    """min_periods=2 gave 18 values in the first 19 rows; they must survive."""
    window = 20
    series = _walk()
    new = MarketContextEnricher._rolling_slope(series, window)
    old = _old_rolling_slope(series, window)

    head = slice(0, window - 1)
    assert np.isnan(new.to_numpy()[0]) and np.isnan(old.to_numpy()[0]), (
        "a single point has no slope"
    )
    assert np.allclose(
        new.to_numpy()[head], old.to_numpy()[head], rtol=1e-9, equal_nan=True
    )
    assert np.count_nonzero(~np.isnan(new.to_numpy()[head])) == window - 2


def test_a_constant_series_has_no_slope():
    flat = pd.Series(np.full(100, 5.0))
    result = MarketContextEnricher._rolling_slope(flat, 20)

    assert np.allclose(result.dropna(), 0.0)


def test_a_straight_line_recovers_its_gradient():
    """The one case where the right answer is known without a reference."""
    line = pd.Series(np.arange(100, dtype=float) * 3.5)
    result = MarketContextEnricher._rolling_slope(line, 20)

    assert np.allclose(result.iloc[19:], 3.5)


def test_a_series_shorter_than_the_window_still_works():
    short = _walk(rows=7)
    new = MarketContextEnricher._rolling_slope(short, 20)
    old = _old_rolling_slope(short, 20)

    assert np.allclose(new.to_numpy(), old.to_numpy(), rtol=1e-9, equal_nan=True)


def test_the_index_is_preserved():
    """The caller assigns this straight into a frame column."""
    series = _walk(rows=50)
    series.index = pd.date_range("2020-01-01", periods=50, freq="D")

    result = MarketContextEnricher._rolling_slope(series, 10)

    assert result.index.equals(series.index)


def test_an_empty_series_returns_empty():
    result = MarketContextEnricher._rolling_slope(pd.Series(dtype=float), 20)
    assert len(result) == 0
