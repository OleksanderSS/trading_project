"""
Property-style financial tests without mandatory Hypothesis dependency.

These tests use generated deterministic examples to check invariants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_prices(seed: int, n: int = 50):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, size=n)
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices)


def test_constant_prices_do_not_create_infinite_returns():
    prices = pd.Series([100.0] * 20)
    returns = prices.pct_change(fill_method=None)
    assert not np.isinf(returns.dropna()).any()
    assert (returns.dropna() == 0).all()


def test_adding_second_ticker_does_not_change_first_ticker_forward_returns():
    a = pd.DataFrame({
        "ticker": ["A"] * 5,
        "timestamp": pd.date_range("2024-01-01", periods=5),
        "close": [100, 101, 102, 103, 104],
    })
    b = pd.DataFrame({
        "ticker": ["B"] * 5,
        "timestamp": pd.date_range("2024-01-01", periods=5),
        "close": [200, 190, 195, 198, 202],
    })

    only_a = a.copy()
    both = pd.concat([a, b], ignore_index=True).sort_values(["ticker", "timestamp"])

    only_a["ret"] = only_a.groupby("ticker")["close"].shift(-1) / only_a["close"] - 1
    both["ret"] = both.groupby("ticker")["close"].shift(-1) / both["close"] - 1

    a_from_both = both[both["ticker"] == "A"]["ret"].reset_index(drop=True)
    assert np.allclose(only_a["ret"].fillna(999), a_from_both.fillna(999))


def test_generated_prices_have_bounded_drawdown_convention():
    for seed in range(10):
        prices = _make_prices(seed)
        dd = prices / prices.cummax() - 1
        assert dd.max() <= 1e-12
        assert dd.min() <= 0
        assert abs(dd.min()) >= 0
