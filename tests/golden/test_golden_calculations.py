"""
Golden calculation tests.

These tests compare deterministic expected values for basic target/risk calculations.
They can be extended to call project calculators once their public API is stable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FIXTURES = Path(__file__).parent / "fixtures"


def test_golden_forward_return_per_ticker():
    prices = pd.read_csv(FIXTURES / "golden_prices_multi_ticker.csv", parse_dates=["timestamp"])
    expected = pd.read_csv(FIXTURES / "expected_forward_return_1.csv", parse_dates=["timestamp"])

    prices["actual_forward_return_1"] = prices.groupby("ticker")["close"].shift(-1) / prices["close"] - 1
    merged = prices.merge(expected, on=["ticker", "timestamp"], how="inner")

    for _, row in merged.iterrows():
        exp = row["expected_forward_return_1"]
        act = row["actual_forward_return_1"]
        if pd.isna(exp):
            assert pd.isna(act)
        else:
            assert np.isclose(act, exp, atol=1e-8)


def test_golden_drawdown_calculation():
    equity = pd.Series([100.0, 120.0, 90.0, 110.0, 130.0])
    drawdown = equity / equity.cummax() - 1
    assert np.isclose(drawdown.min(), -0.25)
    assert np.isclose(abs(drawdown.min()), 0.25)
