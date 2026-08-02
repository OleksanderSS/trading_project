"""True Range compares this bar's extremes against the PREVIOUS close.

    TR = max( H - L , |H - C_prev| , |L - C_prev| )

TechnicalIndicators.calculate_atr shifted the wrong series:

    high_close = np.abs(high.shift(1) - close)

which is the PREVIOUS high against the CURRENT close -- not a range anyone
defines. It disagrees with the real thing on 3 of 4 bars of a worked example,
in both directions, and it is live: technical_analysis_enricher maps this
function into the feature pipeline, so it produces the AATR_* family that
every model trains on.

Measured on stored bars, old value against new:

    AAPL 1d    9.9814 -> 8.0936   -18.91%
    TSLA 1d   17.9771 -> 15.9171  -11.46%
    NVDA 15m   0.7914 ->  0.7216   -8.81%
    SPY  1d    7.8314 ->  7.8200   -0.15%

Systematically OVERSTATED, most for instruments that gap.

Separately, RiskRewardCalculator.calculate_trade_parameters -- the
stop-loss/take-profit machinery -- called VolatilityCalculator.calculate_atr,
a method that does not exist on that class, so it raised AttributeError on
every call. It has no callers, so nothing was crashing; but nothing in
src/trading or src/risk computes ATR-based stops either, which is the more
interesting half.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.calculators.risk_reward_calculator import (
    RiskRewardCalculator,
    TradeParameters,
)
from src.features.utils.technical_indicators_lib import TechnicalIndicators


def _bars(n=40, seed=0):
    rng = np.random.default_rng(seed)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1.0, n)))
    return pd.DataFrame({
        "high": close + rng.uniform(0.5, 2.0, n),
        "low": close - rng.uniform(0.5, 2.0, n),
        "close": close,
    })


def test_true_range_uses_the_previous_close():
    """Stated as arithmetic: one bar, computed by hand."""
    high = pd.Series([10.0, 12.0])
    low = pd.Series([8.0, 9.0])
    close = pd.Series([9.0, 11.0])

    atr = TechnicalIndicators.calculate_atr(high, low, close, period=1)

    # Bar 1: max(12-9, |12-9|, |9-9|) = 3.0
    assert atr.iloc[1] == pytest.approx(3.0)


def test_a_gap_up_widens_the_range():
    """The whole point of using the previous close: an opening gap is part
    of the range even when the bar itself is narrow."""
    high = pd.Series([10.0, 20.5])
    low = pd.Series([9.0, 20.0])
    close = pd.Series([10.0, 20.2])

    atr = TechnicalIndicators.calculate_atr(high, low, close, period=1)

    assert atr.iloc[1] == pytest.approx(10.5)  # |20.5 - 10.0|, not 0.5


def test_the_old_shift_would_have_missed_that_gap():
    high = pd.Series([10.0, 20.5])
    low = pd.Series([9.0, 20.0])
    close = pd.Series([10.0, 20.2])

    old = np.maximum(
        high - low,
        np.maximum(np.abs(high.shift(1) - close), np.abs(low.shift(1) - close)),
    )

    assert old.iloc[1] == pytest.approx(11.2)
    assert old.iloc[1] != TechnicalIndicators.calculate_atr(
        high, low, close, period=1
    ).iloc[1]


def test_atr_is_never_negative():
    frame = _bars()
    atr = TechnicalIndicators.calculate_atr(
        frame["high"], frame["low"], frame["close"], period=14
    )

    assert (atr.dropna() >= 0).all()


def test_atr_needs_a_full_window_before_reporting():
    frame = _bars(n=20)
    atr = TechnicalIndicators.calculate_atr(
        frame["high"], frame["low"], frame["close"], period=14
    )

    # 14 non-null True Range values are needed, and the first TR is NaN
    # because there is no previous close for it -- so the series starts at
    # index 14, not 13.
    assert atr.iloc[:14].isna().all()
    assert not pd.isna(atr.iloc[14])


def test_trade_levels_can_be_computed_at_all():
    """It raised AttributeError on every call."""
    result = RiskRewardCalculator.calculate_trade_parameters(
        TradeParameters(df=_bars(), signal_type="BUY", entry_price=100.0)
    )

    assert result["risk_amount"] > 0


def test_a_buy_puts_the_stop_below_and_the_target_above():
    result = RiskRewardCalculator.calculate_trade_parameters(
        TradeParameters(df=_bars(), signal_type="BUY", entry_price=100.0)
    )

    assert result["stop_loss"] < 100.0 < result["take_profit"]


def test_a_sell_mirrors_it():
    result = RiskRewardCalculator.calculate_trade_parameters(
        TradeParameters(df=_bars(), signal_type="SELL", entry_price=100.0)
    )

    assert result["take_profit"] < 100.0 < result["stop_loss"]


def test_an_unknown_signal_yields_no_levels():
    result = RiskRewardCalculator.calculate_trade_parameters(
        TradeParameters(df=_bars(), signal_type="HOLD", entry_price=100.0)
    )

    assert result["stop_loss"] == 0.0
    assert result["take_profit"] == 0.0


def test_the_risk_reward_ratio_is_the_configured_multiplier_by_construction():
    """Recorded, not changed: risk = ATR*atr_mult and reward = risk*tp_mult,
    so the ratio is always tp_multiplier and carries no information about the
    trade. It has no consumers today; anyone who filters on it would be
    filtering on a constant."""
    result = RiskRewardCalculator.calculate_trade_parameters(
        TradeParameters(df=_bars(), signal_type="BUY", entry_price=100.0)
    )

    assert result["risk_reward_ratio"] == pytest.approx(3.0)
