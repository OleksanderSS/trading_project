"""Two sizing inputs that were wrong in the direction that costs money.

1. RiskRewardCalculator produced negative stop-loss and take-profit prices
   whenever ATR was large relative to price, and reported a reward:risk that
   was tautologically tp_multiplier for every trade ever evaluated.

2. EliteRiskSizer annualised with a flat sqrt(252) regardless of bar
   cadence. This project runs 15m/60m/1d side by side; annualising a
   15-minute series as daily understates volatility by ~sqrt(26), and since
   vol_factor is portfolio_vol / ticker_vol, understating the denominator
   sizes the position LARGER.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.analytics.calculators.risk_reward_calculator import (
    MINIMUM_PRICE_FRACTION,
    RiskRewardCalculator,
    TradeConfig,
    TradeParameters,
)
from src.trading.elite_risk_sizer import EliteRiskSizer


def _bars(*, close, spread):
    """OHLC whose true range is `spread` on every bar, so ATR == spread."""
    highs = [close + spread / 2] * 30
    lows = [close - spread / 2] * 30
    return pd.DataFrame({
        "high": highs, "low": lows, "close": [close] * 30,
    })


def _levels(close, spread, signal="BUY", config=None):
    return RiskRewardCalculator.calculate_trade_parameters(
        TradeParameters(df=_bars(close=close, spread=spread),
                        signal_type=signal, entry_price=close),
        config or TradeConfig(),
    )


def test_an_ordinary_trade_is_unchanged():
    result = _levels(close=100.0, spread=2.0)

    assert result["stop_loss"] == pytest.approx(96.0)
    assert result["take_profit"] == pytest.approx(112.0)
    assert result["risk_reward_ratio"] == pytest.approx(3.0)
    assert result["levels_clamped"] is False


def test_a_stop_loss_can_no_longer_be_a_negative_price():
    """ATR 60 on a $10 instrument: risk 120, so entry - risk was -110."""
    result = _levels(close=10.0, spread=60.0)

    assert result["stop_loss"] > 0
    assert result["stop_loss"] == pytest.approx(10.0 * MINIMUM_PRICE_FRACTION)
    assert result["levels_clamped"] is True


def test_a_take_profit_can_no_longer_be_a_negative_price():
    """The SELL mirror: entry - risk*3 went far below zero."""
    result = _levels(close=10.0, spread=60.0, signal="SELL")

    assert result["take_profit"] > 0
    assert result["levels_clamped"] is True


def test_the_reported_ratio_is_no_longer_tp_multiplier_by_construction():
    """It was reward=risk*3 over risk=risk, i.e. exactly 3.0, always.

    Anything filtering trades on risk_reward_ratio was filtering a constant.
    """
    clamped = _levels(close=10.0, spread=60.0)

    assert clamped["risk_reward_ratio"] != pytest.approx(3.0)
    assert clamped["risk_reward_ratio"] > 0


def test_the_risk_amount_reflects_the_level_actually_used():
    """Reporting the unclamped distance would tell risk sizing the stop sits
    further away than it does."""
    result = _levels(close=10.0, spread=60.0)

    assert result["risk_amount"] == pytest.approx(
        abs(10.0 - result["stop_loss"])
    )


def test_a_clamp_is_announced(caplog):
    with caplog.at_level(logging.WARNING):
        _levels(close=10.0, spread=60.0)

    assert any("clamped" in record.message.lower() for record in caplog.records)


def test_an_unknown_signal_still_returns_zeros():
    assert _levels(close=100.0, spread=2.0, signal="HOLD") == {
        "stop_loss": 0.0, "take_profit": 0.0, "risk_reward_ratio": 0.0,
    }


# --- volatility cadence -----------------------------------------------------

def _returns(periods, freq, sigma=0.01, seed=0):
    generator = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=periods, freq=freq)
    return pd.Series(generator.normal(0, sigma, periods), index=index)


def test_intraday_returns_are_not_annualised_as_daily():
    """The defect, measured: same per-bar sigma, different cadence."""
    sizer = EliteRiskSizer()
    sizer.update_returns_data("DAILY", _returns(400, "D"))
    sizer.update_returns_data("HOURLY", _returns(400, "h"))

    daily = sizer._estimate_ticker_volatility("DAILY")
    hourly = sizer._estimate_ticker_volatility("HOURLY")

    # 252*7 hourly periods against 252 daily -> sqrt(7) apart, not equal.
    assert hourly > daily * 2.0
    assert hourly / daily == pytest.approx(np.sqrt(7), rel=0.15)


def test_daily_returns_still_annualise_by_sqrt_252():
    sizer = EliteRiskSizer()
    series = _returns(400, "D")
    sizer.update_returns_data("AAPL", series)

    assert sizer._estimate_ticker_volatility("AAPL") == pytest.approx(
        float(series.std()) * np.sqrt(252), rel=1e-6
    )


def test_a_series_without_timestamps_falls_back_and_says_so(caplog):
    sizer = EliteRiskSizer()
    sizer.update_returns_data("NOINDEX", pd.Series(np.full(50, 0.01)))

    with caplog.at_level(logging.WARNING):
        volatility = sizer._estimate_ticker_volatility("NOINDEX")

    assert volatility == pytest.approx(0.0)  # constant series, zero std
    assert any("daily" in record.message.lower() for record in caplog.records)


def test_understating_volatility_would_have_oversized_the_position():
    """Why the direction matters: vol_factor = portfolio_vol / ticker_vol."""
    sizer = EliteRiskSizer()
    sizer.update_returns_data("HOURLY", _returns(400, "h"))

    correct = sizer._estimate_ticker_volatility("HOURLY")
    as_if_daily = float(sizer.historical_returns["HOURLY"].std()) * np.sqrt(252)

    assert as_if_daily < correct, "the old formula understated it"
    assert (0.15 / max(as_if_daily, 0.01)) > (0.15 / max(correct, 0.01)), (
        "and a larger vol_factor means a larger position"
    )
