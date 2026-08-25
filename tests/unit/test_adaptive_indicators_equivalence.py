"""The adaptive indicators were rewritten for speed, so equality is the test.

Measured on the 110-ticker rebuild of 2026-08-24, daily frame, 7,507 rows per
ticker: `ARSI_14`, `AATR_14`, `ABB_Upper/Mid/Lower` and `AEMA_20` cost 290.3
seconds per ticker -- 76% of the whole technical-analysis step -- while SMA,
EMA, RSI, MACD, Bollinger, ATR, stochastic, Williams %R and CCI together cost
0.4 seconds. At 110 tickers that one step turned a rebuild into a day.

The cost was never the arithmetic. Each indicator sliced a pandas Series once
per row (`ret.iloc[i - p : i + 1]`), building a fresh Series with its own index
7,507 times over, four times per ticker. The rewrite slices numpy arrays
instead and leaves the arithmetic alone.

"Leaves the arithmetic alone" is a claim, and a rewrite of a feature that
already went into trained models is exactly where a silent change would hurt:
the models would keep training, the pipeline would keep passing, and the
feature would quietly mean something else. So the previous implementations are
kept below verbatim, and every test asserts the two agree exactly rather than
approximately.

The window length varies per row, which is the point of the module, so none of
this can become `.rolling(p)` -- that would change the indicator, not speed it
up.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.features.utils.adaptive_indicators import AdaptiveIndicators


# ----------------------------------------------------------------------
# The previous implementations, copied unchanged from the file's history.
# ----------------------------------------------------------------------

def _old_rsi(ai: AdaptiveIndicators, prices: pd.Series, base_period: int = 14):
    periods = ai._adaptive_period(prices, base_period)
    result = pd.Series(np.nan, index=prices.index, name=f"ARSI_{base_period}")
    ret = prices.diff()
    for i in range(len(prices)):
        p = int(periods.iloc[i]) if not pd.isna(periods.iloc[i]) else base_period
        if i < p:
            continue
        window = ret.iloc[i - p : i + 1]
        gain = window.clip(lower=0).mean()
        loss = -window.clip(upper=0).mean()
        if loss == 0:
            result.iloc[i] = 100.0
        else:
            rs = gain / loss
            result.iloc[i] = 100.0 - (100.0 / (1.0 + rs))
    return result


def _old_bollinger(ai: AdaptiveIndicators, prices: pd.Series,
                   base_period: int = 20, n_std: float = 2.0):
    periods = ai._adaptive_period(prices, base_period)
    upper = pd.Series(np.nan, index=prices.index, name="ABB_Upper")
    middle = pd.Series(np.nan, index=prices.index, name="ABB_Mid")
    lower = pd.Series(np.nan, index=prices.index, name="ABB_Lower")
    for i in range(len(prices)):
        p = int(periods.iloc[i]) if not pd.isna(periods.iloc[i]) else base_period
        if i < p - 1:
            continue
        window = prices.iloc[i - p + 1 : i + 1]
        mu = window.mean()
        sigma = window.std(ddof=1) if len(window) > 1 else 0.0
        if pd.isna(sigma) or sigma < 0:
            sigma = 0.0
        middle.iloc[i] = mu
        upper.iloc[i] = mu + n_std * sigma
        lower.iloc[i] = mu - n_std * sigma
    return upper, middle, lower


def _old_atr(ai: AdaptiveIndicators, high: pd.Series, low: pd.Series,
             close: pd.Series, base_period: int = 14):
    periods = ai._adaptive_period(close, base_period)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    result = pd.Series(np.nan, index=close.index, name=f"AATR_{base_period}")
    for i in range(len(close)):
        p = int(periods.iloc[i]) if not pd.isna(periods.iloc[i]) else base_period
        if i < p:
            continue
        result.iloc[i] = tr.iloc[i - p + 1 : i + 1].mean()
    return result


def _old_moving_average(ai: AdaptiveIndicators, prices: pd.Series,
                        base_period: int = 20, ma_type: str = "ema"):
    periods = ai._adaptive_period(prices, base_period, use_trend=True)
    result = pd.Series(
        np.nan, index=prices.index, name=f"A{ma_type.upper()}_{base_period}"
    )
    for i in range(len(prices)):
        p = int(periods.iloc[i]) if not pd.isna(periods.iloc[i]) else base_period
        if i < p - 1:
            continue
        window = prices.iloc[i - p + 1 : i + 1]
        if ma_type == "ema":
            result.iloc[i] = window.ewm(span=p, adjust=False).mean().iloc[-1]
        else:
            result.iloc[i] = window.mean()
    return result


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

def _prices(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """A random walk with changing volatility, so the period actually varies.

    A constant-volatility series would hold the adaptive period at its base
    for every row and the tests would never exercise the varying window --
    which is the only part where the rewrite could go wrong.
    """
    rng = np.random.default_rng(seed)
    vol = np.concatenate([
        np.full(n // 3, 0.004),
        np.full(n // 3, 0.030),
        np.full(n - 2 * (n // 3), 0.010),
    ])
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 1.0, n) * vol))
    spread = np.abs(rng.normal(0.0, 1.0, n)) * vol * close
    return pd.DataFrame(
        {"close": close, "high": close + spread, "low": close - spread},
        # A non-default index, because the rewrite builds Series from numpy
        # arrays and a positional/label mix-up would be invisible on a
        # RangeIndex -- the failure mode that ended the v8 rebuild.
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def ai() -> AdaptiveIndicators:
    return AdaptiveIndicators()


@pytest.fixture
def frame() -> pd.DataFrame:
    return _prices()


def _identical(new: pd.Series, old: pd.Series) -> None:
    """Equal values, equal NaN positions, equal index and name."""
    pd.testing.assert_series_equal(new, old, check_exact=True)


# ----------------------------------------------------------------------
# Equivalence
# ----------------------------------------------------------------------

def test_rsi_is_unchanged(ai, frame):
    _identical(ai.adaptive_rsi(frame["close"]), _old_rsi(ai, frame["close"]))


def test_bollinger_is_unchanged(ai, frame):
    for new, old in zip(ai.adaptive_bollinger(frame["close"]),
                        _old_bollinger(ai, frame["close"])):
        _identical(new, old)


def test_atr_is_unchanged(ai, frame):
    _identical(
        ai.adaptive_atr(frame["high"], frame["low"], frame["close"]),
        _old_atr(ai, frame["high"], frame["low"], frame["close"]),
    )


def test_ema_is_unchanged(ai, frame):
    """The one place the rewrite reimplements pandas rather than reordering it.

    `ewm(span=p, adjust=False).mean()` is a recursion, and it is now run
    directly instead of being rebuilt per row. If that recursion were seeded
    or scaled differently the values would drift slowly rather than break, so
    exact equality is what is asserted.
    """
    _identical(
        ai.adaptive_moving_average(frame["close"]),
        _old_moving_average(ai, frame["close"]),
    )


def test_sma_branch_is_unchanged(ai, frame):
    _identical(
        ai.adaptive_moving_average(frame["close"], ma_type="sma"),
        _old_moving_average(ai, frame["close"], ma_type="sma"),
    )


# ----------------------------------------------------------------------
# The awkward inputs
# ----------------------------------------------------------------------

def test_gaps_in_the_prices_are_handled_the_same(ai, frame):
    """pandas skips NaN inside a window; numpy does not unless told to.

    Real price series carry gaps -- a halted ticker, a missing bar -- and the
    difference between `mean()` and `np.mean()` on such a window is silent:
    one returns the average of what is there, the other returns NaN. This is
    also the case that sends the EMA back to pandas.
    """
    close = frame["close"].copy()
    close.iloc[50:55] = np.nan
    close.iloc[300] = np.nan

    _identical(ai.adaptive_rsi(close), _old_rsi(ai, close))
    for new, old in zip(ai.adaptive_bollinger(close), _old_bollinger(ai, close)):
        _identical(new, old)
    _identical(ai.adaptive_moving_average(close), _old_moving_average(ai, close))


def test_a_series_shorter_than_the_period(ai):
    """Every row fails the guard, so every output must be NaN, not an error."""
    short = pd.Series([100.0, 101.0, 99.5, 100.2, 101.1])
    assert ai.adaptive_rsi(short).isna().all()
    assert ai.adaptive_moving_average(short).isna().all()


def test_a_flat_series_still_gives_the_saturated_rsi(ai):
    """Zero loss meant RSI 100 before, and it has to still mean that.

    A flat window makes `loss` exactly 0, which is the branch that returns
    100.0 rather than dividing. numpy produces -0.0 where pandas produced 0.0
    here, and `-0.0 == 0` is True -- worth pinning, because if that ever
    stopped holding the result would be NaN instead of 100.
    """
    flat = pd.Series(np.full(120, 100.0))
    new, old = ai.adaptive_rsi(flat), _old_rsi(ai, flat)
    _identical(new, old)
    assert (new.dropna() == 100.0).all()


# ----------------------------------------------------------------------
# The reason for the rewrite
# ----------------------------------------------------------------------

def test_a_full_daily_history_is_not_minutes(ai):
    """7,507 rows is one ticker's daily history, and it cost 290 seconds.

    The threshold is deliberately loose -- this runs on a laptop that may be
    rebuilding at the same time, and the point is the order of magnitude, not
    a benchmark. Anything near the old cost means the pandas slicing is back.
    """
    frame = _prices(7507, seed=11)
    close, high, low = frame["close"], frame["high"], frame["low"]

    start = time.perf_counter()
    ai.adaptive_rsi(close)
    ai.adaptive_atr(high, low, close)
    ai.adaptive_bollinger(close)
    ai.adaptive_moving_average(close)
    elapsed = time.perf_counter() - start

    assert elapsed < 60.0, (
        f"the six adaptive features took {elapsed:.1f}s for 7,507 rows; "
        "they took 290s before this rewrite and should now be seconds"
    )
