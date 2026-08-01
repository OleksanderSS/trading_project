"""Anomaly detection must find anomalous MOVES, not unusual price levels.

The detector used to z-score the close price against the mean of the whole
series. For anything that trends, that mean is a level the price passed
through once, so it measured distance from that level. Measured on real
stored data (NVDA, KO, SPY, TSLA) with a single injected bad tick:

    +15% in one bar -> MISSED on all four
    +30% in one bar -> MISSED on all four
    +100%           -> caught

while simultaneously flagging the two highest closes of untouched KO daily
data as "spikes" -- legitimate trend extremes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.processing.filters.price_filter import PriceFilter


def _trending(n=400, start=100.0, drift=0.0015, vol=0.01, seed=0):
    rng = np.random.default_rng(seed)
    closes = start * np.cumprod(1 + rng.normal(drift, vol, n))
    return pd.DataFrame(
        {"close": closes},
        index=pd.date_range("2024-01-01", periods=n, freq="B"),
    )


def _inject(frame: pd.DataFrame, multiplier: float, at: int | None = None):
    poisoned = frame.copy()
    at = at if at is not None else len(poisoned) // 2
    poisoned.iloc[at, poisoned.columns.get_loc("close")] *= multiplier
    return poisoned, poisoned["close"].iloc[at]


@pytest.mark.parametrize("multiplier", [1.30, 1.50, 2.00, 0.70])
def test_a_single_bad_tick_is_detected(multiplier):
    frame, bad_value = _inject(_trending(), multiplier)
    found = PriceFilter({}).detect_and_classify_anomalies(frame)

    assert any(abs(a["value"] - bad_value) < 1e-6 for a in found), (
        f"a {multiplier:.0%} single-bar move went undetected"
    )


def test_a_strong_uptrend_alone_is_not_anomalous():
    """The old check flagged the highest prices of a trending series."""
    steady = pd.DataFrame(
        {"close": 100 * np.cumprod(np.repeat(1.005, 300))},
        index=pd.date_range("2024-01-01", periods=300, freq="B"),
    )
    assert PriceFilter({}).detect_and_classify_anomalies(steady) == []


def test_detection_scales_with_the_instrument_volatility():
    """The same move is less remarkable in a more volatile instrument.

    Compared by z-score, not by a yes/no flag: injecting +10% ADDS to whatever
    the bar was already doing, so it clears 3 sigma even in a wild series and a
    binary assertion would be testing the construction of the fixture rather
    than the behaviour. The scaling is the real property -- it is what stops a
    volatile name from being condemned for moving the way it always moves.
    """
    detector = PriceFilter({})

    def injected_z(vol):
        frame, value = _inject(_trending(vol=vol, seed=1), 1.10)
        hit = next(
            a for a in detector.detect_and_classify_anomalies(frame)
            if abs(a["value"] - value) < 1e-6
        )
        return abs(hit["z_score"])

    assert injected_z(0.005) > injected_z(0.02) > injected_z(0.05)


def test_reported_fields_describe_the_move():
    frame, bad_value = _inject(_trending(), 1.5)
    hit = next(
        a for a in PriceFilter({}).detect_and_classify_anomalies(frame)
        if abs(a["value"] - bad_value) < 1e-6
    )

    assert hit["type"] == "spike"
    assert hit["return_pct"] > 0.4
    assert abs(hit["z_score"]) > 3


def test_a_downward_tick_is_classified_as_a_dip():
    frame, bad_value = _inject(_trending(), 0.6)
    hit = next(
        a for a in PriceFilter({}).detect_and_classify_anomalies(frame)
        if abs(a["value"] - bad_value) < 1e-6
    )
    assert hit["type"] == "dip"
    assert hit["return_pct"] < 0


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"close": []}),
    pd.DataFrame({"close": [100.0, 101.0]}),
    pd.DataFrame({"open": [1.0, 2.0, 3.0]}),
    pd.DataFrame({"close": [100.0] * 50}),
])
def test_degenerate_input_yields_no_anomalies(frame):
    assert PriceFilter({}).detect_and_classify_anomalies(frame) == []


def test_minimum_candles_is_a_usable_sample():
    """2 was the old floor: a std, a cadence ratio and a duplicate ratio
    computed over two bars are not informative, yet such a series passed the
    gate and was then scored as if it were."""
    assert PriceFilter({}).min_candles >= 30
