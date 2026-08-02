"""ATR_14 was a rolling mean of true ranges, not Wilder's ATR.

A plain rolling mean is a legitimate volatility measure, but "ATR" without
qualification means Wilder's smoothing everywhere it is defined -- every
charting platform, every reference, every threshold taken from the
literature. A feature named ATR_14 that is something else compares silently
wrong against all of them.

The behavioural difference is the memory. Wilder's has an effective span of
about 2n-1 and decays away from a volatility spike gradually; a rolling mean
carries the spike at full weight and then drops it abruptly the moment it
leaves the window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.utils.technical_indicators_lib import TechnicalIndicators


#: A quiet series with one violent bar (index 5), so the two smoothings have
#: to disagree in a way anyone can read.
HIGH = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 15.0, 15.0])
LOW = pd.Series([9.0, 9.5, 11.0, 12.0, 13.0, 13.0, 14.0, 14.0])
CLOSE = pd.Series([9.5, 10.5, 11.5, 12.5, 13.5, 14.0, 14.5, 14.5])


def _true_range():
    previous_close = CLOSE.shift(1)
    return np.maximum(
        HIGH - LOW,
        np.maximum((HIGH - previous_close).abs(), (LOW - previous_close).abs()),
    )


def test_it_matches_a_hand_computed_wilder_series():
    """Seed with the mean of the first `period` VALID true ranges, then
    ATR_t = ((n-1)*ATR_{t-1} + TR_t) / n."""
    true_range = _true_range()
    expected = [np.nan] * len(true_range)
    # TR[0] is NaN (no previous close), so the third valid range is at index 3.
    expected[3] = true_range[1:4].mean()
    for position in range(4, len(true_range)):
        expected[position] = (expected[position - 1] * 2 + true_range[position]) / 3

    result = TechnicalIndicators.calculate_atr(HIGH, LOW, CLOSE, period=3)

    assert np.allclose(np.array(expected, dtype=float)[3:], result.to_numpy()[3:])


def test_no_value_appears_before_enough_valid_ranges_exist():
    """TR[0] cannot be computed, so an ATR at index 2 would be built from
    two ranges and a NaN."""
    result = TechnicalIndicators.calculate_atr(HIGH, LOW, CLOSE, period=3)

    assert result.iloc[:3].isna().all()
    assert not np.isnan(result.iloc[3])


def test_a_spike_decays_instead_of_being_dropped():
    """The reason the distinction matters for a model."""
    wilder = TechnicalIndicators.calculate_atr(HIGH, LOW, CLOSE, period=3)
    sma = TechnicalIndicators.calculate_atr(
        HIGH, LOW, CLOSE, period=3, smoothing='sma'
    )

    # Both react identically to the spike itself...
    assert wilder.iloc[5] == pytest.approx(sma.iloc[5])
    # ...and then diverge: Wilder decays, the rolling mean holds the spike at
    # full weight until it falls out of the window.
    assert wilder.iloc[6] < sma.iloc[6]
    assert wilder.iloc[7] < sma.iloc[7]


def test_the_old_behaviour_is_still_reachable():
    true_range = _true_range()
    expected = true_range.rolling(3, min_periods=3).mean()

    result = TechnicalIndicators.calculate_atr(
        HIGH, LOW, CLOSE, period=3, smoothing='sma'
    )

    assert np.allclose(
        expected.dropna().to_numpy(), result.dropna().to_numpy()
    )


def test_an_unknown_smoothing_is_refused_not_silently_defaulted():
    """Falling back to a default would make a typo change the indicator."""
    with pytest.raises(ValueError, match="wilder"):
        TechnicalIndicators.calculate_atr(HIGH, LOW, CLOSE, smoothing='ema')


def test_atr_is_never_negative():
    result = TechnicalIndicators.calculate_atr(HIGH, LOW, CLOSE, period=3)

    assert (result.dropna() >= 0).all()


def test_a_series_shorter_than_the_period_yields_all_nan():
    short = pd.Series([1.0, 2.0])

    result = TechnicalIndicators.calculate_atr(short, short, short, period=14)

    assert result.isna().all()


def test_the_true_range_still_uses_the_previous_close():
    """Guarding the earlier fix: it was np.abs(high.shift(1) - close), the
    previous HIGH against the current close, which is not a range anyone
    defines."""
    result = TechnicalIndicators.calculate_atr(HIGH, LOW, CLOSE, period=3)
    true_range = _true_range()

    # The seed is exactly the mean of the first three valid true ranges.
    assert result.iloc[3] == pytest.approx(true_range[1:4].mean())
