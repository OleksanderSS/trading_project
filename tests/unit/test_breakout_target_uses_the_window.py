"""A 30% breakout two bars ahead was labelled "no breakout".

`target_hourly_breakout_1h` asks, in its own configured description,
"Close піднімається вище поточної верхньої смуги Боллінджера протягом
наступних 4 барів?" -- does close rise above the current upper Bollinger
band WITHIN the next four bars. The calculator answered a different
question: it took `close.shift(-4)`, the value at exactly the fourth bar.

Demonstrated on a series priced at 100 against a band at 110, spiking to
130 two bars ahead and settling back to 101 by the fourth:

    prices   100  100  100  130  101  100 ...
    label      0    0    0    0
    should     1    1    1    0

A breakout that happened and reverted read as though it never happened. 11
of the 65 champions on the 2026-08-13 batch sit on this target, and every
one of them was trained to predict the wrong event.

The window ends where the shift ends, so purge and horizon accounting are
unchanged.
"""
import numpy as np
import pandas as pd
import pytest

from src.targets.calculators.classification_calculator import (
    ClassificationCalculator,
)

BAND = 110.0


@pytest.fixture
def calculator():
    return ClassificationCalculator()


def _frame(closes, ticker="AAPL"):
    return pd.DataFrame({
        "ticker": [ticker] * len(closes),
        "datetime": pd.date_range("2026-01-01", periods=len(closes),
                                  freq="h", tz="UTC"),
        "close": closes,
        "BB_Upper_60m": [BAND] * len(closes),
    })


def _labels(calculator, frame, shift=-4):
    return calculator.calculate_binary(
        frame, base_col="close", shift=shift, threshold=0.0,
        indicator_col="BB_Upper_60m",
    )


def test_a_break_inside_the_window_counts(calculator):
    labels = _labels(calculator, _frame([100, 100, 100, 130, 101, 100,
                                         100, 100, 100, 100]))

    assert labels.iloc[0] == 1.0, "the spike at bar 3 is inside bar 0's window"
    assert labels.iloc[1] == 1.0
    assert labels.iloc[2] == 1.0


def test_a_break_already_behind_does_not_count(calculator):
    labels = _labels(calculator, _frame([100, 100, 100, 130, 101, 100,
                                         100, 100, 100, 100]))

    assert labels.iloc[3] == 0.0, "bar 3 looks forward, not at itself"


def test_a_window_that_runs_off_the_end_is_unlabelled(calculator):
    labels = _labels(calculator, _frame([100] * 10))

    assert labels.iloc[-4:].isna().all(), (
        "a partial window would label from fewer bars than the target claims"
    )


def test_the_window_never_crosses_into_another_ticker(calculator):
    """MSFT's first bar is 999. It must not break AAPL's band."""
    frame = pd.concat([
        _frame([100] * 5, ticker="AAPL"),
        _frame([999, 100, 100, 100, 100], ticker="MSFT"),
    ], ignore_index=True)

    labels = _labels(calculator, frame)

    assert labels.iloc[0] == 0.0, "AAPL saw MSFT's opening price"


def test_a_flat_series_below_the_band_is_never_a_breakout(calculator):
    labels = _labels(calculator, _frame([100] * 12))

    assert (labels.dropna() == 0.0).all()


def test_the_endpoint_reading_is_gone(calculator):
    """The regression, stated as the difference it makes.

    Under the old rule the only bar that mattered was the last one in the
    window, so a series that ends high scores 1 and one that spikes and
    reverts scores 0. Now both are breakouts, which is what the word means.
    """
    reverted = _labels(calculator, _frame([100, 100, 100, 130, 101, 100,
                                           100, 100, 100, 100])).iloc[0]
    ends_high = _labels(calculator, _frame([100, 100, 100, 100, 130, 100,
                                            100, 100, 100, 100])).iloc[0]

    assert reverted == 1.0 and ends_high == 1.0
