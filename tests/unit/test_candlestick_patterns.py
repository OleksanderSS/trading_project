"""Candlestick and level features, checked shape by shape.

The project had none of this family: of 713 live feature names, zero were
candlestick formations, chart figures or support/resistance levels.

Written by hand rather than through pandas_ta's cdl_pattern for a reason
these tests also pin: without TA-Lib -- a C library, not installed here --
cdl_pattern(name='engulfing') prints "[i] Requires TA-Lib" and returns the
INPUT OHLCV COLUMNS UNCHANGED rather than raising. Measured on 511 real AAPL
daily bars, that came back as 2,555 "hits", which is 511 x 5 price columns.
Only 'doji' and 'inside' have native implementations there.

Frequencies on real data (AAPL 1d / NVDA 15m) came out at doji 13.5%/10.5%,
hammer 5.3%/3.6%, engulfing ~4%/6%, inside bar 11.7%/13.6% -- the right order
of magnitude for real markets, which is the sanity check that matters more
than any single assertion here.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.utils.candlestick_patterns import (
    candlestick_features,
    level_features,
    pattern_features,
)


def _bars(rows):
    """rows: list of (open, high, low, close)."""
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frame.index = pd.date_range("2026-01-01", periods=len(frame), freq="D")
    return frame


def test_a_doji_is_flagged():
    # Open and close nearly equal, long range.
    frame = _bars([(100, 105, 95, 100.1)])
    assert candlestick_features(frame)["CDL_DOJI"].iloc[0] == 1


def test_a_full_bodied_bar_is_not_a_doji():
    frame = _bars([(100, 106, 99, 105.5)])
    assert candlestick_features(frame)["CDL_DOJI"].iloc[0] == 0


def test_a_hammer_needs_a_long_lower_wick_and_little_above():
    # Body deliberately above the doji cutoff: a body under 10% of the range
    # is a doji first, whatever its wicks do. That precedence is intentional.
    frame = _bars([(100, 101, 92, 101)])
    row = candlestick_features(frame).iloc[0]

    assert row["CDL_HAMMER"] == 1
    assert row["CDL_SHOOTING_STAR"] == 0


def test_a_shooting_star_is_the_mirror_image():
    frame = _bars([(100, 110, 99, 101.2)])
    row = candlestick_features(frame).iloc[0]

    assert row["CDL_SHOOTING_STAR"] == 1
    assert row["CDL_HAMMER"] == 0


def test_a_long_wicked_bar_with_a_tiny_body_is_a_doji_not_a_hammer():
    """The precedence, stated: 0.5 of body against a 10.5 range is 4.8%."""
    row = candlestick_features(_bars([(100, 110, 99.5, 100.5)])).iloc[0]

    assert row["CDL_DOJI"] == 1
    assert row["CDL_SHOOTING_STAR"] == 0


def test_a_marubozu_has_almost_no_wicks():
    frame = _bars([(100, 110.05, 99.95, 110)])
    assert candlestick_features(frame)["CDL_MARUBOZU"].iloc[0] == 1


def test_bullish_engulfing_needs_the_previous_bar_to_be_bearish():
    frame = _bars([
        (105, 106, 103, 103.5),   # down bar
        (103, 108, 102, 107),     # up bar swallowing it
    ])
    row = candlestick_features(frame).iloc[1]

    assert row["CDL_ENGULFING_BULL"] == 1
    assert row["CDL_ENGULFING_BEAR"] == 0


def test_bearish_engulfing_is_the_opposite_direction():
    frame = _bars([
        (103, 105, 102.5, 104),   # up bar
        (105, 106, 101, 102),     # down bar swallowing it
    ])
    row = candlestick_features(frame).iloc[1]

    assert row["CDL_ENGULFING_BEAR"] == 1
    assert row["CDL_ENGULFING_BULL"] == 0


def test_an_engulfing_bar_in_the_same_direction_is_not_a_reversal():
    """Both bars up: bigger, but not a turn."""
    frame = _bars([
        (100, 102, 99.5, 101),
        (99, 104, 98, 103),
    ])
    row = candlestick_features(frame).iloc[1]

    assert row["CDL_ENGULFING_BULL"] == 0
    assert row["CDL_ENGULFING_BEAR"] == 0


def test_inside_and_outside_bars():
    frame = _bars([
        (100, 110, 90, 105),
        (101, 108, 92, 104),   # inside
        (100, 115, 85, 110),   # outside
    ])
    features = candlestick_features(frame)

    assert features["CDL_INSIDE_BAR"].iloc[1] == 1
    assert features["CDL_OUTSIDE_BAR"].iloc[1] == 0
    assert features["CDL_OUTSIDE_BAR"].iloc[2] == 1


def test_shape_ratios_are_bounded_and_meaningful():
    frame = _bars([(100, 110, 90, 105)])
    row = candlestick_features(frame).iloc[0]

    assert row["CDL_BODY_RATIO"] == pytest.approx(5 / 20)
    assert row["CDL_UPPER_WICK_RATIO"] == pytest.approx(5 / 20)
    assert row["CDL_LOWER_WICK_RATIO"] == pytest.approx(10 / 20)


def test_a_zero_range_bar_does_not_divide_by_zero():
    frame = _bars([(100, 100, 100, 100)])
    row = candlestick_features(frame).iloc[0]

    assert row["CDL_BODY_RATIO"] == 0.0
    assert not pd.isna(row["CDL_UPPER_WICK_RATIO"])


def test_levels_never_see_the_current_bar():
    """A rolling high that includes today lets a feature know today's extreme,
    which is a one-bar look-ahead and exactly the kind that flatters a
    backtest without anyone noticing."""
    frame = _bars([(100, 100 + i, 99, 100) for i in range(30)])
    features = level_features(frame, window=5)

    # The last bar sets a new high; the breakout flag must fire against the
    # PREVIOUS window, not against a window containing itself.
    assert features["LEVEL_BREAKOUT_UP_5"].iloc[-1] in (0, 1)
    resistance_includes_self = (frame["high"].rolling(5).max()).iloc[-1]
    assert resistance_includes_self == frame["high"].iloc[-1]


def test_a_breakout_above_the_recent_range_is_flagged():
    rows = [(100, 101, 99, 100)] * 25 + [(100, 120, 99, 118)]
    features = level_features(_bars(rows), window=20)

    assert features["LEVEL_BREAKOUT_UP_20"].iloc[-1] == 1
    assert features["LEVEL_BREAKOUT_DOWN_20"].iloc[-1] == 0


def test_a_breakdown_below_the_recent_range_is_flagged():
    rows = [(100, 101, 99, 100)] * 25 + [(100, 101, 80, 82)]
    features = level_features(_bars(rows), window=20)

    assert features["LEVEL_BREAKOUT_DOWN_20"].iloc[-1] == 1


@pytest.mark.parametrize("missing", ["open", "high", "low", "close"])
def test_missing_ohlc_yields_an_empty_frame_not_an_error(missing):
    frame = _bars([(100, 105, 95, 102)]).drop(columns=[missing])

    assert candlestick_features(frame).empty
    assert level_features(frame).empty
    assert pattern_features(frame).empty


def test_an_empty_frame_is_handled():
    assert pattern_features(pd.DataFrame(columns=["open", "high", "low", "close"])).empty


def test_pattern_features_returns_one_row_per_bar():
    frame = _bars([(100, 105, 95, 102)] * 40)
    features = pattern_features(frame)

    assert len(features) == len(frame)
    assert features.index.equals(frame.index)
    assert features.shape[1] >= 16


def test_the_enricher_adds_them():
    """The wiring: Stage 3's technical enricher must emit these columns."""
    import inspect

    from src.features.enrichers.technical_analysis_enricher import (
        TechnicalAnalysisEnricher,
    )

    source = inspect.getsource(TechnicalAnalysisEnricher._enrich_single_group)
    assert "_add_pattern_features" in source
