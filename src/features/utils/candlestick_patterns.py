"""Per-bar candlestick and price-structure features.

The project had none of these. Of 713 live feature names, zero were
candlestick formations, chart figures or support/resistance levels -- the
whole family was missing, and the only module that once addressed it
(src/archive/patterns/pattern_analyzer.py) inspects `iloc[-1]` alone, so it
is a latest-bar verdict rather than something a model can train on.

Written by hand rather than delegated to pandas_ta's `cdl_pattern`, for a
reason worth recording: without TA-Lib installed -- and it is not, being a C
library -- `cdl_pattern(name='engulfing')` prints "[i] Requires TA-Lib" and
RETURNS THE INPUT OHLCV COLUMNS UNCHANGED. It does not raise. Feeding that
into a feature frame would have silently produced five copies of the price
columns under pattern names. Only 'doji' and 'inside' have native
implementations there.

Every function returns one value per bar, aligned to the input index, and
uses only information available at that bar or earlier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# A body smaller than this share of the bar's range counts as "no body".
_DOJI_BODY_RATIO = 0.1
# A wick this many times the body makes a hammer / shooting star.
_WICK_BODY_RATIO = 2.0
# Above this share of the range with no wicks, the bar is a marubozu.
_MARUBOZU_BODY_RATIO = 0.9

REQUIRED_COLUMNS = ("open", "high", "low", "close")


def _parts(frame: pd.DataFrame) -> tuple[pd.Series, ...]:
    """Body, range, and the two wicks -- the vocabulary every pattern uses."""
    open_, high, low, close = (frame[c].astype(float) for c in REQUIRED_COLUMNS)
    body = (close - open_).abs()
    span = (high - low).replace(0.0, np.nan)
    upper = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower = pd.concat([open_, close], axis=1).min(axis=1) - low
    return open_, high, low, close, body, span, upper, lower


def has_required_columns(frame: pd.DataFrame) -> bool:
    return all(column in frame.columns for column in REQUIRED_COLUMNS)


def candlestick_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-bar candlestick flags and shape ratios.

    Returns an empty frame (not an error, and not silence) when the OHLC
    columns are absent, so a caller can tell "not applicable" from "nothing
    found".
    """
    if not has_required_columns(frame) or frame.empty:
        return pd.DataFrame(index=frame.index)

    open_, high, low, close, body, span, upper, lower = _parts(frame)
    body_ratio = (body / span).fillna(0.0)
    upper_ratio = (upper / span).fillna(0.0)
    lower_ratio = (lower / span).fillna(0.0)
    bullish = close > open_

    previous_open = open_.shift(1)
    previous_close = close.shift(1)
    previous_bullish = previous_close > previous_open

    out = pd.DataFrame(index=frame.index)

    # Shape, as continuous features -- a model can use these directly, and
    # they do not throw away information the way a boolean does.
    out["CDL_BODY_RATIO"] = body_ratio
    out["CDL_UPPER_WICK_RATIO"] = upper_ratio
    out["CDL_LOWER_WICK_RATIO"] = lower_ratio

    out["CDL_DOJI"] = (body_ratio < _DOJI_BODY_RATIO).astype("int8")

    # Long lower wick, little above: buyers rejected the low.
    out["CDL_HAMMER"] = (
        (lower > body * _WICK_BODY_RATIO)
        & (upper < body)
        & (body_ratio >= _DOJI_BODY_RATIO)
    ).astype("int8")

    # The mirror image: sellers rejected the high.
    out["CDL_SHOOTING_STAR"] = (
        (upper > body * _WICK_BODY_RATIO)
        & (lower < body)
        & (body_ratio >= _DOJI_BODY_RATIO)
    ).astype("int8")

    out["CDL_MARUBOZU"] = (body_ratio > _MARUBOZU_BODY_RATIO).astype("int8")

    # This bar's body swallows the previous one, and flips direction.
    engulfs = (
        (pd.concat([open_, close], axis=1).min(axis=1)
         <= pd.concat([previous_open, previous_close], axis=1).min(axis=1))
        & (pd.concat([open_, close], axis=1).max(axis=1)
           >= pd.concat([previous_open, previous_close], axis=1).max(axis=1))
        & (body > (previous_close - previous_open).abs())
    )
    out["CDL_ENGULFING_BULL"] = (engulfs & bullish & ~previous_bullish).astype("int8")
    out["CDL_ENGULFING_BEAR"] = (engulfs & ~bullish & previous_bullish).astype("int8")

    # Whole bar inside the previous one: compression, often precedes a break.
    out["CDL_INSIDE_BAR"] = (
        (high <= high.shift(1)) & (low >= low.shift(1))
    ).astype("int8")

    out["CDL_OUTSIDE_BAR"] = (
        (high >= high.shift(1)) & (low <= low.shift(1))
    ).astype("int8")

    return out


def level_features(frame: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Where price sits relative to its recent range.

    Support and resistance as continuous distances rather than flags: how far
    the close is from the rolling high and low, as a share of that range. The
    rolling windows are closed on the left (`shift(1)`) so a bar never sees
    its own extreme -- that would be a look-ahead of exactly one bar, which is
    the sort of thing that quietly flatters a backtest.
    """
    if not has_required_columns(frame) or frame.empty:
        return pd.DataFrame(index=frame.index)

    _, high, low, close, *_ = _parts(frame)
    resistance = high.rolling(window=window, min_periods=2).max().shift(1)
    support = low.rolling(window=window, min_periods=2).min().shift(1)
    span = (resistance - support).replace(0.0, np.nan)

    out = pd.DataFrame(index=frame.index)
    out[f"LEVEL_POSITION_{window}"] = ((close - support) / span).clip(-5, 5)
    out[f"LEVEL_DIST_RESISTANCE_{window}"] = ((resistance - close) / close).clip(-5, 5)
    out[f"LEVEL_DIST_SUPPORT_{window}"] = ((close - support) / close).clip(-5, 5)
    out[f"LEVEL_BREAKOUT_UP_{window}"] = (close > resistance).astype("int8")
    out[f"LEVEL_BREAKOUT_DOWN_{window}"] = (close < support).astype("int8")
    return out


def pattern_features(frame: pd.DataFrame, level_window: int = 20) -> pd.DataFrame:
    """Everything in this module, in one frame."""
    parts = [candlestick_features(frame), level_features(frame, window=level_window)]
    parts = [part for part in parts if not part.empty]
    if not parts:
        return pd.DataFrame(index=frame.index)
    return pd.concat(parts, axis=1)
