"""Leakage is a property of one instrument's history, not of a panel.

The 2026-08-05 run reported EMA_100_1d (0.952) and EMA_200_1d (0.958) as
possible leakage against target_ema_20_f1. Neither leaks. Pooling every
ticker's rows makes any two price-level series correlate at ~0.95, because
both are "the price level of this instrument" and the instruments differ --
median prices in that export span $5 to $955 across 110 tickers.

Measured on the real data:

                pooled      within-ticker median
    close       0.9394      0.0000
    EMA_100_1d  0.9521     -0.0008
    EMA_200_1d  0.9588      0.0012

`close` scoring the same is the giveaway: if the 100-day average leaks then
so does the price, and the warning means nothing.

It failed the other way too. A feature that genuinely leaks WITHIN a ticker
can be diluted below the threshold once a hundred other instruments are
mixed in. Pooling both invented leakage and could hide it. Neither was
visible while the panel was 22 tickers of similar price; growing it to 110
exposed both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.validation.feature_leakage_guard import FeatureLeakageGuard


@pytest.fixture()
def guard():
    return FeatureLeakageGuard(block_on_forbidden=False, report_dir=None)


def _panel(levels, rows=300, seed=0):
    """One frame, several tickers, each at its own price level."""
    generator = np.random.default_rng(seed)
    frames = []
    for name, level in levels.items():
        walk = level * (1.0 + generator.normal(0, 0.01, rows)).cumprod()
        frames.append(pd.DataFrame({
            "ticker": [name] * rows,
            "close": walk,
            # A genuine feature: this bar's return, unrelated to the target.
            "ret": np.r_[0.0, np.diff(walk) / walk[:-1]],
            "target_next_return": np.r_[
                np.diff(walk) / walk[:-1], 0.0
            ] + generator.normal(0, 0.02, rows),
        }))
    return pd.concat(frames, ignore_index=True)


def test_price_levels_across_a_diverse_panel_are_not_leakage(guard):
    """The false positive: $5 and $950 instruments in one frame."""
    frame = _panel({"CHEAP": 5.0, "MID": 100.0, "DEAR": 950.0})

    report = guard.check(
        frame,
        feature_cols=["close", "ret"],
        target_cols=["target_next_return"],
        ticker="panel",
    )

    assert report.status == "clean", report.high_corr_cols


def test_a_feature_leaking_inside_one_ticker_is_still_caught(guard):
    """The check must keep working. One instrument's feature is the target."""
    frame = _panel({"CHEAP": 5.0, "MID": 100.0, "DEAR": 950.0})
    leaking = frame["ticker"] == "MID"
    frame.loc[leaking, "ret"] = frame.loc[leaking, "target_next_return"]

    report = guard.check(
        frame,
        feature_cols=["close", "ret"],
        target_cols=["target_next_return"],
        ticker="panel",
    )

    assert "ret" in report.high_corr_cols


def test_leakage_in_one_ticker_is_not_diluted_by_the_others(guard):
    """The second failure mode. Averaging across a panel would bury this;
    the worst instrument has to decide."""
    levels = {f"T{i}": 50.0 + i for i in range(20)}
    frame = _panel(levels, rows=200)
    leaking = frame["ticker"] == "T7"
    frame.loc[leaking, "ret"] = frame.loc[leaking, "target_next_return"]

    report = guard.check(
        frame,
        feature_cols=["close", "ret"],
        target_cols=["target_next_return"],
        ticker="panel",
    )

    assert "ret" in report.high_corr_cols, (
        "leakage in 1 of 20 tickers was averaged away"
    )


def test_a_single_ticker_frame_still_works(guard):
    """The pooled path remains for frames that are one instrument."""
    frame = _panel({"ONLY": 100.0})
    frame["ret"] = frame["target_next_return"]

    report = guard.check(
        frame,
        feature_cols=["close", "ret"],
        target_cols=["target_next_return"],
        ticker="ONLY",
    )

    assert "ret" in report.high_corr_cols


def test_a_ticker_with_too_few_rows_is_skipped_not_trusted(guard):
    """A correlation from a handful of points is noise. Skipping is honest;
    scoring it 'clean' would be a claim nothing supports."""
    frame = _panel({"BIG": 100.0}, rows=300)
    tiny = _panel({"TINY": 100.0}, rows=5, seed=1)
    tiny["ret"] = tiny["target_next_return"]

    report = guard.check(
        pd.concat([frame, tiny], ignore_index=True),
        feature_cols=["close", "ret"],
        target_cols=["target_next_return"],
        ticker="panel",
    )

    assert report.status == "clean"


def test_a_constant_target_does_not_raise(guard):
    frame = _panel({"A": 100.0, "B": 200.0})
    frame["target_next_return"] = 1.0

    assert guard.check(
        frame,
        feature_cols=["close"],
        target_cols=["target_next_return"],
        ticker="panel",
    ).status in {"clean", "not_checked"}
