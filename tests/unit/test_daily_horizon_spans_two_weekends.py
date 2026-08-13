"""A weekly target existed on Mondays and never on Thursdays.

`mask_targets_across_time_boundaries` blanks a label whose future endpoint
sits further away in wall-clock time than the horizon should reach — a guard
against a data gap being read as a valid forward window. For daily bars the
allowance was a flat three days, which covers one weekend.

A run of n trading days crosses one weekend from a Monday and two from a
Thursday. Measured on the 2026-08-13 batch, AAPL daily, target_weekly_up_1w
(shift 7, expected 7 days, allowance 10):

    from Mon/Tue/Wed   7 trading days =  9 calendar days   kept
    from Thu/Fri       7 trading days = 11 calendar days   blanked

    Mon  93 of 100    Tue  98 of 107    Wed  69 of 104
    Thu   0 of 101    Fri   0 of 102

260 labels survived of the 507 the data supports. Recomputing the target
directly agreed with every stored value, so the labels were correct and
discarded — by weekday, which is the part that matters. A model trained on
this target had never seen a Thursday or a Friday as a starting point.

`target_up_5d` escaped by arithmetic luck: from a Thursday it spans 7
calendar days against an allowance of 8.
"""
import numpy as np
import pandas as pd
import pytest

from src.targets.timeframe_contract import (
    TargetTimeframeContract,
    _maximum_elapsed,
    mask_targets_across_time_boundaries,
)


def _daily_frame(days=60):
    dates = pd.bdate_range("2026-01-05", periods=days, tz="UTC")
    return pd.DataFrame({
        "ticker": ["AAPL"] * days,
        "datetime": dates,
        "close": np.linspace(100, 160, days),
    })


def _forward_target(frame, bars):
    close = pd.to_numeric(frame["close"], errors="coerce")
    target = ((close.shift(-bars) / close - 1) > 0.0).astype(float)
    target[close.shift(-bars).isna()] = np.nan
    return target


def _contract(bars):
    expected = pd.Timedelta(days=bars)
    return TargetTimeframeContract(
        timeframe="1d", horizon=None, shift_bars=bars,
        expected_elapsed=expected,
        maximum_elapsed=_maximum_elapsed("1d", expected, bars),
    )


@pytest.mark.parametrize("bars,minimum_days", [(1, 3), (5, 7), (7, 11), (10, 14)])
def test_the_allowance_covers_the_worst_starting_weekday(bars, minimum_days):
    """n trading days from a Friday is the longest calendar span there is."""
    allowance = _maximum_elapsed("1d", pd.Timedelta(days=bars), bars)

    assert allowance >= pd.Timedelta(days=minimum_days), (
        f"{bars} trading days can span {minimum_days} calendar days; an "
        f"allowance of {allowance.days} blanks the weekdays that reach it"
    )


def test_a_weekly_horizon_survives_on_every_weekday():
    frame = _daily_frame()
    masked = mask_targets_across_time_boundaries(
        frame, _forward_target(frame, 7), _contract(7)
    )

    frame = frame.assign(kept=masked.notna().to_numpy(),
                         dow=frame["datetime"].dt.dayofweek)
    by_day = frame.groupby("dow")["kept"].sum()

    assert (by_day > 0).all(), (
        f"a weekday with no labels at all: {by_day.to_dict()} — this is the "
        f"Thursday/Friday hole, not a tail effect"
    )


def test_only_the_unreachable_tail_is_blanked():
    frame = _daily_frame(days=60)
    masked = mask_targets_across_time_boundaries(
        frame, _forward_target(frame, 7), _contract(7)
    )

    # 60 bars, 7 ahead: the last seven cannot have a future endpoint.
    assert masked.notna().sum() == 53


def test_a_real_gap_is_still_blanked():
    """The guard has to keep working: a hole must not pass as a horizon."""
    frame = _daily_frame(days=30)
    # Drop a month out of the middle: the rows either side are adjacent in
    # position but two months apart in time.
    frame = pd.concat([frame.iloc[:10], frame.iloc[10:].assign(
        datetime=frame["datetime"].iloc[10:] + pd.Timedelta(days=60)
    )], ignore_index=True)

    masked = mask_targets_across_time_boundaries(
        frame, _forward_target(frame, 7), _contract(7)
    )

    # The rows whose forward window jumps the gap lose their label.
    assert masked.iloc[3:10].isna().all()


def test_intraday_timeframes_are_untouched():
    """Only the daily rule changed; 1.5x still governs the rest."""
    expected = pd.Timedelta(hours=4)
    assert _maximum_elapsed("60m", expected, 4) == expected * 1.5
