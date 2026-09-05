"""Beating a coin on every fold is not the same as beating the opponent.

REGISTER #192. Two checks ask different questions of a champion and neither
asked this one:

    the ladder      does the model beat the STRONGEST opponent -- once, on
                    the holdout
    stability       does the model beat CHANCE -- on every fold

So the MARGIN over the strongest opponent was never tested in time, and a
margin is exactly what a narrow promotion lives on. `volatility_spike_1h` was
promoted at 0.7553 against a clock at 0.7415: a gap of 0.0138 with a standard
error of 0.0061, while stability reported "2 of 2 folds" -- against the
majority class.

The regression path has compared each fold against max(train mean,
persistence) all along (`_regression_fold_stability`). The classification path
compared against 0.5. One check, two implementations, different opponents:
family C inside a single class.

LAGGED WITHIN THE NAME, and that is not a detail. A pooled validation window
interleaves 110 tickers at every timestamp, so `y[t-h]` taken by row is
another company minutes earlier. That is #189, and this module already carries
a comment about it twenty lines from where the opponent is now built.

WHERE IT CANNOT BE BUILT IT SAYS SO. A fold with no ticker column, or too few
rows carrying a predecessor, keeps the chance bar and records the reason. An
opponent that could not be measured must not read as an opponent that scored
nothing -- that is #202, and it would hand every such fold a free pass.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.pipeline.stages.modeling import walk_forward_validation as wfv
from src.pipeline.stages.modeling.orchestrator import ModelingStage


def _pooled(names=("AAA", "BBB"), bars=60, seed=0) -> pd.DataFrame:
    """A pooled window: every name at every timestamp, interleaved."""
    rng = np.random.default_rng(seed)
    stamps = pd.date_range("2020-01-01", periods=bars, freq="D", tz="UTC")
    rows = []
    for stamp in stamps:                      # date-major, as the panel is
        for offset, name in enumerate(names):
            rows.append({"ticker": name, "datetime": stamp,
                         "target_up_1d": int(rng.integers(0, 2))})
    return pd.DataFrame(rows)


def test_the_opponent_is_lagged_inside_each_name():
    """The whole point. A row-wise lag on this frame returns the OTHER
    ticker's label at the same timestamp."""
    frame = _pooled()
    # Make each name perfectly persistent, and the two names opposites, so a
    # row-wise lag scores as badly as a per-name lag scores well.
    frame.loc[frame["ticker"] == "AAA", "target_up_1d"] = 1
    frame.loc[frame["ticker"] == "BBB", "target_up_1d"] = 0
    frame.loc[frame.index[:2], "target_up_1d"] = [1, 0]

    target = frame["target_up_1d"].astype(int)
    result = wfv._persistence_opponent(frame, target, "target_up_1d")

    assert result["persistence_reason"] is None
    assert result["persistence_balanced_accuracy"] == pytest.approx(1.0), (
        "the lag is not being taken inside the ticker: with each name constant "
        "and the two names opposite, a per-name lag is perfect and a row-wise "
        "one is not"
    )


def test_a_frame_without_tickers_refuses_rather_than_guessing():
    frame = _pooled().drop(columns=["ticker"])
    result = wfv._persistence_opponent(
        frame, frame["target_up_1d"].astype(int), "target_up_1d")
    assert result["persistence_balanced_accuracy"] is None
    assert "cross names" in result["persistence_reason"]


def test_too_few_predecessors_refuses_rather_than_scoring_zero():
    """"Could not measure" must not read as "the opponent scored nothing"."""
    frame = _pooled(names=("AAA",), bars=4)
    result = wfv._persistence_opponent(
        frame, frame["target_up_1d"].astype(int), "target_up_5d")
    assert result["persistence_balanced_accuracy"] is None
    assert "predecessor" in result["persistence_reason"]
    assert result["persistence_horizon_bars"] >= 1


def test_one_class_among_the_usable_rows_refuses():
    frame = _pooled(names=("AAA", "BBB"), bars=40)
    frame["target_up_1d"] = 1
    result = wfv._persistence_opponent(
        frame, frame["target_up_1d"].astype(int), "target_up_1d")
    assert result["persistence_balanced_accuracy"] is None
    assert "one class" in result["persistence_reason"]


def test_the_horizon_comes_from_the_target_not_from_one():
    """A 5-day target must be lagged five bars, not one -- otherwise the
    opponent is handed a value nobody could know (#238)."""
    frame = _pooled(names=("AAA", "BBB"), bars=60)
    result = wfv._persistence_opponent(
        frame, frame["target_up_1d"].astype(int), "target_up_5d")
    assert result["persistence_horizon_bars"] == 5, (
        f"lagged by {result['persistence_horizon_bars']} bars for a 5-day "
        "target"
    )


def test_the_fold_carries_the_opponent_into_its_metrics():
    source = inspect.getsource(wfv)
    assert "_persistence_opponent(validation, validation_target, target_name)" \
        in source, (
        "the fold no longer computes its own opponent, so the stability check "
        "has nothing to compare against and silently falls back to chance"
    )


def test_the_stability_bar_is_the_opponent_where_one_exists():
    source = inspect.getsource(ModelingStage)
    marker = source.index("THE BAR IS THE STRONGEST OPPONENT")
    window = source[marker:marker + 2200]
    assert "persistence_balanced_accuracy" in window, (
        "the classification stability check is back to comparing against "
        "chance, so a champion promoted on a narrow margin is never asked "
        "whether that margin survives a second window"
    )
    assert "max(bar, float(opponent)" in window
    assert "unmeasured" in window, (
        "folds where the opponent could not be built pass silently on the "
        "chance bar, which is #202 in the check that exists to catch it"
    )


def test_both_stability_paths_now_use_an_opponent_not_a_constant():
    """Regression compared against max(mean, persistence) while classification
    compared against 0.5. That is one check with two answers."""
    source = inspect.getsource(ModelingStage)
    assert "_persistence_r_squared" in source, (
        "the regression path lost its opponent"
    )
    assert source.count("0.5 + self._chance_margin(fold)") <= 1, (
        "the chance bar is used in more than one place again; it is now the "
        "FLOOR under the opponent, not the bar itself"
    )
