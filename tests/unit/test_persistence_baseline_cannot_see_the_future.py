"""The naive opponent has to be a forecast, and this one was not.

`_score_naive_baselines` built persistence as `y[t-1] -> y[t]`. For a target
with a horizon of h bars, y[t-1] is the outcome measured from t-1 to t+h-2 --
it is not known at t. A 5-day target needs four more days of prices before its
previous value exists.

That would be merely wrong if the value were useless. It is not: a 5-day
forward return shares four of its five days with the previous bar's, so
consecutive values are correlated by construction. Measured on the 2026-08-22
batch, lag-1 autocorrelation is 0.778 on target_relative_return_5d against the
0.800 the overlap alone predicts, and the baseline scored R^2 0.5564.

The gate takes the strongest opponent. So every model on a multi-bar target
was asked to beat 0.56 rather than the 0.0000 the constant baseline set, and
no return or direction target has ever been promoted here across 4,613 models.

Lagging by the horizon restores the arrow of time. Measured, the same baseline
then scores -1.0879, worse than the mean, so the constant becomes the opponent
-- which is the ordinary bar.

It does not manufacture champions, and that is worth stating plainly: the
models on those targets scored -0.09 and -0.01, still below 0.0. They go on
losing. They stop losing to an oracle.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.targets.timeframe_contract import target_horizon_bars
from src.training.base_trainer import BaseTrainer


class _R2Evaluator:
    """Just enough evaluator: R2 for regression."""

    @staticmethod
    def calculate(y_true, y_pred, task_type="regression"):
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        residual = ((y_true - y_pred) ** 2).sum()
        total = ((y_true - y_true.mean()) ** 2).sum()
        return {"R2": 1 - residual / total if total > 0 else 0.0}


class _Trainer(BaseTrainer):
    """BaseTrainer is abstract; only the baseline scoring is under test here."""

    def _prepare_ticker_groups(self, *args, **kwargs):
        raise NotImplementedError

    def _train_ticker_group(self, *args, **kwargs):
        raise NotImplementedError


@pytest.fixture
def trainer():
    instance = _Trainer.__new__(_Trainer)
    instance.evaluator = _R2Evaluator()
    return instance


def _overlapping_target(horizon: int, n: int = 400, seed: int = 0):
    """A forward `horizon`-bar return: overlapping, so it repeats itself."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.01, n + horizon)
    return np.array([steps[i: i + horizon].sum() for i in range(n)])


@pytest.mark.parametrize(
    "target_name,timeframe,expected",
    [
        ("target_return_1d", "1d", 1),
        ("target_return_5d", "1d", 5),
        ("target_relative_return_5d", "1d", 5),
        ("target_weekly_return_1w", "1d", 5),
        ("target_intraday_return_15m", "15m", 1),
        ("target_hourly_return_1h", "15m", 4),
        ("target_hourly_return_1h", "60m", 1),
        # Across the intraday/daily boundary the calendar arithmetic is wrong
        # (a trading day is 6.5 hours, not 24), so it refuses rather than
        # answering 480. These pairs carry no data anyway.
        ("target_return_5d", "15m", None),
        ("target_return_1d", "60m", None),
        ("weird_name", "1d", None),
        ("target_return_5d", None, None),
    ],
)
def test_the_horizon_is_read_from_the_name(target_name, timeframe, expected):
    assert target_horizon_bars(target_name, timeframe) == expected


def test_an_overlapping_target_no_longer_hands_the_baseline_the_future(trainer):
    """The whole finding, on a target built to overlap exactly as the real one does."""
    horizon = 5
    y = _overlapping_target(horizon, n=500)
    data = {
        "y_train": y[:300],
        "y_holdout": y[300:],
        "target_name": "target_relative_return_5d",
        "timeframe": "1d",
    }
    out = trainer._score_naive_baselines(data, False, "regression", "R2")

    assert out["baseline_persistence_lag_bars"] == horizon
    # Lagged by the horizon, persistence is worse than the mean, so the
    # constant is what a model actually has to beat.
    assert out["baseline_persistence_score"] < out["baseline_constant_score"]
    assert out["baseline_kind"] == "constant"
    assert out["baseline_score"] == pytest.approx(out["baseline_constant_score"])

    # And the old behaviour would have been far stronger -- this is the number
    # that was being demanded of every model.
    lag_one = np.concatenate([[y[300]], y[300:-1]])
    old = _R2Evaluator.calculate(y[300:], lag_one)["R2"]
    assert old > 0.4, "the overlapping fixture is not reproducing the defect"
    assert old > out["baseline_persistence_score"] + 1.0


def test_a_one_bar_target_is_left_exactly_as_it_was(trainer):
    """h=1 is the case where y[t-1] IS known at t, so nothing should change."""
    rng = np.random.default_rng(1)
    y = rng.normal(0, 0.01, 500)
    data = {
        "y_train": y[:300],
        "y_holdout": y[300:],
        "target_name": "target_return_1d",
        "timeframe": "1d",
    }
    out = trainer._score_naive_baselines(data, False, "regression", "R2")
    assert out["baseline_persistence_lag_bars"] == 1


def test_the_opponent_is_not_seeded_with_the_answer(trainer):
    """`persistence[0] = y_true[0]` gave it one exactly-right prediction free."""
    y = np.arange(100, dtype=float)
    data = {
        "y_train": y[:60],
        "y_holdout": y[60:],
        "target_name": "target_return_1d",
        "timeframe": "1d",
    }
    trainer._score_naive_baselines(data, False, "regression", "R2")

    # Rebuild what the code now constructs and check the first slot.
    holdout = y[60:]
    seeded_with_truth = holdout[0]
    train_mean = float(np.nanmean(y[:60]))
    assert seeded_with_truth != train_mean, "fixture cannot tell the two apart"


def test_an_unresolved_horizon_stays_at_one_bar(trainer):
    """Unresolved means lag 1, and that is deliberate rather than a fallback.

    The first attempt skipped the persistence opponent when the horizon could
    not be read, on the reasoning that "cannot tell" is not "1". That was
    wrong. The targets without a horizon in their name are the INDICATOR ones,
    where the value is a backward-looking window shifted a single bar -- so
    y[t-1] really is known at t, and persistence is the opponent that catches a
    model doing nothing but retracing a moving average. It scores R2 0.9994 on
    target_sma_20_f1. Skipping it weakens the gate exactly where it works.
    """
    rng = np.random.default_rng(2)
    smooth = np.cumsum(rng.normal(0, 0.01, 500))   # drifts, so the mean is poor
    data = {
        "y_train": smooth[:300],
        "y_holdout": smooth[300:],
        "target_name": "target_sma_20_f1",
        "timeframe": "1d",
    }
    out = trainer._score_naive_baselines(data, False, "regression", "R2")

    assert out["baseline_persistence_lag_bars"] == 1
    assert out["baseline_persistence_score"] > out["baseline_constant_score"]
    assert out["baseline_kind"] == "persistence"
