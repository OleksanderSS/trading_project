"""Hyperparameters were chosen on folds that leaked.

`sklearn.model_selection.TimeSeriesSplit` places the validation fold
immediately after the training fold. With a forward-looking target that is not
a subtlety — the label attached to the last training row is computed from
prices that fall INSIDE the validation window, so the hyperparameters that win
are the ones best at exploiting the overlap.

Measured in this repository: `target_hourly_volume_spike_1h` carries a 23-bar
horizon while the configured purge was 5. The pipeline's walk-forward evaluator
catches that and raises the purge automatically. `BayesianOptimizer` and
`OverfittingAnalyzer` never went through the evaluator and used the raw sklearn
splitter, so every hyperparameter this project has selected was selected across
an overlap.

`PurgedTimeSeriesSplit` delegates its fold arithmetic to
`build_purged_expanding_folds`, the same function the evaluator uses, because a
second implementation of a purge is the shape that has left half the fixes here
landing in one copy while the other went on being wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.stages.modeling.walk_forward_validation import (  # noqa: E402
    PurgedTimeSeriesSplit,
)

X = np.arange(2000).reshape(-1, 1)


class TestTheGapIsReal:
    @pytest.mark.parametrize('purge', [1, 5, 23, 60])
    def test_every_fold_leaves_exactly_the_requested_gap(self, purge):
        for train, val in PurgedTimeSeriesSplit(n_splits=3, purge_rows=purge).split(X):
            assert val[0] - train[-1] - 1 == purge

    def test_a_23_bar_horizon_is_actually_honoured(self):
        # The real case: target_hourly_volume_spike_1h.
        for train, val in PurgedTimeSeriesSplit(n_splits=3, purge_rows=23).split(X):
            assert val[0] - train[-1] - 1 >= 23

    def test_training_never_overlaps_validation(self):
        for train, val in PurgedTimeSeriesSplit(n_splits=4, purge_rows=10).split(X):
            assert set(train).isdisjoint(set(val))

    def test_training_is_always_in_the_past(self):
        for train, val in PurgedTimeSeriesSplit(n_splits=4, purge_rows=10).split(X):
            assert train.max() < val.min()


class TestItBehavesLikeASklearnCV:
    def test_get_n_splits_matches_what_split_yields(self):
        # cross_val_score asks first and iterates second; a mismatch raises.
        sp = PurgedTimeSeriesSplit(n_splits=3, purge_rows=23)
        assert sp.get_n_splits(X) == len(list(sp.split(X)))

    def test_it_runs_inside_cross_val_score(self):
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score
        y = X.ravel() * 0.5 + np.random.default_rng(0).normal(0, 1, len(X))
        scores = cross_val_score(Ridge(), X, y,
                                 cv=PurgedTimeSeriesSplit(n_splits=3, purge_rows=23))
        assert len(scores) >= 2 and np.isfinite(scores).all()

    def test_folds_walk_forward(self):
        folds = list(PurgedTimeSeriesSplit(n_splits=4, purge_rows=5).split(X))
        assert all(a[1].min() < b[1].min() for a, b in zip(folds, folds[1:]))


class TestItRefusesRatherThanLeaks:
    def test_too_few_rows_raises_instead_of_dropping_the_purge(self):
        # Silently falling back to an unpurged split would reintroduce exactly
        # the leak this class removes, and nothing downstream could tell.
        with pytest.raises(ValueError, match="purge"):
            list(PurgedTimeSeriesSplit(n_splits=5, purge_rows=500).split(np.arange(20).reshape(-1, 1)))

    def test_a_zero_purge_is_floored_to_one(self):
        for train, val in PurgedTimeSeriesSplit(n_splits=2, purge_rows=0).split(X):
            assert val[0] > train[-1]

    def test_small_data_still_produces_folds_rather_than_none(self):
        # A fixed min_train_rows of 360 would yield zero folds on 300 rows and
        # cross_val_score would raise something unrelated to the cause.
        small = np.arange(300).reshape(-1, 1)
        assert len(list(PurgedTimeSeriesSplit(n_splits=2, purge_rows=5).split(small))) >= 1
