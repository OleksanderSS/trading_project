"""Can one column and a straight line already do what the model does?

Measured 2026-08-20 on `target_hourly_breakout_1h`, the target that produced
more champions than any other. It asks whether price crosses today's upper
Bollinger band within four bars — and the distance from the close to that band,
one arithmetic expression known at forecast time, scores **AUC 0.9666** on it.
The event rate goes from 8.1% overall to 64.0% in the decile nearest the band.

On money, at matched selectivity, on the real holdout:

    model, top 30% by probability      3,494 trades   -0.00022
    closest 30% to the band, no model  3,495 trades   -0.00021

Identical. The existing opponents cannot see this: a constant predictor and a
persistence predictor are both about the SHAPE of the series, and neither asks
whether a single column already contains the answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.base_trainer import BaseTrainer  # noqa: E402

RNG = np.random.default_rng(0)
score = BaseTrainer._score_single_feature_baseline


class Evaluator:
    """Minimal stand-in: R2 for regression, accuracy for classification."""
    def calculate(self, y, pred, task_type='regression'):
        y, pred = np.asarray(y, float), np.asarray(pred, float)
        if task_type == 'classification':
            return {'F1': float((y == pred).mean())}
        ss = float(((y - y.mean()) ** 2).sum())
        return {'R2': 1 - float(((y - pred) ** 2).sum()) / ss if ss else 0.0}


def bundle(n=400, k=6, signal_col=2, strength=3.0, classif=False):
    X = pd.DataFrame(RNG.normal(0, 1, (n, k)), columns=[f'f{i}' for i in range(k)])
    y = X[f'f{signal_col}'] * strength + RNG.normal(0, 0.5, n)
    if classif:
        y = (y > y.median()).astype(int)
    cut = n // 2
    return dict(X_train=X.iloc[:cut], y_train=y.iloc[:cut],
                X_holdout=X.iloc[cut:], y_holdout=y.iloc[cut:])


class TestItFindsTheColumnThatCarriesTheTarget:
    def test_it_names_the_driving_feature(self):
        out = score(bundle(), False, 'regression', 'R2', Evaluator())
        assert out['single_feature_status'] == 'measured'
        assert out['single_feature_name'] == 'f2'

    def test_a_target_that_is_one_feature_scores_high(self):
        out = score(bundle(strength=10.0), False, 'regression', 'R2', Evaluator())
        assert out['single_feature_score'] > 0.9

    def test_a_target_no_single_column_explains_scores_low(self):
        X = pd.DataFrame(RNG.normal(0, 1, (400, 6)), columns=[f'f{i}' for i in range(6)])
        y = pd.Series(RNG.normal(0, 1, 400))          # unrelated to everything
        d = dict(X_train=X.iloc[:200], y_train=y.iloc[:200],
                 X_holdout=X.iloc[200:], y_holdout=y.iloc[200:])
        out = score(d, False, 'regression', 'R2', Evaluator())
        assert out['single_feature_score'] < 0.2

    def test_classification_fires_at_about_the_train_event_rate(self):
        out = score(bundle(classif=True), True, 'classification', 'F1', Evaluator())
        assert out['single_feature_status'] == 'measured'
        assert out['single_feature_score'] > 0.7


class TestTheBaselineCannotCheat:
    def test_the_feature_is_chosen_on_train_not_on_holdout(self):
        # A column that looks predictive only in the holdout must not be picked.
        X = pd.DataFrame(RNG.normal(0, 1, (400, 3)), columns=['a', 'b', 'c'])
        y = pd.Series(RNG.normal(0, 1, 400))
        X.loc[200:, 'c'] = y[200:] * 5           # cheats only in the holdout half
        X.loc[:199, 'a'] = y[:199] * 5           # honest signal in train
        d = dict(X_train=X.iloc[:200], y_train=y.iloc[:200],
                 X_holdout=X.iloc[200:], y_holdout=y.iloc[200:])
        assert score(d, False, 'regression', 'R2', Evaluator())['single_feature_name'] == 'a'


class TestItRefusesRatherThanInventsAScore:
    def test_missing_train_or_holdout_returns_none(self):
        out = score({'X_train': pd.DataFrame({'a': [1, 2]})}, False,
                    'regression', 'R2', Evaluator())
        assert out['single_feature_score'] is None
        assert out['single_feature_status'] == 'no_train_or_holdout'

    def test_mismatched_shapes_return_none(self):
        d = bundle(); d['y_holdout'] = d['y_holdout'].iloc[:10]
        assert score(d, False, 'regression', 'R2', Evaluator())['single_feature_score'] is None

    def test_no_numeric_columns_returns_none(self):
        d = bundle()
        d['X_train'] = pd.DataFrame({'s': ['a'] * 200})
        d['X_holdout'] = pd.DataFrame({'s': ['a'] * 200})
        assert score(d, False, 'regression', 'R2', Evaluator())['single_feature_score'] is None

    def test_a_failure_inside_does_not_raise(self):
        d = bundle(); d['X_holdout'] = 'not a frame'
        out = score(d, False, 'regression', 'R2', Evaluator())
        assert out['single_feature_score'] is None
