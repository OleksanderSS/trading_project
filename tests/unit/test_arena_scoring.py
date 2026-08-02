"""The arena ranks models after every training run, so its score must
prefer the model that is right.

UnifiedTrainingManager builds the arena in __init__ (get_trading_arena) and
calls run_battle whenever training produced results and a y_test, so this is
on the live path -- not the dormant ArenaOrchestrator.

Three of the five scored criteria measured nothing:

  sharpe_ratio     mean(predictions) / std(predictions) -- the predictions'
                   own consistency, with the actual outcomes nowhere in it.
                   A constant predictor scored 9899, capped at 2 by the
                   scoring function, which handed it the full 0.5.
  max_drawdown     hardcoded 0.0, so (1 - abs(drawdown)) was a constant 1.0
                   and its 0.15 weight measured nothing. _calculate_max_
                   drawdown existed and was never called with real data.
  win_rate         a second copy of accuracy, so the score counted the same
                   number twice (0.3 + 0.2) and called it two criteria.

Measured before the fix, against 500 synthetic bars:

    accurate model (97.2% directional)   score 0.698
    constant +0.01 (47.6%, knows nothing) score 0.958   <-- winner
    noise (48.8%)                         score 0.448

after:

    accurate 1.206, noise 0.362, constant 0.281
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.arena.arena_battle import TradingModelArena


@pytest.fixture()
def arena():
    return object.__new__(TradingModelArena)


@pytest.fixture()
def actuals():
    return pd.Series(np.random.default_rng(0).normal(0, 0.02, 500))


def _accurate(actuals, seed=1):
    rng = np.random.default_rng(seed)
    return actuals.to_numpy() * 0.9 + rng.normal(0, 0.002, len(actuals))


def _constant(n, seed=2):
    return np.full(n, 0.01) + np.random.default_rng(seed).normal(0, 1e-6, n)


def _noise(n, seed=3):
    return np.random.default_rng(seed).normal(0, 0.02, n)


def test_the_accurate_model_beats_the_constant_one(arena, actuals):
    """The regression, in one line: a constant predictor used to win."""
    accurate = arena._calculate_metrics(_accurate(actuals), actuals)
    constant = arena._calculate_metrics(_constant(len(actuals)), actuals)

    assert arena._calculate_weighted_score(accurate) > arena._calculate_weighted_score(constant)


def test_the_accurate_model_beats_noise(arena, actuals):
    accurate = arena._calculate_metrics(_accurate(actuals), actuals)
    noise = arena._calculate_metrics(_noise(len(actuals)), actuals)

    assert arena._calculate_weighted_score(accurate) > arena._calculate_weighted_score(noise)


def test_a_constant_predictor_gets_no_sharpe_reward(arena, actuals):
    """mean/std of a near-constant is unbounded; it used to reach 9899."""
    metrics = arena._calculate_metrics(_constant(len(actuals)), actuals)

    assert metrics.sharpe_ratio < 1.0


def test_sharpe_reflects_the_outcome_not_the_prediction_spread(arena, actuals):
    accurate = arena._calculate_metrics(_accurate(actuals), actuals)

    assert accurate.sharpe_ratio > 1.0, (
        "a model that predicts the actual direction should earn a positive "
        "risk-adjusted return"
    )


def test_drawdown_is_measured_rather_than_assumed(arena, actuals):
    """It was hardcoded to 0.0, so 0.15 of the score was a constant."""
    metrics = arena._calculate_metrics(_noise(len(actuals)), actuals)

    assert metrics.max_drawdown < 0.0


def test_a_flawless_model_has_almost_no_drawdown(arena, actuals):
    metrics = arena._calculate_metrics(_accurate(actuals), actuals)

    assert metrics.max_drawdown > -0.05


def test_win_rate_measures_money_not_direction(arena):
    """They coincide only when wins and losses are the same size. Here the
    model is right about direction less often than it is profitable."""
    actuals = pd.Series([0.10, -0.01, -0.01, 0.10, -0.01])
    predictions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    metrics = arena._calculate_metrics(predictions, actuals)

    assert metrics.win_rate == pytest.approx(0.4)
    assert metrics.accuracy == pytest.approx(0.4)


def test_no_valid_rows_yields_zeroed_metrics(arena):
    metrics = arena._calculate_metrics(
        np.array([np.nan, np.inf]), pd.Series([np.nan, np.nan])
    )

    assert metrics.accuracy == 0
    assert metrics.sharpe_ratio == 0


def test_the_score_no_longer_delegates_a_fifth_of_itself_to_a_constant():
    import inspect

    source = inspect.getsource(TradingModelArena._calculate_metrics)

    assert "max_drawdown=0.0" not in source
    assert "np.mean(predictions_clean) / prediction_std" not in source
    assert "calculate_sharpe_ratio" in source
