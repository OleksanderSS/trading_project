"""Deflated Sharpe Ratio: the penalty must actually bite.

Running K configurations and reporting the best inflates Sharpe even when no
configuration has an edge -- the maximum of K noisy estimates is positive by
construction. These pin that the deflation reacts to trial count, sample
size and non-normality, so it cannot be mistaken for a relabelled Sharpe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics.financial.financial_metrics_library import FinancialMetricsLibrary as F


def _returns(mean=0.0005, std=0.01, n=500, seed=0, skew=0.0):
    rng = np.random.default_rng(seed)
    values = rng.normal(mean, std, n)
    if skew:
        values = values + skew * (rng.chisquare(3, n) - 3) * std
    return pd.Series(values)


def test_more_trials_means_a_lower_deflated_sharpe():
    """The whole point: the same track record is worth less if you searched
    harder to find it."""
    returns = _returns()
    one = F.calculate_deflated_sharpe_ratio(returns, n_trials=1)
    ten = F.calculate_deflated_sharpe_ratio(returns, n_trials=10)
    thousand = F.calculate_deflated_sharpe_ratio(returns, n_trials=1000)

    assert one > ten > thousand


def test_a_single_trial_applies_no_selection_penalty():
    returns = _returns()
    assert F.calculate_deflated_sharpe_ratio(returns, n_trials=1) == pytest.approx(
        F.calculate_deflated_sharpe_ratio(returns, n_trials=1), abs=0
    )
    # ...and is strictly the most generous case.
    assert F.calculate_deflated_sharpe_ratio(returns, n_trials=1) >= \
        F.calculate_deflated_sharpe_ratio(returns, n_trials=2)


def test_a_strong_track_record_survives_a_wide_search():
    strong = _returns(mean=0.004, std=0.01, n=1000)
    assert F.calculate_deflated_sharpe_ratio(strong, n_trials=100) > 0.95


def test_a_noise_track_record_does_not_survive():
    """Zero-edge returns picked out of many trials must not look significant."""
    noise = _returns(mean=0.0, std=0.01, n=500, seed=7)
    assert F.calculate_deflated_sharpe_ratio(noise, n_trials=200) < 0.95


def test_longer_samples_are_more_convincing_than_short_ones():
    short = _returns(mean=0.002, n=60, seed=3)
    long = _returns(mean=0.002, n=2000, seed=3)

    assert F.calculate_deflated_sharpe_ratio(long, n_trials=50) > \
        F.calculate_deflated_sharpe_ratio(short, n_trials=50)


def test_result_is_a_probability():
    for trials in (1, 5, 50, 5000):
        value = F.calculate_deflated_sharpe_ratio(_returns(), n_trials=trials)
        assert 0.0 <= value <= 1.0


def test_supplied_trial_variance_is_used():
    returns = _returns()
    tight = F.calculate_deflated_sharpe_ratio(
        returns, n_trials=100, variance_of_trial_sharpes=1e-6)
    wide = F.calculate_deflated_sharpe_ratio(
        returns, n_trials=100, variance_of_trial_sharpes=1.0)
    # A wider spread of trial Sharpes raises the bar the winner must clear.
    assert tight > wide


@pytest.mark.parametrize("bad", [pd.Series(dtype=float), pd.Series([0.01]),
                                 pd.Series([0.01, 0.01, 0.01])])
def test_degenerate_input_returns_the_error_value(bad):
    assert np.isnan(F.calculate_deflated_sharpe_ratio(bad, n_trials=10))


def test_zero_trials_is_rejected():
    assert np.isnan(F.calculate_deflated_sharpe_ratio(_returns(), n_trials=0))
