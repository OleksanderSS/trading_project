"""Sortino must equal its definition, and all implementations must agree.

Three independent Sortino implementations existed and all three disagreed:

    sample        textbook   enricher   metrics_lib   risk_reward
    mild            0.5972     0.5972        0.7099        0.6873
    volatile       -1.2702    -1.2702       -1.2188       -1.4919
    skewed-down    -4.9208    -4.9208       -4.4423       -6.2859

The dangerous part was not the size of the error but its behaviour:
metrics_lib was 1.189x on mild data and 0.903x on downside-skewed data -- the
bias CHANGES SIGN with the shape of the distribution, so it cannot be
corrected for and it reorders strategies. risk_reward overstated most
(1.277x) exactly on downside-skewed returns, i.e. the case Sortino exists to
penalise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.calculators.risk_reward_calculator import RiskRewardCalculator
from src.metrics.financial.financial_metrics_library import FinancialMetricsLibrary as F

PERIODS = 252


def textbook_sortino(returns: pd.Series, target: float = 0.0, periods: int = PERIODS) -> float:
    """Annualised excess return / annualised downside deviation.

    Downside deviation = sqrt(mean(min(0, r - target)^2)) over ALL
    observations -- squared shortfalls averaged over the whole sample, not the
    dispersion of the losing subset.
    """
    shortfalls = np.minimum(0.0, returns - target)
    downside = np.sqrt((shortfalls ** 2).mean()) * np.sqrt(periods)
    return ((returns.mean() - target) * periods) / downside


def _sample(kind: str) -> pd.Series:
    rng = np.random.default_rng({"mild": 1, "volatile": 7, "skewed_down": 13}[kind])
    if kind == "volatile":
        return pd.Series(rng.normal(0.0005, 0.03, 1000))
    if kind == "skewed_down":
        return pd.Series(rng.normal(0.001, 0.01, 1000) - rng.chisquare(2, 1000) * 0.002)
    return pd.Series(rng.normal(0.0008, 0.01, 1000))


@pytest.mark.parametrize("kind", ["mild", "volatile", "skewed_down"])
def test_matches_the_definition(kind):
    returns = _sample(kind)
    assert F.calculate_sortino_ratio(returns, trading_days_per_year=PERIODS) == pytest.approx(
        textbook_sortino(returns), rel=1e-9
    )


@pytest.mark.parametrize("kind", ["mild", "volatile", "skewed_down"])
def test_all_implementations_agree(kind):
    returns = _sample(kind)
    canonical = F.calculate_sortino_ratio(returns, trading_days_per_year=PERIODS)
    delegated = RiskRewardCalculator.calculate_sortino_ratio(returns)
    assert delegated == pytest.approx(canonical, rel=1e-9)


def test_downside_deviation_is_not_the_std_of_the_losing_subset():
    """The two differ whenever losses vary in size, which is always.

    Subset std subtracts the losses' own mean (measuring how much they VARY,
    not how large they are) and divides by the loss count rather than the
    sample size. Both shrink the denominator and inflate the ratio.
    """
    returns = _sample("mild")
    subset_std = returns[returns < 0].std() * np.sqrt(PERIODS)
    shortfalls = np.minimum(0.0, returns)
    downside_deviation = np.sqrt((shortfalls ** 2).mean()) * np.sqrt(PERIODS)

    assert downside_deviation > subset_std
    assert abs(downside_deviation - subset_std) / downside_deviation > 0.05


def test_annualisation_is_consistent_with_sharpe():
    """Sortino used geometric annualisation while Sharpe used arithmetic, so
    the two were not comparable to each other."""
    returns = _sample("mild")
    periods = PERIODS

    sortino = F.calculate_sortino_ratio(returns, trading_days_per_year=periods)
    shortfalls = np.minimum(0.0, returns)
    downside = np.sqrt((shortfalls ** 2).mean()) * np.sqrt(periods)

    # Reconstructing with the ARITHMETIC numerator reproduces it exactly;
    # the geometric one does not.
    arithmetic = (returns.mean() * periods) / downside
    geometric = ((1 + returns.mean()) ** periods - 1) / downside

    assert sortino == pytest.approx(arithmetic, rel=1e-9)
    assert sortino != pytest.approx(geometric, rel=1e-6)


def test_a_strategy_with_larger_losses_scores_worse():
    """Ordering sanity: the metric must punish bigger drawdowns."""
    rng = np.random.default_rng(5)
    base = pd.Series(rng.normal(0.001, 0.01, 800))
    worse = base.copy()
    worse[worse < 0] *= 2.0        # same wins, twice the losses

    assert F.calculate_sortino_ratio(worse) < F.calculate_sortino_ratio(base)


def test_no_losses_yields_nan_rather_than_infinity():
    only_gains = pd.Series([0.01] * 100)
    assert np.isnan(F.calculate_sortino_ratio(only_gains))


@pytest.mark.parametrize("bad", [pd.Series(dtype=float), pd.Series([0.01])])
def test_degenerate_input(bad):
    assert np.isnan(F.calculate_sortino_ratio(bad))
