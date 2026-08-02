"""Conditional VaR must be a conditional expectation.

EliteRiskMetrics is live on the trading path: Stage 6 builds it
(trading/orchestrator.py:54) alongside EliteRiskSizer, and max_exposure_monitor
uses it too. Its numbers set the 'high_risk' status and feed position limits.

The Cornish-Fisher expansion itself is correct -- the z_cf formula matches
the standard one, and scipy's kurtosis returns EXCESS kurtosis, which is
what it expects. Checked, not assumed.

CVaR was not:

    cvar_cf = var_cf * (1 + abs(kurtosis) / 4)

That is an ad-hoc inflation of VaR, not the average loss given a breach. The
abs() also inflates for THIN tails: a platykurtic asset (negative excess
kurtosis) should have CVaR close to VaR, and this pushed it further away.

Measured against real daily returns:

    ticker  excess kurt   VaR      CVaR (ES)   old heuristic
    AAPL       1.79      0.0227     0.0356       0.0329   understated
    NVDA       0.31      0.0363     0.0447       0.0391   understated 13%
    KO         1.90      0.0154     0.0198       0.0227   overstated
    SPY        1.12      0.0130     0.0171       0.0166

Wrong in both directions with no systematic bias -- and understated for the
most volatile name, which is the dangerous direction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.elite_risk_metrics import EliteRiskMetrics


@pytest.fixture()
def metrics():
    return EliteRiskMetrics()


# compute_cornish_fisher_var looks back 252 bars, so fixtures are sized to
# that window -- otherwise a test computing its own expectation over the full
# series is comparing against a different sample than the code used.
LOOKBACK = 252


def _feed(metrics, returns, ticker="TEST"):
    metrics.update_returns(ticker, pd.Series(returns))
    return metrics.compute_cornish_fisher_var(ticker, 0.95)


def _window(returns):
    """The slice the implementation actually scores."""
    series = pd.Series(returns)
    return series.tail(LOOKBACK).reset_index(drop=True)


def test_cvar_is_never_below_var(metrics):
    """Expected Shortfall is an average OVER the tail, so it cannot be
    smaller than the threshold that defines the tail."""
    rng = np.random.default_rng(0)
    var, cvar = _feed(metrics, rng.normal(0.0005, 0.02, LOOKBACK))

    assert cvar >= var


def test_a_fat_tailed_series_has_cvar_well_above_var(metrics):
    rng = np.random.default_rng(1)
    returns = np.concatenate([rng.normal(0.001, 0.01, LOOKBACK - 20), rng.normal(-0.12, 0.02, 20)])

    var, cvar = _feed(metrics, returns)

    assert cvar > var * 1.5


def test_a_thin_tailed_series_keeps_cvar_close_to_var(metrics):
    """The abs() bug: negative excess kurtosis used to push these apart.

    Equality is the limit case and is correct here. VaR is parametric
    (Cornish-Fisher) while ES is empirical, so for a platykurtic sample the
    expansion overshoots the worst observed losses and the ES >= VaR clamp
    binds. What must not happen is the old behaviour, where thin tails
    INFLATED the gap."""
    rng = np.random.default_rng(2)
    uniform = rng.uniform(-0.02, 0.02, LOOKBACK)  # excess kurtosis about -1.2

    var, cvar = _feed(metrics, uniform)

    assert var <= cvar < var * 1.4


def test_cvar_is_the_mean_of_the_worst_five_percent(metrics):
    """Averaged over the worst (1 - confidence) FRACTION, not over whatever
    breaches the Cornish-Fisher threshold: for thin-tailed data that
    threshold can sit beyond every observation, leaving nothing to average."""
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.0005, 0.02, 600))

    _, cvar = _feed(metrics, returns)
    window = _window(returns).sort_values()
    worst = window.head(int(np.ceil(0.05 * len(window))))

    assert cvar == pytest.approx(float(-worst.mean()), rel=1e-6)


def test_both_numbers_are_positive_losses(metrics):
    rng = np.random.default_rng(4)
    var, cvar = _feed(metrics, rng.normal(0.002, 0.01, LOOKBACK))

    assert var > 0
    assert cvar > 0


def test_too_little_history_falls_back_to_the_documented_defaults(metrics):
    var, cvar = _feed(metrics, np.array([0.01, -0.01, 0.005]))

    assert var == metrics.DEFAULT_VAR_LOSS
    assert cvar == metrics.DEFAULT_CVAR_LOSS


def test_the_expansion_still_uses_excess_kurtosis():
    """scipy's kurtosis is Fisher (excess) by default, which is what the
    Cornish-Fisher formula takes. A switch to Pearson would silently shift
    every tail estimate."""
    from scipy import stats

    assert stats.kurtosis(np.random.default_rng(5).normal(size=10000)) == pytest.approx(
        0.0, abs=0.2
    )


def test_the_heuristic_is_gone():
    import inspect

    source = inspect.getsource(EliteRiskMetrics.compute_cornish_fisher_var)
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )

    assert "abs(kurtosis) / 4" not in code


def test_real_returns_produce_a_sane_tail(metrics):
    """Against the live database rather than a fixture."""
    import duckdb

    connection = duckdb.connect("data/trading_data.duckdb", read_only=True)
    frame = connection.execute(
        "SELECT close FROM market_data_raw WHERE interval='1d' AND ticker='SPY' "
        "ORDER BY datetime"
    ).fetchdf()
    connection.close()

    if len(frame) < 60:
        pytest.skip("not enough stored SPY history")

    var, cvar = _feed(metrics, frame["close"].pct_change().dropna().to_numpy(), "SPY")

    assert 0.0 < var < 0.15, "a daily 95% VaR outside 0-15% would be implausible"
    assert var <= cvar < 0.25
