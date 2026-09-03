"""Does the money path find an edge that is definitely there, and none where there is none?

Stage 7 has never computed a profit or loss in this project's history. Not
because the code is wrong -- it may be entirely right -- but because it has
never been handed a return target to compute one on. Every run so far ends at

    holdout_equity: {"status": "no_return_targets"}

and on 2026-09-01 the whole chain finally ran end to end and still ended there
(990 predictions, 0 trades, portfolio_value 0). So the instrument's verdict
"no edge" cannot currently be read: it may be a statement about the market or
a statement about the instrument, and nothing distinguishes them.

This is the calibration. A planted effect of known size goes in, and the
number that comes out is compared with the number arithmetic says must come
out. It is the project's own rule -- "an instrument is checked on a known
answer" (WORKING_METHOD) -- applied to the one path that has never been
checked at all.

THE ARITHMETIC, so the expected values are derived and not fitted:

    actual ~ N(0, sigma), prediction correlated with actual at rho,
    position = sign(prediction), strategy_return = position * actual.

    For jointly normal (prediction, actual):

        E[sign(p) * a] = sigma * sqrt(2/pi) * rho
        Var[sign(p) * a] = sigma^2 * (1 - (2/pi) * rho^2)

    so the per-bar Sharpe is sqrt(2/pi)*rho / sqrt(1 - (2/pi)*rho^2), and the
    annualised Sharpe is that times sqrt(252). At rho = 0.10 that is 1.27
    before the risk-free rate; at rho = 1 it is 21.0; at rho = 0 it is zero.

The risk-free rate is read from the same place the pipeline reads it, so this
tests the pipeline's arithmetic rather than the rate it happens to use.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.pipeline.stages.evaluation.holdout_equity import build_holdout_equity
from src.pipeline.stages.evaluation.metrics_calculator import (
    EvaluationMetricsCalculator,
)

BARS = 5000
SIGMA = 0.01
SEED = 20260901

#: sqrt(2/pi): E[|z|] for a standard normal, the factor that appears whenever
#: a position is the sign of a correlated forecast.
ROOT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def _planted_predictions(rho: float, bars: int = BARS) -> pd.DataFrame:
    """Holdout predictions whose forecast correlates with the realised return at `rho`."""
    rng = np.random.default_rng(SEED)
    base = rng.standard_normal(bars)
    independent = rng.standard_normal(bars)
    actual = SIGMA * base
    forecast = rho * base + math.sqrt(max(0.0, 1.0 - rho * rho)) * independent
    return pd.DataFrame({
        "target": ["target_return_1d"] * bars,
        "context": ["CTRL::1d::target_return_1d"] * bars,
        "ticker": ["CTRL"] * bars,
        "datetime": pd.bdate_range("2000-01-03", periods=bars, tz="UTC"),
        "prediction": forecast,
        "actual": actual,
    })


def _expected_annual_sharpe(rho: float, risk_free_rate: float) -> float:
    mean = SIGMA * ROOT_2_OVER_PI * rho
    variance = SIGMA * SIGMA * (1.0 - (2.0 / math.pi) * rho * rho)
    excess = mean - risk_free_rate / 252.0
    return excess / math.sqrt(variance) * math.sqrt(252.0)


@pytest.fixture(scope="module")
def risk_free_rate() -> float:
    from src.metrics.financial.financial_metrics_library import get_risk_free_rate

    return float(get_risk_free_rate())


def _measured_metrics(rho: float) -> dict:
    curve = build_holdout_equity(_planted_predictions(rho))
    assert curve["status"] == "built", curve
    return EvaluationMetricsCalculator(None)._calculate_basic_metrics(
        curve["portfolio_history"]
    )


def test_the_curve_is_built_at_all_for_a_return_target():
    """The precondition every run has failed so far."""
    curve = build_holdout_equity(_planted_predictions(0.1))
    assert curve["status"] == "built"
    assert curve["bar_count"] == BARS


def test_a_planted_edge_comes_back_at_the_size_it_was_planted(risk_free_rate):
    rho = 0.10
    measured = _measured_metrics(rho)
    expected = _expected_annual_sharpe(rho, risk_free_rate)

    # Sampling error of a Sharpe over N bars is about sqrt(252/N) annualised;
    # 5,000 daily bars gives 0.22, so 0.35 is a little over one standard error
    # and still far from any factor-of-sqrt(252) mistake.
    assert measured["sharpe_ratio"] == pytest.approx(expected, abs=0.35), (
        f"planted rho={rho} implies an annualised Sharpe of {expected:.3f}; "
        f"the pipeline reported {measured['sharpe_ratio']:.3f}. Until these "
        f"agree, a verdict of 'no edge' from this path says nothing about the "
        f"market."
    )


def test_perfect_foresight_comes_back_as_perfect_foresight(risk_free_rate):
    measured = _measured_metrics(1.0)
    expected = _expected_annual_sharpe(1.0, risk_free_rate)
    assert measured["sharpe_ratio"] == pytest.approx(expected, rel=0.05), (
        "with the sign always right the strategy return is |actual|, whose "
        f"annualised Sharpe is {expected:.2f}"
    )


def test_no_edge_comes_back_as_no_edge():
    """The negative control. A pipeline that finds an edge in noise is worse
    than one that finds none anywhere."""
    measured = _measured_metrics(0.0)
    assert abs(measured["sharpe_ratio"]) < 0.6, (
        f"uninformative forecasts produced a Sharpe of "
        f"{measured['sharpe_ratio']:.3f}; over {BARS} bars the sampling "
        f"error is about {math.sqrt(252 / BARS):.2f}"
    )


def test_the_annualisation_factor_is_the_one_the_cadence_implies():
    """A Sharpe off by sqrt(252) is the classic silent error in this code.

    It is caught by the size checks above too, but named here so a failure
    says which of the two things went wrong.
    """
    measured = _measured_metrics(0.10)
    assert measured["periods_per_year_used"] == pytest.approx(252, abs=6), (
        f"business-daily bars were annualised with "
        f"{measured['periods_per_year_used']} periods per year"
    )


def test_a_bigger_planted_edge_reads_as_bigger(risk_free_rate):
    """Monotonicity: the instrument must order edges the way they are planted."""
    sharpes = [_measured_metrics(rho)["sharpe_ratio"] for rho in (0.0, 0.05, 0.10, 0.20)]
    assert sharpes == sorted(sharpes), (
        f"planted edges of 0, 0.05, 0.10, 0.20 came back as {sharpes}"
    )
