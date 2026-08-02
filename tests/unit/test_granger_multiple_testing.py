"""Taking the smallest p-value over several lags is multiple testing.

_calculate_granger_p_value ran the test at every lag up to maxlag and
returned min(p). Under the null that is the best of `maxlag` draws, so the
stated 5% threshold is not 5%.

Measured on 300 pairs of INDEPENDENT random series, maxlag=5:

    min over 5 lags, uncorrected   15.3% declared causal at p<0.05
    a single fixed lag              6.7%
    min with the Sidak correction   2.0%

And the counter-check, because a correction that blinds the test is worse
than no correction -- 100 series where x genuinely drives y at lag 2:

    coefficient 0.60   detected 100%
    coefficient 0.30   detected  96%
    coefficient 0.15   detected  33%

Sidak is conservative here since the per-lag tests are correlated rather
than independent, which is the right direction for a screening tool feeding
feature selection.

Checked and found correct, so left alone: the DIRECTION. statsmodels tests
whether the SECOND column Granger-causes the first, and the code passes
[target, predictor] -- "does the predictor cause the target", as intended.
Getting that backwards is the classic error here.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.analytics.calculators.econometrics_calculator import EconometricsCalculator


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _independent(seed, n=200):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n)})


def _causal(seed, strength=0.6, n=250, lag=2):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(lag, n):
        y[t] = strength * x[t - lag] + rng.normal(0, 1)
    return pd.DataFrame({"y": y, "x": x})


def test_a_p_value_is_a_probability():
    p = EconometricsCalculator._calculate_granger_p_value(_independent(0), 5)

    assert 0.0 <= p <= 1.0


def test_independent_series_are_rarely_called_causal():
    """The regression: uncorrected min-over-lags fired on 15.3% of these."""
    false_positives = sum(
        EconometricsCalculator._calculate_granger_p_value(_independent(seed), 5) < 0.05
        for seed in range(60)
    )

    assert false_positives <= 4, f"{false_positives}/60 independent pairs called causal"


def test_a_strong_real_relationship_is_still_found():
    """A correction that blinds the test would be worse than none."""
    detected = sum(
        EconometricsCalculator._calculate_granger_p_value(_causal(seed), 5) < 0.05
        for seed in range(20)
    )

    assert detected >= 18


def test_a_moderate_relationship_is_mostly_found():
    detected = sum(
        EconometricsCalculator._calculate_granger_p_value(
            _causal(seed, strength=0.3), 5
        ) < 0.05
        for seed in range(20)
    )

    assert detected >= 15


def test_the_correction_only_ever_raises_the_p_value():
    """Sidak is conservative; it must never make a result look stronger."""
    from statsmodels.tsa.stattools import grangercausalitytests

    frame = _causal(3)
    raw = grangercausalitytests(frame, maxlag=5, verbose=False)
    smallest = min(raw[lag][0]["ssr_ftest"][1] for lag in range(1, 6))

    corrected = EconometricsCalculator._calculate_granger_p_value(frame, 5)

    assert corrected >= smallest - 1e-9


def test_the_direction_is_predictor_causes_target():
    """statsmodels tests whether the SECOND column causes the first, so the
    frame must be [target, predictor]."""
    import inspect

    source = inspect.getsource(EconometricsCalculator._test_single_predictor)

    assert "df[[target_col, predictor_col]]" in source


def test_one_sided_causality_is_not_symmetric():
    """x drives y, so testing y as a driver of x must be far weaker."""
    frame = _causal(7, strength=0.8)

    forward = EconometricsCalculator._calculate_granger_p_value(
        frame[["y", "x"]], 5
    )
    backward = EconometricsCalculator._calculate_granger_p_value(
        frame[["x", "y"]], 5
    )

    assert forward < backward


# --- the "comprehensive" sibling ------------------------------------------


def test_the_advanced_validator_no_longer_fails_on_every_call():
    """VAR.resid is 2-D (one column per equation) and acorr_ljungbox needs
    1-D, so this raised ValueError every time. ValueError is in the module's
    handler, so it came back as {'p_value': 1.0, 'is_valid': False} --
    indistinguishable from "tested and found no causality". The whole
    comprehensive analysis therefore never found anything."""
    from src.analytics.calculators.advanced_econometrics_calculator import (
        AdvancedEconometricsCalculator,
    )

    result = AdvancedEconometricsCalculator._run_granger_with_validation(
        _independent(0), 5
    )

    assert "error" not in result
    assert "ljung_box_pvalue" in result["residual_diagnostics"]


def test_the_advanced_validator_finds_a_real_relationship():
    from src.analytics.calculators.advanced_econometrics_calculator import (
        AdvancedEconometricsCalculator,
    )

    result = AdvancedEconometricsCalculator._run_granger_with_validation(
        _causal(0, strength=0.6), 3
    )

    assert result["p_value"] < 0.01
    assert result["best_lag"] == 3


def test_the_advanced_validator_reports_the_selected_lag():
    """_select_optimal_lag has already chosen on an information criterion,
    so taking the smallest p-value across 1..lag would be multiple testing
    for no gain."""
    from src.analytics.calculators.advanced_econometrics_calculator import (
        AdvancedEconometricsCalculator,
    )

    result = AdvancedEconometricsCalculator._run_granger_with_validation(
        _causal(1, strength=0.5), 4
    )

    assert result["best_lag"] == 4
    assert result["p_value"] == pytest.approx(result["all_p_values"][3])


def test_the_advanced_validator_keeps_its_error_rate():
    from src.analytics.calculators.advanced_econometrics_calculator import (
        AdvancedEconometricsCalculator,
    )

    false_positives = sum(
        AdvancedEconometricsCalculator._run_granger_with_validation(
            _independent(seed), 5
        ).get("p_value", 1.0) < 0.05
        for seed in range(40)
    )

    assert false_positives <= 5
