"""Summing per-position VaR assumes every asset is the same asset.

`check_limits` aggregated portfolio VaR as the plain sum of each position's
dollar VaR. That is the portfolio you would have if every holding moved in
lockstep -- correlation 1.0 with everything -- and it is the most pessimistic
one constructible from those positions.

Measured on this project's own daily returns across its 22 tickers, the median
pairwise correlation is 0.427. At that level the sum overstates portfolio VaR
by 1.49x, so a 5% limit trips at a true 3.4% and the kill switch fires on
arithmetic rather than on risk.

    correlation   true VaR    sum   overstatement
          0.000       4.69  22.00          4.69x
          0.300      12.67  22.00          1.74x
          0.427      14.81  22.00          1.49x
          0.700      18.58  22.00          1.18x
          1.000      22.00  22.00          1.00x
"""

from __future__ import annotations

import numpy as np
import pytest

from src.risk.elite_risk_metrics import EliteRiskMetrics


def _metrics(correlated: float = 0.65, tickers: int = 22, rows: int = 300):
    instance = EliteRiskMetrics.__new__(EliteRiskMetrics)
    rng = np.random.default_rng(0)
    common = rng.standard_normal(rows)
    instance.returns_history = {
        f"T{i}": (correlated * common
                  + np.sqrt(max(0.0, 1 - correlated ** 2)) * rng.standard_normal(rows)) * 0.01
        for i in range(tickers)
    }
    return instance


def test_a_correlated_book_is_still_less_risky_than_the_sum():
    metrics = _metrics()
    exposure = {f"T{i}": 100.0 for i in range(22)}
    diversified = metrics._diversified_var(exposure)
    assert diversified is not None
    assert diversified < sum(exposure.values())


def test_independent_assets_diversify_far_more_than_correlated_ones():
    exposure = {f"T{i}": 100.0 for i in range(22)}
    correlated = _metrics(correlated=0.9)._diversified_var(exposure)
    independent = _metrics(correlated=0.0)._diversified_var(exposure)
    assert independent < correlated


def test_perfectly_correlated_assets_recover_the_sum():
    """The old behaviour is the limiting case, not a different formula."""
    instance = EliteRiskMetrics.__new__(EliteRiskMetrics)
    series = np.linspace(-0.02, 0.02, 300)
    instance.returns_history = {f"T{i}": series.copy() for i in range(5)}
    exposure = {f"T{i}": 100.0 for i in range(5)}
    assert instance._diversified_var(exposure) == pytest.approx(500.0, rel=1e-6)


def test_a_single_position_has_nothing_to_diversify():
    assert _metrics()._diversified_var({"T0": 100.0}) is None


def test_positions_without_history_are_treated_as_worst_case():
    """Not knowing is a reason to assume more risk, not less."""
    metrics = _metrics(tickers=3)
    known = metrics._diversified_var({"T0": 100.0, "T1": 100.0, "T2": 100.0})
    mixed = metrics._diversified_var({"T0": 100.0, "T1": 100.0, "UNKNOWN": 100.0})
    assert mixed > known


def test_no_usable_history_leaves_the_sum_alone():
    instance = EliteRiskMetrics.__new__(EliteRiskMetrics)
    instance.returns_history = {}
    assert instance._diversified_var({"A": 100.0, "B": 100.0}) is None
