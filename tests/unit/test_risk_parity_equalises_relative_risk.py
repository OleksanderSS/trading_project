"""Risk parity was ranking a 65/35/0 risk split above an equal one.

`calculate_risk_contribution` returns ABSOLUTE contributions, which sum to the
portfolio's volatility. The ERC objective compared them against `1.0 / n`, so
it was asking for a sum of 1.0 -- a portfolio volatility of exactly 100%.

The audit predicted the optimiser would fail to converge. It converges. It
converges to the wrong portfolio, because piling weight into the most volatile
name moves the sum closer to an unreachable target. Measured on three assets:

    true ERC weights  [0.510, 0.280, 0.210]   relative risk [0.333, 0.333, 0.333]
      scored by the old objective  0.235
    old objective's own optimum
                      [0.010, 0.471, 0.519]   relative risk [0.002, 0.350, 0.648]
      scored                       0.199   -- preferred
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from src.algorithms.risk_parity_allocator import RiskParityAllocator

VOLS = np.array([0.15, 0.25, 0.35])
CORR = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]])


@pytest.fixture
def allocator():
    return RiskParityAllocator(config={})


def _solve(objective, n=3):
    return minimize(
        objective, np.ones(n) / n, bounds=[(0.01, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )


def _relative(allocator, weights):
    contrib = allocator.calculate_risk_contribution(np.asarray(weights), VOLS, CORR)
    return contrib / contrib.sum()


def test_the_optimum_carries_equal_risk_not_equal_weight(allocator):
    result = _solve(allocator._create_erc_objective(VOLS, CORR, 3))
    assert result.success
    assert _relative(allocator, result.x) == pytest.approx([1 / 3] * 3, abs=1e-3)


def test_the_objective_is_near_zero_at_the_solution(allocator):
    """It used to be 0.235 there, and 0.199 somewhere much worse."""
    result = _solve(allocator._create_erc_objective(VOLS, CORR, 3))
    assert result.fun < 1e-6


def test_the_riskiest_asset_gets_the_smallest_weight(allocator):
    """Equal risk from unequal volatility means unequal weight, inverted."""
    result = _solve(allocator._create_erc_objective(VOLS, CORR, 3))
    order = np.argsort(result.x)
    assert list(order) == [2, 1, 0], f"weights {result.x} do not invert the vols {VOLS}"


def test_a_concentrated_portfolio_scores_worse_than_the_equal_one(allocator):
    """The comparison the old objective got backwards."""
    objective = allocator._create_erc_objective(VOLS, CORR, 3)
    equal_risk = _solve(objective).x
    concentrated = np.array([0.01, 0.47, 0.52])
    assert objective(equal_risk) < objective(concentrated)


def test_the_objective_does_not_depend_on_the_scale_of_volatility(allocator):
    """Relative contributions are scale-free; absolute ones are not."""
    objective_small = allocator._create_erc_objective(VOLS, CORR, 3)
    objective_large = allocator._create_erc_objective(VOLS * 10, CORR, 3)
    weights = _solve(objective_small).x
    assert objective_large(weights) == pytest.approx(objective_small(weights), abs=1e-9)


def test_a_degenerate_portfolio_does_not_divide_by_zero(allocator):
    objective = allocator._create_erc_objective(np.zeros(3), CORR, 3)
    assert np.isfinite(objective(np.ones(3) / 3))
