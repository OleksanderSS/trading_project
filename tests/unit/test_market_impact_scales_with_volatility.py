"""Impact must scale with the volatility the function is already given.

`TransactionCostModel.calculate_execution_costs` computes the square-root law,
whose standard form is `impact ~= Y * sigma_daily * sqrt(participation)`. Until
2026-09-08 this engine's version had NO sigma in it -- a constant
`market_impact_coefficient` stood in for one, at 0.1, while a value a thousand
times smaller sat in the config read only by dead archive code (#290).

Measured on the project's own panel: the median daily return sd is 0.0202, so
0.1 implies Y = 4.95 and 0.0001 implies Y = 0.005. Five times too expensive and
two hundred times too cheap. The constant is gone; sigma is used where the law
puts it, and the function was already being handed one -- it had been scaling
its SLIPPAGE by volatility all along and its impact by nothing.

These tests hold the shape, not the number, because the number is the part that
should be allowed to move.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.backtesting.advanced.advanced_engine import TransactionCostModel

#: A median name on this panel, measured 2026-09-08.
ADV = 242_855_016.0
SIGMA = 0.0202


@pytest.fixture()
def model():
    return TransactionCostModel({})


def test_a_more_volatile_name_costs_more_to_move(model):
    """The whole point. A constant coefficient cannot express this."""
    quiet = model.calculate_execution_costs(25_000.0, ADV, 0.01)
    wild = model.calculate_execution_costs(25_000.0, ADV, 0.04)
    assert wild["market_impact"] > quiet["market_impact"] * 3.5, (
        f"quadrupling volatility moved impact from {quiet['market_impact']:.4f} "
        f"to {wild['market_impact']:.4f}; the law is linear in sigma, so it "
        "should roughly quadruple. A constant coefficient gives no change at "
        "all, which is what this replaced.")


def test_impact_follows_the_square_root_of_participation(model):
    """Quadrupling the order should roughly double the impact FRACTION."""
    small = model.calculate_execution_costs(10_000.0, ADV, SIGMA)
    large = model.calculate_execution_costs(40_000.0, ADV, SIGMA)
    small_fraction = small["market_impact"] / 10_000.0
    large_fraction = large["market_impact"] / 40_000.0
    assert large_fraction == pytest.approx(small_fraction * 2, rel=0.02), (
        f"impact fraction went {small_fraction:.6f} -> {large_fraction:.6f}; "
        "the square-root law says double")


def test_the_coefficient_left_over_is_dimensionless_and_of_order_one(model):
    """Y is O(1) in the literature this shape comes from.

    That is a BORROWED convention, not something measured here -- what was
    measured is sigma. The test exists so a future edit that pushes Y to 5, the
    way the old constant effectively did, fails instead of passing quietly.
    """
    assert 0.1 <= model.market_impact_y <= 3.0, (
        f"market_impact_y is {model.market_impact_y}. Outside O(1) it is "
        "standing in for something else again, which is the defect this "
        "replaced.")


def test_impact_no_longer_dominates_the_cost_at_realistic_size(model):
    """It dominated only because the coefficient was five times too large.

    At 0.1 it was 74% of the cost of a $25,000 order. Against a median name,
    $25,000 is a ten-thousandth of a day's volume, and a cost model that says
    impact is most of the bill at that participation is describing a different
    market.
    """
    costs = model.calculate_execution_costs(25_000.0, ADV, SIGMA)
    share = costs["market_impact"] / costs["total"]
    assert share < 0.25, (
        f"market impact is {share:.1%} of the cost of a $25,000 order against "
        f"a ${ADV:,.0f} name -- participation "
        f"{25_000.0 / ADV:.2e}. That is the old coefficient's behaviour.")


def test_zero_volatility_means_no_impact_not_a_crash(model):
    costs = model.calculate_execution_costs(10_000.0, ADV, 0.0)
    assert costs["market_impact"] == 0.0
    assert costs["total"] > 0, "the other cost terms must survive"


def test_the_other_cost_terms_are_untouched(model):
    """Only the impact term changed. Commission and spread are R22's."""
    costs = model.calculate_execution_costs(10_000.0, ADV, SIGMA)
    assert costs["commission"] > 0 and costs["spread"] > 0
    assert costs["total"] == pytest.approx(
        costs["commission"] + costs["spread"] + costs["market_impact"]
        + costs["slippage"])
