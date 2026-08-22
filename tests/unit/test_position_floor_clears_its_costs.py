"""The minimum position size was a round number, not a consequence of costs.

`min_position_size_pct` is 0.005, so on a $100,000 account every position is
$500 -- the product of eight risk multipliers always lands under that clamp.
A $500 round trip costs 0.180% under this repo's own IBKR profile, against a
gross edge last measured at 0.14%. Every position the system opens is below
break-even before any question of skill arises.

These are arithmetic, not opinion, and they hold whether or not the edge is
real: they only say that a floor chosen without reference to a cost model is
not a floor, it is a coincidence.
"""

from __future__ import annotations

import io

import pytest
import yaml


def _cost_profile() -> dict:
    config = yaml.safe_load(io.open("src/config/targets.yaml", encoding="utf-8"))

    def find(node, key):
        if isinstance(node, dict):
            if key in node:
                return node[key]
            for value in node.values():
                found = find(value, key)
                if found is not None:
                    return found
        return None

    return (find(config, "cost_profiles") or {})["ibkr_pro_tiered"]


def round_trip_cost(notional: float, price: float = 150.0) -> float:
    """Commission both ways, plus spread and slippage both ways."""
    p = _cost_profile()
    fee = min(
        max(float(p["per_share_fee"]) * (notional / price), float(p["min_fee_per_order"])),
        float(p["max_fee_pct_of_order"]) * notional,
    )
    friction = 2 * (float(p["spread_pct"]) + float(p["slippage_pct"]))
    return 2 * fee / notional + friction


def break_even_notional(edge: float) -> float:
    p = _cost_profile()
    friction = 2 * (float(p["spread_pct"]) + float(p["slippage_pct"]))
    if edge <= friction:
        return float("inf")
    return 2 * float(p["min_fee_per_order"]) / (edge - friction)


def test_the_cost_of_a_small_position_exceeds_the_measured_edge():
    """$500 is what the current rule delivers on a $100,000 account."""
    assert round_trip_cost(500) > 0.0014


def test_cost_falls_as_the_position_grows_until_the_minimum_stops_binding():
    sizes = [500, 1_000, 2_500, 5_000, 10_000]
    costs = [round_trip_cost(n) for n in sizes]
    assert costs == sorted(costs, reverse=True)
    # Beyond the point where the per-order minimum no longer binds, only
    # spread and slippage remain and the curve flattens.
    assert round_trip_cost(20_000) == pytest.approx(round_trip_cost(50_000), abs=1e-5)


def test_friction_alone_makes_a_small_edge_unreachable_at_any_size():
    """Spread and slippage do not scale away."""
    assert break_even_notional(0.0003) == float("inf")


@pytest.mark.parametrize("edge,expected", [(0.0014, 700), (0.0010, 1167), (0.0007, 2333)])
def test_break_even_position_size_for_a_given_edge(edge, expected):
    assert break_even_notional(edge) == pytest.approx(expected, rel=0.01)


def test_the_configured_floor_is_below_break_even_at_the_measured_edge():
    """Pins the defect: a real fix makes this test fail, which is correct."""
    from src.algorithms.adaptive_position_sizer import AdaptivePositionSizer

    floor_pct = AdaptivePositionSizer(config={}).min_position_size_pct
    floor_notional = 100_000 * floor_pct
    assert floor_notional < break_even_notional(0.0014), (
        "the floor now clears its costs; update this test to assert the new rule"
    )


def test_the_requirement_is_written_down_where_a_decision_would_be_made():
    text = io.open("src/config/pending_decisions.yaml", encoding="utf-8").read()
    assert "what_is_already_decided_by_calculation" in text
    assert "break-even" in text.lower()
