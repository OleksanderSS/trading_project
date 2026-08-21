"""Eight risk adjustments, and a floor that made all of them irrelevant.

`AdaptivePositionSizer` multiplies a base size by eight factors -- VaR, Kelly,
confidence, volatility, drawdown, open positions, regime, liquidity -- and then
clamps the result into [min_pct, max_pct] of the portfolio.

Measured on a 100,000 portfolio, this is the live path
(`calculate_position_size` -> `_calculate_position_size_from_params`):

    calm, confidence 0.85      product 0.0611  ->  122  ->  500
    typical                    product 0.0324  ->   65  ->  500
    drawdown 12%, 8 positions  product 0.0201  ->   40  ->  500

Every case lands under the 0.5% floor, so a 12% drawdown with eight positions
open sizes exactly like a calm market at 85% confidence. Gemini's audit
predicted positions collapsing to 0.01% of the portfolio; the floor prevents
that and replaces it with something quieter -- a constant.

Behaviour is unchanged here. What changes is that the sizer now says when its
own risk logic stopped affecting the answer.
"""

import numpy as np
import pytest

from src.algorithms.adaptive_position_sizer import AdaptivePositionSizer


@pytest.fixture
def sizer():
    return AdaptivePositionSizer(config={})


def _size(sizer, **kwargs):
    defaults = dict(
        portfolio_value=100_000.0,
        volatility=0.02,
        confidence=0.65,
        max_drawdown=0.05,
        active_positions=5,
        market_regime="NORMAL",
        daily_volume=5e6,
        current_price=150.0,
        historical_returns=np.random.default_rng(0).normal(0.0004, 0.015, 500),
        win_rate=0.55,
        payout_ratio=1.2,
    )
    defaults.update(kwargs)
    return sizer.calculate_position_size_legacy(**defaults)


def test_the_floor_binding_is_reported(sizer):
    result = _size(sizer)
    assert result["limit_binding"] == "minimum"
    assert result["unclipped_size"] < result["position_size"]


def test_the_risk_adjustments_currently_change_nothing(sizer):
    """Pins the defect, so a real fix shows up as this test failing."""
    calm = _size(sizer, volatility=0.012, confidence=0.85,
                 max_drawdown=0.0, active_positions=2)
    stressed = _size(sizer, volatility=0.030, confidence=0.60,
                     max_drawdown=0.12, active_positions=8)

    assert stressed["effective_multiplier"] < calm["effective_multiplier"]
    assert stressed["position_size"] == calm["position_size"]


def test_the_unclipped_size_does_respond_to_risk(sizer):
    """The arithmetic works; the clamp is what discards it."""
    calm = _size(sizer, volatility=0.012, confidence=0.85,
                 max_drawdown=0.0, active_positions=2)
    stressed = _size(sizer, volatility=0.030, confidence=0.60,
                     max_drawdown=0.12, active_positions=8)
    assert stressed["unclipped_size"] < calm["unclipped_size"]


def test_a_size_inside_the_limits_reports_no_binding(sizer):
    """Not every call is clamped; the flag must mean something."""
    result = _size(sizer, confidence=1.0, volatility=0.001,
                   max_drawdown=0.0, active_positions=0,
                   win_rate=0.95, payout_ratio=5.0)
    if result["limit_binding"] is None:
        assert result["unclipped_size"] == pytest.approx(result["position_size"])
    else:
        pytest.skip("even the most favourable inputs are clamped here")


def test_which_limit_binds_names_the_right_side(sizer):
    assert sizer._which_limit_binds(100.0, 500.0) == "minimum"
    assert sizer._which_limit_binds(50_000.0, 10_000.0) == "maximum"
    assert sizer._which_limit_binds(700.0, 700.0) is None


def test_kelly_is_pinned_at_its_own_floor_for_a_realistic_edge(sizer):
    """A 55% win rate at 1.2 payout is a real edge, and it lands on 0.1.

    kelly_f = (1.2*0.55 - 0.45)/1.2 = 0.175, halved to 0.0875 by
    kelly_fraction, then floored to 0.1 -- so the factor is a constant for any
    edge below roughly 0.2. It is also a share of CAPITAL being used as a
    dimensionless multiplier on a base size, which is a unit mismatch, not a
    tuning choice.
    """
    assert sizer._calculate_kelly_adjustment(0.65, win_rate=0.55, payout_ratio=1.2) == 0.1
    assert sizer._calculate_kelly_adjustment(0.65, win_rate=0.51, payout_ratio=1.05) == 0.1
