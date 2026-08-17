"""The cost of a trade is not a constant, and the config must be able to say so.

Until 2026-08-17 `targets.yaml` subtracted a flat 0.5% round trip from every
return target, copied into five separate places. Two things were wrong with
that and this file pins both.

1. THE ARITHMETIC. A broker charges per SHARE with a minimum per ORDER, so the
   cost in basis points falls as the order grows and rises as the share price
   falls. The measured break-even for the hourly models is 5-10 bp, which the
   old flat 50 bp cleared by five to ten times -- the assumption alone decided
   the result.
2. THE WIRING. The value has to reach the target series. A correct formula
   behind an unreachable branch is this project's most repeated defect, so the
   last test here drives the real config through the real calculator and reads
   the numbers back off the output rather than trusting the config.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.targets.calculators.regression_calculator import RegressionCalculator

TARGETS_YAML = Path("src/config/targets.yaml")


@pytest.fixture
def calc() -> RegressionCalculator:
    return RegressionCalculator()


def _bp(value: float) -> float:
    return round(float(value) * 1e4, 3)


class TestPerShareArithmetic:
    """$0.0035/share, $0.35 minimum, per side, doubled for the round trip."""

    PROFILE = {
        "model": "per_share",
        "per_share_fee": 0.0035,
        "min_fee_per_order": 0.35,
        "max_fee_pct_of_order": 0.01,
        "spread_pct": 0.0,
        "slippage_pct": 0.0,
    }

    def test_minimum_fee_dominates_a_small_order(self, calc):
        # $1,000 buys 4.3 shares of a $230 stock; 4.3 * $0.0035 = $0.015, far
        # under the $0.35 floor, so both sides pay the floor: 2 * 0.35/1000.
        cost = calc._round_trip_cost(
            pd.Series([230.0]), {**self.PROFILE, "order_value": 1000})
        assert _bp(cost.iloc[0]) == 7.0

    def test_the_same_stock_is_ten_times_cheaper_at_ten_times_the_size(self, calc):
        cost = calc._round_trip_cost(
            pd.Series([230.0]), {**self.PROFILE, "order_value": 10000})
        assert _bp(cost.iloc[0]) == 0.7

    def test_a_cheap_share_has_a_floor_no_order_size_can_beat(self, calc):
        # At $20 a share the per-share fee dominates from $2,000 upward, and
        # 2 * 0.0035/20 = 3.5 bp no matter how large the order gets. This is
        # the reason a cheap ticker can be structurally unprofitable while an
        # expensive one on the same signal is not.
        for order_value in (5_000, 50_000, 500_000):
            cost = calc._round_trip_cost(
                pd.Series([20.0]), {**self.PROFILE, "order_value": order_value})
            assert _bp(cost.iloc[0]) == 3.5

    def test_cost_is_computed_per_row_from_each_bar_own_price(self, calc):
        cost = calc._round_trip_cost(
            pd.Series([20.0, 230.0, 1000.0]),
            {**self.PROFILE, "order_value": 10000})
        assert [_bp(c) for c in cost] == [3.5, 0.7, 0.7]

    def test_spread_and_slippage_are_per_side_and_doubled(self, calc):
        cost = calc._round_trip_cost(
            pd.Series([230.0]),
            {**self.PROFILE, "order_value": 10000,
             "spread_pct": 0.0001, "slippage_pct": 0.0001})
        # 0.7 bp commission + 2 * (1 bp + 1 bp)
        assert _bp(cost.iloc[0]) == 4.7

    def test_broker_cap_bounds_the_fee(self, calc):
        # A $50 order of a $0.10 share is 500 shares = $1.75 of fee, which the
        # 1% cap cuts to $0.50 per side.
        cost = calc._round_trip_cost(
            pd.Series([0.10]), {**self.PROFILE, "order_value": 50})
        assert _bp(cost.iloc[0]) == _bp(2 * 0.50 / 50)

    def test_an_unusable_price_charges_the_minimum_rather_than_voiding_the_row(self, calc):
        cost = calc._round_trip_cost(
            pd.Series([float("nan"), 0.0, -5.0]),
            {**self.PROFILE, "order_value": 10000})
        assert cost.notna().all()
        # the per-order minimum: 2 * 0.35/10000
        assert [_bp(c) for c in cost] == [0.7, 0.7, 0.7]

    def test_a_missing_order_value_raises_instead_of_defaulting(self, calc):
        # There is no safe default: the answer is a function of this number.
        with pytest.raises(ValueError, match="order_value"):
            calc._round_trip_cost(pd.Series([230.0]), self.PROFILE)

    def test_an_unknown_model_raises(self, calc):
        with pytest.raises(ValueError, match="model"):
            calc._round_trip_cost(pd.Series([230.0]), {"model": "guesswork"})


class TestFlatModelStillWorks:
    """The old behaviour is reproducible, not deleted -- every figure produced
    before 2026-08-17 was computed with it."""

    def test_legacy_profile_reproduces_fifty_basis_points(self, calc):
        cost = calc._round_trip_cost(pd.Series([230.0]), {
            "model": "flat", "commission_pct": 0.001,
            "spread_pct": 0.0005, "slippage_pct": 0.001})
        assert _bp(cost) == 50.0

    def test_model_defaults_to_flat_for_a_config_that_predates_the_key(self, calc):
        cost = calc._round_trip_cost(pd.Series([230.0]), {
            "commission_pct": 0.001, "spread_pct": 0.0005,
            "slippage_pct": 0.001})
        assert _bp(cost) == 50.0


class TestTheConfigActuallyReachesTheTarget:
    """Test the wiring, not the function. Three fixes in this project have been
    correct and unreachable; each had a passing test of the pure function."""

    @pytest.fixture
    def profile(self) -> dict:
        raw = yaml.safe_load(TARGETS_YAML.read_text(encoding="utf-8"))
        return raw["cost_profiles"]["ibkr_pro_tiered"]

    def test_the_shipped_profile_is_the_per_share_model(self, profile):
        assert profile["model"] == "per_share"
        assert profile["order_value"] > 0

    def test_every_return_target_shares_one_profile_object(self):
        # YAML anchors resolve to the SAME dict, so identity proves there is
        # one copy rather than five that happen to agree today.
        raw = yaml.safe_load(TARGETS_YAML.read_text(encoding="utf-8"))
        shipped = raw["cost_profiles"]["ibkr_pro_tiered"]
        referents = [
            cfg["params"]["transaction_costs"]
            for cfg in raw["targets"].values()
            if isinstance(cfg, dict) and cfg.get("params", {}).get("adjust_for_costs")
        ]
        assert len(referents) >= 5
        assert all(r is shipped for r in referents)

    def test_a_cheap_and_an_expensive_bar_lose_different_amounts(self, calc, profile):
        # One frame, two tickers, identical 1% raw moves. If the cost reached
        # the series per row, the $20 name keeps less than the $400 name.
        df = pd.DataFrame({
            "ticker": ["CHEAP", "CHEAP", "RICH", "RICH"],
            "close": [20.0, 20.20, 400.0, 404.0],
        })
        adjusted = calc.calculate(
            df, base_col="close", shift=-1,
            adjust_for_costs=True, transaction_costs=profile)
        raw = 0.01
        cheap_cost = raw - float(adjusted.iloc[0])
        rich_cost = raw - float(adjusted.iloc[2])
        assert cheap_cost > rich_cost
        assert _bp(cheap_cost) == pytest.approx(3.5 + 4.0, abs=0.01)
        assert _bp(rich_cost) == pytest.approx(0.7 + 4.0, abs=0.01)

    def test_the_shipped_profile_costs_far_less_than_the_old_flat_fifty(self, calc, profile):
        # The whole reason for the change: at $10k in a liquid large cap the
        # real round trip is single-digit basis points, and the break-even the
        # models were measured against is 5-10 bp. Charging 50 decided the
        # answer before any model was trained.
        cost = calc._round_trip_cost(pd.Series([230.0]), profile)
        assert _bp(cost.iloc[0]) < 10.0
