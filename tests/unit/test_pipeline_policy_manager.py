"""PipelinePolicyManager: the precedence rule and the split resolution.

The point of this component is that configured limits are a HARD CEILING and
regime adaptation may only tighten them. Getting that backwards would let
switching regimes silently widen a risk budget someone set deliberately, so
it is pinned from both directions here.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.policy.pipeline_policy_manager import PipelinePolicyManager


class FakeConfig:
    def __init__(self, sections: dict):
        self._sections = sections

    def get_config(self, name, default=None):
        return self._sections.get(name, default if default is not None else {})


class FakeAdaptive:
    """Stands in for AdaptiveParameterManager without importing its enums."""

    def __init__(self, **values):
        self._values = values

    def compute_adaptive_params(self, regime, asset_class, volatility_percentile):
        return SimpleNamespace(**self._values)


RISK = {"strategy": {"risk_management": {
    "max_daily_loss_pct": 0.03,
    "max_position_size_pct": 0.10,
    "max_drawdown_pct": 0.15,
}}}


def test_configured_limits_are_used_when_no_regime_is_given():
    policy = PipelinePolicyManager(FakeConfig(RISK))
    limits = policy.risk_limits()

    assert limits.max_daily_loss_pct == 0.03
    assert limits.max_position_size_pct == 0.10
    assert limits.tightened_by_regime is False


def test_regime_may_tighten_a_configured_ceiling():
    """The 'dead' regime wants 0.01 against a configured 0.03 -- tighter wins.

    Real preset values: trending_up 0.06, trending_down 0.04, ranging 0.05,
    volatile 0.03, dead 0.01.
    """
    policy = PipelinePolicyManager(
        FakeConfig(RISK), FakeAdaptive(max_daily_drawdown_pct=0.01)
    )
    limits = policy.risk_limits(regime="dead")

    assert limits.max_daily_loss_pct == 0.01
    assert limits.tightened_by_regime is True


def test_an_unknown_regime_name_falls_back_to_configured_ceilings():
    """MarketRegime has no 'crisis' member. An unrecognised name must not
    silently widen anything -- it degrades to the configured ceiling."""
    policy = PipelinePolicyManager(
        FakeConfig(RISK), FakeAdaptive(max_daily_drawdown_pct=0.01)
    )
    limits = policy.risk_limits(regime="crisis")

    assert limits.max_daily_loss_pct == 0.03
    assert limits.tightened_by_regime is False


def test_regime_may_never_loosen_a_configured_ceiling():
    """Trending-up wants 0.06 against a configured 0.03 -- config wins.

    This is the case that matters: AdaptiveParameterManager's trending_up
    preset really does carry 0.06 while risk_management.yaml says 0.03.
    """
    policy = PipelinePolicyManager(
        FakeConfig(RISK), FakeAdaptive(max_daily_drawdown_pct=0.06)
    )
    limits = policy.risk_limits(regime="trending_up")

    assert limits.max_daily_loss_pct == 0.03
    assert limits.tightened_by_regime is False


def test_a_broken_adaptive_manager_falls_back_to_configured_ceilings():
    class Exploding:
        def compute_adaptive_params(self, **_kwargs):
            raise RuntimeError("boom")

    policy = PipelinePolicyManager(FakeConfig(RISK), Exploding())
    limits = policy.risk_limits(regime="crisis")

    assert limits.max_daily_loss_pct == 0.03      # never silently widened
    assert limits.tightened_by_regime is False


def test_risk_per_trade_defaults_to_the_tighter_of_the_two_existing_values():
    """No config key of that name exists; PortfolioManager assumed 0.03 and
    AdaptiveParameters 0.02. The tighter one is the honest default."""
    policy = PipelinePolicyManager(FakeConfig(RISK))
    assert policy.risk_limits().risk_per_trade_pct == 0.02


def test_missing_risk_config_still_yields_conservative_limits():
    policy = PipelinePolicyManager(FakeConfig({}))
    limits = policy.risk_limits()

    assert 0 < limits.max_daily_loss_pct <= 0.03
    assert 0 < limits.max_position_size_pct <= 0.10


@pytest.mark.parametrize(
    "sections,expected,source",
    [
        ({"modeling": {"test_size": 0.25}}, 0.25, "modeling.test_size"),
        ({"processing": {"data_preparation": {"test_size": 0.3}}}, 0.3,
         "processing.data_preparation.test_size"),
        ({"unified_config": {"preparation": {"test_size": 0.15}}}, 0.15,
         "unified_config.preparation.test_size"),
        ({}, 0.2, "builtin_default"),
    ],
)
def test_split_resolution_order_and_provenance(sections, expected, source):
    policy = PipelinePolicyManager(FakeConfig(sections))
    split = policy.split_policy()

    assert split.test_size == expected
    assert split.source == source


def test_split_reports_where_the_number_came_from():
    """Provenance is the point: today's real answer is the builtin default,
    even though two config files declare a test_size, because the modeling
    stage reads a 'modeling' section that does not exist."""
    policy = PipelinePolicyManager(FakeConfig({}))
    assert policy.split_policy().source == "builtin_default"


def test_split_sizes_are_coherent():
    policy = PipelinePolicyManager(FakeConfig({"modeling": {"test_size": 0.2}}))
    split = policy.split_policy()

    assert split.train_size == pytest.approx(1.0 - split.test_size - split.val_size)
    assert 0 < split.train_size < 1


def test_nonsense_split_values_are_ignored():
    for bad in (0, 1, 1.5, -0.2, "0.2", None):
        policy = PipelinePolicyManager(FakeConfig({"modeling": {"test_size": bad}}))
        assert policy.split_policy().source == "builtin_default", bad


def test_hyperparameters_report_only_what_is_configured():
    policy = PipelinePolicyManager(FakeConfig(
        {"models": {"hyperparameters": {"lstm": {"units": 64}}}}
    ))
    assert policy.hyperparameters("lstm") == {"units": 64}
    assert policy.hyperparameters("catboost") == {}


# ── PortfolioManager wiring ──────────────────────────────────────────────

class _Portfolio:
    def __init__(self, drawdown):
        self._drawdown = drawdown

    def get_daily_drawdown(self, _prices):
        return self._drawdown


def _manager(drawdown, config):
    from src.trading.portfolio_manager import PortfolioManager

    return PortfolioManager(
        virtual_portfolio=_Portfolio(drawdown),
        config=config,
    )


def test_kill_switch_is_unchanged_when_no_regime_is_known():
    """Default behaviour must be byte-identical to before this component."""
    pm = _manager(-0.04, {"max_daily_loss_pct": 0.03})
    assert pm.is_trading_allowed({}) is False       # 4% breach of a 3% limit

    pm = _manager(-0.02, {"max_daily_loss_pct": 0.03})
    assert pm.is_trading_allowed({}) is True


def test_kill_switch_tightens_in_a_dead_regime(monkeypatch):
    """Configured 3%, dead-regime 1%: a 2% loss must now block."""
    pm = _manager(-0.02, {"max_daily_loss_pct": 0.03})
    pm.current_regime = "dead"
    pm._policy_manager = PipelinePolicyManager(
        FakeConfig(RISK), FakeAdaptive(max_daily_drawdown_pct=0.01)
    )
    assert pm.is_trading_allowed({}) is False


def test_kill_switch_never_loosens_in_a_trending_regime():
    """Trending-up wants 6% but config says 3%: a 4% loss must still block."""
    pm = _manager(-0.04, {"max_daily_loss_pct": 0.03})
    pm.current_regime = "trending_up"
    pm._policy_manager = PipelinePolicyManager(
        FakeConfig(RISK), FakeAdaptive(max_daily_drawdown_pct=0.06)
    )
    assert pm.is_trading_allowed({}) is False


def test_policy_failure_falls_back_to_the_configured_limit():
    class Exploding:
        def risk_limits(self, **_kwargs):
            raise RuntimeError("boom")

    pm = _manager(-0.02, {"max_daily_loss_pct": 0.03})
    pm.current_regime = "dead"
    pm._policy_manager = Exploding()
    assert pm.is_trading_allowed({}) is True     # 2% < configured 3%
