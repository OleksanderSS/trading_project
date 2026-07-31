"""The DEAN critic filter must actually run, and actually block.

Regression cover for a filter that had been a permanent no-op: register_model
was never called, so ConsensusEngine._apply_critic_filter always took the
"no critic registered" exception path and silently returned the unvetoed
signal. Several contract mismatches (DeanAction vs dict, DataFrame vs flat
dict, a pattern-analyzer method that never existed) meant that simply
registering a model would not have been enough either.
"""
import pandas as pd
import pytest

from src.meta_learning.dean_trading_models import DeanCritic
from src.models.dean.dean_bootstrap_system import DeanBootstrapSystem, ModelRole


@pytest.fixture()
def system():
    sys_ = DeanBootstrapSystem()
    sys_.register_model("dean_critic", ModelRole.CRITIC, DeanCritic())
    return sys_


def test_critique_runs_without_an_actor_registered(system):
    """The consensus IS the actor; requiring a DeanActor would be accidental."""
    action, critique = system.critique_existing_action(
        action_type="buy", confidence=0.7,
        context={"ticker": "NVDA", "regime": "ranging", "anomaly_score": 0.0},
    )
    assert action.action_type == "buy"
    assert -1.0 <= critique.critique_score <= 1.0
    assert action.action_id in system.action_history
    assert action.action_id in system.critique_history


def test_buying_into_a_volatile_regime_is_penalised(system):
    calm = system.critique_existing_action(
        action_type="buy", confidence=0.7,
        context={"ticker": "NVDA", "regime": "ranging", "anomaly_score": 0.0},
    )[1]
    rough = system.critique_existing_action(
        action_type="buy", confidence=0.7,
        context={"ticker": "NVDA", "regime": "volatile", "anomaly_score": 0.0},
    )[1]
    assert rough.critique_score < calm.critique_score


def test_high_anomaly_score_pushes_the_verdict_negative(system):
    _, critique = system.critique_existing_action(
        action_type="buy", confidence=0.7,
        context={"ticker": "NVDA", "regime": "volatile", "anomaly_score": 0.95},
    )
    assert critique.critique_score < 0
    assert any("nomal" in p for p in critique.critique_points)


def test_paradoxical_confidence_is_detected(system):
    _, critique = system.critique_existing_action(
        action_type="buy", confidence=0.99,
        context={"ticker": "NVDA", "regime": "volatile", "anomaly_score": 0.9},
    )
    assert any("Paradoxical" in p for p in critique.critique_points)


def test_high_volatility_is_penalised(system):
    _, critique = system.critique_existing_action(
        action_type="buy", confidence=0.7,
        context={"ticker": "NVDA", "regime": "ranging",
                 "anomaly_score": 0.0, "volatility": 0.25},
    )
    assert any("volatility" in p for p in critique.critique_points)


def test_accepts_a_dict_action_as_well_as_a_dataclass():
    """DeanTradingModels passes a dict; DeanBootstrapSystem passes DeanAction.
    The old code was typed for dict and crashed on the dataclass."""
    critic = DeanCritic()
    as_dict = critic.critique_action(
        {"type": "buy", "confidence": 0.7, "ticker": "NVDA"},
        {"regime": "ranging"},
    )
    assert -1.0 <= as_dict["score"] <= 1.0


def test_unfitted_meta_model_contributes_nothing_rather_than_failing():
    critic = DeanCritic()
    features = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
    out = critic.critique_action(
        {"type": "buy", "confidence": 0.7, "ticker": "NVDA"},
        {"regime": "ranging"},
        features,
    )
    assert out["expected_error"] == 0.0


def test_missing_context_values_do_not_raise():
    critic = DeanCritic()
    out = critic.critique_action({"type": "buy", "confidence": 0.7}, {})
    assert -1.0 <= out["score"] <= 1.0


def test_no_critic_registered_is_an_explicit_error_not_a_silent_pass():
    empty = DeanBootstrapSystem()
    with pytest.raises(ValueError, match="No critic model registered"):
        empty.critique_existing_action("buy", 0.7, {"ticker": "NVDA"})
