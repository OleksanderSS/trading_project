"""The critic's reward loop must close: verdict -> trade -> realised PnL.

`DeanBootstrapSystem.calculate_reward` -- the "+1 to the critic if it
correctly warned" mechanism -- keys off the `action_id` the critic produced.
Nothing carried that id past the ConsensusReport, so a realised PnL could
never be attributed to the verdict that allowed the trade and the method was
uncallable in practice. These pin the whole chain.
"""
from __future__ import annotations

import logging

import pytest

from src.meta_learning.dean_trading_models import DeanCritic
from src.models.dean.dean_bootstrap_system import (
    DeanBootstrapSystem,
    ModelRole,
)
from src.trading.trader import TradeOrder
from src.trading.trading_orchestrator import TradingOrchestrator


class _Trader:
    def execute_order(self, _order):
        return True


class _Portfolio:
    """Records buys; returns a realised PnL on sell."""

    def __init__(self, pnl: float = 0.0, succeed: bool = True):
        self.pnl = pnl
        self.succeed = succeed
        self.bought: list[str] = []

    def buy_stock(self, params):
        self.bought.append(params["ticker"])
        return {"success": True}

    def sell_stock(self, ticker, quantity, price, reason=""):
        if not self.succeed:
            return {"success": False, "error": "no position"}
        return {"success": True, "transaction": {"pnl": self.pnl}}


def _orchestrator(portfolio):
    orch = object.__new__(TradingOrchestrator)
    orch.logger = logging.getLogger("test")
    orch.trader = _Trader()
    orch.portfolio = portfolio
    orch._open_critic_actions = {}
    return orch


@pytest.fixture()
def system():
    sys_ = DeanBootstrapSystem()
    sys_.register_model("dean_critic", ModelRole.CRITIC, DeanCritic())
    return sys_


def _critique(system, action_type="buy", confidence=0.7, regime="ranging"):
    action, _ = system.critique_existing_action(
        action_type=action_type, confidence=confidence,
        context={"ticker": "NVDA", "regime": regime, "anomaly_score": 0.0},
    )
    return action.action_id


def test_trade_order_carries_the_critic_action_id():
    order = TradeOrder(ticker="NVDA", quantity=1, price=10.0, action="BUY",
                       critic_action_id="act_123")
    assert order.critic_action_id == "act_123"


def test_buy_then_sell_scores_the_verdict_that_allowed_it(system, monkeypatch):
    monkeypatch.setattr(
        "src.models.dean.dean_bootstrap_system.get_dean_system", lambda: system
    )
    action_id = _critique(system)
    portfolio = _Portfolio(pnl=250.0)
    orch = _orchestrator(portfolio)

    orch._execute_orders([
        TradeOrder("NVDA", 10, 100.0, "BUY", critic_action_id=action_id),
    ])
    assert orch._open_critic_actions == {"NVDA": action_id}

    orch._execute_orders([TradeOrder("NVDA", 10, 125.0, "SELL")])

    assert orch._open_critic_actions == {}, "the entry must be consumed"
    assert len(system.reward_history) == 1
    assert system.reward_history[0]["action_id"] == action_id
    assert system.reward_history[0]["pnl"] == 250.0


def test_a_critic_that_warned_correctly_is_rewarded(system, monkeypatch):
    """Negative verdict + losing trade => positive critic reward."""
    monkeypatch.setattr(
        "src.models.dean.dean_bootstrap_system.get_dean_system", lambda: system
    )
    # High anomaly and a volatile regime drive the score negative.
    action, critique = system.critique_existing_action(
        action_type="buy", confidence=0.99,
        context={"ticker": "NVDA", "regime": "volatile", "anomaly_score": 0.95},
    )
    assert critique.critique_score < 0

    orch = _orchestrator(_Portfolio(pnl=-500.0))
    orch._execute_orders([
        TradeOrder("NVDA", 10, 100.0, "BUY", critic_action_id=action.action_id),
    ])
    orch._execute_orders([TradeOrder("NVDA", 10, 50.0, "SELL")])

    assert system.reward_history[0]["critic_reward"] > 0


def test_a_critic_that_blocked_a_winner_is_penalised(system, monkeypatch):
    monkeypatch.setattr(
        "src.models.dean.dean_bootstrap_system.get_dean_system", lambda: system
    )
    action, critique = system.critique_existing_action(
        action_type="buy", confidence=0.99,
        context={"ticker": "NVDA", "regime": "volatile", "anomaly_score": 0.95},
    )
    assert critique.critique_score < 0

    orch = _orchestrator(_Portfolio(pnl=400.0))
    orch._execute_orders([
        TradeOrder("NVDA", 10, 100.0, "BUY", critic_action_id=action.action_id),
    ])
    orch._execute_orders([TradeOrder("NVDA", 10, 140.0, "SELL")])

    assert system.reward_history[0]["critic_reward"] < 0


def test_a_failed_sell_scores_nothing(system, monkeypatch):
    monkeypatch.setattr(
        "src.models.dean.dean_bootstrap_system.get_dean_system", lambda: system
    )
    action_id = _critique(system)
    orch = _orchestrator(_Portfolio(pnl=100.0, succeed=False))

    orch._execute_orders([
        TradeOrder("NVDA", 10, 100.0, "BUY", critic_action_id=action_id),
    ])
    orch._execute_orders([TradeOrder("NVDA", 10, 110.0, "SELL")])

    assert system.reward_history == []


def test_a_trade_with_no_critic_verdict_is_simply_skipped(system, monkeypatch):
    monkeypatch.setattr(
        "src.models.dean.dean_bootstrap_system.get_dean_system", lambda: system
    )
    orch = _orchestrator(_Portfolio(pnl=100.0))

    orch._execute_orders([TradeOrder("NVDA", 10, 100.0, "BUY")])
    orch._execute_orders([TradeOrder("NVDA", 10, 110.0, "SELL")])

    assert orch._open_critic_actions == {}
    assert system.reward_history == []


def test_selling_a_ticker_that_was_never_bought_does_not_raise(system, monkeypatch):
    monkeypatch.setattr(
        "src.models.dean.dean_bootstrap_system.get_dean_system", lambda: system
    )
    orch = _orchestrator(_Portfolio(pnl=100.0))
    orch._execute_orders([TradeOrder("AAPL", 5, 200.0, "SELL")])
    assert system.reward_history == []
