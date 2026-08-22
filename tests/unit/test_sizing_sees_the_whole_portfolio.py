"""Sizing was computed from cash plus one holding, and spent more than it had.

Two defects in `_calculate_position_size`, both on the live path.

`get_total_value` skips any position whose ticker is absent from the price map
-- `price = current_prices.get(ticker)` and then `if price:`. Sizing called it
with `{ticker: price}`, one entry, so with ten positions open nine were valued
at zero and the equity behind every size was cash plus a single holding.

And `shares_from_cash = current_balance / price` left nothing for the
commission `VirtualPortfolio` then adds, so an order sized to the whole
balance is refused outright -- "Insufficient funds including transaction
costs". Not trimmed. Refused.
"""

from __future__ import annotations

import logging

import pytest

from src.trading.portfolio_manager import PortfolioManager


class _Portfolio:
    def __init__(self, balance=10_000.0, positions=None):
        self.current_balance = balance
        self.positions = positions or {}
        self.seen_prices = None

    def get_total_value(self, current_prices):
        self.seen_prices = dict(current_prices)
        total = self.current_balance
        for ticker, position in self.positions.items():
            price = current_prices.get(ticker)
            if price:
                total += position["quantity"] * price
        return total


def _manager(portfolio):
    manager = PortfolioManager(virtual_portfolio=portfolio,
                               config={"max_daily_loss_pct": 0.05})
    manager.logger = logging.getLogger("sizing-test")
    return manager


def test_every_open_position_is_priced_when_sizing():
    """Nine of ten were being valued at zero."""
    positions = {f"T{i}": {"quantity": 10, "entry_price": 100.0} for i in range(10)}
    portfolio = _Portfolio(positions=positions)
    manager = _manager(portfolio)
    prices = {f"T{i}": 100.0 for i in range(10)}

    manager._calculate_position_size("T0", 100.0, 0.7, current_prices=prices)

    assert portfolio.seen_prices is not None
    assert len(portfolio.seen_prices) == 10, portfolio.seen_prices


def test_equity_is_larger_once_the_other_holdings_are_visible():
    positions = {f"T{i}": {"quantity": 10, "entry_price": 100.0} for i in range(10)}
    prices = {f"T{i}": 100.0 for i in range(10)}

    full = _Portfolio(positions=positions)
    _manager(full)._calculate_position_size("T0", 100.0, 0.7, current_prices=prices)
    with_all = full.get_total_value(full.seen_prices)

    alone = _Portfolio(positions=positions)
    _manager(alone)._calculate_position_size("T0", 100.0, 0.7, current_prices=None)
    with_one = alone.get_total_value(alone.seen_prices)

    assert with_all > with_one


def test_the_ticker_being_sized_is_priced_even_without_a_map():
    portfolio = _Portfolio(positions={"T0": {"quantity": 10, "entry_price": 100.0}})
    _manager(portfolio)._calculate_position_size("T0", 123.0, 0.7)
    assert portfolio.seen_prices == {"T0": 123.0}


def test_a_missing_price_is_reported_not_swallowed(caplog):
    positions = {"T0": {"quantity": 10, "entry_price": 100.0},
                 "T1": {"quantity": 10, "entry_price": 100.0}}
    portfolio = _Portfolio(positions=positions)
    manager = _manager(portfolio)
    with caplog.at_level(logging.WARNING, logger="sizing-test"):
        manager._calculate_position_size("T0", 100.0, 0.7, current_prices={"T0": 100.0})
    assert any("count as zero in equity" in record.getMessage() for record in caplog.records)


def test_cash_leaves_room_for_the_commission():
    """An order sized to the whole balance is refused, not trimmed."""
    manager = _manager(_Portfolio(balance=10_000.0))
    rate = manager._estimated_cost_rate()
    assert rate > 0

    price = 100.0
    without = 10_000.0 / price
    with_costs = 10_000.0 / (price * (1 + rate))
    assert with_costs < without


def test_the_cost_rate_prefers_the_portfolio_s_own_model():
    """Sizing and execution must not disagree about what a trade costs."""
    portfolio = _Portfolio()
    portfolio.transaction_cost_model = type("M", (), {"total_cost_pct": 0.0123})()
    assert _manager(portfolio)._estimated_cost_rate() == pytest.approx(0.0123)


def test_a_portfolio_without_a_cost_model_still_reserves_something():
    assert _manager(_Portfolio())._estimated_cost_rate() > 0
