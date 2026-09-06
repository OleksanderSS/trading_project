"""The limits written in the config must bind, and must bind at the config's number.

Until 2026-09-06 `VirtualPortfolio.buy_stock` refused an order for exactly one
reason -- not enough cash -- while `risk_management.yaml` declared a 10%
position limit, a 3% daily loss limit, a 15% drawdown limit and 2x leverage.
The config described a risk system that did not exist and read as configured
(REGISTER #101, #288).

Worse, the class asked for `max_position_size` while the config declares
`max_position_size_pct`, so even the value it did read was its own default. The
two are both 0.10, which is why the mismatch survived.

THE TEST THAT MATTERS IS THE THIRD ONE. Enforcing a limit is easy to write and
easy to write wrongly: a limit enforced at a hardcoded number looks identical
from the outside to a limit enforced at the configured one, until the day
somebody changes the config and nothing happens. So the third test moves the
number and requires the behaviour to move with it.

These run against paper trading, which is the point: virtual money is what a
risk control should be exercised on before real money arrives.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trading.virtual_portfolio import VirtualPortfolio  # noqa: E402


@pytest.fixture
def portfolio(tmp_path, monkeypatch):
    """A fresh $100,000 book that writes nowhere the project reads."""
    book = VirtualPortfolio(initial_balance=100_000.0,
                            portfolio_name="contract_limits")
    book.portfolio_file = tmp_path / "contract_limits_portfolio.json"
    book.positions = {}
    book.transactions = []
    book.performance_history = []
    book.current_balance = 100_000.0
    return book


def _order(ticker: str, quantity: int, price: float) -> dict:
    return {"ticker": ticker, "quantity": quantity, "price": price,
            "daily_volume": 10_000_000, "volatility": 0.02}


def test_an_order_inside_the_limit_goes_through(portfolio):
    """$5,000 of a $100,000 book is 5%, under the declared 10%."""
    result = portfolio.buy_stock(_order("AAA", 50, 100.0))
    assert result["success"], result.get("error")


def test_a_single_position_over_the_declared_share_is_refused(portfolio):
    """$15,000 of a $100,000 book is 15%, over the declared 10%."""
    result = portfolio.buy_stock(_order("AAA", 150, 100.0))
    assert not result["success"]
    assert "Position limit" in result["error"], result["error"]
    assert "AAA" not in portfolio.positions, (
        "the order was refused and the position was created anyway")
    assert portfolio.current_balance == 100_000.0, (
        "a refused order still moved the cash")


def test_the_limit_binds_at_the_CONFIGURED_number_not_a_hardcoded_one(portfolio):
    """Move the number, and the refusal must move with it.

    This is the test the defect would have survived: a limit enforced at a
    constant behaves identically to one enforced from config until the config
    changes. Here 15% is refused at the declared 10% and accepted at 20%, and
    nothing but the setting differs between the two runs.
    """
    over = portfolio.buy_stock(_order("AAA", 150, 100.0))
    assert not over["success"], "15% should exceed the declared 10%"

    portfolio.max_position_size = 0.20
    portfolio.positions = {}
    portfolio.current_balance = 100_000.0
    now_fine = portfolio.buy_stock(_order("AAA", 150, 100.0))
    assert now_fine["success"], (
        "the same order was refused at a 10% limit and refused again at 20%, "
        "so the refusal is not reading the limit at all: " + str(now_fine))


def test_positions_accumulate_toward_the_same_limit(portfolio):
    """Three orders of 4% each: the third crosses 10% and must be refused.

    A limit checked only against the incoming order lets a book reach any size
    one slice at a time, which is the shape that makes position limits
    pointless.
    """
    assert portfolio.buy_stock(_order("AAA", 40, 100.0))["success"]
    assert portfolio.buy_stock(_order("AAA", 40, 100.0))["success"]
    third = portfolio.buy_stock(_order("AAA", 40, 100.0))
    assert not third["success"], (
        "three 4% slices of the same name reached 12% without a refusal")
    assert "Position limit" in third["error"]


def test_total_invested_is_capped_across_different_names(portfolio):
    """Eight different names at 5% each: the total limit must stop them.

    Each order passes the per-position limit on its own. Only a check on the
    whole book can see the seventh.
    """
    successes = 0
    for index in range(8):
        if portfolio.buy_stock(_order(f"N{index}", 50, 100.0))["success"]:
            successes += 1
    assert successes < 8, (
        "eight 5% positions were accepted, so nothing caps total exposure and "
        "the declared 30% total risk limit is decoration")
    assert successes >= 5, (
        f"only {successes} of eight 5% positions were accepted; a 30% total "
        "limit should allow six")
