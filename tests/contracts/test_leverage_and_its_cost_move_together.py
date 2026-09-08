"""Borrowing above 1x has a price, and this model does not charge one.

R35 ended by noting that margin financing cost "is not in the friction model at
all". Checked 2026-09-08 and that is true: `TransactionCostModel` returns
commission, spread, market_impact and slippage, and nothing else. There is a
`get_risk_free_rate` in the engine, but it feeds the Sharpe denominator, not the
cost of carrying a borrowed position.

That absence is CORRECT as the project stands. `max_leverage` is 0.51 -- below
one -- so the configured book never borrows and there is nothing to charge. The
financing arithmetic in R34 and R35 belongs to their hypothetical 2x/4x/6x
scenarios, not to the live model, and adding a term for money we do not borrow
would be inventing a cost.

What it leaves is a trap that fires silently. The day somebody raises
`max_leverage` above 1 -- which R34 discusses as a real option, since 4x turns a
4.18% volatility book into a market-risk one -- every backtest keeps reporting
costs that charge nothing for the borrowing, and the results get better for a
reason that is not real.

So this test is the pair, not the term: either leverage stays at or below 1, or
the cost model prices financing. It passes trivially today. It exists for the
day it does not.
"""
from __future__ import annotations

import pytest
import yaml

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RISK = PROJECT_ROOT / "src" / "config" / "risk_management.yaml"

#: The keys `calculate_execution_costs` returns. A financing charge would have
#: to appear as one of these, or the total is not a total.
FINANCING_NAMES = ("financing", "borrow", "margin", "carry", "funding")


def _declared_leverage() -> float:
    document = yaml.safe_load(RISK.read_text(encoding="utf-8"))
    node = document
    for key in ("strategy", "risk_management"):
        node = (node or {}).get(key, {})
    value = node.get("max_leverage")
    assert value is not None, (
        "max_leverage is not declared. It was kept deliberately, because "
        "EliteRiskMetrics falls back to 2.0 when it is missing -- removing the "
        "key re-creates the undeclared-number defect of #288.")
    return float(value)


def test_the_book_does_not_borrow_or_the_model_charges_for_it():
    """The pair. Breaking either half alone is the silent case."""
    from src.backtesting.advanced.advanced_engine import TransactionCostModel

    leverage = _declared_leverage()
    costs = TransactionCostModel({}).calculate_execution_costs(
        10_000.0, 242_855_016.0, 0.0202)
    prices_financing = any(
        any(name in key.lower() for name in FINANCING_NAMES) for key in costs)

    if leverage <= 1.0:
        assert not prices_financing, (
            f"max_leverage is {leverage} so nothing is borrowed, yet the cost "
            f"model returns {sorted(costs)} including a financing term. "
            "Charging for money that is not borrowed is inventing a cost.")
        return

    assert prices_financing, (
        f"max_leverage is {leverage}, so the book borrows "
        f"{leverage - 1:.2f}x its capital -- and calculate_execution_costs "
        f"returns {sorted(costs)}, which charges nothing for it. Every "
        "backtest under this leverage reports costs that are too low, and the "
        "results improve for a reason that is not real (R34, R35).")


def test_the_guard_actually_bites_when_leverage_goes_above_one(monkeypatch):
    """A green test proves nothing until it is shown failing on the real case.

    Today `max_leverage` is 0.51, so the assertion above takes its trivial
    branch and would stay green even if the pairing logic were broken. This
    raises the declared leverage and checks the other branch fires -- otherwise
    the guard sleeps through exactly the day it exists for.
    """
    import tests.contracts.test_leverage_and_its_cost_move_together as module

    monkeypatch.setattr(module, "_declared_leverage", lambda: 4.0)
    with pytest.raises(AssertionError, match="charges nothing for it"):
        module.test_the_book_does_not_borrow_or_the_model_charges_for_it()


def test_the_total_is_the_sum_of_its_parts():
    """A financing term added later must reach the total, not sit beside it."""
    from src.backtesting.advanced.advanced_engine import TransactionCostModel

    costs = TransactionCostModel({}).calculate_execution_costs(
        10_000.0, 242_855_016.0, 0.0202)
    parts = sum(value for key, value in costs.items()
                if key not in {"total", "total_pct"})
    assert costs["total"] == pytest.approx(parts), (
        f"the parts sum to {parts:.6f} and total says {costs['total']:.6f}. A "
        "component that does not reach the total is a cost nobody pays.")
