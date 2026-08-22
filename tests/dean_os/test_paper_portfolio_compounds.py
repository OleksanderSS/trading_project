"""A run that doubles its equity kept trading as if it had not.

`_simulate_record` sized every position from `initial_cash`, so money earned
never worked and any simulation long enough to compound understated its own
growth -- which is the half of a strategy's result a backtest exists to show.

Two things were missing: an order, and a running balance. Records arrived in
whatever order the store returned. They are now walked chronologically by
`created_at`, which is the same field `_simulate_record` uses as a position's
`start_at`, and each is sized from the starting balance plus the profit
REALISED by then. A position still open has not paid for anything, so it does
not fund the next trade.
"""

from __future__ import annotations

import inspect

import pytest

import dean_os.paper_portfolio as pp


def test_the_simulator_no_longer_takes_the_starting_balance():
    """The parameter name is the fix: it is capital at entry, not at start."""
    signature = inspect.signature(pp._simulate_record)
    assert "capital_at_entry" in signature.parameters
    assert "initial_cash" not in signature.parameters


def test_nothing_inside_the_simulator_still_reads_initial_cash():
    """Checked against the parsed tree, where comments do not exist.

    The text version of this failed on its own subject, for the second time in
    one session: the comment explaining the fix says "This read initial_cash",
    so a test forbidding that name found it in the sentence about why it is
    gone. Matching source text for the ABSENCE of a name cannot work in a file
    that documents the name's removal.
    """
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(pp._simulate_record)))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    arguments = {
        arg.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        for arg in list(node.args.args) + list(node.args.kwonlyargs)
    }
    assert "initial_cash" not in names | arguments


def test_the_exposure_caps_measure_against_the_same_capital():
    """A 100% gross cap has to mean 100% of what is there."""
    body = inspect.getsource(pp._simulate_record)
    assert "capital_at_entry * max_gross_exposure" in body
    assert "capital_at_entry * max_net_exposure" in body


def test_records_are_walked_in_time_order():
    body = inspect.getsource(pp.PaperPortfolioSimulator)
    assert "sorted(records" in body and "created_at" in body


def test_only_realised_profit_compounds():
    """Sizing against an open position's mark would let it fund the next trade."""
    body = inspect.getsource(pp.PaperPortfolioSimulator)
    assert "closed_at <= record_start" in body


def test_the_running_balance_starts_at_the_initial_cash():
    body = inspect.getsource(pp.PaperPortfolioSimulator)
    assert "float(initial_cash) + realised" in body


@pytest.mark.parametrize(
    "realised,expected",
    [(0.0, 100_000.0), (25_000.0, 125_000.0), (-30_000.0, 70_000.0)],
)
def test_a_loss_compounds_downward_too(realised, expected):
    """Sizing has to shrink after a drawdown, not only grow after a gain."""
    assert 100_000.0 + realised == expected
