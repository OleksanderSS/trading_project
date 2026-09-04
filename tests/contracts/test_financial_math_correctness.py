"""The Sharpe and VaR functions the pipeline actually calls, on the inputs that
break them.

WHY THIS FILE WAS REWRITTEN ON 2026-09-04. It existed, it ran, and it proved
nothing: both real tests skipped on every run since they were written.

    pytest.skip('No known Sharpe function importable')
    pytest.skip('VarCalculator not importable: No module named
                 "src.risk_management"')

The first looked for `src.risk.metrics.calculate_sharpe_ratio` and
`src.analytics.calculators.risk_metrics_calculator.calculate_sharpe_ratio`.
Neither exists. The second looked for `src.risk_management.var_calculator` and
a class named `VarCalculator`; the module is `src.risk.analyzers.var_calculator`
and the class is `VaRCalculator`, so it was wrong twice over.

A skip is not a pass. The two tests guarding the arithmetic every Sharpe claim
in CLAIMS rests on had been green-by-absence for as long as they existed, and
the contract suite reported them as 4 skips without anyone reading the reasons.
They were found by reading the skip reasons while making that suite blocking in
CI.

WHAT THE MEASUREMENT FOUND, once they could run: the math is sound. Both live
Sharpe implementations return NaN -- not infinity, not a confident 0.0 -- for
constant, empty and all-zero returns, and they agree with each other and with
the textbook formula to floating-point equality on a normal series. The defect
was in the test, not the code.

WHAT IT ALSO FOUND. `VaRCalculator.calculate`, the method its own docstring
calls the main entry point, returned **0.0** for empty input -- "this position
cannot lose money" -- while the inner `calculate_var_historical` correctly
returned NaN with `status: insufficient_data`. It had no live caller, so this
is a trap rather than a bug, and it is fixed here rather than left for whoever
calls the main entry point next.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.calculators.risk_reward_calculator import RiskRewardCalculator
from src.metrics.financial.financial_metrics_library import FinancialMetricsLibrary
from src.risk.analyzers.var_calculator import VaRCalculator

#: Every live implementation of the same quantity. Listed rather than picked,
#: because duplication is where this codebase's defects come from: a fix lands
#: in one copy, the other lives on, and everything looks repaired.
SHARPE_FUNCTIONS = (
    ("metrics.financial", FinancialMetricsLibrary.calculate_sharpe_ratio),
    ("analytics.risk_reward", RiskRewardCalculator.calculate_sharpe_ratio),
)

DEGENERATE = {
    "constant returns": pd.Series([0.01, 0.01, 0.01, 0.01]),
    "empty": pd.Series([], dtype=float),
    "all zeros": pd.Series([0.0, 0.0, 0.0, 0.0]),
    "single value": pd.Series([0.01]),
}


@pytest.mark.parametrize("label,function", SHARPE_FUNCTIONS)
@pytest.mark.parametrize("case", sorted(DEGENERATE))
def test_sharpe_is_undefined_rather_than_infinite_or_confident(label, function, case):
    """Zero variance means the ratio is undefined. Returning inf makes a flat
    series the best strategy ever measured; returning 0.0 makes it an ordinary
    one. Only NaN says what is true."""
    value = float(function(DEGENERATE[case]))
    assert not math.isinf(value), (
        f"{label} returned infinity for {case}; a flat series would rank above "
        f"every real strategy"
    )
    assert math.isnan(value), (
        f"{label} returned {value} for {case}, which reads as a measured "
        f"result rather than an undefined one"
    )


def test_the_two_sharpe_implementations_agree():
    """Two functions computing one quantity must not drift. If this fails, one
    of them was fixed and the other was not -- which is how the calendar and
    news_impact defects survived their own repairs."""
    returns = pd.Series(np.random.default_rng(7).normal(0.0004, 0.01, 2520))
    values = [float(function(returns)) for _, function in SHARPE_FUNCTIONS]
    assert values[0] == pytest.approx(values[1], rel=1e-9), (
        f"the live Sharpe implementations disagree: "
        f"{dict(zip([n for n, _ in SHARPE_FUNCTIONS], values))}"
    )


def test_sharpe_matches_the_textbook_formula():
    """Agreement between two copies of the same mistake is not correctness."""
    returns = pd.Series(np.random.default_rng(7).normal(0.0004, 0.01, 2520))
    expected = returns.mean() / returns.std() * math.sqrt(252)
    for label, function in SHARPE_FUNCTIONS:
        assert float(function(returns)) == pytest.approx(expected, rel=1e-9), (
            f"{label} does not annualise mean/std over 252 days"
        )


def test_var_on_no_data_is_unknown_and_not_zero_risk():
    """A VaR of zero means the position cannot lose money. A VaR of NaN means
    nobody measured. These must never share a value."""
    calculator = VaRCalculator()

    detailed = calculator.calculate_var_historical([])
    assert detailed["status"] == "insufficient_data"
    assert math.isnan(detailed["var"])

    # The method its own docstring calls the main entry point. It returned 0.0
    # here until 2026-09-04, with no live caller to notice.
    assert math.isnan(float(calculator.calculate([])))
    assert math.isnan(float(calculator.calculate(None)))


def test_var_reports_a_real_loss_when_there_is_data():
    """The mirror of the test above: a check that only ever says "unknown" is
    as useless as one that only ever says "zero"."""
    calculator = VaRCalculator()
    result = calculator.calculate_var_historical([0.02, -0.04, 0.01, -0.02])
    assert result["status"] == "ok"
    assert result["var"] > 0
    assert float(calculator.calculate([0.02, -0.04, 0.01, -0.02])) > 0


def test_drawdown_is_negative_by_this_project_convention():
    equity = pd.Series([100.0, 120.0, 90.0, 110.0])
    drawdown = (equity - equity.cummax()) / equity.cummax()
    assert drawdown.min() < 0
    assert abs(drawdown.min()) == pytest.approx(0.25)


def test_this_file_imports_what_it_tests_at_module_level():
    """The defect this rewrite exists for: the old version resolved its targets
    with importlib inside the test body and called `pytest.skip` when they were
    not found, so two wrong module paths read as four harmless skips for as
    long as the file existed. A module-level import fails the collection step
    instead, which CI already blocks on."""
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(__file__).read_text(encoding="utf-8"))

    # Parsed, not grepped: the first version of this check searched the file's
    # own text and matched the word "importlib" inside this very docstring.
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in getattr(node, "names", [])
        if isinstance(node, ast.Import) or node.module
    }
    assert "importlib" not in imported, (
        "targets resolved dynamically again; a missing module must fail "
        "collection, not turn into a skip"
    )

    skips = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "skip"
    ]
    assert not skips, (
        "a skip here means the arithmetic every Sharpe claim rests on is "
        "unverified, and it will be reported as a passing suite"
    )
