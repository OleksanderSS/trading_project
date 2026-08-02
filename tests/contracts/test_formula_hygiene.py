"""The arithmetic mistakes this audit kept finding, counted and held.

Every pattern here comes from a defect that actually shipped. This is a
ratchet and an inventory, NOT a defect list: a hardcoded sqrt(252) is right
for daily bars, and np.std without ddof is fine for descriptive statistics.
The point is that the set is enumerated and cannot grow unnoticed, and that
the next hand-audit of computation modules has a list to work from rather
than a memory to rely on.

  ANNUALISATION    a periods-per-year constant outside the metrics library.
                   This project stores 15m, 60m and 1d bars, so sqrt(252) is
                   correct for one of them. DiaryEngine understated intraday
                   Sharpe 2.6x and 5.1x that way (d151cf7e).
  POPULATION_STD   np.std with no ddof. The default is the population
                   deviation; Sharpe and Sortino want the sample one.
  RIVAL_METRIC     a function named after a metric FinancialMetricsLibrary
                   already owns. FIVE Sharpe implementations existed: three
                   consolidated by an earlier audit, a fourth in
                   diary_engine, a fifth in arena_battle computing
                   mean(predictions)/std(predictions), which let a constant
                   predictor win the model tournament 0.958 to 0.698
                   (8efca119).
  SIGNED_RATIO     comparing against `x * factor` as a "better by N%" test.
                   That inverts below zero: promotion used
                   `challenger > champion * 1.15`, and -2.0 * 1.15 is -2.3,
                   so a WORSE challenger cleared the bar (7c4bd621).
"""
from __future__ import annotations

import pytest

from tests.contracts._formula_scan import by_kind, scan

# Measured 2026-08-02. Lower these as findings are resolved; never raise.
CEILINGS = {
    "ANNUALISATION": 17,
    "POPULATION_STD": 43,
    "RIVAL_METRIC": 22,
    "SIGNED_RATIO": 12,
}


@pytest.fixture(scope="module")
def findings():
    return by_kind(scan())


@pytest.mark.parametrize("kind", sorted(CEILINGS))
def test_formula_shapes_do_not_spread(findings, kind):
    found = findings.get(kind, [])

    assert len(found) <= CEILINGS[kind], (
        f"{kind} rose from {CEILINGS[kind]} to {len(found)}.\n"
        + "\n".join(f"  {finding}" for finding in found[:20])
    )


def test_the_ceilings_are_kept_honest(findings):
    """A count well below its ceiling should be re-pinned, or the ratchet
    stops ratcheting."""
    slack = {
        kind: CEILINGS[kind] - len(findings.get(kind, []))
        for kind in CEILINGS
    }
    stale = {kind: gap for kind, gap in slack.items() if gap > 8}

    assert not stale, f"lower these ceilings; they now sit this far above: {stale}"


def test_the_canonical_library_is_exempt_from_its_own_rules(findings):
    """FinancialMetricsLibrary is where these formulas are supposed to live;
    flagging it would train people to ignore the scanner."""
    canonical = [
        finding for finding in findings.get("RIVAL_METRIC", [])
        if "financial_metrics_library" in finding.module
    ]

    assert not canonical


def test_the_scanner_still_recognises_the_defects_it_was_built_for():
    """A scanner that silently matches nothing is worse than none."""
    import ast

    from tests.contracts._formula_scan import _Scanner

    sample = (
        "import numpy as np\n"
        "def calculate_sharpe(returns):\n"
        "    return np.mean(returns) / np.std(returns) * np.sqrt(252)\n"
        "def promote(a, b):\n"
        "    return a > b * 1.15\n"
    )
    scanner = _Scanner("sample.py")
    scanner.visit(ast.parse(sample))
    kinds = {finding.kind for finding in scanner.findings}

    assert kinds == {"ANNUALISATION", "POPULATION_STD", "RIVAL_METRIC", "SIGNED_RATIO"}


def test_clean_arithmetic_is_not_flagged():
    """Delegating to the library, and a sample std, must stay silent."""
    import ast

    from tests.contracts._formula_scan import _Scanner

    sample = (
        "import numpy as np\n"
        "def score(returns, periods):\n"
        "    spread = np.std(returns, ddof=1)\n"
        "    return FinancialMetricsLibrary.calculate_sharpe_ratio(returns)\n"
    )
    scanner = _Scanner("sample.py")
    scanner.visit(ast.parse(sample))

    assert scanner.findings == []
