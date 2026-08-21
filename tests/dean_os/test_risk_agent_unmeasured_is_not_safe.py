"""The risk gate must not read "nothing was measured" as "no risk".

The only live producer of MarketContext.positions hands RiskAgent two
placeholders -- ``positions={"CASH": 1.0}`` and ``returns={"SPY": 0.0}`` --
and the gate used to answer with a drawdown of 0%, a VaR-95 of 0%, and
"Risk checks passed" at 0.85 confidence. A 95th-percentile tail computed over
a single observation is not a measurement, and the zero it produces is
indistinguishable from a measured zero except that it opens the gate.

Same family as the regime ensemble returning 0.0 for "no model matched",
except the neutral number fails permissive here instead of restrictive.
"""

import asyncio

import numpy as np
import pytest

from dean_os.agents.risk import RiskAgent
from dean_os.schemas import MarketContext

PLACEHOLDER = {"returns": {"SPY": 0.0}, "positions": {"CASH": 1.0}}


def _run(**kwargs):
    return asyncio.run(RiskAgent().run(MarketContext(**kwargs)))


def test_the_orchestrators_placeholder_does_not_pass_the_gate():
    report = _run(phase="post_pipeline", **PLACEHOLDER)
    assert report.verdict != "clear"
    assert "unmeasurable" in report.reasons[0]
    assert report.metrics_snapshot["max_drawdown"] is None
    assert report.metrics_snapshot["daily_var_95"] is None


def test_unmeasurable_risk_is_a_hard_block_before_trading():
    """Post-pipeline it is a caution; before money moves it is a block."""
    assert _run(phase="pre_trade", **PLACEHOLDER).verdict == "blocked"
    assert _run(phase="post_pipeline", **PLACEHOLDER).verdict == "caution"


def test_a_short_history_is_reported_as_short_not_as_calm():
    report = _run(phase="post_pipeline", positions={"AAPL": 0.5},
                  returns=list(np.zeros(19)))
    snapshot = report.metrics_snapshot
    assert snapshot["returns_measurable"] is False
    assert snapshot["sample_count"] == 19
    # Exposure is measurable from positions alone, so it is still reported.
    assert snapshot["gross_exposure"] == pytest.approx(0.5)


def test_enough_history_still_measures_and_still_clears():
    calm = list(np.random.default_rng(0).normal(0.0004, 0.008, 300))
    report = _run(phase="post_pipeline", positions={"AAPL": 0.5}, returns=calm)
    assert report.verdict == "clear"
    assert report.metrics_snapshot["returns_measurable"] is True
    assert report.metrics_snapshot["daily_var_95"] > 0.0


def test_enough_history_still_catches_a_real_drawdown():
    losing = list(np.random.default_rng(1).normal(-0.004, 0.03, 300))
    report = _run(phase="post_pipeline", positions={"AAPL": 0.5}, returns=losing)
    assert report.verdict == "blocked"
    assert "Drawdown" in report.reasons[0]


def test_sample_count_is_in_the_evidence():
    """Whoever reads the report can see what the numbers were computed from."""
    report = _run(phase="post_pipeline", **PLACEHOLDER)
    counts = {item.key: item.value for item in report.evidence}
    assert counts["return_sample_count"] == 1
