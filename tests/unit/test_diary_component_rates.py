"""A driver that appears in most trades tops a count table by being common.

_analyze_fingerprint_components counts raw occurrences, so the "component
vulnerabilities" ranking answered "which driver value shows up most in
losses" rather than "which driver value loses most often". Those differ
whenever the driver values are unevenly distributed, which they always are.

_component_outcome_rates supplies the denominator: of every RESOLVED trade
whose context carried driver i at value v, what share ended in the outcome
being asked about.
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.meta_learning.memory.diary_engine import DecisionOutcome, DiaryEngine


class _StubDataManager:
    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def fetch_all(self, query, params=None):
        self.calls.append((query, list(params or [])))
        return self._rows


def _engine(rows):
    instance = object.__new__(DiaryEngine)
    instance.logger = logging.getLogger("diary-rates-test")
    instance.data_manager = _StubDataManager(rows)
    return instance


def test_a_common_driver_does_not_outrank_a_dangerous_one():
    """Driver 0 value '1' loses 10 of 100. Driver 1 value '1' loses 8 of 10.

    On raw counts the common one wins 10 > 8 and gets blamed. On rates the
    dangerous one wins 0.80 > 0.10, which is the true statement.
    """
    engine = _engine([
        {"context_fingerprint": "1|0", "total_count": 100, "hit_count": 10},
        {"context_fingerprint": "0|1", "total_count": 10, "hit_count": 8},
    ])

    rates = engine._component_outcome_rates("champion", DecisionOutcome.UNPROFITABLE.value)

    assert rates[0]["1"]["rate"] == pytest.approx(0.10)
    assert rates[1]["1"]["rate"] == pytest.approx(0.80)
    assert rates[0]["1"]["count"] > rates[1]["1"]["count"], (
        "the count ranking must genuinely disagree, or this proves nothing"
    )


def test_counts_and_totals_are_reported_alongside_the_rate():
    """A rate of 1.00 from a single trade is not the same as from a hundred."""
    engine = _engine([
        {"context_fingerprint": "1", "total_count": 1, "hit_count": 1},
    ])

    rates = engine._component_outcome_rates("champion", DecisionOutcome.UNPROFITABLE.value)

    assert rates[0]["1"] == {"rate": 1.0, "count": 1.0, "total": 1.0}


def test_unresolved_trades_are_kept_out_of_the_denominator():
    """Every training row is PENDING/NEUTRAL; they never had a chance to lose."""
    engine = _engine([{"context_fingerprint": "1", "total_count": 4, "hit_count": 1}])

    engine._component_outcome_rates("champion", DecisionOutcome.UNPROFITABLE.value)

    query, params = engine.data_manager.calls[0]
    assert "outcome IN (" in query
    assert params == [
        DecisionOutcome.UNPROFITABLE.value,
        "champion",
        DecisionOutcome.PROFITABLE.value,
        DecisionOutcome.UNPROFITABLE.value,
        DecisionOutcome.BREAK_EVEN.value,
    ]
    assert DecisionOutcome.PENDING.value not in params
    assert DecisionOutcome.NOT_APPLICABLE.value not in params


def test_the_rate_query_is_not_limited_to_the_worst_contexts():
    """Numerator from the top ten, denominator from all, inflates every rate."""
    engine = _engine([{"context_fingerprint": "1", "total_count": 4, "hit_count": 1}])

    engine._component_outcome_rates("champion", DecisionOutcome.UNPROFITABLE.value)

    query, _ = engine.data_manager.calls[0]
    assert "LIMIT" not in query.upper()
    assert "HAVING" not in query.upper()


def test_undecodable_fingerprints_yield_nothing_rather_than_zeros():
    hashed = "67a9e31d5fb39bf5212a44f8ff79c9e423d088acc1c18f695f51c59f38091385"
    engine = _engine([{"context_fingerprint": hashed, "total_count": 9, "hit_count": 3}])

    assert engine._component_outcome_rates("champion", DecisionOutcome.UNPROFITABLE.value) == {}


def test_no_history_yields_nothing():
    assert _engine([])._component_outcome_rates(
        "champion", DecisionOutcome.UNPROFITABLE.value
    ) == {}


def test_a_driver_seen_only_in_unresolved_trades_is_skipped_not_divided_by_zero():
    engine = _engine([{"context_fingerprint": "1|0", "total_count": 0, "hit_count": 0}])

    assert engine._component_outcome_rates(
        "champion", DecisionOutcome.UNPROFITABLE.value
    ) == {}


def test_rates_accumulate_across_fingerprints_sharing_a_driver_value():
    engine = _engine([
        {"context_fingerprint": "1|0", "total_count": 30, "hit_count": 3},
        {"context_fingerprint": "1|1", "total_count": 70, "hit_count": 7},
    ])

    rates = engine._component_outcome_rates("champion", DecisionOutcome.UNPROFITABLE.value)

    assert rates[0]["1"] == {"rate": pytest.approx(0.10), "count": 10.0, "total": 100.0}
