"""The phase was cheap to compute and expensive to announce.

`AdvancedAnalyticsEnricher` calls `MarketPhaseAnalyzer.analyze` once per row.
Measured on 2026-08-28:

  * that one `logger.info` produced **156,372 of the run log's 169,622 lines**
    -- 92% of a 19 MB file -- and cost 94% of the analyzer's 449 microseconds
    per row, while parsing and evaluating the rules cost 27;
  * the enricher handed over a one-row slice of all 460 columns, and the
    analyzer then built a set of those 460 names to check what was missing,
    per row. A wide slice is 687 microseconds against 156 for a narrow one.

Together that was 625 seconds, 17% of the frame, for a feature whose own
leading-feature verdict was "sign flipped out of sample".

Neither fix changes a value, so these tests hold the values fixed and pin the
two costs -- the log level and the width of what is passed in -- because a
later change could undo either without altering a single number.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.analytics.context.market_phase_analyzer import MarketPhaseAnalyzer

CONFIG = {
    "indicators": {"vol": "VOLATILITY_20", "regime": "MARKET_REGIME"},
    "rules": [
        {"condition": "vol > 0.02 and regime == 'RANGING'", "phase": "volatile_bull"},
        {"condition": "vol <= 0.02", "phase": "calm_bull"},
    ],
}


@pytest.fixture
def analyzer() -> MarketPhaseAnalyzer:
    return MarketPhaseAnalyzer(CONFIG)


def _row(vol: float, regime: str) -> pd.DataFrame:
    return pd.DataFrame({"VOLATILITY_20": [vol], "MARKET_REGIME": [regime]})


def test_the_phases_are_unchanged(analyzer):
    """The whole point: cheaper to run, identical answers."""
    assert analyzer.analyze({"market_data": _row(0.03, "RANGING")})["market_phase"] \
        == "volatile_bull"
    assert analyzer.analyze({"market_data": _row(0.01, "RANGING")})["market_phase"] \
        == "calm_bull"
    assert analyzer.analyze({"market_data": _row(0.03, "TRENDING_UP")})["market_phase"] \
        == "unknown"


def test_it_does_not_announce_itself_once_per_row(analyzer, caplog):
    """156,372 INFO lines for 156,372 rows, and the value is in the column."""
    with caplog.at_level(logging.INFO):
        analyzer.analyze({"market_data": _row(0.03, "RANGING")})

    said = [r for r in caplog.records if "market phase" in r.message.lower()
            and r.levelno >= logging.INFO]
    assert not said, f"still logging at INFO per row: {[r.message for r in said]}"


def test_it_is_still_available_at_debug(analyzer, caplog):
    """Silenced, not deleted -- someone debugging one row still wants it."""
    with caplog.at_level(logging.DEBUG):
        analyzer.analyze({"market_data": _row(0.03, "RANGING")})

    assert any("market phase" in r.message.lower() for r in caplog.records)


def test_extra_columns_do_not_change_the_answer(analyzer):
    """The narrowing is only valid if width never mattered.

    The enricher now passes the indicator columns alone. If the analyzer ever
    reads anything else, these two must diverge -- and this is where that
    would show.
    """
    narrow = _row(0.03, "RANGING")
    wide = narrow.copy()
    for index in range(50):
        wide[f"unrelated_{index}"] = 1.0

    assert analyzer.analyze({"market_data": narrow}) \
        == analyzer.analyze({"market_data": wide})


def test_a_missing_indicator_is_reported_not_guessed(analyzer, caplog):
    """Narrowing must not turn an absent indicator into a silent default.

    The analyzer already refuses rather than guesses: it names the missing
    column at WARNING and returns its own marker instead of a phase. This test
    pins that, because narrowing the input is only safe while an absent
    indicator stays loud.
    """
    with caplog.at_level(logging.WARNING):
        result = analyzer.analyze(
            {"market_data": pd.DataFrame({"VOLATILITY_20": [0.03]})}
        )

    assert result["market_phase"] not in {"calm_bull", "volatile_bull"}
    assert any("MARKET_REGIME" in record.message for record in caplog.records), (
        "an indicator went missing without a word"
    )
