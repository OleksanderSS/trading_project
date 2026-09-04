"""A holdout equity curve must say how one-sided it is, and what owning
everything would have returned over the same bars.

WHY THIS EXISTS. `build_holdout_equity` turns model predictions into positions
with `sign(prediction)` and reports the curve. That says nothing about whether
the book is a strategy or a market exposure: a model predicting up for almost
every name produces +1 almost everywhere, and the curve is then the market
wearing a champion's name.

This is not hypothetical and not a code-review worry. On 2026-09-04 the net
test reported SEVEN features clearing a Bonferroni correction at net Sharpe
1.016, which was the best result the project had produced. The constant
opponent -- hold every name, same clock, same friction -- scored 1.018. All
seven were that opponent to three decimals, because a column that is 98.9% one
value ranks as ties and `sign(rank - 0.5)` sends every name the same way
(CLAIMS R28). The defect was not in the data. It was in a script that compared
against zero instead of against the market, and this production path had the
same gap.

WHAT IS AND IS NOT ENFORCED. The strategy definition is unchanged: positions
are not demeaned, because making the book dollar-neutral is a decision about
what the pipeline trades and belongs to the owner. What is pinned here is that
the two numbers making the substitution visible are always reported, and that
a heavily one-sided book says so at WARNING rather than in silence.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.pipeline.stages.evaluation.holdout_equity import (
    ONE_SIDED_WARNING,
    build_holdout_equity,
)


def _predictions(signs, actuals, *, names=("AAA", "BBB")) -> pd.DataFrame:
    """One bar per date per name, with the prediction sign we want tested."""
    rows = []
    dates = pd.date_range("2024-01-01", periods=len(actuals), freq="D", tz="UTC")
    for day, moves in zip(dates, actuals):
        for name, move, side in zip(names, moves, signs):
            rows.append({
                "target": "target_return_1d",
                "context": f"CTX::{name}",
                "ticker": name,
                "datetime": day,
                "prediction": float(side),
                "actual": float(move),
            })
    return pd.DataFrame(rows)


def test_a_long_only_book_reports_full_exposure_and_matches_buy_everything():
    """The R28 shape: every position on one side, so the curve IS the market."""
    actuals = [(0.01, 0.02), (-0.03, 0.01), (0.02, -0.01), (0.00, 0.03)] * 10
    result = build_holdout_equity(_predictions((1, 1), actuals))

    assert result["status"] == "built"
    assert result["mean_position"] == pytest.approx(1.0)
    assert result["total_return"] == pytest.approx(
        result["constant_opponent_return"], abs=1e-12), (
        "a book that is long everything cannot differ from owning everything; "
        "if these diverge the opponent is being computed on other bars"
    )
    assert result["excess_over_constant"] == pytest.approx(0.0, abs=1e-12)


def test_a_short_only_book_is_reported_as_one_sided_too():
    """One-sidedness is about the absolute exposure, not the direction. A book
    short everything is equally not a strategy."""
    actuals = [(0.01, 0.02), (-0.03, 0.01), (0.02, -0.01)] * 10
    result = build_holdout_equity(_predictions((-1, -1), actuals))
    assert result["mean_position"] == pytest.approx(-1.0)
    assert abs(result["mean_position"]) >= ONE_SIDED_WARNING


def test_a_two_sided_book_is_neutral_and_its_excess_is_the_part_it_earned():
    actuals = [(0.01, 0.02), (-0.03, 0.01), (0.02, -0.01), (0.00, 0.03)] * 10
    result = build_holdout_equity(_predictions((1, -1), actuals))

    assert result["mean_position"] == pytest.approx(0.0)
    assert result["excess_over_constant"] == pytest.approx(
        result["total_return"] - result["constant_opponent_return"])


def test_the_opponent_is_reported_even_when_the_book_is_neutral():
    """Reporting it only on failure is how a check stops being read at all: the
    number has to be present in the ordinary case to mean anything in the bad
    one."""
    actuals = [(0.01, 0.02), (-0.03, 0.01)] * 15
    result = build_holdout_equity(_predictions((1, -1), actuals))
    for key in ("mean_position", "constant_opponent_return",
                "excess_over_constant", "total_return"):
        assert key in result, f"{key} vanished from the holdout result"


def test_a_one_sided_book_says_so_at_warning(caplog):
    """Silence is what let the seven candidates through for nine minutes."""
    actuals = [(0.01, 0.02), (-0.03, 0.01), (0.02, -0.01)] * 10
    with caplog.at_level(logging.WARNING):
        build_holdout_equity(_predictions((1, 1), actuals))
    said = " ".join(record.getMessage() for record in caplog.records)
    assert "one-sided" in said
    assert "Owning" in said or "owning" in said


def test_a_neutral_book_does_not_cry_wolf(caplog):
    """A check that fires on ordinary runs gets switched off -- `|| true` sat
    in ci.yml for six weeks for exactly that reason."""
    actuals = [(0.01, 0.02), (-0.03, 0.01), (0.02, -0.01)] * 10
    with caplog.at_level(logging.WARNING):
        build_holdout_equity(_predictions((1, -1), actuals))
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warnings, f"neutral book warned anyway: {warnings}"


def test_the_threshold_is_a_number_and_not_a_feeling():
    """Without a stated limit "the book is balanced" is true forever."""
    assert isinstance(ONE_SIDED_WARNING, float)
    assert 0.0 < ONE_SIDED_WARNING < 1.0


def test_the_opponent_uses_the_same_bars_as_the_curve():
    """The comparison is worthless if the two are measured over different
    periods -- which is the easiest way for this check to become decorative."""
    actuals = [(0.05, -0.05)] * 40
    result = build_holdout_equity(_predictions((1, -1), actuals))
    # Longs and shorts cancel in the opponent (it owns both) but not in the
    # book (it is long the first name and short the second).
    assert result["constant_opponent_return"] == pytest.approx(0.0, abs=1e-9)
    assert result["total_return"] > 0.0
    assert result["bar_count"] == 40
