"""Promotion decides which model becomes champion, so the comparison must
not invert.

The rule was

    agent_sharpe > champion_sharpe * 1.15

which inverts the moment the champion's Sharpe is negative: -2.0 * 1.15 is
-2.3, so a challenger at -2.25 -- WORSE than the champion -- cleared the bar
and was recommended for promotion. Verified before the fix, all three cases:

    champion +1.0, challenger +1.2            -> promote   (correct)
    champion -2.0, challenger -2.25 (worse)   -> promote   (inverted)
    champion -1.0, challenger with no data    -> promote   (inverted)

The third came from `.get('sharpe_ratio', 0)`: an agent whose metrics are
{"error": "No data"} scored 0, which beats any negative champion, so "no
evidence" outranked "measured and losing".

A relative margin only means anything above zero. It is now combined with an
absolute one, and the stricter of the two applies.
"""
from __future__ import annotations

import pytest

from src.meta_learning.memory.diary_engine import DiaryEngine


@pytest.fixture()
def engine():
    return object.__new__(DiaryEngine)


def _promotes(engine, champion, challenger):
    results = {"champ": champion, "x": challenger}
    return bool(engine._check_promotion_criteria(["champ", "x"], "champ", results))


def test_a_clearly_better_challenger_is_promoted(engine):
    assert _promotes(engine, {"sharpe_ratio": 1.0}, {"sharpe_ratio": 1.5})


def test_a_marginally_better_challenger_is_not(engine):
    assert not _promotes(engine, {"sharpe_ratio": 1.0}, {"sharpe_ratio": 1.05})


def test_a_worse_challenger_is_never_promoted_when_the_champion_is_negative(engine):
    """The inversion: -2.0 * 1.15 = -2.3, so -2.25 used to qualify."""
    assert not _promotes(engine, {"sharpe_ratio": -2.0}, {"sharpe_ratio": -2.25})


def test_a_genuinely_better_challenger_is_promoted_when_the_champion_is_negative(engine):
    assert _promotes(engine, {"sharpe_ratio": -2.0}, {"sharpe_ratio": -1.0})


def test_a_barely_better_challenger_is_not_promoted_below_zero(engine):
    """Still requires the absolute margin, not merely 'less bad'."""
    assert not _promotes(engine, {"sharpe_ratio": -2.0}, {"sharpe_ratio": -1.95})


def test_an_agent_with_no_data_never_outranks_a_measured_one(engine):
    assert not _promotes(engine, {"sharpe_ratio": -1.0}, {"error": "No data"})


def test_a_champion_with_no_data_produces_no_recommendations(engine):
    assert not _promotes(engine, {"error": "No data"}, {"sharpe_ratio": 2.0})


@pytest.mark.parametrize("champion,challenger", [
    (0.0, 0.10),
    (0.0, 0.20),
    (-0.5, -0.40),
    (-0.5, -0.30),
])
def test_the_margin_is_monotonic_around_zero(engine, champion, challenger):
    """Promotion must depend on the size of the improvement, never on which
    side of zero the champion happens to sit."""
    promoted = _promotes(
        engine, {"sharpe_ratio": champion}, {"sharpe_ratio": challenger}
    )
    assert promoted == (challenger - champion > DiaryEngine._MIN_PROMOTION_MARGIN)


def test_the_reason_states_the_numbers(engine):
    recommendations = engine._check_promotion_criteria(
        ["champ", "x"], "champ",
        {"champ": {"sharpe_ratio": 1.0}, "x": {"sharpe_ratio": 2.0}},
    )

    assert "2.000" in recommendations[0]["reason"]
    assert "1.000" in recommendations[0]["reason"]


def test_history_is_ordered_so_tail_means_recent():
    """suggest_threshold_adjustments takes .tail(20) and calls it recent."""
    import inspect

    source = inspect.getsource(DiaryEngine.get_history_by_agent)
    assert "ORDER BY decision_timestamp" in source
