"""Significance was measured against a scale the score never reaches.

`significance_thresholds` were absolute -- 0.8 for high, 0.3 for medium --
while the impact score spans 0.058 to 0.102 in magnitude on the 2026-08-15
batch. Everything landed in "low", so `news_significance_level` arrived at the
models as the constant 0 on all three timeframes.

No absolute number fixes that for long. The impact score is a time-decayed sum
of weighted sentiment, so its scale follows how many articles were collected:
extending the news history, adding a source or changing the half-life all move
it. A threshold calibrated against today's batch is wrong after the next
collection.

So significance is relative -- above the 90th percentile of the magnitudes
seen up to and including this one is "high", above the 70th is "medium". The
window is EXPANDING, which is the part that matters: at row i it reads rows
0..i and never the future, so the label a bar carries could have been computed
at that bar. A whole-series `quantile()` would have been one character shorter
and would have leaked the distribution backwards into every earlier row.

After: three levels on every timeframe (15m 3,928/15,158/7,209 low/medium/high
across 26,295 bars; 1d 10,146/892/352).
"""
import numpy as np
import pandas as pd
import pytest

from src.analytics.analyzers.news_impact_analyzer import NewsImpactAnalyzer


def _scores(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.05, n),
                     index=pd.date_range("2026-07-01", periods=n, freq="h"))


def test_a_score_that_never_reaches_the_old_cut_offs_still_gets_graded():
    analyzer = NewsImpactAnalyzer()

    levels = analyzer._determine_significance_levels(_scores())

    assert set(levels.unique()) == {"low", "medium", "high"}, (
        "the live scores span 0.058-0.102 and were all 'low'"
    )


def test_the_grade_uses_only_what_came_before_it():
    """An expanding window, not the whole series: no leak backwards.

    Truncating the series must not change the labels of the rows that remain.
    A whole-series quantile fails this outright.
    """
    analyzer = NewsImpactAnalyzer()
    full = _scores(200)

    on_full = analyzer._determine_significance_levels(full)
    on_prefix = analyzer._determine_significance_levels(full.iloc[:120])

    assert on_full.iloc[:120].tolist() == on_prefix.tolist(), (
        "a later observation changed an earlier row's label"
    )


def test_nothing_is_unusual_before_there_is_a_baseline():
    """Ignorance is reported as 'low', not as a measurement."""
    analyzer = NewsImpactAnalyzer({"significance_thresholds": {"min_history": 50}})

    levels = analyzer._determine_significance_levels(_scores(60))

    assert set(levels.iloc[:49].unique()) == {"low"}


def test_the_absolute_mode_still_works_and_complains_when_it_is_useless(caplog):
    """Kept for anyone who wants it, but it can no longer fail silently."""
    import logging

    analyzer = NewsImpactAnalyzer({"significance_thresholds": {
        "significance_mode": "absolute", "high_impact": 0.8, "medium_impact": 0.3,
    }})

    with caplog.at_level(logging.ERROR):
        levels = analyzer._determine_significance_levels(_scores())

    assert set(levels.unique()) == {"low"}
    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "single level" in messages, (
        "a constant feature must announce itself; this one was silent for "
        "every run the project has made"
    )


def test_a_large_impact_outranks_a_small_one():
    """Relative must not mean arbitrary."""
    analyzer = NewsImpactAnalyzer()
    scores = pd.Series(
        list(np.full(60, 0.01)) + [0.9],
        index=pd.date_range("2026-07-01", periods=61, freq="h"),
    )

    levels = analyzer._determine_significance_levels(scores)

    assert levels.iloc[-1] == "high"
