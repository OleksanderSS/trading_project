"""A method's best feature and its thirtieth-best used to score the same.

`_run_feature_selection_methods` combined three rankings with

    scores.loc[ranked_features.head(top_n).index] += weight

which keeps only MEMBERSHIP of the top 30% and throws the ranking itself away.
A feature every method ranks first and a feature every method ranks thirtieth
came out tied, so the threshold that picks the final set could not separate
them — on a pool of 2,203 columns, where telling them apart is the entire job.

Borda keeps the order. Separately, `corrwith` replaces a per-column Python
loop: measured on the 2026-08-20 experiment the loop cost about ten minutes a
run against seconds, and a selection step nobody re-runs is how unmeasured
choices survive.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.selection.smart_selector import SmartFeatureSelector  # noqa: E402

RNG = np.random.default_rng(0)


@pytest.fixture
def selector():
    return SmartFeatureSelector()


def frame(n=300, cols=20):
    return pd.DataFrame(RNG.normal(0, 1, (n, cols)),
                        columns=[f'f{i}' for i in range(cols)])


class TestCorrelationFilter:
    def test_it_ranks_the_genuinely_correlated_feature_first(self, selector):
        f = frame()
        target = f['f7'] * 3 + RNG.normal(0, 0.1, len(f))
        assert selector._correlation_filter(f, target, False).index[0] == 'f7'

    def test_it_returns_absolute_correlation(self, selector):
        f = frame()
        target = -f['f3'] * 3 + RNG.normal(0, 0.1, len(f))
        out = selector._correlation_filter(f, target, False)
        assert out.index[0] == 'f3' and out.iloc[0] > 0

    def test_a_constant_column_does_not_become_a_nan_row(self, selector):
        # corrwith yields NaN for zero-variance columns; a NaN sorted into the
        # ranking would take a slot in the top 30% while meaning nothing.
        f = frame(); f['dead'] = 1.0
        out = selector._correlation_filter(f, f['f1'], False)
        assert 'dead' not in out.index
        assert out.notna().all()

    def test_it_is_ordered_descending(self, selector):
        f = frame()
        out = selector._correlation_filter(f, f['f2'] * 2 + RNG.normal(0, 1, len(f)), False)
        assert list(out) == sorted(out, reverse=True)


class TestBordaVotingKeepsTheRanking:
    def _scores(self, selector, ranking, weight=1.0):
        """Drive the real voting path with one controlled ranking."""
        f = pd.DataFrame(columns=ranking, dtype=float)
        selector._get_methods_for_regime = lambda regime: {
            (lambda a, b, c: pd.Series(np.linspace(1.0, 0.1, len(ranking)),
                                       index=ranking)): weight
        }
        return selector._run_feature_selection_methods(
            pd.DataFrame(RNG.normal(0, 1, (50, len(ranking))), columns=ranking),
            pd.Series(RNG.normal(0, 1, 50)), False, 'normal', 'ctx',
        )

    def test_the_top_ranked_feature_outscores_the_last_one_kept(self, selector):
        names = [f'f{i}' for i in range(30)]
        scores = self._scores(selector, names)
        kept = scores[scores > 0].sort_values(ascending=False)
        assert len(kept) >= 2
        assert kept.iloc[0] > kept.iloc[-1], "flat voting would tie these"

    def test_scores_decrease_with_rank(self, selector):
        names = [f'f{i}' for i in range(30)]
        scores = self._scores(selector, names)
        kept = [scores[n] for n in names if scores[n] > 0]
        assert all(a >= b for a, b in zip(kept, kept[1:]))

    def test_features_outside_the_top_share_still_score_zero(self, selector):
        names = [f'f{i}' for i in range(30)]
        scores = self._scores(selector, names)
        assert (scores == 0).any(), "the top-share cut must still apply"

    def test_a_heavier_method_moves_scores_more(self, selector):
        names = [f'f{i}' for i in range(30)]
        light = self._scores(selector, names, weight=1.0).max()
        heavy = self._scores(selector, names, weight=2.0).max()
        assert heavy > light
