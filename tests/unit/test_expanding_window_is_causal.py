"""An expanding window is only "the past" if the rows are in time order.

`AdvancedAnalyticsEnricher` computed expanding sentiment statistics with
`groupby.transform`, which walks the FRAME'S row order. By the time it runs
(priority 78) several earlier enrichers have reordered the frame —
nlp_features, keyword_entity, news_quality and volatility all do, and the
orchestrator restores order only AFTER each one.

Measured on the real batch, 2026-08-20:

    AAPL 1996-08-26   nlp_sentiment_score 0.0   sentiment_mean 0.011112

Every raw score before the news era (2026-03-16) is exactly zero across all
151,640 rows, so an expanding mean over them cannot be anything but zero. The
value decayed with time — 0.0111 in 1996, 0.0015 in 2010 — the signature of a
window filled from the wrong end. A 1996 bar was carrying 2026 sentiment.

It inflated a real conclusion: adding the "news" feature family to the core
doubled the measured IC, +0.0184 to +0.0381, and that increment was the leak.

**The orchestrator's row-order invariant does not catch this.** It restores
order after the enricher, so rows end up where they belong carrying values
computed over somebody else's sequence. An order invariant is not a value
invariant — the same lesson the August date catastrophe taught in a different
costume.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.enrichers.advanced_analytics_enricher import (  # noqa: E402
    AdvancedAnalyticsEnricher,
)


def frame(shuffled: bool, seed: int = 0) -> pd.DataFrame:
    """Zeros for the first 40 bars, then a positive burst — like a news era."""
    dates = pd.date_range('2000-01-03', periods=50, freq='B')
    score = np.concatenate([np.zeros(40), np.full(10, 0.5)])
    d = pd.DataFrame({'datetime': list(dates) * 2,
                      'ticker': ['AAA'] * 50 + ['BBB'] * 50,
                      'nlp_sentiment_score': np.concatenate([score, score]),
                      'close': 100.0})
    if shuffled:
        d = d.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return d


def sentiment_mean(d: pd.DataFrame) -> pd.DataFrame:
    e = AdvancedAnalyticsEnricher()
    out = d.copy()
    # a non-empty news frame only gets past the guard; the statistic
    # itself is computed from df_enriched['nlp_sentiment_score']
    e._add_sentiment_statistics(out, pd.DataFrame({'sentiment': [0.1]}))
    return out


class TestTheWindowRunsForward:
    def test_early_bars_with_no_sentiment_have_no_sentiment_mean(self):
        """The defect in one line: a zero-only past cannot average to nonzero."""
        out = sentiment_mean(frame(shuffled=False))
        early = out.sort_values(['ticker', 'datetime']).groupby('ticker').head(30)
        vals = early['sentiment_mean'].dropna()
        assert (vals.abs() < 1e-12).all(), (
            'a bar before any sentiment existed is carrying sentiment')

    def test_shuffling_the_frame_does_not_change_the_result(self):
        """The whole point: the statistic must depend on time, not row order."""
        key = ['ticker', 'datetime']
        a = sentiment_mean(frame(shuffled=False)).set_index(key)['sentiment_mean']
        b = sentiment_mean(frame(shuffled=True)).set_index(key)['sentiment_mean']
        joined = pd.concat([a.rename('ordered'), b.rename('shuffled')], axis=1).dropna()
        assert len(joined) > 50
        assert np.allclose(joined['ordered'], joined['shuffled'], atol=1e-12), (
            'the expanding window followed row order rather than time')

    def test_the_burst_does_raise_later_bars(self):
        # A control: if nothing ever moves, the test above passes vacuously.
        out = sentiment_mean(frame(shuffled=False))
        last = out.sort_values(['ticker', 'datetime']).groupby('ticker').tail(1)
        assert (last['sentiment_mean'] > 0).all()

    def test_each_ticker_is_independent(self):
        d = frame(shuffled=False)
        d.loc[d.ticker == 'BBB', 'nlp_sentiment_score'] = 0.0
        out = sentiment_mean(d)
        bbb = out[out.ticker == 'BBB']['sentiment_mean'].dropna()
        assert (bbb.abs() < 1e-12).all(), "one ticker's sentiment reached another"
