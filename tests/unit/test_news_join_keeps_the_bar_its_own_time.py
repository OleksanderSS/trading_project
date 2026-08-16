"""Attaching news to a bar must not move the bar.

`NLPFeaturesEnricher` joined news onto bars with
`merge_asof(bars, news, left_index=True, right_on='datetime')`, which keeps the
bar timestamps as the result's index and carries the ARTICLE's publication time
along as a `datetime` column. The next line was

    merged_group.set_index('datetime', inplace=True)   # "Restore the original index"

so every bar was stamped with the publication time of whatever article it had
matched. A backward asof match means many consecutive bars resolve to the same
article, so they also collapsed onto a single timestamp.

Measured on the v14 rebuild: 15m entered stage 3 with 26,295 bars, zero
duplicates, every timestamp on a quarter-hour boundary, and came out with
24,143 bars on somebody else's date and 16,886 duplicate (ticker, datetime)
rows, spread over 592 distinct times shared across all 22 tickers.

Nothing downstream caught it, and the reason is worth keeping: the row count
never changed, the row order never changed, and every collector hash was
intact. The invariant added after the August date corruption watches row ORDER,
and the rows never moved. Only the values did.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.nlp_features_enricher import NLPFeaturesEnricher


@pytest.fixture
def enricher():
    return NLPFeaturesEnricher()


def _bars(tickers=('AAPL', 'MSFT'), n=8):
    """Bars shaped the way the enricher really receives them.

    `_ensure_datetime_index` runs first and does `set_index('datetime')`, so by
    the time the merge happens the timestamp lives ONLY in the index. Keeping a
    duplicate `datetime` column here would give the join a collision the real
    frame never has, and the test would be measuring the fixture.
    """
    stamps = pd.date_range('2026-03-16 13:30', periods=n, freq='15min')
    return pd.concat([
        pd.DataFrame({
            'ticker': t,
            'datetime': stamps,
            'close': np.linspace(100, 110, n),
            'hash': [f'{t}-{i}' for i in range(n)],
        })
        for t in tickers
    ], ignore_index=True).set_index('datetime')


def _news():
    # Deliberately off the bar grid and sparse, which is what real news is:
    # one article covers many bars, and it is published at 09:03:27, not at
    # a quarter past the hour.
    return pd.DataFrame({
        'datetime': pd.to_datetime(['2026-03-16 09:03:27', '2026-03-16 14:12:41']),
        'nlp_sentiment_score': [0.4, -0.2],
    })


def test_the_bar_keeps_its_own_timestamp(enricher):
    bars = _bars()
    merged = enricher._merge_features_by_ticker(bars, _news())

    assert len(merged) == len(bars)
    stamps = pd.to_datetime(merged.index)
    assert set(stamps) == set(pd.to_datetime(bars.index)), (
        'bars were stamped with the publication times of the articles they '
        'matched'
    )


def test_bars_do_not_collapse_onto_one_article(enricher):
    bars = _bars()
    merged = enricher._merge_features_by_ticker(bars, _news())

    key = pd.DataFrame({
        'ticker': merged['ticker'].to_numpy(),
        'datetime': pd.to_datetime(merged.index),
    })
    assert key.duplicated().sum() == 0, (
        'a backward asof match gives many bars the same article; taking the '
        "article's time as the bar's time collapses them"
    )


def test_every_bar_keeps_the_timestamp_it_arrived_with(enricher):
    bars = _bars()
    truth = dict(zip(bars['hash'], pd.to_datetime(bars.index)))
    merged = enricher._merge_features_by_ticker(bars, _news())

    moved = [
        h for h, stamp in zip(merged['hash'], pd.to_datetime(merged.index))
        if truth[h] != stamp
    ]
    assert not moved, f'{len(moved)} of {len(bars)} bars moved to another date'


def test_the_publication_time_does_not_travel_on_as_datetime(enricher):
    # An unsuffixed `datetime` column holding a non-bar meaning is how this
    # started; downstream rescues datetime from wherever it can find it.
    merged = enricher._merge_features_by_ticker(_bars(), _news())
    assert 'datetime' not in merged.columns or (
        pd.to_datetime(merged['datetime']).equals(pd.Series(
            pd.to_datetime(merged.index), index=merged.index))
    )


def test_the_news_still_actually_attaches(enricher):
    # A join that keeps every timestamp by attaching nothing would pass every
    # assertion above.
    merged = enricher._merge_features_by_ticker(_bars(), _news())
    assert 'nlp_sentiment_score' in merged.columns
    assert merged['nlp_sentiment_score'].notna().any()
