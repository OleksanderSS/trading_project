"""`market_context` must read the bar's time whatever index the frame carries.

It is the twenty-first enricher in the chain, so by the time it runs the frame
carries whatever index survived the journey. Its timestamp lookup ended with

    pd.Series(pd.to_datetime(values, ...), index=range(len(df)))

which does NOT relabel a series positionally — pandas REINDEXES, looking up
each requested label in the series it was handed. Measured before the fix:

    index 0..n-1        every hour correct
    index 500..500+n    every hour NaT, replaced by the default 0.0
    index 0,2,4,...     SOME hours correct and the rest defaulted, interleaved

The third is the one that matters. A wholly constant column is inert; a column
that is half real and half filler is one a model will learn from.

The fallback then turned a missing answer into a wrong one: `to_datetime` on a
RangeIndex reads the integers as nanoseconds since the epoch, so a frame with
no timestamp came out as 1970-01-01 — hour 0, weekday 3, non-null on every row,
and therefore silently past the check that reports fully-defaulted features.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.market_context_enricher import MarketContextEnricher

N = 40
STAMPS = pd.date_range('2026-03-16 13:30', periods=N, freq='15min')
EXPECTED_HOURS = list(pd.Series(STAMPS).dt.hour)


@pytest.fixture
def enricher():
    return MarketContextEnricher()


def _frame():
    return pd.DataFrame({
        'ticker': 'AAPL',
        'datetime': STAMPS,
        'close': np.linspace(100, 110, N),
        'volume': np.linspace(1e6, 2e6, N),
        'RSI_14_15m': 50.0,
    })


@pytest.mark.parametrize(
    'label, reindex',
    [
        ('a plain 0..n index', lambda d: d),
        ('an index that does not start at zero', lambda d: d.set_index(pd.RangeIndex(500, 500 + N))),
        ('an index with gaps', lambda d: d.set_index(pd.Index(np.arange(0, 2 * N, 2)))),
        ('a non-monotonic index', lambda d: d.set_index(pd.Index(list(range(N))[::-1]))),
    ],
)
def test_the_hour_is_right_whatever_the_index(enricher, label, reindex):
    context = enricher._build_single_series_context(reindex(_frame()))
    assert list(context['hour_of_day']) == EXPECTED_HOURS, (
        f'the timestamps were reindexed away with {label}'
    )


def test_the_time_is_found_in_the_index_when_there_is_no_column(enricher):
    frame = _frame().drop(columns=['datetime']).set_index(STAMPS)
    context = enricher._build_single_series_context(frame)
    assert list(context['hour_of_day']) == EXPECTED_HOURS


def test_a_frame_with_no_time_does_not_get_1970(enricher):
    """The old fallback read a RangeIndex as nanoseconds since the epoch."""
    frame = _frame().drop(columns=['datetime'])
    context = enricher._build_single_series_context(frame)
    # It falls back to the neutral default, which is fine. What must not happen
    # is a plausible-looking hour derived from row numbers.
    assert context['hour_of_day'].nunique() == 1
    assert context['day_of_week'].nunique() == 1
    assert float(context['day_of_week'].iloc[0]) == 0.0, (
        'weekday 3 is 1970-01-01, i.e. the row index read as a timestamp'
    )


def test_a_frame_with_no_time_says_so(enricher, caplog):
    frame = _frame().drop(columns=['datetime'])
    with caplog.at_level(logging.WARNING):
        enricher._build_single_series_context(frame)
    defaulted = [r for r in caplog.messages if 'filled entirely by their default' in r]
    assert defaulted, 'silently defaulting is how a constant column goes unnoticed'
    assert 'hour_of_day' in defaulted[0] and 'day_of_week' in defaulted[0]


def test_an_empty_string_date_column_is_not_a_date(enricher):
    """`''` is not NaN and satisfies every existence test in this codebase."""
    frame = _frame().drop(columns=['datetime']).assign(date='')
    context = enricher._build_single_series_context(frame)
    assert context['hour_of_day'].nunique() == 1


def test_an_empty_date_column_does_not_shadow_a_real_one(enricher):
    frame = _frame().assign(date='')
    context = enricher._build_single_series_context(frame)
    assert list(context['hour_of_day']) == EXPECTED_HOURS
