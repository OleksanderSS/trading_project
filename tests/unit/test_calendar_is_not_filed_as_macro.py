"""The calendar must reach stage 3 under its own name.

It did not, twice, for two different reasons. First its table name sat in the
macro set, so its rows were concatenated into the shared macro frame. That was
removed — but the collector's *type* was still listed as macro, in a branch
that ran before the rule written for the calendar, so the fix changed nothing
and the run looked exactly as before.

Both routes into a table's family are checked here, because a rule that is
correct and unreachable is indistinguishable from no rule at all.
"""

import pytest

from src.pipeline.stages.collection.orchestrator import classify_source_table


def test_calendar_is_its_own_family_by_collector_type():
    # This is what the real config declares: a type, no explicit data_type.
    family = classify_source_table(
        'economic_calendar', {'type': 'economic_calendar', 'table_name': 'economic_calendar'}
    )
    assert family == 'economic_calendar', (
        'the calendar was claimed by an earlier branch and folded into macro_data'
    )


def test_calendar_is_its_own_family_by_table_name():
    # Same answer when nothing is known about the collector at all.
    assert classify_source_table('economic_calendar', {}) == 'economic_calendar'
    assert classify_source_table('economic_calendar', None) == 'economic_calendar'


def test_calendar_is_never_macro():
    for info in ({'type': 'economic_calendar'}, {}, {'type': ''}):
        assert classify_source_table('economic_calendar', info) != 'macro_data'


def test_an_explicit_data_type_wins():
    assert classify_source_table('anything', {'data_type': 'news', 'type': 'fred'}) == 'news'


@pytest.mark.parametrize(
    'table_name, info, expected',
    [
        ('fred_data', {'type': 'fred'}, 'macro_data'),
        ('fred_data', {}, 'macro_data'),
        ('google_news', {'type': 'google_news'}, 'news'),
        ('market_data_raw', {'type': 'yahoo_finance'}, 'market_data'),
        ('reddit_sentiment', {'type': 'reddit_sentiment'}, 'reddit_sentiment'),
        ('google_trends_data', {}, 'google_trends'),
        # Historically handled as news via the name fragment; unchanged.
        ('news_patterns', {}, 'news'),
    ],
)
def test_the_other_sources_keep_the_family_they_had(table_name, info, expected):
    assert classify_source_table(table_name, info) == expected


def test_an_unknown_table_is_claimed_by_nobody():
    # Backups and scratch tables must fall through, so they keep their own name
    # instead of being concatenated into somebody else's frame.
    assert classify_source_table('market_data_raw_backup_15m_20260805', {}) is None
    assert classify_source_table('test_table', {}) is None
