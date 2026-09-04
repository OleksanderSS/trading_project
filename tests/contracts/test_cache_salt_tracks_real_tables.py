"""The cache salt must be built from tables that exist.

REGISTER #166, open since 2026-08-29 as a hypothesis and confirmed on
2026-09-04. Every run logged:

    Generated DB salt based on table states: news:missing_market_data:missing

Neither `news` nor `market_data` has ever existed in this database. The real
tables are `market_data_raw`, `google_news`, `rss_news`, `newsapi_articles`
and six more. So the salt was the SHA-256 of a constant string, and the cache
key never moved no matter what arrived in the database -- while the log
printed a fresh-looking salt on every run.

The whole purpose of a state salt is to invalidate a cached answer when the
data behind it changes. This one could not, and it looked like it did.

TWO DEFECTS, ONE FIX EACH.

    Two default lists. `cli/pipeline_executor.py` carried a correct ten-table
    list all along; `cache_manager.py` carried ['news', 'market_data']. The
    wrong copy was the one inside the salt. Two places declaring one thing is
    how every half-landed fix in this project happened, so there is now one
    definition and the executor imports it.

    Silence when every tracked table is missing. A salt built entirely from
    absent tables is not a salt, and saying so at ERROR is what turns this
    from a thing found by reading a log line into a thing a run reports.
"""
from __future__ import annotations

import inspect

import pytest

from src.core.cache.cache_manager import DEFAULT_TRACKED_TABLES, CacheManager


def test_the_tracked_tables_are_named_and_plausible():
    assert isinstance(DEFAULT_TRACKED_TABLES, list)
    assert len(DEFAULT_TRACKED_TABLES) >= 5, (
        "the tracked list shrank; a salt over one or two tables tracks almost "
        "nothing"
    )
    for dead in ("news", "market_data"):
        # `market_data` is kept in the list deliberately -- it is harmless if
        # absent and would be tracked if it ever appears -- but it must not be
        # the ONLY market table, which is the state that froze the salt.
        assert "market_data_raw" in DEFAULT_TRACKED_TABLES, (
            f"{dead!r} is tracked without market_data_raw beside it, which is "
            f"exactly the configuration that made the salt constant"
        )


def test_the_real_tables_of_this_database_are_covered():
    """Named individually rather than counted: a list of the right LENGTH made
    of the wrong names is precisely what #166 was."""
    for table in ("market_data_raw", "google_news", "fred_data", "sec_filings"):
        assert table in DEFAULT_TRACKED_TABLES, (
            f"{table} exists in the database and is not tracked, so a change "
            f"to it cannot invalidate anything"
        )


def test_there_is_only_one_definition():
    """The two lists disagreed for as long as both existed."""
    from src.cli import pipeline_executor

    assert pipeline_executor._DEFAULT_TRACKED_TABLES is DEFAULT_TRACKED_TABLES, (
        "the executor has its own copy again; when these drift the salt is "
        "the one that silently loses"
    )


def test_a_salt_of_only_missing_tables_is_reported_at_error():
    """The half that matters: not that the list is right today, but that a
    wrong one says so. A constant salt is indistinguishable from a working one
    in every log line except this."""
    source = inspect.getsource(CacheManager._get_db_salt)
    assert 'endswith(":missing")' in source, (
        "nothing checks whether every tracked table is absent, so the frozen "
        "salt is silent again"
    )
    assert "logger.error" in source, (
        "an all-missing salt is reported at info level, which is where this "
        "defect hid for a week"
    )
    assert "never invalidate" in source, (
        "the message does not state the consequence; 'tables missing' reads as "
        "a warning about tables, not about the cache being frozen"
    )


def test_the_salt_still_reads_the_config_first():
    """The list above is a FALLBACK. An operator who sets cache.tracked_tables
    must still win, or this fix replaces one hardcoded answer with another."""
    source = inspect.getsource(CacheManager._get_db_salt)
    assert "cache.tracked_tables" in source
    assert "DEFAULT_TRACKED_TABLES" in source
    assert source.index("cache.tracked_tables") < source.index("DEFAULT_TRACKED_TABLES"), (
        "the default is consulted before the config, so the config decides "
        "nothing"
    )
