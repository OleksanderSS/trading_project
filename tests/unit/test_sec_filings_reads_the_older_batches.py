"""`filings.recent` is capped at 1000, and the cap decided what we could measure.

How far back `recent` reaches is decided by how OFTEN a company files, not by
any window we choose. Measured on the stored table on 2026-09-08, before the
older batches were read:

    1997-2012    11 to 157 filings a year, from 2 to 6 tickers
    2019-2023    5,894 to 9,036 a year, from 70 to 94 tickers

So a filing-based feature measured over the explorable period would have been a
statement about the last five years and about the names that file most, wearing
the label of a thirty-year result. That is a validity problem, not a thin-data
problem, and `filings.files` fixes it for free: AAPL has one older batch of
1,246 filings reaching 1994-01-26, KO and XOM two each holding 2,301 and 2,554.

The older batch has the SAME shape as `recent` -- parallel arrays keyed by
field name -- so both go through one parser. These tests hold that parser to
the shape, and hold the batch skip to not skipping anything it should keep.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from src.data.collectors.sec_filings_collector import SECFilingsCollector

START = datetime(2000, 1, 1)


def _block(dates, forms=None):
    """A filings block in EDGAR's shape: parallel arrays, one per field."""
    return {
        "accessionNumber": [f"000-{i}" for i in range(len(dates))],
        "filingDate": list(dates),
        "form": list(forms or ["10-K"] * len(dates)),
        "items": [[] for _ in dates],
    }


def test_the_same_parser_reads_recent_and_an_older_batch():
    """One shape, one parser. Two copies of a date filter is how they drift."""
    rows = SECFilingsCollector._block_to_filings(
        _block(["2020-05-01", "1994-01-26", "2010-07-07"]),
        "AAPL", "0000320193", START)
    assert [r["filingDate"] for r in rows] == ["2020-05-01", "2010-07-07"], (
        "the 1994 filing should have been dropped by the window and the other "
        "two kept, in order")
    assert {r["ticker"] for r in rows} == {"AAPL"}
    assert {r["cik"] for r in rows} == {"0000320193"}


def test_a_list_field_is_serialised_not_dropped():
    """`items` arrives as an array and DuckDB is handed a string."""
    rows = SECFilingsCollector._block_to_filings(
        {"accessionNumber": ["000-1"], "filingDate": ["2020-01-02"],
         "items": [["2.02", "9.01"]]},
        "KO", "0000021344", START)
    assert rows[0]["items"] == '["2.02", "9.01"]', rows[0]["items"]


def test_an_empty_or_shapeless_block_is_not_an_error():
    """A company with no older batches, and a malformed one, both yield []."""
    assert SECFilingsCollector._block_to_filings({}, "X", "1", START) == []
    assert SECFilingsCollector._block_to_filings(
        {"filingDate": ["2020-01-01"]}, "X", "1", START) == []


def test_an_unparseable_date_skips_its_row_not_the_batch():
    rows = SECFilingsCollector._block_to_filings(
        _block(["2020-01-02", "not-a-date", "2021-03-04"]),
        "X", "1", START)
    assert [r["filingDate"] for r in rows] == ["2020-01-02", "2021-03-04"]


@pytest.mark.parametrize("filing_to,skipped", [
    ("1999-12-31", True),    # ends before the window opens
    ("2000-01-01", False),   # ends exactly on it
    ("2015-07-20", False),   # overlaps
    (None, False),           # unknown: must be fetched, not assumed empty
])
def test_the_batch_skip_only_skips_what_is_wholly_before_the_window(
        filing_to, skipped):
    """The skip saves a request; skipping one batch too many loses history.

    This mirrors the condition in `_fetch_filings_for_cik` rather than calling
    it, because calling it needs a live SEC session. If the condition there
    changes, this is the line that has to change with it.
    """
    boundary = START.strftime("%Y-%m-%d")
    assert bool(filing_to and filing_to < boundary) is skipped
