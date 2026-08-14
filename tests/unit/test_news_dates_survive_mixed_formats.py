"""Two thirds of the news lost its date to pandas format inference.

pandas 2 infers ONE format from the first non-null value of an object column
and coerces everything that does not match it to NaT. The news frame is the
concatenation of four tables, each with its own convention:

    newsapi   2026-05-13T18:54:31Z        ISO with time
    rss       Timestamp(..., Europe/Kiev) already parsed, tz-aware
    sec       2026-05-12                  date only

So `pd.to_datetime(col, errors='coerce', utc=True)` locked onto ISO and threw
away every filing. Measured on the live database: 12,252 of 35,673 rows
survived — exactly newsapi (2,510) plus rss (9,742), with all 23,421 SEC
filings turned to NaT. After the fix, 35,673 of 35,673.

The second-order damage was worse than the missing rows. Of the four news
tables only sec_filings carries a ticker, so discarding it left the hourly
keyword aggregation with nothing but market-wide rows: every group key was
'general'. With the filings restored the aggregation groups by AAPL, AMD,
AMZN, BAC, GOOGL, GS, INTC, JPM, KO, MSFT, NVDA, QQQ as intended.

Three rebuilds reported "Avg keywords: 0.0" and one crashed outright on
columns that were never merged in, and the visible symptom each time was a
missing feature — never a missing date.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.keyword_entity_enricher import KeywordEntityEnricher
from src.features.utils.datetime_utils import (
    ensure_datetime_column,
    parse_mixed_datetimes,
)


MIXED = ["2026-05-13T18:54:31Z", "2026-05-18T09:00:02Z", "2026-05-12", "2026-05-08"]


def test_a_date_only_row_after_an_iso_row_is_not_discarded():
    naive = pd.to_datetime(pd.Series(MIXED), errors="coerce", utc=True)
    assert int(naive.isna().sum()) == 2, (
        "this is the pandas behaviour being worked around; if it ever changes "
        "the workaround can go, but silently keeping it is harmless"
    )

    parsed = parse_mixed_datetimes(pd.Series(MIXED), utc=True)

    assert int(parsed.isna().sum()) == 0
    assert parsed.iloc[2] == pd.Timestamp("2026-05-12", tz="UTC")


def test_the_order_of_the_formats_does_not_decide_which_survive():
    """Whichever came first used to win. Neither should lose."""
    for order in (MIXED, list(reversed(MIXED))):
        assert int(parse_mixed_datetimes(pd.Series(order), utc=True).isna().sum()) == 0


def test_a_value_that_is_no_date_at_all_still_becomes_nat():
    """The retry recovers rows lost to inference, not rows with no date."""
    parsed = parse_mixed_datetimes(
        pd.Series(["2026-05-12", "not a date", None]), utc=True
    )

    assert parsed.notna().tolist() == [True, False, False]


def test_an_already_parsed_column_is_left_alone():
    stamps = pd.Series(pd.date_range("2026-05-01", periods=3, freq="D", tz="UTC"))

    parsed = parse_mixed_datetimes(stamps, utc=True)

    assert parsed.tolist() == stamps.tolist()


def test_the_shared_normaliser_carries_the_fix():
    """Every stage reaches this through ensure_datetime_column."""
    frame = pd.DataFrame({"published_at": MIXED, "title": ["a", "b", "c", "d"]})

    out = ensure_datetime_column(frame)

    assert "datetime" in out.columns
    assert int(out["datetime"].isna().sum()) == 0


# --- the enricher that the lost dates were starving ------------------------


@pytest.fixture
def enricher():
    return KeywordEntityEnricher()


def _bars(tickers):
    stamps = list(pd.date_range("2026-05-10", periods=20, freq="D", tz="UTC"))
    return pd.DataFrame({
        "ticker": sum(([t] * 20 for t in tickers), []),
        "datetime": stamps * len(tickers),
        "close": np.linspace(100, 120, 20 * len(tickers)),
    })


def _news(ticker):
    body = "Apple earnings beat as AI semiconductor demand eases inflation"
    return pd.DataFrame({
        "ticker": [ticker] * 10,
        "published_at": pd.date_range("2026-05-09", periods=10, freq="D", tz="UTC"),
        "title": [body] * 10,
    })


def test_a_ticker_is_matched_whatever_its_case(enricher):
    """'general' was folded and the ticker was not, so neither branch matched.

    When both are empty the loop appends each group without counts, and the
    finalizer then indexes into columns that were never merged in — the
    KeyError that cost three rebuilds their keyword features.
    """
    enriched = enricher._enrich_impl(_bars(["aapl"]), news=_news("AAPL"))

    counts = pd.to_numeric(enriched["keyword_count"], errors="coerce").fillna(0)
    assert counts.sum() > 0, (
        "a case difference between news and bars must not cost every keyword"
    )


def test_a_merge_that_attached_nothing_reports_instead_of_raising(enricher, caplog):
    """The exception cost the counts AND the extraction that produced them.

    `_enrich_impl` catches it and returns the ORIGINAL frame, so 45 to 137
    seconds of keyword extraction per timeframe were discarded along with the
    result.
    """
    import logging

    bars = _bars(["AAPL"])
    merged = bars.copy()  # no keyword_count / entity_count: the merge attached nothing

    with caplog.at_level(logging.ERROR):
        out = enricher._finalize_merge_result(merged)

    assert len(out) == len(bars), "the bars must survive a merge that found nothing"
    assert "never attached" in "\n".join(r.message for r in caplog.records)
