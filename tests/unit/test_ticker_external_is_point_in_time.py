"""Two per-ticker sources collected for months and merged by nothing.

    wikipedia_attention   11,417 rows, 309 articles, daily since 2026-06-30
    insider_trades         1,395 rows, 778 tickers,  since 2026-07-29

Both are keyed by (ticker, date), and the part everyone expects to be hard —
mapping an article name to a symbol — does not exist: the `article` column IS
the ticker. AAPL, ABBV, ABT.

Each carries the same timing trap the CFTC report has, in its own dialect:
the day a fact happened is not the day it became knowable.

  Wikipedia publishes a day's pageviews the following day, so a bar inside
  day D cannot read D's count.

  An insider trade is private until the Form 4 is filed — a median of two
  days later in the stored rows. Joining on `trade_date` would hand every bar
  those two days. `filing_date` is the only honest key.

Measured after wiring, and worth stating rather than discovering later:
wikipedia reaches 61.6% of 15-minute bars with 604 distinct attention scores,
while insider reaches 4.7% with a single value, because only 9 of its 1,395
filings name one of our 22 tickers. One is useful now; the other is wired so
it becomes useful as the collector accumulates.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.ticker_external_enricher import TickerExternalEnricher


@pytest.fixture
def enricher():
    return TickerExternalEnricher()


@pytest.fixture
def bars():
    days = pd.date_range("2026-07-10", periods=12, freq="D", tz="UTC")
    return pd.DataFrame({
        "ticker": ["AAPL"] * 12 + ["MSFT"] * 12,
        "datetime": list(days) * 2,
        "close": np.linspace(100, 130, 24),
    })


def _wiki(ticker: str, views: list[int], start: str = "2026-07-01") -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range(start, periods=len(views), freq="D"),
        "article": [ticker] * len(views),
        "views": views,
        "project": ["en.wikipedia"] * len(views),
    })


def test_yesterdays_pageviews_are_the_newest_a_bar_can_see(enricher, bars):
    """The API publishes a day's count the following day."""
    wiki = _wiki("AAPL", [100, 100, 100, 100, 100, 100, 100, 100, 100, 9999],
                 start="2026-07-05")   # the spike is 2026-07-14

    enriched = enricher._enrich_impl(bars, wikipedia_attention=wiki)

    stamps = pd.to_datetime(enriched["datetime"]).dt.tz_localize(None)
    same_day = enriched.loc[
        (stamps == pd.Timestamp("2026-07-14")) & (enriched["ticker"] == "AAPL"),
        "wiki_views",
    ]
    next_day = enriched.loc[
        (stamps == pd.Timestamp("2026-07-15")) & (enriched["ticker"] == "AAPL"),
        "wiki_views",
    ]
    assert (same_day != 9999).all(), "14 July read its own day's pageviews"
    assert (next_day == 9999).all()


def test_attention_is_measured_against_the_tickers_own_normal(enricher, bars):
    """AAPL's floor is another company's spike, so a raw count says little."""
    wiki = pd.concat([
        _wiki("AAPL", [1000] * 9 + [3000]),
        _wiki("MSFT", [10] * 9 + [30]),
    ], ignore_index=True)

    enriched = enricher._enrich_impl(bars, wikipedia_attention=wiki)

    z = pd.to_numeric(enriched["wiki_attention_z"], errors="coerce")
    aapl = z[enriched["ticker"] == "AAPL"].dropna()
    msft = z[enriched["ticker"] == "MSFT"].dropna()
    assert len(aapl) and len(msft)
    # A tripling is a tripling on either scale.
    assert aapl.max() == pytest.approx(msft.max(), rel=0.05)


def test_one_tickers_attention_does_not_leak_into_another(enricher, bars):
    wiki = _wiki("AAPL", [500] * 10)

    enriched = enricher._enrich_impl(bars, wikipedia_attention=wiki)

    msft = enriched.loc[enriched["ticker"] == "MSFT", "wiki_views"]
    assert msft.isna().all(), "MSFT has no article in this fixture"


def test_an_insider_trade_is_private_until_it_is_filed(enricher, bars):
    """Median gap between trade and filing is two days in the stored rows."""
    insider = pd.DataFrame({
        "ticker": ["AAPL"],
        "trade_date": [pd.Timestamp("2026-07-11")],
        "filing_date": [pd.Timestamp("2026-07-15")],
        "trade_type": ["P - Purchase"],
        "value": [1_000_000.0],
    })

    enriched = enricher._enrich_impl(bars, insider_trades=insider)

    stamps = pd.to_datetime(enriched["datetime"]).dt.tz_localize(None)
    is_aapl = enriched["ticker"] == "AAPL"
    before = enriched.loc[is_aapl & (stamps < pd.Timestamp("2026-07-15")),
                          "insider_net_value_30d"]
    after = enriched.loc[is_aapl & (stamps >= pd.Timestamp("2026-07-15")),
                         "insider_net_value_30d"]
    assert before.isna().all(), "the trade reached bars before it was filed"
    assert (after == 1_000_000.0).all()


def test_sales_count_against_purchases(enricher, bars):
    insider = pd.DataFrame({
        "ticker": ["AAPL", "AAPL"],
        "trade_date": [pd.Timestamp("2026-07-10")] * 2,
        "filing_date": [pd.Timestamp("2026-07-12")] * 2,
        "trade_type": ["P - Purchase", "S - Sale"],
        "value": [1_000_000.0, 400_000.0],
    })

    enriched = enricher._enrich_impl(bars, insider_trades=insider)

    net = enriched.loc[enriched["ticker"] == "AAPL",
                       "insider_net_value_30d"].dropna()
    assert (net == 600_000.0).all()


def test_both_key_spellings_are_accepted(enricher, bars):
    wiki = _wiki("AAPL", [500] * 10)

    by_table = enricher._enrich_impl(bars, wikipedia_attention_data=wiki)
    by_stem = enricher._enrich_impl(bars, wikipedia_attention=wiki)

    assert by_table["wiki_views"].equals(by_stem["wiki_views"])


def test_row_order_and_ticker_alignment_survive(enricher, bars):
    """A positional copy here once put 54,000 bars on other bars' dates."""
    wiki = pd.concat([_wiki("AAPL", [100] * 10), _wiki("MSFT", [7] * 10)],
                     ignore_index=True)

    enriched = enricher._enrich_impl(bars, wikipedia_attention=wiki)

    assert enriched["ticker"].tolist() == bars["ticker"].tolist()
    assert enriched["close"].tolist() == bars["close"].tolist()
    aapl = enriched.loc[enriched["ticker"] == "AAPL", "wiki_views"].dropna()
    assert (aapl == 100).all(), "AAPL rows carry MSFT's counts"


def test_a_missing_source_is_reported_not_invented(enricher, bars, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        enriched = enricher._enrich_impl(bars)

    assert "wiki_views" not in enriched.columns
    assert "reached this enricher" in "\n".join(
        r.getMessage() for r in caplog.records
    )
