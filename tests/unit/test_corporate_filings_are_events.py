"""SEC filings are events with dates, not prose with a mood.

They were classified as a news source and fed into the news frame, where the
alias list said `filing_date` and the table said `filingDate`, so 24,365
dated, ticker-tagged filings were discarded every run over one capital letter.

Renaming the column would have been the wrong fix. A filing carries `form` and
`primaryDocDescription` -- codes like "10-Q" -- so the sentiment model would
return a number for the string "10-Q" and that number would be noise wearing
the label of sentiment. What a filing is: a company told the regulator
something on a date.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.corporate_filings_enricher import (
    CorporateFilingsEnricher,
)


def _bars(ticker, dates):
    return pd.DataFrame({
        "datetime": pd.to_datetime(dates),
        "ticker": [ticker] * len(dates),
        "close": np.arange(len(dates), dtype=float),
    })


def _filings(rows):
    return pd.DataFrame(rows, columns=["ticker", "filingDate", "form"])


@pytest.fixture
def enricher():
    return CorporateFilingsEnricher({"window_days": 30})


def test_a_bar_before_any_filing_reports_absence_not_zero(enricher):
    """The mistake that put a neutral 0.0 in front of every gate in this system."""
    bars = _bars("AAPL", ["2026-01-10"])
    out = enricher._enrich_impl(bars, sec_filings=_filings([("AAPL", "2026-03-01", "8-K")]))

    assert out["filing_data_available"].iloc[0] == 0
    assert pd.isna(out["filing_days_since_last"].iloc[0])
    assert pd.isna(out["filing_count_30d"].iloc[0])


def test_recency_is_counted_from_the_filing_date(enricher):
    bars = _bars("AAPL", ["2026-03-11"])
    out = enricher._enrich_impl(bars, sec_filings=_filings([("AAPL", "2026-03-01", "8-K")]))

    assert out["filing_days_since_last"].iloc[0] == 10
    assert out["filing_data_available"].iloc[0] == 1


def test_the_window_counts_only_what_fell_inside_it(enricher):
    filings = _filings([
        ("AAPL", "2026-01-05", "8-K"),   # 65 days before the bar
        ("AAPL", "2026-03-01", "8-K"),
        ("AAPL", "2026-03-05", "10-Q"),
    ])
    out = enricher._enrich_impl(_bars("AAPL", ["2026-03-11"]), sec_filings=filings)

    assert out["filing_count_30d"].iloc[0] == 2
    assert out["filing_material_30d"].iloc[0] == 1     # the 8-K only
    assert out["filing_periodic_30d"].iloc[0] == 1     # the 10-Q only


def test_a_filing_belongs_to_one_company(enricher):
    filings = _filings([("NVDA", "2026-03-01", "8-K"), ("NVDA", "2026-03-02", "8-K")])
    out = enricher._enrich_impl(_bars("AAPL", ["2026-03-11"]), sec_filings=filings)

    assert out["filing_data_available"].iloc[0] == 0


def test_a_bar_never_sees_a_filing_from_its_own_future(enricher):
    filings = _filings([("AAPL", "2026-03-20", "8-K")])
    out = enricher._enrich_impl(_bars("AAPL", ["2026-03-10", "2026-03-25"]), sec_filings=filings)

    assert out["filing_data_available"].tolist() == [0, 1]
    assert out["filing_days_since_last"].iloc[1] == 5


def test_report_date_is_refused_as_the_event_time():
    """`reportDate` is the period covered; using it backdates every disclosure."""
    frame = pd.DataFrame({"ticker": ["AAPL"], "reportDate": ["2026-03-31"], "form": ["10-Q"]})
    assert CorporateFilingsEnricher._date_column(frame) is None

    frame["filingDate"] = ["2026-05-02"]
    assert CorporateFilingsEnricher._date_column(frame) == "filingDate"


def test_bars_come_back_unchanged_when_there_are_no_filings(enricher):
    bars = _bars("AAPL", ["2026-03-11"])
    out = enricher._enrich_impl(bars, sec_filings=pd.DataFrame())
    assert list(out.columns) == list(bars.columns)


def test_bars_come_back_unchanged_without_a_ticker_column(enricher):
    bars = _bars("AAPL", ["2026-03-11"]).drop(columns=["ticker"])
    out = enricher._enrich_impl(bars, sec_filings=_filings([("AAPL", "2026-03-01", "8-K")]))
    assert "filing_days_since_last" not in out.columns


def test_rows_keep_their_own_values_when_tickers_interleave(enricher):
    """The as-of merge sorts by time; results must land back on the right row."""
    bars = pd.DataFrame({
        "datetime": pd.to_datetime(
            ["2026-03-11", "2026-03-11", "2026-03-12", "2026-03-12"]
        ),
        "ticker": ["AAPL", "NVDA", "NVDA", "AAPL"],
    })
    filings = _filings([("AAPL", "2026-03-01", "8-K"), ("NVDA", "2026-03-09", "10-Q")])
    out = enricher._enrich_impl(bars, sec_filings=filings)

    by_row = dict(zip(zip(out["ticker"], out["datetime"].dt.day),
                      out["filing_days_since_last"]))
    assert by_row[("AAPL", 11)] == 10
    assert by_row[("NVDA", 11)] == 2
    assert by_row[("NVDA", 12)] == 3
    assert by_row[("AAPL", 12)] == 11


def test_the_feature_names_it_declares_are_the_ones_it_adds(enricher):
    bars = _bars("AAPL", ["2026-03-11"])
    out = enricher._enrich_impl(bars, sec_filings=_filings([("AAPL", "2026-03-01", "8-K")]))
    for name in enricher.get_feature_names():
        assert name in out.columns, name
