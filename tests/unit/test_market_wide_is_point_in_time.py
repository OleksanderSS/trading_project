"""Two market-wide sources were collected for years and reached no model.

Neither carries a ticker, because neither is about one company: CNN's Fear &
Greed index reads the whole market's mood, and the CFTC Commitments of Traders
report is how futures traders are positioned in the S&P, Nasdaq, Dow, gold and
crude. Both belong on every bar of every ticker, which is exactly why no
per-ticker merge ever picked them up. Measured against 11,390 daily bars:

    cftc         2,610 rows, weekly since 2016-08  -> 100% of bars covered
    fear_greed     267 rows, daily  since 2025-08  ->  49%

CFTC is the deepest history of anything unused in this project.

The publication lag is the whole difficulty, and getting it wrong would have
been worse than leaving the data unused. A COT report is stamped with the
Tuesday it describes and released the following Friday: the stored rows show
`date` 2026-08-11, a Tuesday, with `collected_at` 2026-08-14, the Friday.
Joining on the stamp would hand every bar three days of the future — a leak
identical in every fold, so walk-forward validation would not catch it, and it
would simply raise every score.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.market_wide_enricher import MarketWideEnricher


@pytest.fixture
def enricher():
    return MarketWideEnricher()


@pytest.fixture
def bars():
    return pd.DataFrame({
        "ticker": ["AAPL"] * 20 + ["MSFT"] * 20,
        "datetime": list(pd.date_range("2026-08-10", periods=20, freq="D", tz="UTC")) * 2,
        "close": np.linspace(100, 120, 40),
    })


def _cot(net_pct: float, date: str) -> pd.DataFrame:
    return pd.DataFrame({
        "date": [pd.Timestamp(date)] * 2,
        "instrument": ["S&P", "NASDAQ"],
        "net_position_pct": [net_pct, net_pct / 2],
        "long_short_ratio": [1.5, 2.0],
    })


def test_a_report_is_invisible_until_the_day_it_is_released(enricher, bars):
    """Stamped Tuesday, released Friday: three days that must not leak."""
    tuesday = "2026-08-11"
    cftc = pd.concat([_cot(10.0, "2026-08-04"), _cot(99.0, tuesday)],
                     ignore_index=True)

    enriched = enricher._enrich_impl(bars, cftc=cftc)

    stamps = pd.to_datetime(enriched["datetime"]).dt.tz_localize(None)
    before = enriched.loc[stamps < pd.Timestamp("2026-08-14"), "cftc_sp500_net_pct"]
    after = enriched.loc[stamps >= pd.Timestamp("2026-08-14"), "cftc_sp500_net_pct"]

    assert not (before == 99.0).any(), (
        "Tuesday's report reached bars before Friday's release"
    )
    assert (after == 99.0).all(), "after release it must be the current report"


def test_the_previous_report_still_applies_in_the_gap(enricher, bars):
    """Absence of the newest figure is not absence of all information."""
    cftc = pd.concat([_cot(10.0, "2026-08-04"), _cot(99.0, "2026-08-11")],
                     ignore_index=True)

    enriched = enricher._enrich_impl(bars, cftc=cftc)

    stamps = pd.to_datetime(enriched["datetime"]).dt.tz_localize(None)
    gap = enriched.loc[
        (stamps >= pd.Timestamp("2026-08-10")) & (stamps < pd.Timestamp("2026-08-14")),
        "cftc_sp500_net_pct",
    ]
    assert (gap == 10.0).all()


def test_todays_fear_and_greed_is_not_read_during_today(enricher, bars):
    """The index moves through the session, so the day's value closes it."""
    fg = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-10", "2026-08-11"]),
        "value": [20.0, 80.0],
    })

    enriched = enricher._enrich_impl(bars, fear_greed=fg)

    stamps = pd.to_datetime(enriched["datetime"]).dt.tz_localize(None)
    same_day = enriched.loc[stamps == pd.Timestamp("2026-08-11"), "fear_greed_index"]
    next_day = enriched.loc[stamps == pd.Timestamp("2026-08-12"), "fear_greed_index"]
    assert (same_day == 20.0).all(), "11 August read its own closing value"
    assert (next_day == 80.0).all()


def test_the_last_reading_of_a_day_is_the_one_kept(enricher, bars):
    """267 rows cover 252 days: the index is stored several times a day."""
    fg = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-10"] * 3),
        "value": [10.0, 40.0, 55.0],
    })

    enriched = enricher._enrich_impl(bars, fear_greed=fg)

    assert (enriched["fear_greed_index"].dropna() == 55.0).all()


def test_both_key_spellings_are_accepted(enricher, bars):
    """Stage 3 keys its inputs by table name: cftc_data, fear_greed_data."""
    cftc = _cot(10.0, "2026-08-04")

    by_table = enricher._enrich_impl(bars, cftc_data=cftc)
    by_stem = enricher._enrich_impl(bars, cftc=cftc)

    assert by_table["cftc_sp500_net_pct"].equals(by_stem["cftc_sp500_net_pct"])


def test_every_ticker_gets_the_same_market_wide_value(enricher, bars):
    """It is one market. A per-ticker merge is what left these unused."""
    fg = pd.DataFrame({"date": pd.to_datetime(["2026-08-05"]), "value": [42.0]})

    enriched = enricher._enrich_impl(bars, fear_greed=fg)

    for ticker in ("AAPL", "MSFT"):
        rows = enriched.loc[enriched["ticker"] == ticker, "fear_greed_index"]
        assert (rows == 42.0).all()


def test_a_missing_source_is_reported_not_invented(enricher, bars, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        enriched = enricher._enrich_impl(bars)

    assert "fear_greed_index" not in enriched.columns
    assert "reached this enricher" in "\n".join(
        r.getMessage() for r in caplog.records
    )


def test_row_order_survives_the_as_of_join(enricher, bars):
    """merge_asof returns a fresh RangeIndex; the caller's order must win."""
    fg = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-05", "2026-08-15"]),
        "value": [1.0, 2.0],
    })

    enriched = enricher._enrich_impl(bars, fear_greed=fg)

    assert enriched["ticker"].tolist() == bars["ticker"].tolist()
    assert enriched["close"].tolist() == bars["close"].tolist()
