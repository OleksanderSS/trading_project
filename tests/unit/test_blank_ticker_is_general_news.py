"""The fourth thing an empty string broke, this time the ticker.

`fillna('general')` assumed a news row with no company is NaN. This database
writes blanks as '' instead, and on the 2026-08-14 batch every one of the
15,661 cleaned news rows carried ticker == '' with not a single NaN among
them. So the fill did nothing, the hourly aggregation grouped all of them
under '', and the per-ticker merge then looked for 'general' and for 'AAPL'
and matched neither. Both branches empty means every bar group is passed
through without counts:

    Keyword/entity counts were extracted but never attached to the bars:
    ['keyword_count', 'entity_count'] absent after the merge. Frame has
    ['datetime', 'ticker', 'open', 'high', 'low', 'close', 'volume', ...]

Four rebuilds produced a feature set with no keyword_count and no
entity_count in it, after paying 42 to 137 seconds per timeframe to compute
them.

The identical '' -- not NaN -- distinction had already fooled three enrichers
on the TEXT column. NlpFeaturesEnricher was the one place that had the right
predicate (`isin({'general', 'nan', ''})`), inline; it now lives on
BaseEnricher so the next caller inherits it.

Verified on the run's own artifacts (news_20260814_225152.parquet against
prices_15m of the same batch): keyword_count sums to 80,405 over 15,905 of
26,260 bars, entity_count to 175,414 over 17,884, and all 22 tickers are
covered at roughly 720 bars each.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.base import BaseEnricher
from src.features.enrichers.keyword_entity_enricher import KeywordEntityEnricher


@pytest.fixture
def enricher():
    return KeywordEntityEnricher()


@pytest.mark.parametrize("blank", ["", "  ", "nan", "NaN", "None", None])
def test_a_ticker_that_names_no_company_is_general(blank):
    folded = BaseEnricher.normalise_news_ticker(pd.Series([blank]))

    assert folded.iloc[0] == "general"


def test_a_real_symbol_survives_and_is_case_folded():
    folded = BaseEnricher.normalise_news_ticker(pd.Series(["AAPL", "aapl", " MsFt "]))

    assert folded.tolist() == ["aapl", "aapl", "msft"]


def test_counts_reach_the_bars_when_the_ticker_column_is_blank(enricher):
    """The live shape: a ticker column present, filled, and empty."""
    body = "Apple earnings beat as AI semiconductor demand eases inflation"
    news = pd.DataFrame({
        "ticker": [""] * 20,                       # not NaN -- '' , as stored
        "published_at": pd.date_range("2026-07-05", periods=20, freq="D", tz="UTC"),
        "title": [body] * 20,
    })
    bars = pd.DataFrame({
        "ticker": ["AAPL"] * 20 + ["MSFT"] * 20,
        "datetime": list(pd.date_range("2026-07-05", periods=20, freq="D", tz="UTC")) * 2,
        "close": np.linspace(100, 120, 40),
    })

    enriched = enricher._enrich_impl(bars, news=news)

    assert "keyword_count" in enriched.columns, (
        "four rebuilds produced a feature set without this column"
    )
    counts = pd.to_numeric(enriched["keyword_count"], errors="coerce").fillna(0)
    for ticker in ("AAPL", "MSFT"):
        assert counts[enriched["ticker"] == ticker].sum() > 0, (
            f"{ticker} saw no market-wide news"
        )


def test_a_daily_bar_counts_a_days_news_not_the_last_hour_of_it(enricher):
    """`merge_asof` reads the most recent CLOSED bucket, and only that one.

    With the window fixed at an hour, a bar stamped midnight read the
    23:00-00:00 bucket, so a story published at 09:00 the same session sat
    twenty-three buckets behind it and was never counted. Measured on the
    2026-08-14 batch, bars receiving a non-zero keyword count:

        15m   60.6%     60m  42.5%     1d   8.4%

    With the window taken from the bars, daily reaches 19.0% and its total
    count rises from 3,503 to 172,376. Intraday is deliberately unchanged --
    the window is never narrowed below an hour -- because a 15-minute bucket
    is not wrong, merely a different and much emptier question.
    """
    body = "Apple earnings beat as AI semiconductor demand eases inflation"
    news = pd.DataFrame({
        "ticker": [""] * 20,
        # Published mid-session, not just before midnight.
        "published_at": pd.date_range("2026-07-05 09:00", periods=20, freq="D", tz="UTC"),
        "title": [body] * 20,
    })
    bars = pd.DataFrame({
        "ticker": ["AAPL"] * 20,
        "datetime": pd.date_range("2026-07-06", periods=20, freq="D", tz="UTC"),
        "close": np.linspace(100, 120, 20),
    })

    assert enricher._bar_interval(bars) == pd.Timedelta(days=1)

    enriched = enricher._enrich_impl(bars, news=news)

    assert pd.to_numeric(enriched["keyword_count"], errors="coerce").fillna(0).sum() > 0, (
        "a story published during the session must reach that session's bar"
    )


def test_the_window_is_never_narrowed_below_an_hour(enricher):
    """Intraday behaviour must not change: 15m bars keep the hourly window."""
    bars = pd.DataFrame({
        "ticker": ["AAPL"] * 40,
        "datetime": pd.date_range("2026-07-05", periods=40, freq="15min", tz="UTC"),
        "close": np.linspace(100, 110, 40),
    })

    assert enricher._bar_interval(bars) == pd.Timedelta(hours=1)


def test_an_unreadable_index_falls_back_to_an_hour(enricher):
    assert enricher._bar_interval(pd.DataFrame({"ticker": []})) == pd.Timedelta(hours=1)


def test_a_named_ticker_still_gets_its_own_news_plus_the_market(enricher):
    market = "Federal Reserve holds rates as AI semiconductor earnings beat"
    news = pd.concat([
        pd.DataFrame({
            "ticker": [""] * 10,
            "published_at": pd.date_range("2026-07-04", periods=10, freq="D", tz="UTC"),
            "title": [market] * 10,
        }),
        pd.DataFrame({
            "ticker": ["aapl"] * 10,               # lower case, as sources write it
            "published_at": pd.date_range("2026-07-04", periods=10, freq="D", tz="UTC"),
            "title": ["Apple quarterly revenue and dividend announcement"] * 10,
        }),
    ], ignore_index=True)
    bars = pd.DataFrame({
        "ticker": ["AAPL"] * 20,
        "datetime": pd.date_range("2026-07-05", periods=20, freq="D", tz="UTC"),
        "close": np.linspace(100, 120, 20),
    })

    enriched = enricher._enrich_impl(bars, news=news)

    assert pd.to_numeric(enriched["keyword_count"], errors="coerce").fillna(0).sum() > 0
