"""Four columns, one value each, computed from news the bars had not seen.

`advanced_analytics` wrote a single mean, a single standard deviation and
two thresholds derived from them onto every row of the frame. The enricher
diagnostic flagged it plainly:

    advanced_analytics    4 cols    0 live    4 const

Two things wrong at once. A column with one value cannot inform a model.
And the value came from the whole series -- news published long after most
of the bars it was attached to.

Expanding statistics answer the same question causally: how does sentiment
now compare with its own history up to this bar. They vary, which is the
point of a feature, and they contain nothing a bar could not have known.

min_periods=2 because the standard deviation of one observation is not a
number.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.advanced_analytics_enricher import (
    AdvancedAnalyticsEnricher,
)

COLUMNS = [
    "sentiment_mean",
    "sentiment_std_stat",
    "sentiment_pos_threshold",
    "sentiment_neg_threshold",
]


@pytest.fixture
def enricher():
    return AdvancedAnalyticsEnricher()


def _bars(per_ticker=30):
    rng = np.random.default_rng(0)
    total = per_ticker * 2
    return pd.DataFrame({
        "ticker": ["AAPL"] * per_ticker + ["MSFT"] * per_ticker,
        "datetime": list(pd.date_range("2026-07-01", periods=per_ticker,
                                       freq="D", tz="UTC")) * 2,
        "close": np.linspace(100, 110, total),
        "nlp_sentiment_score": rng.normal(0, 0.3, total),
    })


def _news():
    return pd.DataFrame({
        "ticker": ["AAPL"] * 5,
        "published_at": pd.date_range("2026-07-01", periods=5, freq="D", tz="UTC"),
        "sentiment": [0.1, -0.2, 0.3, 0.0, 0.5],
    })


def test_the_statistics_vary(enricher):
    enriched = enricher._enrich_impl(_bars(), news=_news())

    for column in COLUMNS:
        values = pd.to_numeric(enriched[column], errors="coerce")
        assert values.nunique(dropna=True) > 5, (
            f"{column} has {values.nunique(dropna=True)} distinct value(s) — "
            f"a constant column carries no signal"
        )


def test_each_bar_sees_only_its_own_history(enricher):
    """The first bar of a ticker has no standard deviation to report."""
    enriched = enricher._enrich_impl(_bars(), news=_news())

    for ticker in ("AAPL", "MSFT"):
        rows = enriched[enriched["ticker"] == ticker]
        assert pd.isna(rows["sentiment_std_stat"].iloc[0]), (
            f"{ticker}'s first bar was given a statistic drawn from later bars"
        )


def test_one_tickers_history_does_not_enter_anothers(enricher):
    bars = _bars()
    enriched = enricher._enrich_impl(bars.copy(), news=_news())

    aapl = enriched[enriched["ticker"] == "AAPL"]
    own = pd.to_numeric(aapl["nlp_sentiment_score"], errors="coerce")
    expected = own.expanding(min_periods=2).mean()

    assert np.allclose(
        pd.to_numeric(aapl["sentiment_mean"], errors="coerce").to_numpy(),
        expected.to_numpy(), equal_nan=True,
    )


def test_thresholds_stay_one_deviation_from_the_mean(enricher):
    """The relationship the original computed globally, kept."""
    enriched = enricher._enrich_impl(_bars(), news=_news())

    mean = pd.to_numeric(enriched["sentiment_mean"], errors="coerce")
    std = pd.to_numeric(enriched["sentiment_std_stat"], errors="coerce")

    assert np.allclose(
        pd.to_numeric(enriched["sentiment_pos_threshold"], errors="coerce"),
        mean + std, equal_nan=True,
    )
    assert np.allclose(
        pd.to_numeric(enriched["sentiment_neg_threshold"], errors="coerce"),
        mean - std, equal_nan=True,
    )


def test_news_only_sentiment_is_skipped_rather_than_broadcast(enricher, caplog):
    """A corpus scalar spread across bars is what produced the constants."""
    import logging

    bars = _bars().drop(columns=["nlp_sentiment_score"])

    with caplog.at_level(logging.INFO):
        enriched = enricher._enrich_impl(bars, news=_news())

    assert "sentiment_mean" not in enriched.columns
    assert any("statistics skipped" in r.message.lower() for r in caplog.records)
