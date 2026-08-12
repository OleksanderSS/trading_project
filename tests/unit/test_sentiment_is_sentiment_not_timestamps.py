"""Every sentiment feature was computed from epoch nanoseconds.

`_aggregate_news_sentiment` ended with a loop that renamed "the first column
that is not ticker or datetime" to `nlp_sentiment_score`. The aggregations
above it return [ticker, <time_col>, <sentiment_col>], and the time column is
`published_at`, not `datetime` -- so the loop renamed the TIMESTAMP and left
the sentiment untouched under its own name. Measured on 2026-08-12:

    {'ticker': 'AAPL',
     'nlp_sentiment_score': Timestamp('2026-03-27 00:00:00'),
     'sentiment': 0.0377...}

`pd.to_numeric` on that timestamp gives ~1.7e18, which is never null. Two
consequences, both visible in the batch:

  * `sentiment_available` read 1.0 on every bar -- including 9,070 daily bars
    that predate any news in the database. The flag whose whole purpose is to
    say "there is a reading here" could not say no.
  * All thirteen sentiment features -- rolling mean and std over 5/20/50,
    velocity, decay weighting, news intensity -- were statistics of
    nanoseconds since 1970. Monotonically increasing ones, so `sentiment_sma`
    was a clock and `sentiment_velocity` was a constant.

The real sentiment reached nothing.

Both column names are known to the caller, so they are renamed by name now.
The guessing loop is gone.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.sentiment_features_enricher import (
    SentimentFeaturesEnricher,
)

NEWS_START = pd.Timestamp("2026-03-27")


@pytest.fixture
def enricher():
    return SentimentFeaturesEnricher()


def _news(n=100):
    return pd.DataFrame({
        "ticker": ["AAPL"] * n,
        "published_at": pd.date_range(NEWS_START, periods=n, freq="D", tz="UTC"),
        "sentiment": np.random.default_rng(0).normal(0, 0.3, n),
    })


def _bars(n=400, start="2025-06-01"):
    return pd.DataFrame({
        "ticker": ["AAPL"] * n,
        "datetime": pd.date_range(start, periods=n, freq="D", tz="UTC"),
        "close": np.linspace(100, 200, n),
    })


def test_the_aggregate_carries_the_sentiment_not_the_clock(enricher):
    aggregated = enricher._aggregate_news_sentiment(_news(), "published_at", "sentiment")

    assert "nlp_sentiment_score" in aggregated.columns
    assert "datetime" in aggregated.columns

    values = pd.to_numeric(aggregated["nlp_sentiment_score"], errors="coerce")
    assert values.notna().all(), "the sentiment column must survive as numbers"
    # A timestamp in nanoseconds is ~1.7e18. A sentiment is not.
    assert values.abs().max() < 10, (
        f"nlp_sentiment_score holds values up to {values.abs().max():.3g}; "
        f"that is a timestamp, not a sentiment"
    )
    assert pd.api.types.is_datetime64_any_dtype(aggregated["datetime"])


def test_bars_older_than_the_news_carry_no_sentiment(enricher):
    """Nine months of bars before the first article must read as empty."""
    enriched = enricher._enrich_impl(_bars(), news=_news())

    older = np.array([
        pd.Timestamp(stamp).tz_localize(None) < NEWS_START
        if pd.Timestamp(stamp).tzinfo else pd.Timestamp(stamp) < NEWS_START
        for stamp in enriched.index
    ])
    assert older.sum() > 100, "the fixture must contain pre-news bars"

    sentiment = pd.to_numeric(
        enriched["nlp_sentiment_score"], errors="coerce"
    ).to_numpy(dtype=float)

    assert int((np.nan_to_num(sentiment[older]) != 0).sum()) == 0, (
        "a bar predating every article cannot carry a sentiment reading"
    )


def test_the_availability_flag_can_say_no(enricher):
    """It read 1.0 on all 11,324 daily bars in the batch. A constant flag
    informs nothing, and this one asserted a reading that did not exist."""
    enriched = enricher._enrich_impl(_bars(), news=_news())

    older = np.array([
        pd.Timestamp(stamp).tz_localize(None) < NEWS_START
        if pd.Timestamp(stamp).tzinfo else pd.Timestamp(stamp) < NEWS_START
        for stamp in enriched.index
    ])
    flag = pd.to_numeric(
        enriched["sentiment_available"], errors="coerce"
    ).to_numpy(dtype=float)

    assert np.nanmean(flag[older]) == pytest.approx(0.0)
    assert np.nanmean(flag[~older]) > 0.9
    assert len(np.unique(flag)) > 1, "a flag with one value carries no information"


def test_the_derived_features_are_not_statistics_of_a_clock(enricher):
    """sentiment_sma over nanoseconds is a clock: it only ever increases."""
    enriched = enricher._enrich_impl(_bars(), news=_news())

    rolling = pd.to_numeric(enriched["sentiment_sma_5"], errors="coerce").dropna()

    assert rolling.abs().max() < 10, (
        f"sentiment_sma_5 reaches {rolling.abs().max():.3g}; a mean of "
        f"sentiments cannot leave the sentiment scale"
    )
    assert not rolling.is_monotonic_increasing, (
        "a rolling mean that only rises is measuring time, not mood"
    )


@pytest.mark.parametrize("time_column", ["published_at", "ts", "created", "datetime"])
def test_the_time_column_never_becomes_the_sentiment_whatever_it_is_called(
    enricher, time_column
):
    """The bug in one line: the loop keyed on the NAME `datetime`.

    Any feed whose timestamp is called anything else -- and every feed here
    calls it something else -- had its clock promoted to the sentiment. Both
    names are passed in, so neither is guessed now.
    """
    news = _news().rename(columns={"published_at": time_column})

    aggregated = enricher._aggregate_news_sentiment(news, time_column, "sentiment")

    values = pd.to_numeric(aggregated["nlp_sentiment_score"], errors="coerce")
    assert values.abs().max() < 10, (
        f"with the time column named '{time_column}', nlp_sentiment_score "
        f"reached {values.abs().max():.3g} -- the timestamp again"
    )
