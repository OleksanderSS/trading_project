"""A bar at 14:00 read news published at 14:50.

Both news enrichers aggregate articles into one-hour windows, and pandas
labels a window with its START: `pd.Grouper(freq='1h')` and `resample('1h')`
file an article published at 14:50 under 14:00.

The sentiment enricher then merges on an exact timestamp and the keyword
enricher merges backward-asof, so a bar at 14:00 -- and, for the keyword
path, 14:15, 14:30 and 14:45 too -- received counts and scores drawn from
articles it could not have read. Up to 59 minutes of look-ahead in every
news-derived feature on every intraday bar.

A window covering [H, H+1) is knowable at H+1. The label now sits there, so
it means "available from", which is what a merge on time is entitled to
assume.

Checked here on the smallest case that can show it: one article at 14:50,
bars every fifteen minutes around it.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.keyword_entity_enricher import KeywordEntityEnricher
from src.features.enrichers.sentiment_features_enricher import (
    SentimentFeaturesEnricher,
)

PUBLISHED = pd.Timestamp("2026-07-01 14:50", tz="UTC")
HEADLINE = "Apple earnings beat as AI semiconductor demand surges"


def _bars(count=16):
    return pd.DataFrame({
        "ticker": ["AAPL"] * count,
        "datetime": pd.date_range("2026-07-01 13:00", periods=count,
                                  freq="15min", tz="UTC"),
        "close": np.linspace(100, 102, count),
    })


def _news():
    return pd.DataFrame({
        "ticker": ["AAPL"],
        "published_at": [PUBLISHED],
        "text": [HEADLINE],
        "title": [HEADLINE],
        "sentiment": [0.9],
    })


def _timestamps(frame):
    if frame.index.name == "datetime":
        return pd.to_datetime(frame.index, utc=True)
    return pd.to_datetime(frame["datetime"], utc=True)


def test_keyword_counts_do_not_precede_the_article():
    enriched = KeywordEntityEnricher()._enrich_impl(_bars(), news=_news())

    stamps = _timestamps(enriched)
    counts = pd.to_numeric(enriched["keyword_count"], errors="coerce").fillna(0)

    early = [str(t) for t, v in zip(stamps, counts) if v > 0 and t < PUBLISHED]
    assert not early, f"bars before the article already carry its keywords: {early}"


def test_the_article_does_arrive_once_its_window_closes():
    """The other half: a causal feature that never fires is not a fix."""
    enriched = KeywordEntityEnricher()._enrich_impl(_bars(), news=_news())

    stamps = _timestamps(enriched)
    counts = pd.to_numeric(enriched["keyword_count"], errors="coerce").fillna(0)

    arrived = [t for t, v in zip(stamps, counts) if v > 0]
    assert arrived, "the article never reached any bar"
    assert arrived[0] == pd.Timestamp("2026-07-01 15:00", tz="UTC"), (
        "the window covering 14:00-14:59 becomes knowable at 15:00"
    )


def test_sentiment_does_not_precede_the_article():
    enriched = SentimentFeaturesEnricher()._enrich_impl(_bars(), news=_news())

    stamps = _timestamps(enriched)
    flag = pd.to_numeric(enriched["sentiment_available"], errors="coerce").fillna(0)

    early = [str(t) for t, v in zip(stamps, flag) if v > 0 and t < PUBLISHED]
    assert not early, f"sentiment marked available before publication: {early}"


def test_sentiment_arrives_at_the_window_close():
    enriched = SentimentFeaturesEnricher()._enrich_impl(_bars(), news=_news())

    stamps = _timestamps(enriched)
    flag = pd.to_numeric(enriched["sentiment_available"], errors="coerce").fillna(0)

    arrived = [t for t, v in zip(stamps, flag) if v > 0]
    assert arrived and arrived[0] == pd.Timestamp("2026-07-01 15:00", tz="UTC")


def test_pandas_still_labels_windows_by_their_start():
    """The premise, pinned: if this ever changes, the shift becomes a bug."""
    news = pd.DataFrame({
        "published_at": [PUBLISHED.tz_convert(None)],
        "value": [1.0],
    })
    grouped = news.groupby(pd.Grouper(key="published_at", freq="1h"))["value"].sum()

    assert grouped.index[0] == pd.Timestamp("2026-07-01 14:00"), (
        "an article at 14:50 is filed under 14:00 — that is why the label "
        "has to be moved to the window's end"
    )
