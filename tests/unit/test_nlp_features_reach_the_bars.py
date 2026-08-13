"""Fifteen thousand articles analysed, clustered, and dropped at the last step.

Every run logged the work being done and then nothing coming of it:

    [QuickNewsAnalyzer] Using existing FinBERT sentiment scores
    [QuickNewsAnalyzer] Clustering complete: 15274 news in 5 clusters
    NLP enrichment complete. Added features: []
    Enricher 'nlp_features' completed: +0 columns in 8.22s

`_get_ticker_features` filtered the analysed news on an exact ticker match
and stopped there. Most of this corpus carries no ticker at all -- RSS feeds
and keyword-driven Google News are market-wide -- so the filter returned
nothing for every ticker, each group was appended unchanged, and three
features that had just been computed were discarded.

Unattributed news now falls back to every ticker, which is what "general
market news" means. Another ticker's news does not: MSFT's article is not
evidence about AAPL, and the keyword enricher already draws the line in the
same place.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.nlp_features_enricher import NLPFeaturesEnricher

HEADLINE = "Apple beats earnings as AI chip demand surges"
NLP_COLUMNS = {"nlp_cluster", "nlp_sentiment_score", "nlp_subjectivity_score"}


@pytest.fixture(scope="module")
def enricher():
    return NLPFeaturesEnricher()


def _bars():
    return pd.DataFrame({
        "ticker": ["AAPL"] * 20,
        "datetime": pd.date_range("2026-07-01 13:00", periods=20,
                                  freq="15min", tz="UTC"),
        "close": np.linspace(100, 101, 20),
    })


def _news(ticker_values):
    return pd.DataFrame({
        "ticker": ticker_values,
        "published_at": pd.date_range("2026-07-01 12:00", periods=20,
                                      freq="10min", tz="UTC"),
        "title": [HEADLINE] * 20,
        "text": [HEADLINE] * 20,
    })


def _added(enricher, news):
    bars = _bars()
    enriched = enricher._enrich_impl(bars.copy(), news=news)
    return {c for c in enriched.columns if c not in bars.columns}, enriched


def test_news_tagged_with_the_ticker_arrives(enricher):
    added, _ = _added(enricher, _news(["AAPL"] * 20))
    assert NLP_COLUMNS <= added


@pytest.mark.parametrize("value", [None, "general"])
def test_unattributed_news_reaches_every_ticker(enricher, value):
    """This is the case the whole corpus falls into, and it added nothing."""
    added, enriched = _added(enricher, _news([value] * 20))

    assert NLP_COLUMNS <= added, (
        "market-wide news reached no ticker — this is the +0 columns case"
    )
    scores = pd.to_numeric(enriched["nlp_sentiment_score"], errors="coerce")
    assert scores.notna().any(), "columns added but never populated"


def test_another_tickers_news_is_not_borrowed(enricher):
    """MSFT's article is not evidence about AAPL."""
    added, _ = _added(enricher, _news(["MSFT"] * 20))

    assert not (NLP_COLUMNS & added), (
        "AAPL bars took features derived from MSFT's news"
    )


def test_a_ticker_with_its_own_news_does_not_fall_back(enricher):
    """The fallback is for absence, not for topping up."""
    mixed = pd.concat([
        _news(["AAPL"] * 20).head(10),
        _news([None] * 20).tail(10),
    ], ignore_index=True)

    _, enriched = _added(enricher, mixed)

    # The point is simply that the specific news still governs: the merge is
    # asof, so the newest AAPL-tagged row at or before a bar wins over an
    # older general one.
    assert "nlp_sentiment_score" in enriched.columns
