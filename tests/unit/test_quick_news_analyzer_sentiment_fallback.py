"""Tests for QuickNewsAnalyzer.cluster_and_analyze's sentiment source
selection.

Reproduces a bug found while reviewing a real pipeline run's
feature_lineage_report.json: the 'sentiment' feature (and everything
derived from it downstream) was permanently 100% NaN. Root cause: the
'sentiment' presence check only asked whether the column existed, not
whether it had usable content — the upstream news schema always includes
a 'sentiment' column as a placeholder for the async FinBERT cloud
function (see README's "Serverless NLP Microservice"), which stays empty
strings until that pipeline populates it. The real run's news data had
'sentiment' present on 15344/15344 rows, all empty strings — 100% "non-null"
by pandas' own accounting, 0% usable.
"""
import pandas as pd

from src.features.nlp.processors.news_analyzer import QuickNewsAnalyzer


def _news_df(sentiment_values, n=6):
    return pd.DataFrame({
        "content": [f"Company announces strong quarterly earnings report number {i}" for i in range(n)],
        "sentiment": sentiment_values,
    })


def test_falls_back_to_textblob_when_sentiment_column_is_all_empty_strings():
    """The exact scenario found in production: column present, values ''."""
    df = _news_df([""] * 6)
    analyzer = QuickNewsAnalyzer(n_clusters=2)

    result = analyzer.cluster_and_analyze(df)

    # TextBlob fallback must have actually run — real (non-null) polarity
    # scores in [-1, 1], not a silent pass-through of empty strings.
    assert result["sentiment_score"].notna().all()
    assert result["sentiment_score"].between(-1.0, 1.0).all()


def test_falls_back_to_textblob_when_sentiment_column_is_all_nan():
    df = _news_df([None] * 6)
    analyzer = QuickNewsAnalyzer(n_clusters=2)

    result = analyzer.cluster_and_analyze(df)

    assert result["sentiment_score"].notna().all()


def test_uses_existing_sentiment_when_genuinely_populated():
    """Must not regress the intended fast path: real FinBERT scores
    should still be reused as-is, not recomputed with TextBlob."""
    real_scores = [0.8, -0.3, 0.5, 0.1, -0.9, 0.0]
    df = _news_df(real_scores)
    analyzer = QuickNewsAnalyzer(n_clusters=2)

    result = analyzer.cluster_and_analyze(df)

    assert result["sentiment_score"].tolist() == real_scores


def test_uses_existing_sentiment_when_partially_populated():
    """A column that's mostly-empty but has at least one real value is
    still "usable existing sentiment" — pd.to_numeric coerces the empty
    entries to NaN rather than discarding the real signal that is there."""
    df = _news_df(["", "0.4", "", "-0.2", "", ""])
    analyzer = QuickNewsAnalyzer(n_clusters=2)

    result = analyzer.cluster_and_analyze(df)

    assert result["sentiment_score"].iloc[1] == 0.4
    assert result["sentiment_score"].iloc[3] == -0.2


def test_computes_textblob_sentiment_when_no_sentiment_column_at_all():
    df = pd.DataFrame({
        "content": [f"Company announces strong quarterly earnings report number {i}" for i in range(6)],
    })
    analyzer = QuickNewsAnalyzer(n_clusters=2)

    result = analyzer.cluster_and_analyze(df)

    assert result["sentiment_score"].notna().all()
