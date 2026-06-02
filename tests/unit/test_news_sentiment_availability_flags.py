import pandas as pd

from src.features.enrichers.hype_enricher import HypeEnricher
from src.features.enrichers.news_impact_enricher import NewsImpactEnricher
from src.features.enrichers.sentiment_features_enricher import SentimentFeaturesEnricher


def test_hype_enricher_marks_missing_news_count_availability():
    enricher = object.__new__(HypeEnricher)
    df = pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=2, freq="h")})
    news_count = pd.DataFrame({"datetime": [df.loc[0, "datetime"]], "news_count": [2]})

    enriched = enricher._merge_hype_scores(df, news_count, ["datetime"], "global")

    assert enriched["hype_score"].tolist() == [2.0, 0.0]
    assert enriched["hype_available"].tolist() == [1, 0]


def test_news_impact_enricher_marks_unavailable_impact_before_first_news():
    enricher = object.__new__(NewsImpactEnricher)
    df = pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=2, freq="h")})
    impact_scores = pd.Series([0.7], index=[df.loc[1, "datetime"]])
    significance = pd.Series(["high"], index=[df.loc[1, "datetime"]])

    enriched = enricher._merge_impact_scores(
        df,
        impact_scores,
        significance,
        "datetime",
        "test",
    )

    assert enriched["news_impact_score"].tolist() == [0.0, 0.7]
    assert enriched["news_impact_available"].tolist() == [0, 1]
    assert enriched["news_significance_level"].tolist() == [0, 2]


def test_sentiment_preparation_tracks_carried_sentiment_availability():
    enricher = object.__new__(SentimentFeaturesEnricher)
    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "sentiment_score": [None, 0.2, None],
        }
    )

    prepared = enricher._prepare_dataframe(df, "sentiment_score")

    assert prepared["sentiment_score"].tolist() == [0.0, 0.2, 0.2]
    assert prepared["sentiment_available"].tolist() == [0, 1, 1]
