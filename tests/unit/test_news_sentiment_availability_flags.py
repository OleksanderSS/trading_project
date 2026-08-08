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


def test_sentiment_preparation_marks_only_rows_with_their_own_reading():
    """sentiment_available now means what the other two flags in this file
    mean: a value at THIS row.

    It used to be read off the forward-filled series, so it was [0, 1, 1]
    here -- the third row's 1 describing a value carried from the second.
    Extended over a real ticker's history that makes it the constant 1.0:
    measured on the 2026-08-06 export it never varies, on any of the three
    timeframes, including 5,757 daily rows predating any sentiment source
    in the database. A column with one distinct value cannot inform a
    model, and this one also asserts a reading exists where it does not.

    hype_available and news_impact_available, two tests above, have always
    meant "present here". One name with two meanings across three sibling
    flags is the shape this codebase keeps paying for.

    Note what is NOT lost: "some value is present" stays derivable from
    sentiment_score != 0.0. "This value is fresh" is derivable from
    nothing else, so that is what the flag carries.
    """
    enricher = object.__new__(SentimentFeaturesEnricher)
    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "sentiment_score": [None, 0.2, None],
        }
    )

    prepared = enricher._prepare_dataframe(df, "sentiment_score")

    # The value is still carried forward -- the fix is to the flag, not the fill.
    assert prepared["sentiment_score"].tolist() == [0.0, 0.2, 0.2]
    assert prepared["sentiment_available"].tolist() == [0, 1, 0]


def test_the_three_availability_flags_agree_on_what_available_means():
    """All three answer "is there a reading at this row", so a row with no
    reading of its own reads 0 in every one of them."""
    hype = object.__new__(HypeEnricher)
    frame = pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=2, freq="h")})
    counts = pd.DataFrame({"datetime": [frame.loc[0, "datetime"]], "news_count": [2]})
    assert hype._merge_hype_scores(
        frame, counts, ["datetime"], "global"
    )["hype_available"].tolist() == [1, 0]

    impact = object.__new__(NewsImpactEnricher)
    assert impact._merge_impact_scores(
        frame,
        pd.Series([0.7], index=[frame.loc[1, "datetime"]]),
        pd.Series(["high"], index=[frame.loc[1, "datetime"]]),
        "datetime",
        "test",
    )["news_impact_available"].tolist() == [0, 1]

    sentiment = object.__new__(SentimentFeaturesEnricher)
    assert sentiment._prepare_dataframe(
        pd.DataFrame({"ticker": ["AAPL"] * 2, "sentiment_score": [0.2, None]}),
        "sentiment_score",
    )["sentiment_available"].tolist() == [1, 0]
