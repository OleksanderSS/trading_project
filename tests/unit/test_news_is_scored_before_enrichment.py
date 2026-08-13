"""The news carried an empty sentiment column and FinBERT ran too late.

Collectors store `sentiment` and leave it blank. On the 2026-08-13 batch the
enricher said so in as many words:

    News sentiment column 'sentiment': 15165/15165 non-null, values: ['']

`pd.to_numeric` turned all 15,165 into NaN, so every sentiment feature was
built from nothing and `sentiment_available` read 0 on all 55,565 rows.

FinBERT was loaded during that same run — inside `news_impact`, which the
enricher priority order places AFTER `sentiment_features`. Its scores went
into that analyzer and never back onto the news frame the other enrichers
read. The model ran, the texts were scored, and the result went nowhere.

Scoring once in the stage, before the per-timeframe loop, gives every
enricher and all three timeframes the same values and loads the model once.
"""
import numpy as np
import pandas as pd
import pytest

from src.pipeline.stages.stage_3_feature_engineering import FeatureEngineeringStage


HEADLINES = [
    "Company beats earnings expectations, shares surge",
    "Firm faces major lawsuit and reports heavy losses",
    "Board meeting scheduled for Tuesday",
]


def _news(sentiment=("", "", "")):
    return pd.DataFrame({
        "text": HEADLINES,
        "sentiment": list(sentiment),
        "published_at": pd.date_range("2026-08-01", periods=3, tz="UTC"),
    })


def test_an_empty_sentiment_column_gets_scored():
    scored = FeatureEngineeringStage._score_news_sentiment(_news())

    values = pd.to_numeric(scored["sentiment"], errors="coerce")
    assert values.notna().all()
    assert values.abs().max() <= 1.0, "a signed sentiment cannot leave [-1, 1]"
    assert (values != 0).any(), "three headlines cannot all be neutral"


def test_the_sign_follows_the_meaning():
    """+score for positive, -score for negative, 0 for neutral -- the
    convention already used by SentimentIntegrator._extract_sentiment_scores."""
    scored = FeatureEngineeringStage._score_news_sentiment(_news())
    values = pd.to_numeric(scored["sentiment"], errors="coerce").to_numpy()

    assert values[0] > 0, "beating earnings is not bearish"
    assert values[1] < 0, "a lawsuit and heavy losses are not bullish"


def test_existing_numeric_sentiment_is_left_alone():
    """Rescoring what a collector already scored would be a second opinion
    nobody asked for, and would cost the model load every run."""
    scored = FeatureEngineeringStage._score_news_sentiment(
        _news(sentiment=(0.42, -0.17, 0.0))
    )

    assert pytest.approx(0.42) == float(scored["sentiment"].iloc[0])
    assert pytest.approx(-0.17) == float(scored["sentiment"].iloc[1])


def test_news_without_any_text_is_reported_not_guessed(caplog):
    import logging

    frame = pd.DataFrame({"sentiment": ["", ""], "ticker": ["AAPL", "MSFT"]})
    with caplog.at_level(logging.ERROR):
        result = FeatureEngineeringStage._score_news_sentiment(frame)

    assert (result["sentiment"] == "").all(), "nothing to score, nothing invented"
    assert any("no usable text" in r.message.lower() for r in caplog.records)


def test_an_empty_frame_passes_through():
    empty = pd.DataFrame()
    assert FeatureEngineeringStage._score_news_sentiment(empty) is empty
    assert FeatureEngineeringStage._score_news_sentiment(None) is None


def test_a_length_mismatch_is_refused_rather_than_aligned_by_position(monkeypatch):
    """Attaching by position is how 54,552 bars got the wrong dates."""
    import logging
    import src.sentiment.sentiment_models as models

    monkeypatch.setattr(
        models, "analyze_sentiment",
        lambda texts, **kw: pd.DataFrame({"label": ["positive"], "score": [0.9]}),
    )

    frame = _news()
    result = FeatureEngineeringStage._score_news_sentiment(frame)

    # Unchanged: two of the three rows would otherwise have been paired with
    # nothing, or worse, with each other's scores.
    assert (result["sentiment"] == "").all()


def test_the_text_column_is_chosen_by_content_not_by_presence():
    """`notna().any()` was true for a column of 15,274 empty strings.

    This database stores blanks as '' rather than NaN. FinBERT then scored
    the word "neutral" -- what _prepare_batch_texts substitutes for an empty
    string -- hit its cache 15,273 times and returned all-neutral for the
    whole corpus in 2.8 seconds on CPU. A real forward pass over that many
    texts takes minutes; the timing was the tell.
    """
    news = pd.DataFrame({
        "text": ["", "", ""],
        "title": HEADLINES,
        "sentiment": ["", "", ""],
        "published_at": pd.date_range("2026-08-01", periods=3, tz="UTC"),
    })

    scored = FeatureEngineeringStage._score_news_sentiment(news)

    values = pd.to_numeric(scored["sentiment"], errors="coerce")
    assert (values != 0).any(), (
        "an empty 'text' column was preferred over a populated 'title'"
    )
    assert values.iloc[0] > 0 and values.iloc[1] < 0
