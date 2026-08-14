"""The third enricher fooled by an empty string, and the worst-behaved one.

Every run reported success over nothing:

    NewsImpactAnalyzer - Starting sentiment analysis for 15661 news items...
    NewsImpactAnalyzer - Successfully calculated news impact and significance
    NewsImpactEnricher - Added news impact features. Impact score range:
                         [0.000, 0.000]

15,661 items in 0.226 seconds — not a model running, a cache returning the
same hash 15,661 times. `_find_text_column` took `title` because `title` is
in the candidate list and the column is present, and this database stores
blanks as '' rather than NaN. `_extract_news_rows` then kept all of them,
because `notna()` is true for a blank.

Why the zero is worse than a missing column: FinBERT labels an empty string
"neutral", and enrichment.yaml weights neutral at 0.0 — correctly, for real
neutral news. So the whole feature became 0.0, which reads exactly like "no
news mattered on any bar", on all three timeframes, and is what the
2026-08-14 export contains. The weight is right; what was wrong is that
15,661 blanks were labelled neutral in the first place.

The same mistake was fixed in the sentiment path on 2026-08-13 and in the
keyword path after that. It is fixed here by moving the check onto
BaseEnricher rather than writing it a third time — which is what the keyword
enricher's own comment had already concluded was needed.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.keyword_entity_enricher import KeywordEntityEnricher
from src.features.enrichers.news_impact_enricher import NewsImpactEnricher


HEADLINES = (
    ["Shares surge as record profit beats every estimate"] * 20
    + ["Regulator opens fraud probe, shares plunge on the news"] * 20
)


@pytest.fixture
def enricher():
    return NewsImpactEnricher()


@pytest.fixture
def news():
    """`title` present but blank, the body in `text` — the live shape."""
    return pd.DataFrame({
        "published_at": pd.date_range("2026-07-01", periods=40, freq="h", tz="UTC"),
        "title": [""] * 40,
        "text": HEADLINES,
    })


@pytest.fixture
def bars():
    return pd.DataFrame({
        "ticker": ["AAPL"] * 40,
        "datetime": pd.date_range("2026-07-01", periods=40, freq="h", tz="UTC"),
        "close": np.linspace(100, 110, 40),
    })


def test_the_column_with_words_in_it_wins_over_the_one_that_merely_exists(enricher, news):
    assert enricher._find_text_column(news) == "text"


def test_blank_rows_are_not_news_items_with_no_sentiment(enricher, news):
    """`notna()` let 15,661 blanks through to be scored as neutral."""
    half_blank = news.copy()
    half_blank.loc[:19, "text"] = ""

    rows = enricher._extract_news_rows(half_blank, "text")

    assert len(rows) == 20


def test_news_with_no_text_anywhere_is_refused_not_scored(enricher, news):
    assert enricher._find_text_column(news.assign(text="")) is None


def test_the_impact_score_varies_once_real_text_reaches_the_model(enricher, bars, news):
    enriched = enricher._enrich_impl(bars, news=news)

    score = pd.to_numeric(enriched["news_impact_score"], errors="coerce")
    assert score.nunique() > 1, (
        "a single value across every bar is what the live export contains"
    )
    assert float(score.abs().max()) > 0.0, (
        "[0.000, 0.000] was reported as a successful calculation"
    )


def test_the_check_is_shared_rather_than_written_a_third_time():
    """Both enrichers reach the same implementation on BaseEnricher."""
    from src.features.enrichers.base import BaseEnricher

    assert hasattr(BaseEnricher, "choose_text_column")

    frame = pd.DataFrame({"title": [""] * 5, "text": ["real words here"] * 5})
    assert KeywordEntityEnricher()._find_text_column(frame) == "text"
    assert NewsImpactEnricher()._find_text_column(frame) == "text"
