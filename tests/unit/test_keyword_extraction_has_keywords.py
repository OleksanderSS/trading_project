"""The keyword extractor was handed an empty dictionary and matched nothing.

`KeywordExtractor` matches pre-defined terms; it does not discover them. The
enricher passed `self.config.get('keywords', {})`, which is empty, so on the
2026-08-13 batch `keyword_count` was 0 on all 55,565 rows of every
timeframe, with `keyword_entity_available` 0 beside it.

The enricher still cost 34-90 seconds per timeframe, because entity
extraction does work: half the output was real and half was structurally
zero, at full price.

`knowledge_base.keywords` meanwhile holds 167 terms across 14 categories,
and the collection stage already uses that list to decide which news to
keep. Its shape -- category -> list of terms -- is what KeywordExtractor
wants. Both now read it.

Measured after: 137 terms loaded (the extractor drops noise words and
ticker-shaped entries), and on twenty identical news items the aggregated
keyword_count over forty bars went from 0 to 144.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.keyword_entity_enricher import KeywordEntityEnricher


@pytest.fixture(scope="module")
def enricher():
    return KeywordEntityEnricher()


def test_the_extractor_is_given_terms_to_match(enricher):
    assert len(enricher.keyword_extractor.keywords) > 50, (
        "an empty keyword list makes keyword_count a constant zero, which is "
        "what the batch contained"
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Apple announces record iPhone revenue as AI chips drive earnings", "ai"),
        ("Federal Reserve signals interest rates hold amid inflation data",
         "federal reserve"),
    ],
)
def test_financial_language_produces_keywords(enricher, text, expected):
    found = enricher.keyword_extractor.extract(text)

    assert found, f"no keyword found in: {text}"
    assert expected in found


def test_unrelated_text_produces_none(enricher):
    """A matcher that fires on everything is as useless as one that never does."""
    assert enricher.keyword_extractor.extract("Nothing relevant happened") == []


def test_the_count_reaches_the_bars(enricher):
    bars = pd.DataFrame({
        "ticker": ["AAPL"] * 40,
        "datetime": pd.date_range("2026-07-01", periods=40, freq="D", tz="UTC"),
        "close": np.linspace(100, 120, 40),
    })
    news = pd.DataFrame({
        "ticker": ["AAPL"] * 20,
        "published_at": pd.date_range("2026-07-05", periods=20, freq="D", tz="UTC"),
        "text": ["Apple earnings beat as AI semiconductor demand eases inflation"] * 20,
    })

    enriched = enricher._enrich_impl(bars.copy(), news=news)

    counts = pd.to_numeric(enriched["keyword_count"], errors="coerce")
    assert np.nansum(counts) > 0, "keywords were extracted but never merged onto bars"
    assert counts.nunique(dropna=True) > 1, "a constant column carries no signal"


def test_the_keywords_come_from_the_shared_knowledge_base():
    """Not a second list. The collection stage filters news with this one."""
    from src.config.unified_config_manager import get_current_config

    declared = get_current_config().get_config("knowledge_base").get("keywords")
    assert isinstance(declared, dict) and declared

    loaded = KeywordEntityEnricher._knowledge_base_keywords()
    assert loaded == declared
