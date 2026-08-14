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


def test_the_text_column_is_chosen_by_content_not_by_presence(enricher):
    """`_find_text_column` returned 'title' because 'title' exists.

    Presence is not content. This database stores blanks as '' rather than
    NaN, and on the 2026-08-13 batch the enricher spent 32 seconds per
    timeframe to report "Avg keywords: 0.0, Avg entities: 0.0". Reproduced:
    with `title` populated the same twenty articles yield 144 keywords and
    72 entities across forty bars; with `title` empty and the body in `text`,
    both were zero.

    The identical mistake was in the sentiment path, where `notna().any()`
    was true for 15,274 empty strings.
    """
    bars = pd.DataFrame({
        "ticker": ["AAPL"] * 40,
        "datetime": pd.date_range("2026-07-01", periods=40, freq="D", tz="UTC"),
        "close": np.linspace(100, 120, 40),
    })
    body = "Apple earnings beat as AI semiconductor demand eases inflation"
    news = pd.DataFrame({
        "ticker": ["AAPL"] * 20,
        "published_at": pd.date_range("2026-07-05", periods=20, freq="D", tz="UTC"),
        "title": [""] * 20,
        "text": [body] * 20,
    })

    enriched = enricher._enrich_impl(bars, news=news)

    assert np.nansum(pd.to_numeric(enriched["keyword_count"], errors="coerce")) > 0
    assert np.nansum(pd.to_numeric(enriched["entity_count"], errors="coerce")) > 0


def test_news_with_no_text_anywhere_is_refused(enricher):
    news = pd.DataFrame({
        "ticker": ["AAPL"] * 5,
        "published_at": pd.date_range("2026-07-05", periods=5, freq="D", tz="UTC"),
        "title": [""] * 5,
        "text": [""] * 5,
    })

    assert enricher._find_text_column(news) is None


def test_general_news_is_added_to_a_tickers_own_not_replaced_by_it(enricher):
    """Six tickers were starved by having any news of their own.

    Of the four news tables only sec_filings carries a ticker column, and it
    names six. Every RSS, Google News and NewsAPI headline therefore arrives
    with ticker NaN -> 'general'. The old rule took a ticker's own rows and
    fell back to general only when there were none, so those six received
    nothing but 8-K and 10-Q titles -- no market vocabulary -- while the
    15,000 headlines carrying every keyword hit went to the other sixteen.

    Counts of attention in an hour add up: a filing about AAPL and a
    market-wide headline in the same hour are both attention.
    """
    bars = pd.DataFrame({
        "ticker": ["AAPL"] * 20 + ["MSFT"] * 20,
        "datetime": list(pd.date_range("2026-07-05", periods=20,
                                       freq="D", tz="UTC")) * 2,
        "close": np.linspace(100, 120, 40),
    })
    market = "Federal Reserve holds rates as AI semiconductor earnings beat"
    news = pd.concat([
        pd.DataFrame({                       # market-wide, no ticker
            "ticker": [None] * 10,
            "published_at": pd.date_range("2026-07-04", periods=10,
                                          freq="D", tz="UTC"),
            "title": [market] * 10,
        }),
        pd.DataFrame({                       # AAPL's own filings, no keywords
            "ticker": ["AAPL"] * 10,
            "published_at": pd.date_range("2026-07-04", periods=10,
                                          freq="D", tz="UTC"),
            "title": ["8-K"] * 10,
        }),
    ], ignore_index=True)

    enriched = enricher._enrich_impl(bars, news=news)

    counts = pd.to_numeric(enriched["keyword_count"], errors="coerce").fillna(0)
    for ticker in ("AAPL", "MSFT"):
        rows = counts[enriched["ticker"] == ticker]
        assert rows.sum() > 0, (
            f"{ticker} saw no market keywords — having filings of its own "
            f"must not cost it the market-wide corpus"
        )
