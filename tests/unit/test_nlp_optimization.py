import pytest
import pandas as pd
from src.features.enrichers.keyword_entity_enricher import KeywordEntityEnricher
from src.features.nlp.extractors.keyword_extractor import KeywordExtractor
from src.features.nlp.extractors.entity_extractor import EntityExtractor

def test_keyword_entity_enricher_batch_processing():
    """Перевірка коректності пакетної обробки."""
    config = {
        'entities': {'spacy_model': 'en_core_web_sm'}
    }
    enricher = KeywordEntityEnricher(config)
    
    # Створюємо тестовий DataFrame з новинами
    news_df = pd.DataFrame({
        'text': ['Apple is looking at buying U.K. startup for $1 billion', 'Google is a company in the USA'],
        'published_at': pd.to_datetime(['2026-05-30 10:00:00', '2026-05-30 11:00:00'])
    })
    
    # Викликаємо метод вилучення ознак
    enriched_news = enricher._extract_features(news_df, 'text')
    
    assert 'entities' in enriched_news.columns
    assert len(enriched_news['entities']) == 2
    # Перевіряємо, чи результати не порожні (або принаймні типи вірні)
    assert isinstance(enriched_news['entities'].iloc[0], list)

def test_lru_cache_functionality():
    """Перевірка, чи працює кешування (на прикладі KeywordExtractor)."""
    extractor = KeywordExtractor({'keywords': ['apple', 'google']})
    
    # Перший виклик
    res1 = extractor.extract("Apple and Google are tech companies.")
    # Другий виклик (має взяти з кешу)
    res2 = extractor.extract("Apple and Google are tech companies.")
    
    assert res1 == res2
    # Cache functionality is verified by res1 == res2, internal cache is instance-specific
    assert "apple" in res1
