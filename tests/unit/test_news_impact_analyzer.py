import pytest
import pandas as pd
from src.analytics.analyzers.news_impact_analyzer import NewsImpactAnalyzer
from src.core.exceptions import DataProcessingError
from unittest.mock import MagicMock

def test_news_impact_analyzer_valid(monkeypatch):
    """Тест успішного аналізу коректних даних."""
    df = pd.DataFrame({
        'text': ['good news', 'bad news', 'neutral news'],
        'score': [0.5, -0.5, 0.0],
        'label': ['positive', 'negative', 'neutral']
    }, index=pd.date_range('2026-05-01', periods=3, freq='h'))

    def fake_analyze_sentiment(texts):
        return pd.DataFrame(
            {
                "text": texts,
                "label": ["positive", "negative", "neutral"],
                "score": [0.5, 0.5, 0.0],
            }
        )

    monkeypatch.setattr(
        "src.analytics.analyzers.news_impact_analyzer.analyze_sentiment",
        fake_analyze_sentiment,
    )
    
    analyzer = NewsImpactAnalyzer()
    results = analyzer.analyze(df)
    
    assert 'news_impact_scores' in results
    assert 'news_significance_levels' in results
    assert len(results['news_impact_scores']) == 3

def test_news_impact_analyzer_invalid_data():
    """Перевірка обробки некоректних даних (порожній DataFrame)."""
    analyzer = NewsImpactAnalyzer()
    with pytest.raises(DataProcessingError, match="Input data must be a non-empty DataFrame"):
        analyzer.analyze(pd.DataFrame())

def test_news_impact_analyzer_missing_column():
    """Перевірка обробки відсутності колонки 'text'."""
    df = pd.DataFrame({'other': [1, 2]}, index=pd.date_range('2026-05-01', periods=2))
    analyzer = NewsImpactAnalyzer()
    with pytest.raises(DataProcessingError, match="Input data must be a non-empty DataFrame with a 'text' column."):
        analyzer.analyze(df)

def test_news_impact_analyzer_aggregation_error():
    """Перевірка обробки помилки агрегації за допомогою мока."""
    analyzer = NewsImpactAnalyzer()
    
    # Створюємо дані
    df = pd.DataFrame({
        'text': ['text', 'text'],
        'score': [0.1, 0.2],
        'label': ['positive', 'positive']
    }, index=pd.date_range('2026-05-01', periods=2, freq='h'))
    
    # Мокаємо попередні кроки
    analyzer._perform_sentiment_analysis = MagicMock(return_value=df)
    analyzer._calculate_weighted_scores = MagicMock(return_value=df)
    
    # Мокаємо метод агрегації, щоб він викидав DataProcessingError
    analyzer._aggregate_scores_by_timestamp = MagicMock(side_effect=DataProcessingError("Simulated error"))
    # Мокаємо метод decay, щоб він не викликався
    analyzer._apply_time_decay = MagicMock()
    
    with pytest.raises(DataProcessingError, match="Simulated error"):
        analyzer.analyze(df)
