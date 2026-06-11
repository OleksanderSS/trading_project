import pytest
import pandas as pd
import numpy as np
from src.models.analysis.regime.detector import RegimeDetector
from src.models.analysis.regime.metrics import RegimeMetrics
from src.models.analysis.regime.patterns import RegimePatternAnalyzer
from src.models.analysis.regime.stability import RegimeStabilityAnalyzer
from src.models.analysis.regime.recommendations import RegimeRecommendationEngine
from src.core.exceptions import DataProcessingError

# Фіктивна конфігурація для тестів
REGIME_TYPES = {
    'bull': {
        'description': 'Strong upward trend',
        'volatility_range': (0.0, 0.02),
        'trend_strength': (0.01, 0.1),
        'typical_winners': ['trend_follower', 'ensemble']
    },
    'normal': {
        'description': 'Normal market',
        'volatility_range': (0.01, 0.03),
        'trend_strength': (-0.01, 0.01),
        'typical_winners': ['rf']
    }
}

def test_regime_detector():
    detector = RegimeDetector(REGIME_TYPES)
    # Створюємо фіктивний DataFrame
    data = pd.DataFrame({'close': [100, 101, 102, 103, 104]})
    regime = detector.detect_regime(data)
    assert regime in REGIME_TYPES.keys() or regime == 'normal'

def test_regime_metrics():
    metrics = {'accuracy': 0.8, 'mse': 0.1}
    score = RegimeMetrics.calculate_performance_score(metrics)
    assert 0.0 <= score <= 1.0

    ranked = [('model1', {'performance_score': 0.9}), ('model2', {'performance_score': 0.7})]
    gap = RegimeMetrics.calculate_score_gap(ranked)
    assert gap == pytest.approx(0.2)

def test_regime_stability_analyzer_error():
    analyzer = RegimeStabilityAnalyzer()
    # Перевірка на підняття помилки при невалідній структурі даних
    with pytest.raises(DataProcessingError):
        analyzer.get_most_frequent_switch([{'bad': 'data'}])

def test_regime_recommendation_engine():
    engine = RegimeRecommendationEngine(REGIME_TYPES)
    metrics = {'overall_consistency': 0.4}
    patterns = {'pattern_deviations': [{'expected': 'A', 'actual': 'B'}, {'expected': 'C', 'actual': 'D'}, {'expected': 'E', 'actual': 'F'}]}
    
    recs = engine.generate_regime_recommendations('bull', metrics, patterns)
    assert any("Low model consistency" in r for r in recs)
    assert any("High pattern deviations" in r for r in recs)
