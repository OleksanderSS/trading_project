import pytest
import pandas as pd
from src.models.analysis.baseline.comparison import BaselineComparisonEngine
from src.models.analysis.baseline.recommendations import BaselineRecommendationEngine

def test_comparison_engine():
    engine = BaselineComparisonEngine(dominance_threshold=0.01)
    complex_metrics = {'mse': 0.1, 'r2': 0.8}
    baseline_results = {
        'simple': {'metrics': {'mse': 0.05, 'r2': 0.9}, 'complexity_score': 1}
    }
    
    comp = engine.compare(complex_metrics, baseline_results)
    assert comp['dominance_detected'] is True
    assert len(comp['dominant_baselines']) == 1

def test_recommendation_engine():
    engine = BaselineRecommendationEngine(complexity_penalty=0.01)
    dominance = {'dominance_detected': True, 'dominant_baselines': [{'baseline_name': 'simple', 'dominance_strength': 0.1, 'complexity_savings': 0.9}]}
    cost_benefit = engine.perform_cost_benefit_analysis({}, dominance)
    
    recs = engine.generate_simplification_recommendations(dominance, cost_benefit)
    assert any("Consider" in r for r in recs)
    assert cost_benefit['recommendation'] == 'simplify'
