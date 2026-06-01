import pytest
from src.analytics.analyzers.adaptive_confidence_analyzer import AdaptiveConfidenceAnalyzer
from src.core.exceptions import DataProcessingError

def test_adaptive_confidence_analyzer_init():
    config = {
        'base_confidence': 0.6,
        'rules': []
    }
    analyzer = AdaptiveConfidenceAnalyzer(config)
    assert analyzer.base_confidence == 0.6
    assert analyzer.rules == []

def test_adaptive_confidence_analyzer_analyze_no_rules():
    analyzer = AdaptiveConfidenceAnalyzer({'base_confidence': 0.5})
    data = {'market_regime': 'normal'}
    result = analyzer.analyze(data)
    assert result['adaptive_confidence_threshold'] == 0.5

def test_adaptive_confidence_analyzer_increase_threshold():
    config = {
        'base_confidence': 0.5,
        'rules': [{
            'name': 'test_rule',
            'if': {'all': [{'context_feature': 'volatility', 'greater_than': 0.5}]},
            'then': {'action': 'increase_threshold', 'value': 0.1}
        }]
    }
    analyzer = AdaptiveConfidenceAnalyzer(config)
    data = {'volatility': 0.6}
    result = analyzer.analyze(data)
    assert result['adaptive_confidence_threshold'] == 0.6

def test_adaptive_confidence_analyzer_decrease_threshold():
    config = {
        'base_confidence': 0.5,
        'rules': [{
            'name': 'test_rule',
            'if': {'all': [{'context_feature': 'sentiment', 'is': 'positive'}]},
            'then': {'action': 'decrease_threshold', 'value': 0.1}
        }]
    }
    analyzer = AdaptiveConfidenceAnalyzer(config)
    data = {'sentiment': 'positive'}
    result = analyzer.analyze(data)
    assert result['adaptive_confidence_threshold'] == 0.4

def test_adaptive_confidence_analyzer_cap_threshold():
    config = {
        'base_confidence': 0.5,
        'max_confidence': 0.7,
        'rules': [{
            'name': 'test_rule',
            'if': {'all': [{'context_feature': 'volatility', 'greater_than': 0.5}]},
            'then': {'action': 'increase_threshold', 'value': 0.5}
        }]
    }
    analyzer = AdaptiveConfidenceAnalyzer(config)
    data = {'volatility': 0.6}
    result = analyzer.analyze(data)
    # 0.5 + 0.5 = 1.0, but capped at 0.7
    assert result['adaptive_confidence_threshold'] == 0.7

def test_adaptive_confidence_analyzer_invalid_rule():
    config = {
        'base_confidence': 0.5,
        'rules': [{
            'name': 'bad_rule',
            # 'if' missing 'all' or 'any' will cause `_evaluate_rule_conditions` to return False, 
            # NOT raise an exception. Let's pass invalid conditions structure.
            'if': {'invalid_key': []}, 
            'then': {}
        }]
    }
    analyzer = AdaptiveConfidenceAnalyzer(config)
    # We need something that forces an exception inside `_evaluate_rule_conditions`
    # Let's mock a method to force exception
    analyzer._evaluate_rule_conditions = lambda conditions, data: exec("raise Exception('Force Fail')")
    
    with pytest.raises(DataProcessingError):
        analyzer.analyze({})
