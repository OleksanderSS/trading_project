import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.models.analysis.regime.stability import RegimeStabilityAnalyzer
from src.core.exceptions import DataProcessingError

def test_calculate_average_stable_period_valid():
    analyzer = RegimeStabilityAnalyzer()
    switches = [
        {'timestamp': datetime(2026, 1, 1, 10, 0)},
        {'timestamp': datetime(2026, 1, 1, 12, 0)},
        {'timestamp': datetime(2026, 1, 1, 15, 0)}
    ]
    # Period 1: 2h, Period 2: 3h. Mean: 2.5h
    result = analyzer.calculate_average_stable_period(switches)
    assert result == 2.5

def test_calculate_average_stable_period_insufficient_data():
    analyzer = RegimeStabilityAnalyzer()
    assert analyzer.calculate_average_stable_period([]) == float('inf')
    assert analyzer.calculate_average_stable_period([{'timestamp': datetime.now()}]) == float('inf')

def test_calculate_average_stable_period_error():
    analyzer = RegimeStabilityAnalyzer()
    # Passing data without 'timestamp' to force TypeError
    with pytest.raises(DataProcessingError):
        analyzer.calculate_average_stable_period([{}, {}])

def test_get_most_frequent_switch_valid():
    analyzer = RegimeStabilityAnalyzer()
    switches = [
        {'from_regime': 'normal', 'to_regime': 'bull'},
        {'from_regime': 'normal', 'to_regime': 'bull'},
        {'from_regime': 'bull', 'to_regime': 'bear'}
    ]
    result = analyzer.get_most_frequent_switch(switches)
    assert result['from_regime'] == 'normal'
    assert result['to_regime'] == 'bull'
    assert result['count'] == 2

def test_get_most_frequent_switch_empty():
    analyzer = RegimeStabilityAnalyzer()
    assert analyzer.get_most_frequent_switch([]) == {}

def test_calculate_regime_stability():
    analyzer = RegimeStabilityAnalyzer()
    records = [
        {'regime': 'normal'},
        {'regime': 'normal'},
        {'regime': 'bull'},
        {'regime': 'bull'}
    ]
    # 3 records evaluated for switch (total 4), 1 switch occurred.
    # Stability = 1.0 - (1 / 4) = 0.75
    assert analyzer.calculate_regime_stability(records) == 0.75
