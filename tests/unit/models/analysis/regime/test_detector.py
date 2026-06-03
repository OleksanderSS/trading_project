import pytest
import pandas as pd
import numpy as np
from src.models.analysis.regime.detector import RegimeDetector

@pytest.fixture
def regime_config():
    return {
        'bull': {'volatility_range': [0, 0.5], 'trend_strength': [0.001, 1.0]},
        'bear': {'volatility_range': [0, 0.5], 'trend_strength': [-1.0, -0.001]},
        'normal': {'volatility_range': [0, 0.5], 'trend_strength': [-0.0001, 0.0001]}
    }

def test_detect_regime_bull(regime_config):
    detector = RegimeDetector(regime_config)
    # Створюємо дані для bull режиму: зростаючий тренд (>20 точок)
    data = pd.DataFrame({'close': np.linspace(100, 150, 25)})
    regime = detector.detect_regime(data)
    assert regime == 'bull'

def test_detect_regime_bear(regime_config):
    detector = RegimeDetector(regime_config)
    # Створюємо дані для bear режиму: падаючий тренд (>20 точок)
    data = pd.DataFrame({'close': np.linspace(150, 100, 25)})
    regime = detector.detect_regime(data)
    assert regime == 'bear'

def test_detect_regime_normal(regime_config):
    detector = RegimeDetector(regime_config)
    # Створюємо дані для normal режиму (майже плоский)
    data = pd.DataFrame({'close': np.ones(25) * 100 + np.random.normal(0, 0.01, 25)})
    regime = detector.detect_regime(data)
    assert regime == 'normal'

def test_detect_regime_empty_data(regime_config):
    detector = RegimeDetector(regime_config)
    data = pd.DataFrame()
    regime = detector.detect_regime(data)
    assert regime == 'normal'
