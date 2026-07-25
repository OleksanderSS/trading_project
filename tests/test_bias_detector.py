import os
import sys

import numpy as np
import pandas as pd

# Додаємо корінь проекту до шляху
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtesting.advanced.advanced_engine import BiasDetector


def test_look_ahead_bias_detection():
    """Перевірка виявлення look-ahead bias."""
    detector = BiasDetector()

    # detect_look_ahead_bias's second argument is a raw PRICE series -- it
    # derives future returns internally via pct_change(lag).shift(-lag), it
    # does not accept a pre-computed returns series directly.
    n = 100
    prices = pd.Series(100.0 + np.linspace(0, 10, n))
    future_returns = prices.pct_change(1, fill_method=None).shift(-1)
    signals = future_returns  # Очевидний витік: сигнал t = майбутня дохідність t
    result = detector.detect_look_ahead_bias(signals, prices)

    assert bool(result['lookahead_bias_detected']) is True
    # print(f"\n✅ Look-ahead bias успішно виявлено")

def test_no_look_ahead_bias():
    """Перевірка того, що випадкові сигнали не викликають bias."""
    detector = BiasDetector()
    
    # Випадкові сигнали
    signals = pd.Series(np.random.randn(100))
    returns = pd.Series(np.random.randn(100))
    
    result = detector.detect_look_ahead_bias(signals, returns)
    
    assert bool(result['lookahead_bias_detected']) is False
    # print(f"✅ Відсутність bias підтверджена")
