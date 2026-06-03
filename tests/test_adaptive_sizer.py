import os
import sys

# Додаємо корінь проекту до шляху
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.algorithms.adaptive_position_sizer import AdaptivePositionSizer, PositionSizingParams


def test_adaptive_sizer_initialization():
    """Перевірка ініціалізації з дефолтним конфігом."""
    config = {
        'base_position_size_pct': 0.05,
        'use_kelly_criterion': True
    }
    sizer = AdaptivePositionSizer(config)
    assert sizer.base_position_size_pct == 0.05
    assert sizer.use_kelly is True

def test_calculate_position_size_logic():
    """Перевірка базової логіки розрахунку."""
    sizer = AdaptivePositionSizer()
    params = PositionSizingParams(
        portfolio_value=100000.0,
        volatility=0.02, # 2%
        confidence=0.8,
        market_regime='RANGING'
    )
    
    result = sizer.calculate_position_size(params)
    
    assert 'position_size' in result
    assert result['position_size'] > 0
    assert result['position_size'] <= 100000.0 * 0.10 # Max position limit

def test_regime_adaptation():
    """Перевірка того, що різні режими впливають на розмір позиції."""
    sizer = AdaptivePositionSizer()

    # Для уникнення лімітів, задамо такі параметри, щоб розмір був у межах норми
    params_normal = PositionSizingParams(100000, 0.01, 0.9, market_regime='RANGING')
    params_crisis = PositionSizingParams(100000, 0.01, 0.9, market_regime='CRISIS')

    result_normal = sizer.calculate_position_size(params_normal)
    result_crisis = sizer.calculate_position_size(params_crisis)

    size_normal = result_normal['position_size']
    size_crisis = result_crisis['position_size']

    print(f"\nDEBUG: Normal={size_normal}, Crisis={size_crisis}")

    # В режимі CRISIS розмір має бути значно меншим
    assert size_crisis < size_normal
    assert size_crisis < (size_normal * 0.8) # Має бути менше ніж 80% від норми

