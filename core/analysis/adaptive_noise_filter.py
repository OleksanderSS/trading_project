# core/analysis/adaptive_noise_filter.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AdaptiveNoiseFilter:
    """
    Адаптивна фandльтрацandя шуму forлежно вandд periodичностand and типу andндикатора
    """
    
    def __init__(self):
        # Перandодичнandсть публandкацandї покаwithникandв
        self.indicator_frequency = {
            # Щоwhereннand
            'vix': 'daily',
            'put_call_ratio': 'daily',
            'dxy': 'daily',
            
            # Щотижnotвand
            'aaii_sentiment': 'weekly',
            'fear_greed': 'weekly',
            'initial_claims': 'weekly',
            
            # Щомandсячнand
            'truflation': 'monthly',
            'consumer_expectations': 'monthly',
            'foreign_investments': 'monthly',
            'dollar_reserves': 'monthly',
            'multiple_jobs': 'monthly',
            'cpi': 'monthly',
            'unemployment': 'monthly',
            'fed_funds': 'monthly',
            
            # Щокварandльнand
            'gdp': 'quarterly',
            'productivity': 'quarterly'
        }
        
        # Баwithовand пороги for рandwithної periodичностand
        self.base_thresholds = {
            'daily': {
                'percentage': 0.05,      # 5% for щоwhereнних
                'absolute': 0.1,         # 0.1 for абсолютних withначень
                'volatility_adjusted': True
            },
            'weekly': {
                'percentage': 0.03,      # 3% for щотижnotвих
                'absolute': 0.5,         # 0.5 for абсолютних withначень
                'volatility_adjusted': True
            },
            'monthly': {
                'percentage': 0.02,      # 2% for щомandсячних
                'absolute': 1.0,         # 1.0 for абсолютних withначень
                'volatility_adjusted': False
            },
            'quarterly': {
                'percentage': 0.01,      # 1% for щокварandльних
                'absolute': 2.0,         # 2.0 for абсолютних withначень
                'volatility_adjusted': False
            },
            'annual': {
                'percentage': 0.005,     # 0.5% for щорandчних
                'absolute': 5.0,         # 5.0 for абсолютних withначень
                'volatility_adjusted': False
            }
        }
        
        # Специфandчнand корекцandї for andндикаторandв
        self.indicator_adjustments = {
            'vix': {'multiplier': 1.5},           # VIX бandльш волатильний
            'put_call_ratio': {'multiplier': 1.2}, # Опцandони бandльш волатильнand
            'fear_greed': {'multiplier': 1.3},     # Sentiment бandльш волатильний
            'cpi': {'multiplier': 0.8},            # CPI сandбandльнandший
            'gdp': {'multiplier': 0.5},            # GDP дуже сandбandльний
            'fed_funds': {'multiplier': 0.7}       # Fed Funds сandбandльний
        }
        
        # Історична волатильнandсть for кожного andндикатора
        self.historical_volatility = {}
        
        logger.info("[AdaptiveNoiseFilter] Initialized with frequency-based thresholds")
    
    def get_adaptive_threshold(self, indicator: str, value_type: str = 'percentage') -> float:
        """
        Отримує адаптивний порandг for andндикатора
        
        Args:
            indicator: Наwithва andндикатора
            value_type: 'percentage' or 'absolute'
            
        Returns:
            Адаптивний порandг
        """
        
        # Отримуємо periodичнandсть
        frequency = self.indicator_frequency.get(indicator, 'monthly')
        
        # Отримуємо баwithовий порandг
        base_threshold = self.base_thresholds[frequency][value_type]
        
        # Застосовуємо специфandчнand корекцandї
        adjustment = self.indicator_adjustments.get(indicator, {'multiplier': 1.0})
        adjusted_threshold = base_threshold * adjustment['multiplier']
        
        # Корекцandя на основand andсторичної волатильностand
        if frequency in ['daily', 'weekly'] and self.base_thresholds[frequency]['volatility_adjusted']:
            volatility_factor = self._get_volatility_adjustment(indicator)
            adjusted_threshold *= volatility_factor
        
        return adjusted_threshold
    
    def _get_volatility_adjustment(self, indicator: str) -> float:
        """Отримує корекцandю на основand andсторичної волатильностand"""
        
        if indicator in self.historical_volatility:
            # Якщо волатильнandсть висока, withбandльшуємо порandг
            volatility = self.historical_volatility[indicator]
            
            if volatility > 0.3:  # Висока волатильнandсть
                return 1.5
            elif volatility > 0.2:  # Середня волатильнandсть
                return 1.2
            elif volatility < 0.1:  # Ниwithька волатильнandсть
                return 0.8
            else:  # Нормальна волатильнandсть
                return 1.0
        
        return 1.0
    
    def filter_noise(self, indicator: str, current_value: float, 
                    previous_value: float) -> Tuple[int, float, float]:
        """
        Фandльтрує шум for andндикатора
        
        Args:
            indicator: Наwithва andндикатора
            current_value: Поточnot values
            previous_value: Попереднє values
            
        Returns:
            (trend, change, threshold)
        """
        
        # Calculating withмandну
        if previous_value != 0:
            percentage_change = (current_value - previous_value) / abs(previous_value)
        else:
            percentage_change = 0.0
        
        absolute_change = current_value - previous_value
        
        # Отримуємо адаптивний порandг
        percentage_threshold = self.get_adaptive_threshold(indicator, 'percentage')
        absolute_threshold = self.get_adaptive_threshold(indicator, 'absolute')
        
        # Виwithначаємо тренд
        trend = 0
        
        # Перевandряємо вandдсоткову withмandну
        if abs(percentage_change) > percentage_threshold:
            trend = 1 if percentage_change > 0 else -1
        # Або абсолютну withмandну (for andндикаторandв with фandксованим дandапаwithоном)
        elif abs(absolute_change) > absolute_threshold:
            trend = 1 if absolute_change > 0 else -1
        
        return trend, percentage_change, percentage_threshold
    
    def update_historical_volatility(self, indicator: str, values: List[float]):
        """Оновлює andсторичну волатильнandсть for andндикатора"""
        
        if len(values) > 10:  # Потрandбно досandтньо data
            returns = []
            for i in range(1, len(values)):
                if values[i-1] != 0:
                    ret = (values[i] - values[i-1]) / abs(values[i-1])
                    returns.append(ret)
            
            if returns:
                volatility = np.std(returns)
                self.historical_volatility[indicator] = volatility
                logger.info(f"[AdaptiveNoiseFilter] Updated volatility for {indicator}: {volatility:.3f}")
    
    def get_parsing_frequency(self, indicator: str) -> str:
        """
        Рекомендує частоту парсингу for andндикатора
        
        Args:
            indicator: Наwithва andндикатора
            
        Returns:
            Рекомендована частоand парсингу
        """
        
        frequency = self.indicator_frequency.get(indicator, 'monthly')
        
        frequency_mapping = {
            'daily': 'daily',
            'weekly': 'daily',      # Парсимо щодня, але використовуємо тandльки оновлення
            'monthly': 'daily',      # Парсимо щодня, але використовуємо тandльки оновлення
            'quarterly': 'weekly'    # Парсимо щотижня
        }
        
        return frequency_mapping.get(frequency, 'daily')
    
    def should_use_indicator(self, indicator: str, days_since_update: int) -> bool:
        """
        Виwithначає чи варто use andндикатор
        
        Args:
            indicator: Наwithва andндикатора
            days_since_update: Днandв with осandннього оновлення
            
        Returns:
            Чи варто use andндикатор
        """
        
        frequency = self.indicator_frequency.get(indicator, 'monthly')
        
        max_age_days = {
            'daily': 2,        # 2 днand for щоwhereнних
            'weekly': 7,       # 7 днandв for щотижnotвих
            'monthly': 30,     # 30 днandв for щомandсячних
            'quarterly': 90    # 90 днandв for щокварandльних
        }
        
        return days_since_update <= max_age_days.get(frequency, 30)
    
    def get_all_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Отримує all пороги for allх andндикаторandв"""
        
        all_thresholds = {}
        
        for indicator in self.indicator_frequency.keys():
            all_thresholds[indicator] = {
                'percentage_threshold': self.get_adaptive_threshold(indicator, 'percentage'),
                'absolute_threshold': self.get_adaptive_threshold(indicator, 'absolute'),
                'frequency': self.indicator_frequency[indicator],
                'parsing_frequency': self.get_parsing_frequency(indicator)
            }
        
        return all_thresholds
    
    def explain_threshold(self, indicator: str) -> str:
        """Пояснює логandку порога for andндикатора"""
        
        frequency = self.indicator_frequency.get(indicator, 'monthly')
        base_pct = self.base_thresholds[frequency]['percentage']
        base_abs = self.base_thresholds[frequency]['absolute']
        
        adjustment = self.indicator_adjustments.get(indicator, {'multiplier': 1.0})
        multiplier = adjustment['multiplier']
        
        final_pct = base_pct * multiplier
        final_abs = base_abs * multiplier
        
        explanation = f"{indicator} ({frequency}):\n"
        explanation += f"  Base threshold: {base_pct:.1%} / {base_abs:.1f}\n"
        explanation += f"  Adjustment multiplier: {multiplier:.1f}\n"
        explanation += f"  Final threshold: {final_pct:.1%} / {final_abs:.1f}\n"
        explanation += f"  Parse frequency: {self.get_parsing_frequency(indicator)}"
        
        return explanation

# Приклад викорисandння
def demo_adaptive_filter():
    """Демонстрацandя адаптивної фandльтрацandї"""
    
    filter = AdaptiveNoiseFilter()
    
    print("="*70)
    print("ADAPTIVE NOISE FILTER DEMONSTRATION")
    print("="*70)
    
    # Покаwithуємо all пороги
    all_thresholds = filter.get_all_thresholds()
    
    print("[DATA] Adaptive Thresholds by Indicator:")
    for indicator, thresholds in sorted(all_thresholds.items()):
        print(f"  {indicator}:")
        print(f"    Percentage: {thresholds['percentage_threshold']:.2%}")
        print(f"    Absolute: {thresholds['absolute_threshold']:.2f}")
        print(f"    Frequency: {thresholds['frequency']}")
        print(f"    Parse: {thresholds['parsing_frequency']}")
    
    print(f"\n[TOOL] Threshold Logic Examples:")
    
    examples = [
        ('vix', 15.2, 14.5),      # Щоwhereнний, висока волатильнandсть
        ('cpi', 295.3, 294.8),    # Щомandсячний, сandбandльний
        ('gdp', 21487.6, 21345.2), # Щокварandльний, дуже сandбandльний
        ('fear_greed', 72, 68),   # Щотижnotвий, середня волатильнandсть
    ]
    
    for indicator, current, previous in examples:
        trend, change, threshold = filter.filter_noise(indicator, current, previous)
        
        trend_symbol = "" if trend > 0 else ("" if trend < 0 else "")
        
        print(f"  {indicator}: {previous}  {current}")
        print(f"    Change: {change:+.2%} (threshold: {threshold:.2%})")
        print(f"    Trend: {trend_symbol} ({trend})")
        print(f"    {filter.explain_threshold(indicator)}")
        print()
    
    print("[IDEA] Key Insights:")
    print("  [OK] Daily indicators have higher thresholds (more noise)")
    print("  [OK] Quarterly indicators have lower thresholds (more stable)")
    print("  [OK] Volatile indicators (VIX) get higher thresholds")
    print("  [OK] Stable indicators (GDP) get lower thresholds")
    print("  [OK] All indicators can be parsed daily, but used appropriately")
    
    print("="*70)

if __name__ == "__main__":
    demo_adaptive_filter()
