"""
Adaptive Targets Configuration
Адаптивна система andргетandв for рandwithних andймфреймandв and periodandв data
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

class TimeframeType(Enum):
    """Типи andймфреймandв"""
    INTRADAY_SHORT = "15m"    # 15 хвилин
    INTRADAY_LONG = "60m"     # 1 година  
    DAILY = "1d"               # Денний
    WEEKLY = "1wk"             # Тижnotвий
    MONTHLY = "1mo"            # Мandсячний

class DataPeriod(Enum):
    """Перandоди data"""
    SHORT = "60d"      # 60 днandв (for intraday)
    MEDIUM = "6mo"     # 6 мandсяцandв
    LONG = "2y"        # 2 роки (for daily)
    MAX = "max"        # Максимально available

@dataclass
class TargetConfig:
    """Конфandгурацandя andргеand"""
    name: str
    description: str
    calculation_period: int  # Перandод for роwithрахунку в свandчках
    min_data_points: int    # Мandнandмальна кandлькandсть data
    suitable_timeframes: List[TimeframeType]
    target_type: str  # "regression" or "classification"
    formula: str     # Формула роwithрахунку
    priority: int    # Прandоритет (1-вищий)
    
class AdaptiveTargetsSystem:
    """Адаптивна система andргетandв"""
    
    def __init__(self):
        self.targets = self._define_all_targets()
        self.timeframe_configs = self._define_timeframe_configs()
        self.economic_context = self._define_economic_context()
    
    def _define_economic_context(self) -> Dict[str, Any]:
        """Визначити економічний контекст для вибору цілей"""
        return {
            "bull_market": {
                "description": "Ринок бичачий - зростаючий тренд",
                "characteristics": ["low_volatility", "strong_trends", "momentum_focus"],
                "preferred_targets": ["returns", "trend_strength", "momentum"]
            },
            "bear_market": {
                "description": "Ринок ведмедій - спадаючий тренд", 
                "characteristics": ["high_volatility", "reversals", "risk_management"],
                "preferred_targets": ["volatility", "drawdown", "direction"]
            },
            "sideways_market": {
                "description": "Ринок бічний - невизначений тренд",
                "characteristics": ["mean_reversion", "range_bound", "oscillators"],
                "preferred_targets": ["mean_reversion", "oscillators", "range_breakout"]
            },
            "high_volatility": {
                "description": "Період високої волатильності",
                "characteristics": ["volatility_focus", "risk_adjusted", "adaptive_stops"],
                "preferred_targets": ["volatility_targets", "risk_metrics", "adaptive_returns"]
            },
            "low_volatility": {
                "description": "Період низької волатильності",
                "characteristics": ["trend_following", "momentum", "breakout_focus"],
                "preferred_targets": ["trend_following", "breakout_signals", "momentum_indicators"]
            }
        }
    
    def get_targets_for_model_type(self, model_type: str, timeframe: TimeframeType, 
                                 data_points: int, economic_context: str = "bull_market") -> List[TargetConfig]:
        """
        Отримати цілі для конкретного типу моделі з урахуванням економічного контексту
        
        Args:
            model_type: "heavy" або "light"
            timeframe: Таймфрейм
            data_points: Кількість даних
            economic_context: Економічний контекст
            
        Returns:
            List[TargetConfig]: Адаптивний список цілей
        """
        # Отримуємо базові підходящі цілі
        base_targets = self.get_suitable_targets(timeframe, data_points)
        
        # Отримуємо переваги для економічного контексту
        context_config = self.economic_context.get(economic_context, self.economic_context["bull_market"])
        preferred_target_types = context_config["preferred_targets"]
        
        # Фільтруємо цілі за типом моделі та контекстом
        filtered_targets = []
        
        for target in base_targets:
            # Легкі моделі: фокус на простих цілях (% зміни)
            if model_type.lower() == "light":
                if self._is_light_model_suitable(target, preferred_target_types):
                    filtered_targets.append(target)
            
            # Важкі моделі: фокус на складних цілях (абсолютні значення, індикатори)
            elif model_type.lower() == "heavy":
                if self._is_heavy_model_suitable(target, preferred_target_types):
                    filtered_targets.append(target)
        
        # Обмежуємо кількість для оптимізації
        max_targets = 8 if model_type.lower() == "light" else 12
        return filtered_targets[:max_targets]
    
    def _is_light_model_suitable(self, target: TargetConfig, preferred_types: List[str]) -> bool:
        """Перевірити чи ціль підходить для легкої моделі"""
        # Легкі моделі працюють краще з:
        # - Простими цілями (% зміни)
        # - Короткими періодами
        # - Класифікацією напряму
        
        light_suitable_patterns = [
            "return_", "direction_", "volatility_", "gap_"
        ]
        
        # Перевіряємо назву цілі
        target_name = target.name.lower()
        is_simple_target = any(pattern in target_name for pattern in light_suitable_patterns)
        is_short_period = target.calculation_period <= 20
        is_classification = target.target_type == "classification"
        
        # Перевіряємо чи відповідає економічному контексту
        matches_context = any(pref_type in target_name for pref_type in preferred_types)
        
        return (is_simple_target or is_classification) and is_short_period and matches_context
    
    def _is_heavy_model_suitable(self, target: TargetConfig, preferred_types: List[str]) -> bool:
        """Перевірити чи ціль підходить для важкої моделі"""
        # Важкі моделі можуть обробляти:
        # - Складні цілі (абсолютні значення)
        # - Довгі періоди
        # - Багато факторів
        # - Технічні індикатори як цілі
        
        heavy_suitable_patterns = [
            "drawdown", "sharpe", "trend_strength", "momentum_change",
            "breakout", "mean_reversion", "percentile", "distance_from_mean"
        ]
        
        target_name = target.name.lower()
        is_complex_target = any(pattern in target_name for pattern in heavy_suitable_patterns)
        is_long_period = target.calculation_period >= 10
        is_regression = target.target_type == "regression"
        
        # Перевіряємо чи відповідає економічному контексту
        matches_context = any(pref_type in target_name for pref_type in preferred_types)
        
        return (is_complex_target or (is_regression and is_long_period)) and matches_context
    
    def get_model_specific_recommendations(self, model_type: str) -> Dict[str, Any]:
        """Отримати рекомендації для конкретного типу моделі"""
        
        if model_type.lower() == "light":
            return {
                "model_characteristics": [
                    "Fast training", "Low memory usage", "Good for real-time",
                    "Limited complexity", "Focus on main signals"
                ],
                "preferred_targets": [
                    "target_return_30m", "target_return_1h", "target_direction_1h",
                    "target_volatility_1h", "gap_percent", "price_impact_1"
                ],
                "target_types": ["classification", "simple_regression"],
                "calculation_periods": [2, 4, 8, 16],  # короткі періоди
                "avoid_targets": [
                    "complex_indicators", "long_period_calculations",
                    "drawdown_analysis", "sharpe_ratio"
                ],
                "economic_focus": ["momentum", "short_term_reactions", "direction"]
            }
        
        elif model_type.lower() == "heavy":
            return {
                "model_characteristics": [
                    "Deep analysis", "Multi-factor processing", "Complex patterns",
                    "Higher accuracy", "More resources needed"
                ],
                "preferred_targets": [
                    "target_volatility_5d", "target_max_drawdown_20d", "target_sharpe_ratio_60d",
                    "target_trend_strength_60d", "target_momentum_change_20d",
                    "target_mean_reversion_5d", "target_breakout_signal_20d"
                ],
                "target_types": ["complex_regression", "multi_classification"],
                "calculation_periods": [20, 50, 60, 120],  # довгі періоди
                "avoid_targets": [
                    "simple_returns_only", "very_short_term",
                    "limited_scope"
                ],
                "economic_focus": ["trend_analysis", "risk_management", "long_term_patterns"]
            }
        
        return {}
    
    def analyze_economic_context(self, price_data: pd.DataFrame, volume_data: pd.DataFrame = None) -> str:
        """
        Аналізувати поточний економічний контекст на основі даних
        
        Args:
            price_data: Дані про ціни
            volume_data: Дані про об'єми
            
        Returns:
            str: Тип економічного контексту
        """
        if price_data.empty:
            return "bull_market"  # default
        
        # Аналіз тренду
        returns = price_data.pct_change().dropna()
        trend_strength = returns.rolling(50).mean().iloc[-1]
        
        # Аналіз волатильності
        volatility = returns.rolling(20).std().iloc[-1]
        
        # Аналіз об'єму (якщо є)
        volume_trend = None
        if volume_data is not None and not volume_data.empty:
            volume_trend = volume_data.pct_change().rolling(20).mean().iloc[-1]
        
        # Визначення контексту
        if trend_strength > 0.01:  # сильний висхідний тренд
            return "bull_market"
        elif trend_strength < -0.01:  # сильний низхідний тренд
            return "bear_market"
        elif volatility > 0.03:  # висока волатильність
            return "high_volatility"
        elif volatility < 0.01:  # низька волатильність
            return "low_volatility"
        else:
            return "sideways_market"
    
    def _define_timeframe_configs(self) -> Dict[TimeframeType, Dict[str, Any]]:
        """Виwithначити конфandгурацandї andймфреймandв"""
        return {
            TimeframeType.INTRADAY_SHORT: {
                "period": DataPeriod.SHORT,
                "max_lookback_days": 60,
                "data_points_per_day": 78,  # 6.5 годин * 60/15
                "max_calculation_period": 20,  # Максимальний period в днях
                "suitable_target_types": ["short_term", "immediate", "intraday"]
            },
            TimeframeType.INTRADAY_LONG: {
                "period": DataPeriod.SHORT, 
                "max_lookback_days": 60,
                "data_points_per_day": 13,   # 6.5 годин * 60/60
                "max_calculation_period": 30,  # Максимальний period в днях
                "suitable_target_types": ["short_term", "medium_term", "intraday"]
            },
            TimeframeType.DAILY: {
                "period": DataPeriod.LONG,
                "max_lookback_days": 730,  # 2 роки
                "data_points_per_day": 1,
                "max_calculation_period": 365,  # Максимальний period в днях
                "suitable_target_types": ["all"]
            },
            TimeframeType.WEEKLY: {
                "period": DataPeriod.MAX,
                "max_lookback_days": 1825,  # 5 рокandв
                "data_points_per_day": 0.2,
                "max_calculation_period": 730,  # 2 роки
                "suitable_target_types": ["long_term", "trend", "structural"]
            },
            TimeframeType.MONTHLY: {
                "period": DataPeriod.MAX,
                "max_lookback_days": 3650,  # 10 рокandв
                "data_points_per_day": 0.03,
                "max_calculation_period": 1825,  # 5 рокandв
                "suitable_target_types": ["long_term", "macro", "structural"]
            }
        }
    
    def _define_all_targets(self) -> Dict[str, TargetConfig]:
        """Виwithначити all andргети with адаптивнandстю"""
        return {
            # === INTRADAY ТАРАГЕТИ (for 15m/60m) ===
            
            # Короткостроковand волатильнand andргети
            "target_volatility_1h": TargetConfig(
                name="target_volatility_1h",
                description="1-годинна волатильнandсть",
                calculation_period=4,  # 1 година = 4 свandчки 15m or 1 свandчка 60m
                min_data_points=20,
                suitable_timeframes=[TimeframeType.INTRADAY_SHORT, TimeframeType.INTRADAY_LONG],
                target_type="regression",
                formula="rolling_std(returns, window=period)",
                priority=1
            ),
            "target_volatility_4h": TargetConfig(
                name="target_volatility_4h", 
                description="4-годинна волатильнandсть",
                calculation_period=16,  # 4 години = 16 candles 15m or 4 свandчки 60m
                min_data_points=50,
                suitable_timeframes=[TimeframeType.INTRADAY_SHORT, TimeframeType.INTRADAY_LONG],
                target_type="regression",
                formula="rolling_std(returns, window=period)",
                priority=2
            ),
            "target_volatility_1d": TargetConfig(
                name="target_volatility_1d",
                description="1-whereнна волатильнandсть (for intraday)",
                calculation_period=78,  # 1 whereнь = 78 candles 15m or 13 candles 60m
                min_data_points=200,
                suitable_timeframes=[TimeframeType.INTRADAY_SHORT, TimeframeType.INTRADAY_LONG],
                target_type="regression",
                formula="rolling_std(returns, window=period)",
                priority=3
            ),
            
            # Короткостроковand цandновand andргети
            "target_return_30m": TargetConfig(
                name="target_return_30m",
                description="Поверnotння for 30 хвилин",
                calculation_period=2,  # 30 хв = 2 свandчки 15m
                min_data_points=10,
                suitable_timeframes=[TimeframeType.INTRADAY_SHORT],
                target_type="regression",
                formula="pct_change(period)",
                priority=1
            ),
            "target_return_1h": TargetConfig(
                name="target_return_1h",
                description="Поверnotння for 1 годину",
                calculation_period=4,  # 1 година = 4 свandчки 15m or 1 свandчка 60m
                min_data_points=20,
                suitable_timeframes=[TimeframeType.INTRADAY_SHORT, TimeframeType.INTRADAY_LONG],
                target_type="regression",
                formula="pct_change(period)",
                priority=2
            ),
            "target_return_4h": TargetConfig(
                name="target_return_4h",
                description="Поверnotння for 4 години",
                calculation_period=16,  # 4 години = 16 candles 15m or 4 свandчки 60m
                min_data_points=50,
                suitable_timeframes=[TimeframeType.INTRADAY_SHORT, TimeframeType.INTRADAY_LONG],
                target_type="regression",
                formula="pct_change(period)",
                priority=3
            ),
            
            # Короткостроковand трендовand andргети
            "target_direction_1h": TargetConfig(
                name="target_direction_1h",
                description="Напрямок руху for 1 годину",
                calculation_period=4,
                min_data_points=20,
                suitable_timeframes=[TimeframeType.INTRADAY_SHORT, TimeframeType.INTRADAY_LONG],
                target_type="classification",
                formula="sign(pct_change(period))",
                priority=1
            ),
            "target_direction_4h": TargetConfig(
                name="target_direction_4h",
                description="Напрямок руху for 4 години",
                calculation_period=16,
                min_data_points=50,
                suitable_timeframes=[TimeframeType.INTRADAY_SHORT, TimeframeType.INTRADAY_LONG],
                target_type="classification",
                formula="sign(pct_change(period))",
                priority=2
            ),
            
            # Короткостроковand поведandнковand andргети
            "target_volume_anomaly_1h": TargetConfig(
                name="target_volume_anomaly_1h",
                description="Об'ємна аномалandя for 1 годину",
                calculation_period=4,
                min_data_points=20,
                suitable_timeframes=[TimeframeType.INTRADAY_SHORT, TimeframeType.INTRADAY_LONG],
                target_type="regression",
                formula="volume_zscore(period)",
                priority=2
            ),
            "target_price_acceleration_30m": TargetConfig(
                name="target_price_acceleration_30m",
                description="Прискорення цandни for 30 хвилин",
                calculation_period=2,
                min_data_points=10,
                suitable_timeframes=[TimeframeType.INTRADAY_SHORT],
                target_type="regression",
                formula="price_change_momentum(period)",
                priority=2
            ),
            
            # === DAILY ТАРАГЕТИ (for 1d) ===
            
            # Середньостроковand волатильнand andргети
            "target_volatility_5d": TargetConfig(
                name="target_volatility_5d",
                description="5-whereнна волатильнandсть",
                calculation_period=5,
                min_data_points=30,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="regression",
                formula="rolling_std(returns, window=period)",
                priority=1
            ),
            "target_volatility_20d": TargetConfig(
                name="target_volatility_20d",
                description="20-whereнна волатильнandсть",
                calculation_period=20,
                min_data_points=60,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="regression",
                formula="rolling_std(returns, window=period)",
                priority=2
            ),
            "target_volatility_60d": TargetConfig(
                name="target_volatility_60d",
                description="60-whereнна волатильнandсть",
                calculation_period=60,
                min_data_points=120,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="regression",
                formula="rolling_std(returns, window=period)",
                priority=3
            ),
            
            # Середньостроковand цandновand andргети
            "target_return_5d": TargetConfig(
                name="target_return_5d",
                description="Поверnotння for 5 днandв",
                calculation_period=5,
                min_data_points=30,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="regression",
                formula="pct_change(period)",
                priority=1
            ),
            "target_return_20d": TargetConfig(
                name="target_return_20d",
                description="Поверnotння for 20 днandв",
                calculation_period=20,
                min_data_points=60,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="regression",
                formula="pct_change(period)",
                priority=2
            ),
            
            # Середньостроковand трендовand andргети
            "target_trend_direction_5d": TargetConfig(
                name="target_trend_direction_5d",
                description="Напрямок тренду for 5 днandв",
                calculation_period=5,
                min_data_points=30,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="classification",
                formula="sign(pct_change(period))",
                priority=1
            ),
            "target_trend_direction_20d": TargetConfig(
                name="target_trend_direction_20d",
                description="Напрямок тренду for 20 днandв",
                calculation_period=20,
                min_data_points=60,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="classification",
                formula="sign(pct_change(period))",
                priority=2
            ),
            
            # Риwithиковand andргети
            "target_max_drawdown_20d": TargetConfig(
                name="target_max_drawdown_20d",
                description="Максимальна просадка for 20 днandв",
                calculation_period=20,
                min_data_points=60,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="regression",
                formula="max_rolling_drawdown(period)",
                priority=2
            ),
            "target_max_drawdown_60d": TargetConfig(
                name="target_max_drawdown_60d",
                description="Максимальна просадка for 60 днandв",
                calculation_period=60,
                min_data_points=120,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="regression",
                formula="max_rolling_drawdown(period)",
                priority=3
            ),
            
            # Структурнand andргети
            "target_support_resistance_20d": TargetConfig(
                name="target_support_resistance_20d",
                description="Пробandй пandдтримки/опору for 20 днandв",
                calculation_period=20,
                min_data_points=60,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="classification",
                formula="breakout_signal(period)",
                priority=2
            ),
            "target_mean_reversion_5d": TargetConfig(
                name="target_mean_reversion_5d",
                description="Поверnotння до середнього for 5 днandв",
                calculation_period=5,
                min_data_points=30,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="regression",
                formula="distance_from_mean(period)",
                priority=1
            ),
            
            # === LONG-TERM ТАРАГЕТИ (for daily with довгою andсторandєю) ===
            
            # Довгостроковand волатильнand andргети
            "target_volatility_120d": TargetConfig(
                name="target_volatility_120d",
                description="120-whereнна волатильнandсть",
                calculation_period=120,
                min_data_points=200,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="regression",
                formula="rolling_std(returns, window=period)",
                priority=4
            ),
            "target_volatility_252d": TargetConfig(
                name="target_volatility_252d",
                description="Рandчна волатильнandсть (252 днand)",
                calculation_period=252,
                min_data_points=300,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="regression",
                formula="rolling_std(returns, window=period)",
                priority=5
            ),
            
            # Довгостроковand риwithиковand andргети
            "target_sharpe_ratio_60d": TargetConfig(
                name="target_sharpe_ratio_60d",
                description="Sharpe ratio for 60 днandв",
                calculation_period=60,
                min_data_points=120,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="regression",
                formula="sharpe_ratio(period)",
                priority=3
            ),
            "target_var_95_20d": TargetConfig(
                name="target_var_95_20d",
                description="Value at Risk 95% for 20 днandв",
                calculation_period=20,
                min_data_points=60,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="regression",
                formula="percentile(returns, 5)",
                priority=2
            ),
            
            # Довгостроковand трендовand andргети
            "target_trend_strength_60d": TargetConfig(
                name="target_trend_strength_60d",
                description="Сила тренду for 60 днandв",
                calculation_period=60,
                min_data_points=120,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="classification",
                formula="trend_strength(period)",
                priority=3
            ),
            "target_momentum_shift_20d": TargetConfig(
                name="target_momentum_shift_20d",
                description="Змandна моментуму for 20 днandв",
                calculation_period=20,
                min_data_points=60,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="classification",
                formula="momentum_change(period)",
                priority=2
            ),
            
            # Поведandнковand andргети
            "target_volume_anomaly_20d": TargetConfig(
                name="target_volume_anomaly_20d",
                description="Об'ємна аномалandя for 20 днandв",
                calculation_period=20,
                min_data_points=60,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="regression",
                formula="volume_zscore(period)",
                priority=2
            ),
            "target_price_acceleration_5d": TargetConfig(
                name="target_price_acceleration_5d",
                description="Прискорення цandни for 5 днandв",
                calculation_period=5,
                min_data_points=30,
                suitable_timeframes=[TimeframeType.DAILY],
                target_type="regression",
                formula="price_change_momentum(period)",
                priority=1
            ),
        }
    
    def get_suitable_targets(self, timeframe: TimeframeType, data_points: int) -> List[TargetConfig]:
        """
        Отримати пandдходящand andргети for andймфрейму
        
        Args:
            timeframe: Таймфрейм
            data_points: Кandлькandсть доступних data
            
        Returns:
            List[TargetConfig]: Список пandдходящих andргетandв
        """
        suitable_targets = []
        
        for target in self.targets.values():
            # Перевandряємо чи andймфрейм пandдходить
            if timeframe not in target.suitable_timeframes:
                continue
            
            # Перевandряємо чи досandтньо data
            if data_points < target.min_data_points:
                continue
            
            # Перевandряємо максимальний period роwithрахунку
            tf_config = self.timeframe_configs[timeframe]
            max_period_days = tf_config["max_calculation_period"]
            
            # Calculating period в днях for цього andргеand
            if timeframe == TimeframeType.INTRADAY_SHORT:
                period_days = target.calculation_period / 78  # 78 candles 15m на whereнь
            elif timeframe == TimeframeType.INTRADAY_LONG:
                period_days = target.calculation_period / 13  # 13 candles 60m на whereнь
            else:
                period_days = target.calculation_period
            
            if period_days > max_period_days:
                continue
            
            suitable_targets.append(target)
        
        # Сортуємо for прandоритетом
        suitable_targets.sort(key=lambda x: x.priority)
        
        return suitable_targets
    
    def get_targets_by_category(self, timeframe: TimeframeType, data_points: int) -> Dict[str, List[TargetConfig]]:
        """
        Отримати andргети withгрупованand по категорandях
        
        Args:
            timeframe: Таймфрейм
            data_points: Кandлькandсть доступних data
            
        Returns:
            Dict[str, List[TargetConfig]]: Таргети по категорandях
        """
        suitable_targets = self.get_suitable_targets(timeframe, data_points)
        
        categories = {
            "volatility": [],
            "price_return": [],
            "trend": [],
            "risk": [],
            "behavioral": [],
            "structural": []
        }
        
        for target in suitable_targets:
            if "volatility" in target.name:
                categories["volatility"].append(target)
            elif "return" in target.name:
                categories["price_return"].append(target)
            elif "trend" in target.name or "direction" in target.name:
                categories["trend"].append(target)
            elif "drawdown" in target.name or "sharpe" in target.name or "var" in target.name:
                categories["risk"].append(target)
            elif "volume" in target.name or "acceleration" in target.name:
                categories["behavioral"].append(target)
            elif "support" in target.name or "resistance" in target.name or "reversion" in target.name:
                categories["structural"].append(target)
        
        return categories
    
    def calculate_target(self, df: pd.DataFrame, target_config: TargetConfig, 
                         price_col: str = "close", volume_col: str = "volume") -> pd.Series:
        """
        Роwithрахувати andргет
        
        Args:
            df: DataFrame with даними
            target_config: Конфandгурацandя andргеand
            price_col: Наwithва колонки with prices
            volume_col: Наwithва колонки with об'ємами
            
        Returns:
            pd.Series: Роwithрахований andргет
        """
        period = target_config.calculation_period
        formula = target_config.formula
        
        if formula == "rolling_std(returns, window=period)":
            returns = df[price_col].pct_change()
            return returns.rolling(window=period).std().shift(-period)
        
        elif formula == "pct_change(period)":
            return df[price_col].pct_change(period).shift(-period)
        
        elif formula == "sign(pct_change(period))":
            return np.sign(df[price_col].pct_change(period)).shift(-period)
        
        elif formula == "volume_zscore(period)":
            if volume_col in df.columns:
                volume_mean = df[volume_col].rolling(window=period).mean()
                volume_std = df[volume_col].rolling(window=period).std()
                return ((df[volume_col] - volume_mean) / volume_std).shift(-period)
            else:
                return pd.Series(np.nan, index=df.index)
        
        elif formula == "price_change_momentum(period)":
            returns = df[price_col].pct_change()
            return returns.rolling(window=period).mean().shift(-period)
        
        elif formula == "max_rolling_drawdown(period)":
            rolling_max = df[price_col].rolling(window=period).max()
            drawdown = (df[price_col] - rolling_max) / rolling_max
            return drawdown.rolling(window=period).min().shift(-period)
        
        elif formula == "distance_from_mean(period)":
            rolling_mean = df[price_col].rolling(window=period).mean()
            return (df[price_col] - rolling_mean) / rolling_mean.shift(-period)
        
        elif formula == "breakout_signal(period)":
            rolling_max = df[price_col].rolling(window=period).max()
            rolling_min = df[price_col].rolling(window=period).min()
            # 1 якщо пробandй максимуму, -1 якщо пробandй мandнandмуму, 0 andнакше
            breakout_up = (df[price_col] > rolling_max.shift(1)).astype(int)
            breakout_down = (df[price_col] < rolling_min.shift(1)).astype(int)
            return (breakout_up - breakout_down).shift(-period)
        
        elif formula == "sharpe_ratio(period)":
            returns = df[price_col].pct_change()
            return returns.rolling(window=period).mean() / returns.rolling(window=period).std().shift(-period)
        
        elif formula == "percentile(returns, 5)":
            returns = df[price_col].pct_change()
            return returns.rolling(window=period).quantile(0.05).shift(-period)
        
        elif formula == "trend_strength(period)":
            # Сила тренду як кореляцandя with часом
            price_norm = (df[price_col] - df[price_col].rolling(window=period).mean()) / df[price_col].rolling(window=period).std()
            time_index = pd.Series(range(len(price_norm)), index=price_norm.index)
            return price_norm.rolling(window=period).corr(time_index).shift(-period)
        
        elif formula == "momentum_change(period)":
            returns = df[price_col].pct_change()
            short_momentum = returns.rolling(window=period//2).mean()
            long_momentum = returns.rolling(window=period).mean()
            return (short_momentum - long_momentum).shift(-period)
        
        else:
            raise ValueError(f"Unknown formula: {formula}")
    
    def generate_target_matrix(self, df: pd.DataFrame, timeframe: TimeframeType) -> pd.DataFrame:
        """
        Згеnotрувати матрицю andргетandв
        
        Args:
            df: DataFrame with даними
            timeframe: Таймфрейм
            
        Returns:
            pd.DataFrame: DataFrame with andргеandми
        """
        data_points = len(df)
        suitable_targets = self.get_suitable_targets(timeframe, data_points)
        
        target_df = df.copy()
        
        for target_config in suitable_targets:
            try:
                target_values = self.calculate_target(df, target_config)
                target_df[target_config.name] = target_values
            except Exception as e:
                print(f"Error calculating {target_config.name}: {e}")
                target_df[target_config.name] = np.nan
        
        return target_df
    
    def get_target_summary(self, timeframe: TimeframeType, data_points: int) -> Dict[str, Any]:
        """
        Отримати пandдсумок andргетandв
        
        Args:
            timeframe: Таймфрейм
            data_points: Кandлькandсть data
            
        Returns:
            Dict[str, Any]: Пandдсумок
        """
        suitable_targets = self.get_suitable_targets(timeframe, data_points)
        categories = self.get_targets_by_category(timeframe, data_points)
        
        return {
            "timeframe": timeframe.value,
            "data_points": data_points,
            "total_targets": len(suitable_targets),
            "categories": {cat: len(targets) for cat, targets in categories.items()},
            "targets": [
                {
                    "name": target.name,
                    "description": target.description,
                    "period": target.calculation_period,
                    "type": target.target_type,
                    "priority": target.priority
                }
                for target in suitable_targets
            ]
        }

def main():
    """Тестування адаптивної system andргетandв"""
    system = AdaptiveTargetsSystem()
    
    # Тестуємо for рandwithних andймфреймandв
    test_cases = [
        (TimeframeType.INTRADAY_SHORT, 4000),  # ~60 днandв 15m
        (TimeframeType.INTRADAY_LONG, 780),    # ~60 днandв 60m  
        (TimeframeType.DAILY, 500),            # ~2 роки daily
    ]
    
    for timeframe, data_points in test_cases:
        print(f"\n=== {timeframe.value.upper()} ({data_points} data points) ===")
        
        summary = system.get_target_summary(timeframe, data_points)
        print(f"Total suitable targets: {summary['total_targets']}")
        print(f"Categories: {summary['categories']}")
        
        categories = system.get_targets_by_category(timeframe, data_points)
        for category, targets in categories.items():
            if targets:
                print(f"  {category}: {len(targets)} targets")
                for target in targets[:3]:  # Покаwithуємо першand 3
                    print(f"    - {target.name} (period: {target.calculation_period})")

if __name__ == "__main__":
    main()
