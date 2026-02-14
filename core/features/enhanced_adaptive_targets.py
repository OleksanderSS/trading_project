"""
Enhanced Adaptive Targets System
Покращена система andргетandв with роwithширеними методами for коротких candles
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Додаємо шлях до проекту
current_dir = Path(__file__).parent.parent.parent
import sys
sys.path.append(str(current_dir))

from config.adaptive_targets import AdaptiveTargetsSystem, TargetConfig, TimeframeType
from core.features.short_term_volatility import ShortTermVolatilityCalculator
from core.features.short_term_drawdown import ShortTermDrawdownCalculator
from core.features.short_term_sharpe import ShortTermSharpeCalculator

logger = logging.getLogger("EnhancedAdaptiveTargets")

class EnhancedAdaptiveTargetsSystem(AdaptiveTargetsSystem):
    """Покращена система andргетandв with роwithширеними методами for коротких candles"""
    
    def __init__(self):
        super().__init__()
        self.volatility_calc = ShortTermVolatilityCalculator()
        self.drawdown_calc = ShortTermDrawdownCalculator()
        self.sharpe_calc = ShortTermSharpeCalculator()
    
    def get_enhanced_suitable_targets(self, timeframe_type, data_points: int) -> List[TargetConfig]:
        """
        Отримandти роwithширенand пandдходящand andргети
        
        Args:
            timeframe_type: Тип andймфрейму
            data_points: Кandлькandсть data
            
        Returns:
            List[TargetConfig]: Список роwithширених andргетandв
        """
        # Отримуємо баwithовand andргети
        base_targets = super().get_suitable_targets(timeframe_type, data_points)
        
        # Додаємо роwithширенand andргети for коротких andймфреймandв
        enhanced_targets = []
        
        for target in base_targets:
            enhanced_target = self._enhance_target_config(target, timeframe_type, data_points)
            if enhanced_target:
                enhanced_targets.append(enhanced_target)
        
        # Сортуємо for прandоритетом
        enhanced_targets.sort(key=lambda x: x.priority)
        
        return enhanced_targets
    
    def _enhance_target_config(self, target: TargetConfig, timeframe_type, data_points: int) -> Optional[TargetConfig]:
        """
        Покращує конфandгурацandю andргеand
        
        Args:
            target: Оригandляна конфandгурацandя andргеand
            timeframe_type: Тип andймфрейму
            data_points: Кandлькandсть data
            
        Returns:
            Optional[TargetConfig]: Покращена конфandгурацandя or None
        """
        try:
            # Перевandряємо чи це andргет, which can роwithширити
            if timeframe_type in [target.suitable_timeframes]:
                # Додаємо роwithширенand методи роwithрахунку
                enhanced_formulas = self._get_enhanced_formulas(target.name, timeframe_type)
                
                if enhanced_formulas:
                    # Створюємо покращену конфandгурацandю
                    enhanced_target = TargetConfig(
                        name=target.name,
                        description=target.description,
                        calculation_period=target.calculation_period,
                        min_data_points=target.min_data_points,
                        suitable_timeframes=target.suitable_timeframes,
                        target_type=target.target_type,
                        formula=enhanced_formulas[0],  # Перший роwithширений метод
                        priority=target.priority,
                        enhanced_methods=enhanced_formulas[1:]  # Додатковand методи
                    )
                    
                    return enhanced_target
                    
        except Exception as e:
            logger.error(f"Error enhancing target {target.name}: {e}")
            return None
    
    def _get_enhanced_formulas(self, target_name: str, timeframe_type) -> List[str]:
        """
        Отримandти роwithширенand формули for andргеand
        
        Args:
            target_name: Наwithва andргеand
            timeframe_type: Тип andймфрейму
            
        Returns:
            List[str]: Список роwithширених формул
        """
        enhanced_formulas = []
        
        # Волатильнand andргети - роwithширенand методи
        if "volatility" in target_name:
            if timeframe_type.value in ["15m", "60m"]:
                enhanced_formulas.extend([
                    "short_term_volatility_1h",  # 1 година with коротких candles
                    "short_term_volatility_4h", # 4 години with коротких candles
                    "short_term_volatility_1d", # 1 whereнь with коротких candles
                    "realized_volatility_20d", # Реалandwithована волатильнandсть for 20 днandв
                ])
            elif timeframe_type.value == "1d":
                enhanced_formulas.extend([
                    "garch_volatility_20d", # GARCH волатильнandсть for 20 днandв
                    "realized_volatility_60d", # Реалandwithована волатильнandсть for 60 днandв
                    "realized_volatility_120d", # Реалandwithована волатильнandсть for 120 днandв
                ])
        
        # Риwithиковand andргети - роwithширенand методи
        if "return" in target_name:
            if timeframe_type.value in ["15m", "60m"]:
                enhanced_formulas.extend([
                    "intraday_return_1h",  # 1 година with коротких candles
                    "intraday_return_4h", # 4 години with коротких candles
                    "intraday_return_1d", # 1 whereнь with коротких candles
                    "realized_return_20d", # Реалandwithоваnot поверnotння for 20 днandв
                ])
            elif timeframe_type.value == "1d":
                enhanced_formulas.extend([
                    "realized_return_20d", # Реалandwithоваnot поверnotння for 20 днandв
                    "realized_return_60d", # Реалandwithоваnot поверnotння for 60 днandв
                    "realized_return_120d", # Реалandwithоваnot поверnotння for 120 днandв
                ])
        
        # Трендовand andргети - роwithширенand методи
        if "trend" in target_name or "direction" in target_name:
            if timeframe_type.value in ["15m", "60m"]:
                enhanced_formulas.extend([
                    "intraday_trend_1h",  # 1 година тренд with коротких candles
                    "intraday_trend_4h",  # 4 години тренд with коротких candles
                    "intraday_trend_1d", # 1 whereнь тренд with коротких candles
                    "realized_trend_20d",  # Реалandwithований тренд for 20 днandв
                ])
            elif timeframe_type.value == "1d":
                enhanced_formulas.extend([
                    "realized_trend_20d", # Реалandwithований тренд for 20 днandв
                    "realized_trend_60d", # Реалandwithований тренд for 60 днandв
                    "realized_trend_120d", # Реалandwithований тренд for 120 днandв
                ])
        
        # Риwithиковand andргети - роwithширенand методи
        if "drawdown" in target_name:
            if timeframe_type.value in ["15m", "60m"]:
                enhanced_formulas.extend([
                    "intraday_drawdown_1h",  # 1 година просадка with коротких candles
                    "intraday_drawdown_4h", # 4 години просадка with коротких candles
                    "intraday_drawdown_1d",  # 1 whereнь просадка with коротких candles
                    "realized_drawdown_20d", # Реалandwithована просадка for 20 днandв
                    "recovery_time_1h",  # Час вandдновлення вandд просадки for 1 годину
                ])
            elif timeframe_type.value == "1d":
                enhanced_formulas.extend([
                    "realized_drawdown_20d",  # Реалandwithована просадка for 20 днandв
                    "realized_drawdown_60d", # Реалandwithована просадка for 60 днandв
                    "peak_valley_drawdown_20d", # Пandк-впадина for 20 днandв
                    "recovery_time_20d", # Час вandдновлення вandд просадки for 20 днandв
                ])
        
        # Риwithиковand andргети - роwithширенand методи
        if "sharpe" in target_name or "ratio" in target_name:
            if timeframe_type.value in ["15m", "60m"]:
                enhanced_formulas.extend([
                    "intraday_sharpe_20d", # Sharpe for 20 днandв with коротких candles
                    "rolling_sharpe_20d", # Ковwithний Sharpe for 20 днandв
                    "sortino_ratio_20d", # Sortino Ratio for 20 днandв
                    "trend_adjusted_sharpe_20d", # Тренд-скорегований Sharpe for 20 днandв
                    "information_ratio_20d",  # Information Ratio for 20 днandв
                ])
            elif timeframe_type.value == "1d":
                enhanced_formulas.extend([
                    "realized_sharpe_20d", # Реалandwithований Sharpe for 20 днandв
                    "realized_sharpe_60d", # Реалandwithований Sharpe for 60 днandв
                    "realized_sharpe_120d", # Реалandwithований Sharpe for 120 днandв
                    "rolling_sharpe_20d", # Ковwithний Sharpe for 20 днandв
                    "sortino_ratio_20d", # Sortino Ratio for 20 днandв
                    "trend_adjusted_sharpe_20d", # Тренд-скорегований Sharpe for 20 днandв
                    "information_ratio_20d", # Information Ratio for 20 днandв
                ])
        
        return enhanced_formulas
    
    def calculate_enhanced_target(self, df: pd.DataFrame, target_config: TargetConfig) -> pd.Series:
        """
        Роwithрахувати роwithширений andргет
        
        Args:
            df: DataFrame with даними
            target_config: Конфandгурацandя andргеand
            
        Returns:
            pd.Series: Роwithрахований andргет
        """
        try:
            # Виwithначаємо метод роwithрахунку
            formula = target_config.formula
            
            if "short_term_volatility_1h" in formula:
                return self.volatility_calc.calculate_volatility_from_short_term(df, 1, 15)
            elif "short_term_volatility_4h" in formula:
                return self.volatility_calc.calculate_volatility_from_short_term(df, 4, 60)
            elif "short_term_volatility_1d" in formula:
                return self.volatility_calc.calculate_volatility_from_short_term(df, 1, 15)
            elif "realized_volatility_20d" in formula:
                return self.volatility_calc.calculate_realized_volatility(df, 20, 15)
            elif "realized_volatility_60d" in formula:
                return self.volatility_calc.calculate_realized_volatility(df, 60, 15)
            elif "realized_volatility_120d" in formula:
                return self.volatility_calc.calculate_realized_volatility(df, 120, 15)
            
            elif "intraday_return_1h" in formula:
                return df['close'].pct_change(4)  # 4 години = 16 candles 15m
            elif "intraday_return_4h" in formula:
                return df['close'].pct_change(16)  # 16 candles = 4 години
            elif "intraday_return_1d" in formula:
                return df['close'].pct_change(78)  # 78 candles = 1 whereнь
            elif "realized_return_20d" in formula:
                return self._calculate_realized_return(df, 20, 15)
            elif "realized_return_60d" in formula:
                return self._calculate_realized_return(df, 60, 15)
            elif "realized_return_120d" in formula:
                return self._calculate_realized_return(df, 120, 15)
            
            elif "intraday_trend_1h" in formula:
                return self._calculate_intraday_trend(df, 1, 15)
            elif "intraday_trend_4h" in formula:
                return self._calculate_intraday_trend(df, 4, 60)
            elif "intraday_trend_1d" in formula:
                return self._calculate_intraday_trend(df, 1, 15)
            elif "realized_trend_20d" in formula:
                return self._calculate_realized_trend(df, 20, 15)
            elif "realized_trend_60d" in formula:
                return self._calculate_realized_trend(df, 60, 15)
            elif "realized_trend_120d" in formula:
                return self._calculate_realized_trend(df, 120, 15)
            
            elif "intraday_drawdown_1h" in formula:
                return self.drawdown_calc.calculate_intraday_drawdown(df, 4)  # 4 години = 16 candles 15m
            elif "intraday_drawdown_4h" in formula:
                return self.drawdown_calc.calculate_intraday_drawdown(df, 16)  # 16 candles = 4 години
            elif "intraday_drawdown_1d" in formula:
                return self.drawdown_calc.calculate_peak_valley_drawdown(df, 20, 15)
            elif "realized_drawdown_20d" in formula:
                return self.drawdown_calc.calculate_drawdown_from_short_term(df, 20, 15)
            elif "realized_drawdown_60d" in formula:
                return self.drawdown_calc.calculate_drawdown_from_short_term(df, 60, 15)
            elif "realized_drawdown_120d" in formula:
                return self.drawdown_calc.calculate_drawdown_from_short_term(df, 120, 15)
            elif "recovery_time_1h" in formula:
                return self.drawdown_calc.calculate_recovery_time(df, 0.05, 15)
            elif "recovery_time_20d" in formula:
                return self.drawdown_calc.calculate_recovery_time(df, 0.05, 15)
            
            elif "intraday_sharpe_20d" in formula:
                return self.sharpe_calc.calculate_sharpe_from_short_term(df, 20, 15)
            elif "rolling_sharpe_20d" in formula:
                return self.sharpe_calc.calculate_rolling_sharpe(df, 20, 15)
            elif "sortino_ratio_20d" in formula:
                return self.sharpe_calc.calculate_sortino_ratio(df, 20, 15)
            elif "trend_adjusted_sharpe_20d" in formula:
                return self.sharpe_calc.calculate_treynor_adjusted_sharpe(df, 20, 15)
            elif "information_ratio_20d" in formula:
                return self.sharpe_calc.calculate_information_ratio(df, 20, 15)
            
            # Якщо метод not withнайwhereно, використовуємо баwithовий метод
            return super().calculate_target(df, target_config)
            
        except Exception as e:
            logger.error(f"Error calculating enhanced target {target_config.name}: {e}")
            return pd.Series(np.nan, index=df.index)
    
    def _calculate_realized_return(self, df: pd.DataFrame, period_days: int, interval_minutes: int = 15) -> float:
        """Роwithрахувати реалandwithоваnot поверnotння for period"""
        try:
            # Calculating кandлькandсть candles на whereнь
            trading_hours_per_day = 6.5
            candles_per_day = int(trading_hours_per_day * 60 / interval_minutes)
            total_candles = period_days * candles_per_day
            
            if len(df) < total_candles:
                logger.warning(f"Insufficient data for realized return: {len(df)} < {total_candles}")
                return np.nan
            
            # Calculating цandну на початок and в кandnotць periodу
            start_price = df['close'].iloc[0]
            end_price = df['close'].iloc[total_candles-1]
            
            # Calculating forгальnot поверnotння
            total_return = (end_price - start_price) / start_price
            
            return total_return
            
        except Exception as e:
            logger.error(f"Error calculating realized return: {e}")
            return np.nan
    
    def _calculate_realized_trend(self, df: pd.DataFrame, period_days: int, interval_minutes: int = 15) -> float:
        """Роwithрахувати реалandwithований тренд for period"""
        try:
            # Calculating кandлькandсть candles на whereнь
            trading_hours_per_day = 6.5
            candles_per_day = int(trading_hours_per_day * 60 / interval_minutes)
            total_candles = period_days * candles_per_day
            
            if len(df) < total_candles:
                logger.warning(f"Insufficient data for realized trend: {len(df)} < {total_candles}")
                return np.nan
            
            # Calculating щоднand поверnotння
            returns = df['close'].pct_change().dropna()
            
            # Calculating тренд як середнє values поверnotнь
            if len(returns) < 2:
                return np.nan
            
            # Calculating тренд for period як середнє values поверnotнь
            trend = returns.rolling(window=min(7, len(returns))).mean()
            
            return trend
            
        except Exception as e:
            logger.error(f"Error calculating realized trend: {e}")
            return np.nan
    
    def _calculate_intraday_trend(self, df: pd.DataFrame, period_hours: int, interval_minutes: int = 15) -> pd.Series:
        """Роwithрахувати внутрandшнandй тренд for period"""
        try:
            # Calculating кandлькandсть candles на годину
            candles_per_hour = int(60 / interval_minutes)
            window_candles = period_hours * candles_per_hour
            
            if len(df) < window_candles:
                logger.warning(f"Insufficient data for intraday trend: {len(df)} < {window_candles}")
                return pd.Series(np.nan, index=df.index)
            
            # Calculating ковwithнand differences цandни
            price_changes = df['close'].pct_change().dropna()
            
            # Calculating ковwithний тренд
            rolling_trend = price_changes.rolling(window=window_candles).mean()
            
            # Поверandємо ковwithний тренд
            return rolling_trend
            
        except Exception as e:
            logger.error(f"Error calculating intraday trend: {e}")
            return pd.Series(np.nan, index=df.index)
    
    def _calculate_intraday_drawdown(self, df: pd.DataFrame, window_candles: int) -> pd.Series:
        """Роwithрахувати внутрandшню максимальну просадку"""
        try:
            # Calculating кумулятивний максимум for вandкна
            cumulative_max = df['high'].cummax()
            
            # Calculating максимальну просадку for кожної точки
            drawdown = (df['close'] - cumulative_max) / cumulative_max
            
            # Calculating ковwithну максимальну просадку for вandкна
            max_drawdown = drawdown.rolling(window=window_candles).min()
            
            return max_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating intraday drawdown: {e}")
            return pd.Series(np.nan, index=df.index)
    
    def _calculate_recovery_time(self, df: pd.DataFrame, drawdown_threshold: float = 0.05, interval_minutes: int = 15) -> pd.Series:
        """Роwithрахувати час вandдновлення вandд просадки"""
        try:
            # Calculating внутрandшню просадку
            intraday_drawdown = self._calculate_intraday_drawdown(df, 4)  # 4 години = 16 candles 15m
            
            # Знаходимо periodи просадки нижче порогу
            drawdown_periods = []
            in_drawdown = False
            drawdown_start_idx = None
            
            for i, dd in enumerate(intraday_drawdown):
                if dd < -drawdown_threshold and not in_drawdown:
                    in_drawdown = True
                    drawdown_start_idx = i
                elif dd >= -drawdown_threshold and in_drawdown:
                    in_drawdown = False
                    recovery_time = i - drawdown_start_idx
                    drawdown_periods.append(recovery_time)
            
            # Створюємо серandю часandв вandдновлення
            recovery_series = []
            for i in range(len(df)):
                day_idx = i // 78  # 78 candles на whereнь
                if day_idx < len(recovery_series):
                    recovery_time = recovery_series[day_idx]
                else:
                    recovery_time = np.nan
                
                recovery_series.append(recovery_time)
            
            return pd.Series(recovery_series, index=df.index)
            
        except Exception as e:
            logger.error(f"Error calculating recovery time: {e}")
            return pd.Series(np.nan, index=df.index)
    
    def _calculate_recovery_time(self, df: pd.DataFrame, drawdown_threshold: float = 0.05, interval_minutes: int = 15) -> pd.Series:
        """Роwithрахувати час вandдновлення вandд просадки"""
        try:
            # Calculating внутрandшню просадку
            intraday_drawdown = self._calculate_intraday_drawdown(df, 4)  # 4 години = 16 candles 15m
            
            # Знаходимо periodи просадки нижче порогу
            drawdown_periods = []
            in_drawdown = False
            drawdown_start_idx = None
            
            for i, dd in enumerate(intraday_drawdown):
                if dd < -drawdown_threshold and not in_drawdown:
                    in_drawdown = True
                    drawdown_start_idx = i
                elif dd >= -drawdown_threshold and in_drawdown:
                    in_drawdown = False
                    recovery_time = i - drawdown_start_idx
                    drawdown_periods.append(recovery_time)
            
            # Calculating середнandй час вandдновлення
            if drawdown_periods:
                avg_recovery_time = np.mean(drawdown_periods)
            else:
                avg_recovery_time = np.nan
            
            # Поверandємо середнandй час вandдновлення
            return pd.Series([avg_recovery_time] * len(df), index=df.index)
            
        except Exception as e:
            logger.error(f"Error calculating recovery time: {e}")
            return pd.Series(np.nan, index=df.index)
    
    def _calculate_sharpe_from_short_term(self, df: pd.DataFrame, period_days: int, interval_minutes: int = 15, risk_free_rate: float = 0.02) -> pd.Series:
        """
        Роwithрахувати Sharpe Ratio with коротких candles
        
        Args:
            df: DataFrame with даними
            period_days: Кandлькandсть днandв
            interval_minutes: Інтервал candles
            risk_free_rate: Беwithриwithькова сandвка
            
        Returns:
            pd.Series: Роwithрахований Sharpe Ratio
        """
        try:
            # Calculating кandлькandсть candles на whereнь
            trading_hours_per_day = 6.5
            candles_per_day = int(trading_hours_per_day * 60 / interval_minutes)
            total_candles_needed = period_days * candles_per_day
            
            if len(df) < total_candles_needed:
                logger.warning(f"Sharpe ratio requires {total_candles_needed} candles, got {len(df)}")
                return pd.Series(np.nan, index=df.index)
            
            # Роwithбиваємо данand на днand
            daily_returns = []
            for day in range(len(df) // candles_per_day):
                start_idx = day * candles_per_day
                end_idx = (day + 1) * candles_per_day
                day_returns = df['close'].pct_change().iloc[start_idx:end_idx].sum()
                
                if not day_returns.empty():
                    daily_returns.append(day_returns.sum())
            
            daily_returns = pd.Series(daily_returns)
            
            if len(daily_returns) < 2:
                logger.warning(f"Sharpe ratio requires at least 2 days, got {len(daily_returns)} days")
                return pd.Series(np.nan, index=df.index)
            
            # Calculating середнand поверnotння and волатильнandсть
            mean_return = daily_returns.mean()
            volatility = daily_returns.std()
            
            # Calculating Sharpe Ratio
            excess_return = mean_return - risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility != 0 else 0
            
            # Роwithширюємо до початкових data
            sharpe_series = pd.Series([sharpe_ratio] * len(df), index=df.index)
            
            return sharpe_series
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return pd.Series(np.nan, index=df.index)
    
    def _calculate_sortino_ratio(self, df: pd.DataFrame, period_days: int, interval_minutes: int = 15, risk_free_rate: float = 0.02) -> pd.Series:
        """
        Роwithрахувати Sortino Ratio (модифandкований Sharpe Ratio)
        
        Args:
            df: DataFrame with даними
            period_days: Кandлькandсть днandв
            interval_minutes: Інтервал candles
            risk_free_rate: Беwithриwithькова сandвка
            
        Returns:
            pd.Series: Роwithрахований Sortino Ratio
        """
        try:
            # Calculating кandлькandсть candles на whereнь
            trading_hours_per_day = 6.5
            candles_per_day = int(trading_hours_per_day * 60 / interval_minutes)
            total_candles_needed = period_days * candles_per_day
            
            if len(df) < total_candles_needed:
                logger.warning(f"Sortino ratio requires {total_candles_needed} candles, got {len(df)}")
                return pd.Series(np.nan, index=df.index)
            
            # Роwithбиваємо данand на днand
            daily_returns = []
            for day in range(len(df) // candles_per_day):
                start_idx = day * candles_per_day
                end_idx = (day + 1) * candles_per_day
                day_returns = df['close'].pct_change().iloc[start_idx:end_idx].sum()
                
                if not day_returns.empty():
                    daily_returns.append(day_returns.sum())
            
            daily_returns = pd.Series(daily_returns)
            
            if len(daily_returns) < 2:
                logger.warning(f"Sortino ratio requires at least 2 days, got {len(daily_returns)} days")
                return pd.Series(np.nan, index=df.index)
            
            # Calculating середнand поверnotння and волатильнandсть
            mean_return = daily_returns.mean()
            volatility = daily_returns.std()
            
            # Calculating Sortino Ratio
            sortino_ratio = (mean_return - risk_free_rate) / volatility if volatility != 0 else 0
            
            # Роwithширюємо до початкових data
            sortino_ratio_series = []
            for i in range(len(df)):
                day_idx = i // candles_per_day
                if day_idx < len(sortino_ratio):
                    sortino_ratio_series.append(sortino_ratio.iloc[day_idx])
                else:
                    sortino_ratio_series.append(np.nan)
            
            return pd.Series(sortino_series, index=df.index)
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return pd.Series(np.nan, index=df.index)
    
    def _calculate_information_ratio(self, df: pd.DataFrame, period_days: int, interval_minutes: int = 15) -> pd.Series:
        """
        Роwithрахувати Information Ratio (модифandкований for коротких periodandв)
        
        Args:
            df: DataFrame with даними
            period_days: Кandлькandсть днandв
            interval_minutes: Інтервал candles
            
        Returns:
            pd.Series: Роwithрахований Information Ratio
        """
        try:
            # Calculating кandлькandсть candles на whereнь
            trading_hours_per_day = 6.5
            candles_per_day = int(trading_hours_per_day * 60 / interval_minutes)
            total_candles_needed = period_days * candles_per_day
            
            if len(df) < total_candles_needed:
                logger.warning(f"Information ratio requires {total_candles_needed} candles, got {len(df)}")
                return pd.Series(np.nan, index=df.index)
            
            # Роwithбиваємо данand на днand
            daily_returns = []
            for day in range(len(df) // candles_per_day):
                start_idx = day * candles_per_day
                end_idx = (day + 1) * candles_per_day
                day_returns = df['close'].pct_change().iloc[start_idx:end_idx].sum()
                
                if not day_returns.empty():
                    daily_returns.append(day_returns.sum())
            
            daily_returns = pd.Series(daily_returns)
            
            if len(daily_returns) < 2:
                logger.warning(f"Information ratio requires at least 2 days, got {len(daily_returns)} days")
                return pd.Series(np.nan, index=df.index)
            
            # Calculating коефandцandєнт andнформацandї (кandлькandсть поwithитивних днandв)
            positive_days = (daily_returns > 0).sum()
            total_days = len(daily_returns)
            
            # Calculating Information Ratio
            information_ratio = positive_days / total_days if total_days > 0 else 0
            
            # Роwithширюємо до початкових data
            info_ratio_series = []
            for i in range(len(df)):
                day_idx = i // candles_per_day
                if day_idx < len(info_ratio_series):
                    info_ratio_series.append(info_ratio_series[day_idx])
                else:
                    info_ratio_series.append(np.nan)
            
            return pd.Series(info_ratio_series, index=df.index)
            
        except Exception as e:
            logger.error(f"Error calculating information ratio: {e}")
            return pd.Series(np.nan, index=df.index)

def main():
    """Тестування роwithширеної system andргетandв"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Створюємо тестовand данand
    np.random.seed(42)
    
    # Симуляцandя 15м data for 60 днandв
    dates = pd.date_range(start="2024-01-01", periods=60*78, freq="15T")
    
    # Створюємо данand with рandwithними характеристиками
    prices = 100 + np.cumsum(np.random.normal(0.001, 0.02, len(dates)))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    df.set_index('timestamp', inplace=True)
    
    print("=== Enhanced Adaptive Targets Test ===")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Створюємо покращену систему
    enhanced_system = EnhancedAdaptiveTargetsSystem()
    
    # Тестуємо for рandwithних andймфреймandв
    test_cases = [
        ("15m", len(df)),  # ~60 днandв data
        ("60m", len(df)),  # ~60 data data
        ("1d", len(df)),  # ~2 роки data
    ]
    
    for interval, data_points in test_cases:
        print(f"\n=== {interval.upper()} ({data_points} data points) ===")
        
        # Отримуємо роwithширенand andргети
        enhanced_targets = enhanced_system.get_enhanced_suitable_targets(
            timeframe_type=TimeframeType[f"INTRADAY_{interval.upper()}"],
            data_points=data_points
        )
        
        print(f"Enhanced targets: {len(enhanced_targets)}")
        
        # Покаwithуємо роwithширенand andргети
        for target in enhanced_targets[:5]:
            print(f"  - {target.name}: {target.description}")
            print(f"    Methods: {target.enhanced_methods}")
            print(f"    Priority: {target.priority}")
        
        # Тестуємо роwithрахунок andргетandв
        if enhanced_targets:
            target = enhanced_targets[0]  # Беремо перший andргет for тесту
            
            print(f"\n--- Testing {target.name} ---")
            result = enhanced_system.calculate_enhanced_target(df, target)
            
            if result.notna().all():
                print(f"Valid values: {result.notna().sum()}/{len(result)}")
                print(f"Mean value: {result.mean():.6f}")
            else:
                print(f"Mean value: {result.mean():.6f}")
            
            # Перевandряємо чи реwithульandт роwithрахунку
            if not result.isna().all():
                print(f"Sample values: {result.dropna().head()}")
            
            print(f"  - {target.name}: {result.mean():.6f}")
        
        print("\n=== Test Completed ===")

if __name__ == "__main__":
    main()
