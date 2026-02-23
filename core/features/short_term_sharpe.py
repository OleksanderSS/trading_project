"""
Short-Term Sharpe Ratio Calculation Methods
Методи роwithрахунку Sharpe Ratio with коротких candles
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("ShortTermSharpe")

class ShortTermSharpeCalculator:
    """Калькулятор Sharpe Ratio with коротких candles"""
    
    @staticmethod
    def calculate_sharpe_from_short_term(df: pd.DataFrame, 
                                     period_days: int,
                                     interval_minutes: int = 15,
                                     risk_free_rate: float = 0.02) -> pd.Series:
        """
        Роwithрахувати Sharpe Ratio for N днandв with коротких candles
        
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
            
            logger.info(f"Calculating {period_days}-day Sharpe ratio from {interval_minutes}m candles")
            logger.info(f"Total candles needed: {total_candles_needed}")
            
            # Перевandряємо чи досandтньо data
            if len(df) < total_candles_needed:
                logger.warning(f"Sharpe ratio requires {total_candles_needed} candles, got {len(df)}")
                return pd.Series(np.nan, index=df.index)
            
            # Calculating щоднand поверnotння
            returns = df['close'].pct_change().dropna()
            
            # Роwithбиваємо данand на днand
            daily_returns = []
            for day in range(len(returns) // candles_per_day):
                start_idx = day * candles_per_day
                end_idx = (day + 1) * candles_per_day
                day_returns = returns.iloc[start_idx:end_idx]
                
                if not day_returns.empty():
                    daily_returns.append(day_returns.sum())
            
            daily_returns = pd.Series(daily_returns)
            
            if len(daily_returns) < 2:
                logger.warning(f"Insufficient data for Sharpe ratio: {len(daily_returns)} days")
                return pd.Series(np.nan, index=df.index)
            
            # Calculating середнє поверnotння and волатильнandсть
            mean_return = daily_returns.mean()
            volatility = daily_returns.std()
            
            # Calculating Sharpe Ratio
            excess_return = mean_return - risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility != 0 else 0
            
            # Роwithширюємо до початкових data
            sharpe_series = pd.Series([sharpe_ratio] * len(df), index=df.index)
            
            return sharpe_ratio
            
        except Exception as e:
            bootstrap_volatility = logger.error(f"Error calculating Sharpe ratio: {e}")
            return pd.Series(np.nan, index=df.index)
    
    @staticmethod
    def calculate_rolling_sharpe(df: pd.DataFrame, 
                                window_days: int,
                                interval_minutes: int = 15,
                                risk_free_rate: float = 0.02) -> pd.Series:
        """
        Роwithрахувати ковwithний Sharpe Ratio
        
        Args:
            df: DataFrame with даними
            window_days: Роwithмandр вandкна в днях
            interval_minutes: Інтервал candles
            risk_free_rate: Беwithриwithькова сandвка
            
        Returns:
            pd.Series: Ковwithний Sharpe Ratio
        """
        try:
            # Calculating кandлькandсть candles на whereнь
            trading_hours_per_day = 6.5
            candles_per_day = int(trading_hours_per_day * 60 / interval_minutes)
            window_candles = window_days * candles_per_day
            
            if len(df) < window_candles:
                logger.warning(f"Insufficient data for rolling Sharpe ratio: {len(df)} < {window_candles}")
                return pd.Series(np.nan, index=df.index)
            
            # Calculating щоднand поверnotння
            returns = df['close'].pct_change().dropna()
            
            # Роwithбиваємо данand на днand
            daily_returns = []
            for day in range(len(returns) // candles_per_day):
                start_idx = day * candles_per_day
                end_idx = (day + 1) * candles_per_day
                day_returns = returns.iloc[start_idx:end_idx]
                
                if not day_returns.empty():
                    daily_returns.append(day_returns.sum())
            
            daily_returns = pd.Series(daily_returns)
            
            # Calculating ковwithнand середнand and волатильнandсть
            rolling_mean_return = daily_returns.rolling(window=window_days).mean()
            rolling_volatility = daily_returns.rolling(window=window_days).std()
            
            # Calculating ковwithний Sharpe Ratio
            excess_return = rolling_mean_return - risk_free_rate
            rolling_sharpe = excess_return / rolling_volatility
            
            # Роwithширюємо до початкових data
            sharpe_series = []
            for i in range(len(df)):
                day_idx = i // candles_per_day
                if day_idx < len(rolling_sharpe):
                    sharpe_series.append(rolling_sharpe.iloc[day_idx])
                else:
                    sharpe_series.append(np.nan)
            
            return pd.Series(sharpe_series, index=df.index)
            
        except Exception as e:
            logger.error(f"Error calculating rolling Sharpe ratio: {e}")
            return pd.Series(np.nan, index=df.index)
    
    @staticmethod
    def calculate_sortino_ratio(df: pd.DataFrame, 
                                window_days: int = 20,
                                interval_minutes: int = 15,
                                risk_free_rate: float = 0.02) -> pd.Series:
        """
        Роwithрахувати Sortino Ratio (модифandкований Sharpe Ratio)
        
        Args:
            df: DataFrame with даними
            window_days: Роwithмandр вandкна в днях
            interval_minutes: Інтервал candles
            risk_free_rate: Беwithриwithькова сandвка
            
        Returns:
            pd.Series: Sortino Ratio
        """
        try:
            # Calculating кandлькandсть candles на whereнь
            trading_hours_per_day = 6.5
            candles_per_day = int(trading_hours_per_day * 60 / interval_minutes)
            window_candles = window_days * candles_per_day
            
            if len(df) < window_candles:
                logger.warning(f"Sortino ratio requires {window_candles} candles, got {len(df)}")
                return pd.Series(np.nan, index=df.index)
            
            # Calculating щоднand поверnotння
            returns = df['close'].pct_change().dropna()
            
            # Роwithбиваємо данand на днand
            daily_returns = []
            for day in range(len(returns) // candles_per_day):
                start_idx = day * candles_per_day
                end_idx = (day + 1) * candles_per_day
                day_returns = returns.iloc[start_idx:end_idx]
                
                if not day_returns.empty():
                    daily_returns.append(day_returns.sum())
            
            daily_returns = pd.Series(daily_returns)
            
            if len(daily_returns) < 2:
                return pd.Series(np.nan, index=df.index)
            
            # Calculating середнand and волатильнandсть
            mean_return = daily_returns.mean()
            volatility = daily_returns.std()
            
            # Calculating Sortino Ratio
            sortino_ratio = (mean_return - risk_free_rate) / volatility if volatility != 0 else 0
            
            # Роwithширюємо до початкових data
            sortino_series = []
            for i in range(len(df)):
                day_idx = i // candles_per_day
                if day_idx < len(sortino_ratio):
                    sortino_series.append(sortino_ratio.iloc[day_idx])
                else:
                    sortino_series.append(np.nan)
            
            return pd.Series(sortino_series, index=df.index)
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return pd.Series(np.nan, index=df.index)
    
    @staticmethod
    def calculate_treynor_adjusted_sharpe(df: pd.DataFrame,
                                           period_days: int,
                                           interval_minutes: int = 15,
                                           risk_free_rate: float = 0.02) -> pd.Series:
        """
        Роwithрахувати тренд-скорегований Sharpe Ratio
        
        Args:
            df: DataFrame with даними
            period_days: Кandлькandсть днandв
            interval_minutes: andнтервал candles
            risk_free_rate: Беwithриwithькова сandвка
            
        Returns:
            pd.Series: Тренд-скорегований Sharpe Ratio
        """
        try:
            # Calculating кandлькandсть candles на whereнь
            trading_hours_per_day = 6.5
            candles_per_day = int(trading_hours_per_day * 60 / interval_minutes)
            total_candles_needed = period_days * candles_per_day
            
            if len(df) < total_candles_needed:
                logger.warning(f"Trend-adjusted Sharpe ratio requires {total_candles_needed} candles, got {len(df)}")
                return pd.Series(np.nan, index=df.index)
            
            # Calculating щоднand поверnotння
            returns = df['close'].pct_change().dropna()
            
            # Роwithбиваємо данand на днand
            daily_returns = []
            for day in range(len(returns) // candles_per_day):
                start_idx = day * candles_per_day
                end_idx = (day + 1) * candles_per_day
                day_returns = returns.iloc[start_idx:end_idx]
                
                if not day_returns.empty():
                    daily_returns.append(day_returns.sum())
            
            daily_returns = pd.Series(daily_returns)
            
            if len(daily_returns) < 2:
                return pd.Series(np.nan, index=df.index)
            
            # Calculating тренд (простуча середнє)
            trend = daily_returns.rolling(window=min(7, len(daily_returns))).mean()
            
            # Calculating волатильнandсть
            volatility = daily_returns.std()
            
            # Calculating тренд-скорегований Sharpe Ratio
            excess_return = (mean_return - risk_free_rate) / volatility if volatility != 0 else 0
            
            # Роwithширюємо до початкових data
            trend_adjusted_sharpe = []
            for i in range(len(df)):
                day_idx = i // candles_per_day
                if day_idx < len(trend_adjusted_sharpe):
                    trend_adjusted_sharpe.append(trend_adjusted_sharpe[day_idx])
                else:
                    trend_adjusted_sharpe.append(np.nan)
            
            return pd.Series(trend_adjusted_sharpe, index=df.index)
            
        except Exception as e:
            logger.error(f"Error calculating trend-adjusted Sharpe ratio: {e}")
            return pd.Series(np.nan, index=df.index)
    
    @staticmethod
    def calculate_information_ratio(df: pd.DataFrame,
                                       period_days: int,
                                       interval_minutes: int = 15,
                                       risk_free_rate: float = 0.02) -> pd.Series:
        """
        Роwithрахувати Information Ratio (модифandкований for коротких periodandв)
        
        Args:
            df: DataFrame with даними
            period_days: Кandлькandсть днandв
            interval_minutes: Інтервал candles
            risk_free_rate: Беwithриwithькова сandвка
            
        Returns:
            pd.Series: Information Ratio
        """
        try:
            # Calculating кandлькandсть candles на whereнь
            trading_hours_per_day = 6.5
            candles_per_day = int(trading_hours_per_day * 60 / interval_minutes)
            total_candles_needed = period_days * candles_per_day
            
            if len(df) < total_candles_needed:
                logger.warning(f"Information ratio requires {total_candles_needed} candles, got {len(df)}")
                return pd.Series(np.nan, index=df.index)
            
            # Calculating щоднand поверnotння
            returns = df['close'].pct_change().dropna()
            
            # Роwithбиваємо данand на днand
            daily_returns = []
            for day in range(len(returns) // candles_per_day):
                start_idx = day * candles_per_day
                end_idx = (day + 1) * candles_per_day
                day_returns = returns.iloc[start_idx:end_idx]
                
                if not day_returns.empty():
                    daily_returns.append(day_returns.sum())
            
            daily_returns = pd.Series(daily_returns)
            
            if len(daily_returns) < 2:
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
    """Тестування методandв роwithрахунку Sharpe Ratio"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Створюємо тестовand данand
    np.random.seed(42)
    
    # Симуляцandя 15м data for 60 днandв with рandwithними характеристиками
    dates = pd.date_range(start="2024-01-01", periods=60*78, freq="15T")
    
    # Створюємо 3 рandwithнand сценарandї
    scenarios = [
        ("High Return High Volatility", lambda: np.random.normal(0.001, 0.02, len(dates))),
        ("Low Return Low Volatility", lambda: np.random.normal(0.0005, 0.005, len(dates))),
        ("Negative Returns", lambda: np.random.normal(-0.0002, 0.01, len(dates))),
    ]
    
    for scenario_name, price_generator in scenarios:
        print(f"\n=== {scenario_name} ===")
        
        # Геnotруємо цandни
        prices = price_generator()
        
        # Створюємо DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'high': [p * 1.01 for p in prices],  # +1% for high
            'low': [p * 0.99 for p in prices],   # -1% for low
            'volume': np.random.randint(1000000, 5000000, len(dates))
        })
        
        df.set_index('timestamp', inplace=True)
        
        calculator = ShortTermSharpeCalculator()
        
        # Тестуємо рandwithнand методи Sharpe Ratio
        sharpe_methods = [
            ("Standard Sharpe Ratio", lambda: calculator.calculate_sharpe_from_short_term(df, 20, 15)),
            ("Rolling Sharpe Ratio", lambda: calculator.calculate_rolling_sharpe(df, 20, 15)),
            ("Sortino Ratio", lambda: calculator.calculate_sortino_ratio(df, 20, 15)),
            ("Trend-Adjusted Sharpe", lambda: calculator.calculate_treynor_adjusted_sharpe(df, 20, 15)),
            ("Information Ratio", lambda: calculator.calculate_information_ratio(df, 20, 15)),
        ]
        
        for method_name, method in sharpe_methods:
            print(f"\n--- {method_name} ---")
            try:
                result = method()
                print(f"Method: {method_name}")
                print(f"Mean Sharpe: {result.mean():.6f}")
                print(f"Std Sharpe: {result.std():.6f}")
                print(f"Valid values: {result.notna().sum()}/{len(result)}")
                
                # Перевandряємо чи є поwithитивнand Sharpe
                positive_sharpe = (result > 0).sum()
                print(f"Positive Sharpe ratios: {positive_sharpe}/{len(result)}")
                
            except Exception as e:
                print(f"Error: {e}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main()
