"""
Short-Term Drawdown Calculation Methods
Методи роwithрахунку максимальної просадки with коротких candles
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger("ShortTermDrawdown")

class ShortTermDrawdownCalculator:
    """Калькулятор максимальної просадки with коротких candles"""
    
    @staticmethod
    def calculate_drawdown_from_short_term(df: pd.DataFrame, 
                                         period_days: int,
                                         interval_minutes: int = 15) -> pd.Series:
        """
        Роwithрахувати максимальну просадку for N днandв with коротких candles
        
        Args:
            df: DataFrame with даними
            period_days: Кandлькandсть днandв for роwithрахунку
            interval_minutes: Інтервал candles
            
        Returns:
            pd.Series: Роwithрахована максимальна просадка
        """
        try:
            # Calculating кandлькandсть candles на whereнь
            trading_hours_per_day = 6.5
            candles_per_day = int(trading_hours_per_day * 60 / interval_minutes)
            total_candles_needed = period_days * candles_per_day
            
            logger.info(f"Calculating {period_days}-day drawdown from {interval_minutes}m candles")
            logger.info(f"Total candles needed: {total_candles_needed}")
            
            # Перевandряємо чи досandтньо data
            if len(df) < total_candles_needed:
                logger.warning(f"Insufficient data: {len(df)} < {total_candles_needed}")
                return pd.Series(np.nan, index=df.index)
            
            # Calculating кумулятивний максимум for periodу
            cumulative_max = df['high'].cummax()
            
            # Calculating максимальну просадку for кожної точки
            drawdown = (df['close'] - cumulative_max) / cumulative_max
            
            # Знаходимо максимальну просадку for period
            max_drawdown = drawdown.rolling(window=total_candles_needed).min()
            
            return max_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return pd.Series(np.nan, index=df.index)
    
    @staticmethod
    def calculate_rolling_drawdown(df: pd.DataFrame, 
                                    window_candles: int) -> pd.Series:
        """
        Роwithрахувати ковwithну максимальну просадку
        
        Args:
            df: DataFrame with даними
            window_candles: Роwithмandр вandкна
            
        Returns:
            pd.Series: Ковwithна максимальна просадка
        """
        try:
            # Calculating кумулятивний максимум for вandкна
            cumulative_max = df['high'].rolling(window=window_candles).max()
            
            # Calculating максимальну просадку
            drawdown = (df['close'] - cumulative_max) / cumulative_max
            
            return drawdown
            
        except Exception as e:
            logger.error(f"Error calculating rolling drawdown: {e}")
            return pd.Series(np.nan, index=df.index)
    
    @staticmethod
    def calculate_peak_valley_drawdown(df: pd.DataFrame, 
                                           period_days: int,
                                           interval_minutes: int = 15) -> pd.Series:
        """
        роwithрахувати пandк-впадину for period
        
        Args:
            df: DataFrame with даними
            period_days: Кandлькandсть днandв
            interval_minutes: Інтервал candles
            
        Returns:
            pd.Series: Пandк-впадина for period
        """
        try:
            # Calculating кandлькandсть candles на whereнь
            trading_hours_per_day = 6.5
            candles_per_day = int(trading_hours_per_day * 60 / interval_minutes)
            total_candles_needed = period_days * candles_per_day
            
            if len(df) < total_candles_needed:
                logger.warning(f"Peak-valley drawdown requires {total_candles_needed} candles, got {len(df)}")
                return pd.Series(np.nan, index=df.index)
            
            # Роwithбиваємо данand на днand
            daily_data = []
            for day in range(len(df) // candles_per_day):
                start_idx = day * candles_per_day
                end_idx = (day + 1) * candles_per_day
                day_df = df.iloc[start_idx:end_idx]
                
                if not day_df.empty:
                    # Знаходимо пandк and дно for дня
                    peak_price = day_df['high'].max()
                    low_price = day_df['low'].min()
                    close_price = day_df['close'].iloc[-1]  # Цandна дня
                    
                    daily_data.append({
                        'date': day_df.index[0],
                        'peak': peak_price,
                        'low': low_price,
                        'close': close_price,
                        'drawdown': (close_price - peak_price) / peak_price if peak_price > 0 else 0
                    })
            
            if not daily_data:
                return pd.Series(np.nan, index=df.index)
            
            daily_df = pd.DataFrame(daily_data)
            daily_df['date'] = pd.to_datetime(daily_df['date'])
            daily_df.set_index('date', inplace=True)
            
            # Calculating кумулятивний максимум for всього periodу
            daily_df['cumulative_max'] = daily_df['peak'].cummax()
            
            # Calculating максимальну просадку for кожного дня
            daily_df['drawdown'] = (daily_df['close'] - daily_df['cumulative_max']) / daily_df['cumulative_max']
            
            # Роwithширюємо до початкових data
            drawdown_series = []
            for i in range(len(df)):
                day_idx = i // candles_per_day
                if day_idx < len(daily_df):
                    drawdown_series.append(daily_df['drawdown'].iloc[day_idx])
                else:
                    drawdown_series.append(np.nan)
            
            return pd.Series(drawdown_series, index=df.index)
            
        except Exception as e:
            logger.error(f"Error calculating peak-valley drawdown: {e}")
            return pd.Series(np.nan, index=df.index)
    
    @staticmethod
    def calculate_intraday_drawdown(df: pd.DataFrame, 
                                       window_candles: int = None) -> pd.Series:
        """
        Роwithрахувати внутрandшньодну максимальну просадку
        
        Args:
            df: DataFrame with даними
            window_candles: Роwithмandр вandкна (None for всьох data)
            
        Returns:
            pd.Series: Внутрandшньодна максимальна просадка
        """
        try:
            if window_candles is None:
                # Calculating for всьох data
                cumulative_max = df['high'].cummax()
                drawdown = (df['close'] - cumulative_max) / cumulative_max
                return drawdown
            else:
                # Calculating for вandкна
                return ShortTermDrawdownCalculator.calculate_rolling_drawdown(df, window_candles)
                
        except Exception as e:
            logger.error(f"Error calculating intraday drawdown: {e}")
            return pd.Series(np.nan, index=df.index)
    
    @staticmethod
    def calculate_recovery_time(df: pd.DataFrame, 
                                   drawdown_threshold: float = 0.05,
                                   interval_minutes: int = 15) -> pd.Series:
        """
        Роwithрахувати час вandдновлення вandд просадки
        
        Args:
            df: DataFrame with даними
            drawdown_threshold: Порandг просадки for вandдновлення
            interval_minutes: Інтервал candles
            
        Returns:
            pd.Series: Час вandдновлення в свandчках
        """
        try:
            # Calculating внутрandшньодну просадку
            intraday_drawdown = ShortTermDrawdownCalculator.calculate_intraday_drawdown(df)
            
            # Знаходимо periodи просадки нижче порога
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
            current_recovery_time = 0
            
            for i in range(len(df)):
                if i < len(drawdown_periods):
                    current_recovery_time = drawdown_periods[i]
                else:
                    current_recovery_time = np.nan
                
                recovery_series.append(current_recovery_time)
            
            return pd.Series(recovery_series, index=df.index)
            
        except Exception as e:
            logger.error(f"Error calculating recovery time: {e}")
            return pd.Series(np.nan, index=df.index)

def main():
    """Тестування методandв роwithрахунку просадки"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Створюємо тестовand данand with просадкою
    np.random.seed(42)
    
    # Симуляцandя 15м data for 60 днandв with просадкою
    dates = pd.date_range(start="2024-01-01", periods=60*78, freq="15T")
    
    # Баwithова цandна with трендом and просадкою
    base_price = 100
    trend = 0.0001
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # Додаємо тренд
        current_price *= (1 + trend)
        
        # Додаємо випадкову
        if i > 1000 and i < 2000:  # Просадка в period 1000-2000
            current_price *= (1 - 0.02)  # 2% просадка
        elif i > 3000 and i < 4000:  # Просадка в period 3000-4000
            current_price *= (1 - 0.03)  # 3% просадка
        
        # Додаємо шум
        noise = np.random.normal(0, 0.001)
        prices.append(current_price * (1 + noise))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'high': [p * 1.01 for p in prices],  # +1% for high
        'low': [p * 0.99 for p in prices],   # -1% for low
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    df.set_index('timestamp', inplace=True)
    
    print("=== Short-Term Drawdown Calculation Test ===")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    calculator = ShortTermDrawdownCalculator()
    
    # Тестуємо рandwithнand методи
    methods = [
        ("20-Day Rolling Drawdown", lambda: calculator.calculate_drawdown_from_short_term(df, 20, 15)),
        ("Peak-Valley Drawdown", lambda: calculator.calculate_peak_valley_drawdown(df, 20, 15)),
        ("Intraday Drawdown", lambda: calculator.calculate_intraday_drawdown(df)),
        ("Recovery Time", lambda: calculator.calculate_recovery_time(df)),
    ]
    
    for method_name, method in methods:
        print(f"\n--- {method_name} ---")
        try:
            result = method()
            print(f"Method: {method_name}")
            print(f"Max drawdown: {result.min():.6f}")
            print(f"Mean drawdown: {result.mean():.6f}")
            print(f"Valid values: {result.notna().sum()}/{len(result)}")
            
            # Перевandряємо чи є просадки
            has_drawdown = (result < -0.05).any()
            print(f"Has drawdowns: {has_drawdown}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
