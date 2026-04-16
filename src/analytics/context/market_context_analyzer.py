
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime

from ..interfaces import IAnalyzer

logger = logging.getLogger(__name__)

class MarketContextAnalyzer(IAnalyzer):
    """
    Analyzes raw market data to generate a standardized context vector.
    This vector captures the 'DNA' of the market at a specific moment, including
    volatility, trend, momentum, and other user-defined features.
    """

    def __init__(self, context_features: List[str]):
        """
        Initializes the MarketContextAnalyzer.

        Args:
            context_features (List[str]): A list of feature names that define the market context.
        """
        if not context_features:
            raise ValueError("The context_features list cannot be empty.")
        self.context_features = context_features
        logger.info(f"MarketContextAnalyzer initialized with {len(context_features)} features.")

    def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Analyzes the provided market data to compute a context vector.

        Args:
            data (pd.DataFrame): A DataFrame containing market data (OHLCV, indicators, etc.).
                                 Must have a datetime index.
            **kwargs: Can include external data like 'vix_level' or 'macro_bias'.

        Returns:
            Dict[str, Any]: A dictionary containing the computed context vector as a pd.Series.
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            logger.error("Invalid input: Market data must be a non-empty pd.DataFrame.")
            return {"error": "Invalid input data"}

        context_vector = pd.Series(index=self.context_features, dtype=float)
        
        # ✅ FIX: Знайти колонки з OHLCV даними (підтримка event-centric формату)
        self._find_price_columns(data)
        
        # This is a simplified calculation logic. A real implementation would be more robust.
        # It dynamically calls calculation methods based on feature names.
        for feature in self.context_features:
            calc_method_name = f"_calculate_{feature}"
            if hasattr(self, calc_method_name):
                try:
                    value = getattr(self, calc_method_name)(data, **kwargs)
                    context_vector[feature] = value
                except Exception as e:
                    logger.warning(f"Could not calculate feature '{feature}': {e}")
                    context_vector[feature] = np.nan
            elif feature in kwargs:
                context_vector[feature] = kwargs[feature] # Allow passing features directly

        # Fill any remaining NaNs with 0, as a fallback
        final_vector = context_vector.fillna(0)
        
        return {"market_context_vector": final_vector}
    
    def _find_price_columns(self, df: pd.DataFrame):
        """
        Знаходить колонки з OHLCV даними в DataFrame.
        Підтримує як стандартний формат (close, open, high, low, volume),
        так і event-centric формат (AMD_1d_close, AMD_15m_open, тощо).
        """
        # Шукаємо колонки з 'close', 'open', 'high', 'low', 'volume'
        self.close_col = None
        self.open_col = None
        self.high_col = None
        self.low_col = None
        self.volume_col = None
        self.rsi_col = None
        
        # Пріоритет: спочатку шукаємо стандартні назви
        if 'close' in df.columns:
            self.close_col = 'close'
        else:
            # Шукаємо колонки з '_close' (event-centric формат)
            close_cols = [col for col in df.columns if col.endswith('_close') and not col.endswith('_+1') and not col.endswith('_+2')]
            if close_cols:
                # Віддаємо перевагу 1d таймфрейму
                self.close_col = next((col for col in close_cols if '_1d_' in col), close_cols[0])
        
        if 'open' in df.columns:
            self.open_col = 'open'
        else:
            open_cols = [col for col in df.columns if col.endswith('_open') and not col.endswith('_+1') and not col.endswith('_+2')]
            if open_cols:
                self.open_col = next((col for col in open_cols if '_1d_' in col), open_cols[0])
        
        if 'high' in df.columns:
            self.high_col = 'high'
        else:
            high_cols = [col for col in df.columns if col.endswith('_high') and not col.endswith('_+1') and not col.endswith('_+2')]
            if high_cols:
                self.high_col = next((col for col in high_cols if '_1d_' in col), high_cols[0])
        
        if 'low' in df.columns:
            self.low_col = 'low'
        else:
            low_cols = [col for col in df.columns if col.endswith('_low') and not col.endswith('_+1') and not col.endswith('_+2')]
            if low_cols:
                self.low_col = next((col for col in low_cols if '_1d_' in col), low_cols[0])
        
        if 'volume' in df.columns:
            self.volume_col = 'volume'
        else:
            volume_cols = [col for col in df.columns if col.endswith('_volume') and not col.endswith('_+1') and not col.endswith('_+2')]
            if volume_cols:
                self.volume_col = next((col for col in volume_cols if '_1d_' in col), volume_cols[0])
        
        if 'rsi' in df.columns:
            self.rsi_col = 'rsi'
        else:
            rsi_cols = [col for col in df.columns if 'rsi' in col.lower()]
            if rsi_cols:
                self.rsi_col = rsi_cols[0]
        
        logger.debug(f"Found price columns: close={self.close_col}, open={self.open_col}, high={self.high_col}, low={self.low_col}, volume={self.volume_col}, rsi={self.rsi_col}")

    # --- Feature Calculation Methods ---
    # Each method is responsible for a single feature.

    def _calculate_volatility_5d(self, df: pd.DataFrame, **kwargs) -> float:
        if self.close_col and self.close_col in df.columns:
            return df[self.close_col].pct_change().tail(5).std()
        return 0.0

    def _calculate_volatility_20d(self, df: pd.DataFrame, **kwargs) -> float:
        if self.close_col and self.close_col in df.columns:
            return df[self.close_col].pct_change().tail(20).std()
        return 0.0

    def _calculate_volatility_ratio(self, df: pd.DataFrame, **kwargs) -> float:
        vol_5d = self._calculate_volatility_5d(df)
        vol_20d = self._calculate_volatility_20d(df)
        return vol_5d / (vol_20d + 1e-9) # Add epsilon to avoid division by zero

    def _calculate_trend_5d(self, df: pd.DataFrame, **kwargs) -> float:
        if self.close_col and self.close_col in df.columns:
            prices = df[self.close_col].tail(5).values
            if len(prices) >= 2:
                return np.polyfit(np.arange(len(prices)), prices, 1)[0]
        return 0.0

    def _calculate_trend_20d(self, df: pd.DataFrame, **kwargs) -> float:
        if self.close_col and self.close_col in df.columns:
            prices = df[self.close_col].tail(20).values
            if len(prices) >= 2:
                return np.polyfit(np.arange(len(prices)), prices, 1)[0]
        return 0.0

    def _calculate_trend_alignment(self, df: pd.DataFrame, **kwargs) -> float:
        trend_5d = self._calculate_trend_5d(df)
        trend_20d = self._calculate_trend_20d(df)
        return np.sign(trend_5d * trend_20d)

    def _calculate_rsi_current(self, df: pd.DataFrame, **kwargs) -> float:
        # Assuming RSI is pre-calculated and available in the DataFrame
        if self.rsi_col and self.rsi_col in df.columns:
            return df[self.rsi_col].iloc[-1]
        return 50.0  # Neutral RSI

    def _calculate_volume_ratio(self, df: pd.DataFrame, **kwargs) -> float:
        if self.volume_col and self.volume_col in df.columns and len(df) >= 20:
            avg_vol_5 = df[self.volume_col].tail(5).mean()
            avg_vol_20 = df[self.volume_col].tail(20).mean()
            return avg_vol_5 / (avg_vol_20 + 1e-9)
        return 1.0  # Neutral volume ratio

    def _calculate_price_to_ma20(self, df: pd.DataFrame, **kwargs) -> float:
        if self.close_col and self.close_col in df.columns and len(df) >= 20:
            ma20 = df[self.close_col].tail(20).mean()
            return (df[self.close_col].iloc[-1] / ma20) - 1
        return 0.0  # Neutral price to MA ratio

    def _calculate_hour_of_day(self, df: pd.DataFrame, **kwargs) -> int:
        return df.index[-1].hour if isinstance(df.index, pd.DatetimeIndex) else datetime.now().hour

    def _calculate_day_of_week(self, df: pd.DataFrame, **kwargs) -> int:
        return df.index[-1].weekday() if isinstance(df.index, pd.DatetimeIndex) else datetime.now().weekday()

    # ✅ НОВІ КОНТЕКСТНІ ПОКАЗНИКИ (з CONTEXT_FEATURES_RECOMMENDATIONS.md)
    
    def _calculate_yield_curve_slope(self, df: pd.DataFrame, **kwargs) -> float:
        """
        Розраховує нахил кривої дохідності (10Y - 2Y).
        Негативне значення = інверсія кривої = можлива рецесія.
        """
        dgs10 = kwargs.get('DGS10', df.get('DGS10', pd.Series([np.nan])).iloc[-1] if 'DGS10' in df.columns else np.nan)
        dgs2 = kwargs.get('DGS2', df.get('DGS2', pd.Series([np.nan])).iloc[-1] if 'DGS2' in df.columns else np.nan)
        
        if pd.isna(dgs10) or pd.isna(dgs2):
            return 0.0
        
        slope = dgs10 - dgs2
        logger.debug(f"Yield curve slope: {slope:.4f} (10Y={dgs10:.2f}%, 2Y={dgs2:.2f}%)")
        return slope
    
    def _calculate_yield_curve_inverted(self, df: pd.DataFrame, **kwargs) -> int:
        """
        Прапорець інверсії кривої дохідності (0 або 1).
        1 = інверсія (10Y < 2Y) = сигнал рецесії.
        """
        slope = self._calculate_yield_curve_slope(df, **kwargs)
        return 1 if slope < 0 else 0
    
    def _calculate_fed_funds_trend(self, df: pd.DataFrame, **kwargs) -> float:
        """
        Тренд ставки Fed Funds (зміна за останні 3 місяці).
        Позитивне = підвищення ставок = жорсткіша монетарна політика.
        """
        if 'FEDFUNDS' not in df.columns or len(df) < 60:
            return 0.0
        
        current = df['FEDFUNDS'].iloc[-1]
        three_months_ago = df['FEDFUNDS'].iloc[-60] if len(df) >= 60 else df['FEDFUNDS'].iloc[0]
        
        trend = current - three_months_ago
        logger.debug(f"Fed Funds trend: {trend:.4f}% (current={current:.2f}%, 3m ago={three_months_ago:.2f}%)")
        return trend
    
    def _calculate_fed_funds_velocity(self, df: pd.DataFrame, **kwargs) -> float:
        """
        Швидкість зміни ставки Fed Funds (% за місяць).
        Висока швидкість = агресивна зміна політики.
        """
        if 'FEDFUNDS' not in df.columns or len(df) < 20:
            return 0.0
        
        current = df['FEDFUNDS'].iloc[-1]
        one_month_ago = df['FEDFUNDS'].iloc[-20] if len(df) >= 20 else df['FEDFUNDS'].iloc[0]
        
        velocity = current - one_month_ago
        logger.debug(f"Fed Funds velocity: {velocity:.4f}%/month")
        return velocity
    
    def _calculate_market_breadth(self, df: pd.DataFrame, **kwargs) -> float:
        """
        Ширина ринку (advance/decline ratio).
        > 1 = більше акцій зростає = здоровий тренд.
        < 1 = більше акцій падає = слабкий тренд.
        
        Якщо немає даних про advance/decline, використовуємо proxy:
        % акцій вище SMA(50) / % акцій нижче SMA(50).
        """
        # Спробуємо знайти advance/decline дані
        if 'advances' in df.columns and 'declines' in df.columns:
            advances = df['advances'].iloc[-1]
            declines = df['declines'].iloc[-1]
            breadth = advances / (declines + 1e-9)
            logger.debug(f"Market breadth: {breadth:.2f} (advances={advances}, declines={declines})")
            return breadth
        
        # Proxy: використовуємо close vs SMA(50)
        if 'close' in df.columns and len(df) >= 50:
            sma50 = df['close'].tail(50).mean()
            current_price = df['close'].iloc[-1]
            breadth_proxy = 1.0 if current_price > sma50 else 0.5
            logger.debug(f"Market breadth (proxy): {breadth_proxy:.2f} (price vs SMA50)")
            return breadth_proxy
        
        return 1.0  # Neutral
    
    def _calculate_dollar_strength(self, df: pd.DataFrame, **kwargs) -> float:
        """
        Сила долара (DXY index).
        Висока сила долара = тиск на commodities та emerging markets.
        """
        if 'DXY' in df.columns:
            dxy = df['DXY'].iloc[-1]
            logger.debug(f"Dollar strength (DXY): {dxy:.2f}")
            return dxy
        
        dxy = kwargs.get('DXY', 100.0)  # Default neutral value
        return dxy
    
    def _calculate_put_call_ratio(self, df: pd.DataFrame, **kwargs) -> float:
        """
        Put/Call ratio (страх на ринку).
        > 1 = більше puts = песимізм.
        < 1 = більше calls = оптимізм.
        """
        if 'put_call_ratio' in df.columns:
            ratio = df['put_call_ratio'].iloc[-1]
            logger.debug(f"Put/Call ratio: {ratio:.2f}")
            return ratio
        
        ratio = kwargs.get('put_call_ratio', 1.0)  # Default neutral
        return ratio
