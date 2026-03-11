
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
        
        # This is a simplified calculation logic. A real implementation would be more robust.
        # It dynamically calls calculation methods based on feature names.
        for feature in self.context_features:
            calc_method_name = f"_calculate_{feature}"
            if hasattr(self, calc_method_name):
                try:
                    value = getattr(self, calc_method_name)(data, **kwargs)
                    context_vector[feature] = value
                except Exception as e:
                    logger.warning(f"Could not calculate feature '{feature}': {e}", exc_info=True)
                    context_vector[feature] = np.nan
            elif feature in kwargs:
                context_vector[feature] = kwargs[feature] # Allow passing features directly

        # Fill any remaining NaNs with 0, as a fallback
        final_vector = context_vector.fillna(0)
        
        return {"market_context_vector": final_vector}

    # --- Feature Calculation Methods ---
    # Each method is responsible for a single feature.

    def _calculate_volatility_5d(self, df: pd.DataFrame, **kwargs) -> float:
        return df['close'].pct_change().tail(5).std()

    def _calculate_volatility_20d(self, df: pd.DataFrame, **kwargs) -> float:
        return df['close'].pct_change().tail(20).std()

    def _calculate_volatility_ratio(self, df: pd.DataFrame, **kwargs) -> float:
        vol_5d = self._calculate_volatility_5d(df)
        vol_20d = self._calculate_volatility_20d(df)
        return vol_5d / (vol_20d + 1e-9) # Add epsilon to avoid division by zero

    def _calculate_trend_5d(self, df: pd.DataFrame, **kwargs) -> float:
        prices = df['close'].tail(5).values
        return np.polyfit(np.arange(len(prices)), prices, 1)[0]

    def _calculate_trend_20d(self, df: pd.DataFrame, **kwargs) -> float:
        prices = df['close'].tail(20).values
        return np.polyfit(np.arange(len(prices)), prices, 1)[0]

    def _calculate_trend_alignment(self, df: pd.DataFrame, **kwargs) -> float:
        trend_5d = self._calculate_trend_5d(df)
        trend_20d = self._calculate_trend_20d(df)
        return np.sign(trend_5d * trend_20d)

    def _calculate_rsi_current(self, df: pd.DataFrame, **kwargs) -> float:
        # Assuming RSI is pre-calculated and available in the DataFrame
        return df['rsi'].iloc[-1] if 'rsi' in df.columns else np.nan

    def _calculate_volume_ratio(self, df: pd.DataFrame, **kwargs) -> float:
        if 'volume' not in df.columns or len(df) < 20:
            return np.nan
        avg_vol_5 = df['volume'].tail(5).mean()
        avg_vol_20 = df['volume'].tail(20).mean()
        return avg_vol_5 / (avg_vol_20 + 1e-9)

    def _calculate_price_to_ma20(self, df: pd.DataFrame, **kwargs) -> float:
        if 'close' not in df.columns or len(df) < 20:
            return np.nan
        ma20 = df['close'].tail(20).mean()
        return (df['close'].iloc[-1] / ma20) - 1

    def _calculate_hour_of_day(self, df: pd.DataFrame, **kwargs) -> int:
        return df.index[-1].hour if isinstance(df.index, pd.DatetimeIndex) else datetime.now().hour

    def _calculate_day_of_week(self, df: pd.DataFrame, **kwargs) -> int:
        return df.index[-1].weekday() if isinstance(df.index, pd.DatetimeIndex) else datetime.now().weekday()
