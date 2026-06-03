"""
News Candle Seeker
Handles finding relevant candles around news events.
"""
import logging
import pandas as pd
from typing import List, Optional, Any
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

def feature_to_key(col: Any) -> str:
    """Helper to convert column name to lowercase string key."""
    return str(col).lower()

class NewsCandleSeeker:
    def __init__(self, candle_features: List[str]):
        self.candle_features = candle_features

    @staticmethod
    def normalize_datetime(dt: Any) -> pd.Timestamp:
        """Normalize datetime by removing timezone."""
        ts = pd.to_datetime(dt)
        if ts.tz is not None:
            ts = ts.tz_localize(None)
        return ts

    def get_candles_before(self, df: pd.DataFrame, published_at: pd.Timestamp, timeframe: str, n: int = 1) -> List[pd.Series]:
        """Gets the last N closed candles strictly BEFORE publication."""
        if df is None or df.empty or n <= 0:
            return []
            
        pub_at = self.normalize_datetime(published_at)
        dt_col = self._find_datetime_column(df)
        
        if dt_col:
            df_temp = df.copy()
            df_temp[dt_col] = pd.to_datetime(df_temp[dt_col], utc=True).dt.tz_localize(None)
            df_before = df_temp[df_temp[dt_col] <= pub_at]
        else:
            df_copy = df.copy()
            if isinstance(df_copy.index, pd.DatetimeIndex) and df_copy.index.tz is not None:
                df_copy.index = df_copy.index.tz_localize(None)
            df_before = df_copy[df_copy.index <= pub_at]
            
        if df_before.empty:
            return []
            
        # Take last N
        last_n = df_before.tail(n)
        return [last_n.iloc[i] for i in range(len(last_n))]

    def get_candles_after(self, df: pd.DataFrame, published_at: pd.Timestamp, timeframe: str, n: int = 2) -> List[pd.Series]:
        """Gets N candles strictly AFTER publication."""
        if df is None or df.empty or n <= 0:
            return []
            
        pub_at = self.normalize_datetime(published_at)
        dt_col = self._find_datetime_column(df)
        
        if dt_col:
            df_temp = df.copy()
            df_temp[dt_col] = pd.to_datetime(df_temp[dt_col], utc=True).dt.tz_localize(None)
            df_after = df_temp[df_temp[dt_col] > pub_at]
        else:
            df_copy = df.copy()
            if isinstance(df_copy.index, pd.DatetimeIndex) and df_copy.index.tz is not None:
                df_copy.index = df_copy.index.tz_localize(None)
            df_after = df_copy[df_copy.index > pub_at]
            
        if len(df_after) < n:
            return []
        return [df_after.iloc[i] for i in range(n)]

    def extract_features(self, ticker: str, timeframe: str, candle: pd.Series, suffix: str = '') -> dict:
        """Extracts candle features into a flat dictionary."""
        features = {}
        
        # If candle_features is empty, extract all except service columns
        if not self.candle_features:
            service_cols = {'datetime', 'ticker', 'interval'}
            for col in candle.index:
                if col not in service_cols:
                    key = f'{ticker}_{timeframe}_{feature_to_key(col)}{suffix}'
                    features[key] = candle[col]
            return features
            
        for feature in self.candle_features:
            if feature in candle.index:
                key = f'{ticker}_{timeframe}_{feature.lower()}{suffix}'
                features[key] = candle[feature]
        return features

    def _find_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Finds the datetime column in a DataFrame."""
        for col in ['datetime', 'published_at', 'date', 'timestamp']:
            if col in df.columns:
                return col
        return None
