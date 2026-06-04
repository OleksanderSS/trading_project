"""
News Global Enricher
Handles adding macro, long-term MA, and context fingerprint features.
"""
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class NewsGlobalEnricher:
    def __init__(self):
        self.context_thresholds = {
            'macro_vixcls': {'low': 15.0, 'high': 25.0},
            'macro_dgs10': {'low': 2.0, 'high': 4.5},
            'macro_fedfunds': {'low': 1.0, 'high': 4.0},
            'macro_unrate': {'low': 4.0, 'high': 6.0},
        }

    def enrich_record(self, record: dict[str, Any], macro_data: pd.DataFrame,
                      published_at: pd.Timestamp, tickers: list[str],
                      price_data: dict[str, pd.DataFrame]) -> bool:
        """Enriches the record with global features. Returns False if critical data is missing."""
        macro_features = self.get_macro_features(macro_data, published_at)
        if not macro_features:
            return False

        record.update(macro_features)
        record.update(self.get_long_term_mas(tickers, price_data, published_at))
        record.update(self.calculate_context_map(record))
        return True

    def get_macro_features(self, macro_data: pd.DataFrame, published_at: pd.Timestamp) -> dict[str, Any]:
        """Gets macro indicators at publication time."""
        if macro_data.empty:
            return {}

        pub_at = self._normalize_timestamp(published_at)
        macro_before = self._filter_macro_data_before_date(macro_data, pub_at)

        if macro_before.empty:
            return {}

        latest_macro = macro_before.iloc[-1]
        features = {}
        excluded_cols = ['ticker', 'datetime', 'date', 'timestamp', 'hash', 'realtime_start', 'realtime_end', 'series_id']

        for col in macro_data.columns:
            if col not in excluded_cols:
                key = f'macro_{col.lower()}'
                value = latest_macro[col]
                if pd.notna(value):
                    features[key] = value
        return features

    def get_long_term_mas(self, tickers: list[str], price_data: dict[str, pd.DataFrame],
                          published_at: pd.Timestamp) -> dict[str, Any]:
        """Calculates long-term MAs (SMA_200, EMA_200) for all tickers."""
        if '1d' not in price_data:
            return {}

        daily_data = price_data['1d']
        pub_at = self._normalize_timestamp(published_at)
        features = {}

        for ticker in tickers:
            if ticker not in daily_data.columns:
                continue

            ticker_data = daily_data[ticker].dropna()
            if ticker_data.empty:
                continue

            if ticker_data.index.tz is not None:
                ticker_data = ticker_data.tz_localize(None)

            data_before = ticker_data[ticker_data.index <= pub_at]
            if len(data_before) >= 200:
                features[f'{ticker}_sma_200_1d'] = data_before.rolling(window=200).mean().iloc[-1]
                features[f'{ticker}_ema_200_1d'] = data_before.ewm(span=200).mean().iloc[-1]

        return features

    def calculate_context_map(self, record: dict[str, Any]) -> dict[str, Any]:
        """Calculates context fingerprint and stability."""
        context_features = {}
        states = []
        key_indicators = ['macro_vixcls', 'macro_dgs10', 'macro_fedfunds', 'macro_cpiaucsl', 'macro_unrate']

        for indicator in key_indicators:
            if indicator in record:
                state = self._macro_indicator_state(indicator, record[indicator])
                state_key = f"state_{indicator.replace('macro_', '')}"
                context_features[state_key] = state
                states.append(state)

        if states:
            context_features['context_fingerprint'] = '|'.join(map(str, states))
            context_features['context_stability'] = states.count(0) / len(states)
        else:
            context_features['context_fingerprint'] = ''
            context_features['context_stability'] = 0.0

        return context_features

    def _macro_indicator_state(self, indicator: str, value: Any) -> int:
        """Map macro values into risk states: -1 benign, 0 neutral, 1 stressed."""
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return 0

        thresholds = self.context_thresholds.get(indicator)
        if not thresholds:
            return 0
        if numeric_value >= thresholds['high']:
            return 1
        if numeric_value <= thresholds['low']:
            return -1
        return 0

    def _normalize_timestamp(self, ts: pd.Timestamp) -> pd.Timestamp:
        normalized = pd.to_datetime(ts)
        if normalized.tz is not None:
            normalized = normalized.tz_localize(None)
        return normalized

    def _filter_macro_data_before_date(self, macro_data: pd.DataFrame, published_at: pd.Timestamp) -> pd.DataFrame:
        df = macro_data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            date_col = next((c for c in ['date', 'datetime', 'timestamp'] if c in df.columns), None)
            if not date_col:
                return pd.DataFrame()
            df[date_col] = pd.to_datetime(df[date_col])
            if df[date_col].dt.tz is not None:
                df[date_col] = df[date_col].dt.tz_localize(None)
            return df[df[date_col] <= published_at]
        else:
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df[df.index <= published_at]
