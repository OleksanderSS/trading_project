"""
Specialized filter for market price data.
Handles quality assessment, gap detection, and anomaly classification.
"""
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("PriceFilter")

class PriceFilter:
    def __init__(self, config: dict[str, Any]):
        self.min_candles = config.get('min_candles_per_timeframe', 2)
        self.min_quality = config.get('min_data_quality_score', 0.6)
        self.anomaly_threshold = config.get('anomaly_std_dev_threshold', 3)

    def filter(self, price_data: dict[str, pd.DataFrame]) -> tuple[dict[str, Any], dict[str, Any]]:
        filtered_prices = {}
        quality_report = {}

        for timeframe, df in price_data.items():
            if df is None or df.empty or len(df) < self.min_candles:
                quality_report[timeframe] = {'status': 'rejected', 'reason': 'insufficient_data'}
                continue

            quality = self._assess_quality(df)
            if quality['overall_score'] < self.min_quality:
                quality_report[timeframe] = {'status': 'low_quality', **quality}
                continue

            filtered_prices[timeframe] = {
                'data': df,
                'quality': quality,
                'anomalies': self._detect_anomalies(df),
                'gaps': self._detect_gaps(df)
            }
            quality_report[timeframe] = {'status': 'accepted', **quality}

        return filtered_prices, quality_report

    def _assess_quality(self, df: pd.DataFrame) -> dict[str, float]:
        total = df.size
        nulls = df.isnull().sum().sum()
        completeness = 1 - (nulls / total) if total > 0 else 0

        # Simple consistency check
        consistency = 1.0
        if 'close' in df.columns:
            extreme_moves = (df['close'].pct_change().abs() > 0.5).sum()
            consistency -= (extreme_moves / len(df)) * 0.5

        return {
            'completeness': float(completeness),
            'consistency': float(max(0, consistency)),
            'overall_score': float(completeness * 0.5 + max(0, consistency) * 0.5)
        }

    def _detect_anomalies(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if 'close' not in df.columns: return []
        prices = df['close']
        mean, std = prices.mean(), prices.std()
        if std == 0: return []

        anomalies = []
        mask = (prices - mean).abs() > self.anomaly_threshold * std
        for idx in df.index[mask]:
            anomalies.append({
                'timestamp': idx,
                'price': float(df.loc[idx, 'close']),
                'deviation': float((df.loc[idx, 'close'] - mean) / std)
            })
        return anomalies

    def _detect_gaps(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        # Assume index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex): return []
        # ✅ Preserving original DatetimeIndex so that gaps.items() yields real Timestamps instead of integer row indices
        diffs = pd.Series(df.index, index=df.index).diff()
        median_diff = diffs.median()
        gaps = diffs[diffs > median_diff * 3]

        return [{
            'end_time': idx.isoformat() if hasattr(idx, 'isoformat') else idx,
            'duration_sec': dur.total_seconds()
        } for idx, dur in gaps.items()]
