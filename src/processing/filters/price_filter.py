from datetime import timedelta
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("PriceFilter")

class PriceFilter:
    """Specialized filter for market price data with anomaly and gap detection."""

    def __init__(self, config: dict[str, Any]):
        self.min_candles = config.get('min_candles_per_timeframe', 2)
        self.min_quality = config.get('min_data_quality_score', 0.6)
        self.anomaly_threshold = config.get('anomaly_std_dev_threshold', 3)
        self.max_gap_duration = timedelta(hours=config.get('max_gap_duration_hours', 24))

    def filter_price_data(self, price_data: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
        """Intelligently filters price data for each timeframe."""
        filtered_prices = {}
        quality_report = {}

        for timeframe, tf_data in price_data.items():
            if not isinstance(tf_data, pd.DataFrame) or tf_data.empty:
                quality_report[timeframe] = {'status': 'empty', 'reason': 'no_data'}
                continue

            if len(tf_data) < self.min_candles:
                quality_report[timeframe] = {
                    'status': 'insufficient_data',
                    'reason': f'only_{len(tf_data)}_candles'
                }
                continue

            data_quality = self.assess_price_quality(tf_data)

            if data_quality['overall_score'] < self.min_quality:
                quality_report[timeframe] = {
                    'status': 'low_quality',
                    'reason': f'quality_score_{data_quality["overall_score"]:.2f}',
                    **data_quality
                }
                continue

            gaps = self.detect_and_classify_gaps(tf_data)
            anomalies = self.detect_and_classify_anomalies(tf_data)

            filtered_prices[timeframe] = {
                'data': tf_data,
                'quality': data_quality,
                'gaps': gaps,
                'anomalies': anomalies
            }

            quality_report[timeframe] = {
                'status': 'accepted',
                'quality_score': data_quality['overall_score'],
                'gaps_count': len(gaps),
                'anomalies_count': len(anomalies),
                **data_quality
            }

        return filtered_prices, quality_report

    def assess_price_quality(self, price_data: pd.DataFrame) -> dict[str, float]:
        """Assess quality of price data based on completeness and consistency."""
        total_rows = len(price_data)
        if total_rows == 0: return {'overall_score': 0.0}

        # Completeness (non-null values)
        null_counts = price_data.isnull().sum().sum()
        completeness = 1.0 - (null_counts / (total_rows * len(price_data.columns)))

        # Consistency (price sanity)
        consistency = 1.0
        if 'close' in price_data.columns and 'open' in price_data.columns:
            # Check for zero or negative prices
            bad_prices = ((price_data['close'] <= 0) | (price_data['open'] <= 0)).sum()
            consistency -= (bad_prices / total_rows)

        return {
            'completeness': float(completeness),
            'consistency': float(max(0, consistency)),
            'overall_score': float((completeness + consistency) / 2)
        }

    def detect_and_classify_gaps(self, price_data: pd.DataFrame) -> list[dict]:
        """Detect and classify gaps in time series."""
        if not isinstance(price_data.index, pd.DatetimeIndex):
            return []

        diffs = pd.Series(price_data.index).diff().dropna()
        median_diff = diffs.median()

        gaps = []
        for i, diff in diffs.items():
            if diff > median_diff * 3:
                gaps.append({
                    'timestamp': price_data.index[i],
                    'duration': diff.total_seconds(),
                    'severity': 'high' if diff > self.max_gap_duration else 'medium'
                })
        return gaps

    def detect_and_classify_anomalies(self, price_data: pd.DataFrame) -> list[dict]:
        """Detect and classify price anomalies."""
        if 'close' not in price_data.columns: return []

        prices = price_data['close']
        mean = prices.mean()
        std = prices.std()

        if std == 0: return []

        anomalies = []
        z_scores = (prices - mean) / std
        mask = z_scores.abs() > self.anomaly_threshold

        for idx in price_data.index[mask]:
            anomalies.append({
                'timestamp': idx,
                'value': float(price_data.loc[idx, 'close']),
                'z_score': float(z_scores.loc[idx]),
                'type': 'spike' if z_scores.loc[idx] > 0 else 'dip'
            })
        return anomalies
