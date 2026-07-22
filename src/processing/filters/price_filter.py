import math
from datetime import timedelta
from typing import Any

import numpy as np
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
        self.max_cross_ticker_duplicate_ratio = config.get(
            'max_cross_ticker_duplicate_ratio',
            0.001,
        )
        self.min_cadence_match_ratio = config.get('min_cadence_match_ratio', 0.60)
        self.max_extreme_return_ratio = config.get('max_extreme_return_ratio', 0.01)

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

            hard_failures = []
            if data_quality.get('cross_ticker_duplicate_ratio', 0.0) > self.max_cross_ticker_duplicate_ratio:
                hard_failures.append('cross_ticker_duplicate_ohlcv')
            cadence_ratio = data_quality.get('cadence_match_ratio')
            if cadence_ratio is not None and cadence_ratio < self.min_cadence_match_ratio:
                hard_failures.append('timeframe_cadence_mismatch')
            if data_quality.get('extreme_return_ratio', 0.0) > self.max_extreme_return_ratio:
                hard_failures.append('extreme_return_contamination')

            if data_quality['overall_score'] < self.min_quality or hard_failures:
                quality_report[timeframe] = {
                    'status': 'low_quality',
                    'reason': (
                        ','.join(hard_failures)
                        if hard_failures
                        else f'quality_score_{data_quality["overall_score"]:.2f}'
                    ),
                    'hard_failures': hard_failures,
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
        if total_rows == 0:
            return {'overall_score': 0.0}

        # Completeness (non-null values)
        null_counts = price_data.isnull().sum().sum()
        completeness = 1.0 - (null_counts / (total_rows * len(price_data.columns)))

        # Consistency (price sanity)
        consistency = 1.0
        if 'close' in price_data.columns and 'open' in price_data.columns:
            # Check for zero or negative prices
            bad_prices = ((price_data['close'] <= 0) | (price_data['open'] <= 0)).sum()
            consistency -= (bad_prices / total_rows)

        temporal_identity = self._assess_temporal_identity(price_data)
        return {
            'completeness': float(completeness),
            'consistency': float(max(0, consistency)),
            'overall_score': float((completeness + consistency) / 2),
            **temporal_identity,
        }

    def _assess_temporal_identity(self, price_data: pd.DataFrame) -> dict[str, float | None]:
        required_identity = {'datetime', 'ticker', 'open', 'high', 'low', 'close', 'volume'}
        cross_ticker_duplicate_ratio = 0.0
        if required_identity.issubset(price_data.columns):
            identity_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            duplicate_mask = price_data.duplicated(identity_columns, keep=False)
            duplicate_rows = price_data.loc[duplicate_mask]
            if not duplicate_rows.empty:
                cross_ticker = duplicate_rows.groupby(
                    identity_columns,
                    dropna=False,
                )['ticker'].transform('nunique') > 1
                cross_ticker_duplicate_ratio = float(cross_ticker.sum() / len(price_data))

        cadence_match_ratio = None
        if 'datetime' in price_data.columns:
            timestamps = pd.to_datetime(price_data['datetime'], errors='coerce', utc=True)
            interval_values = (
                price_data['interval'].astype(str).str.lower()
                if 'interval' in price_data.columns
                else pd.Series('', index=price_data.index)
            )
            ticker_values = (
                price_data['ticker'].astype(str)
                if 'ticker' in price_data.columns
                else pd.Series('__single__', index=price_data.index)
            )
            cadence_parts = []
            expected_minutes = {'15m': 15.0, '60m': 60.0, '1h': 60.0, '1d': 1440.0}
            timing = pd.DataFrame(
                {
                    'datetime': timestamps,
                    'ticker': ticker_values,
                    'interval': interval_values,
                }
            ).dropna(subset=['datetime'])
            for (_ticker, interval), group in timing.groupby(['ticker', 'interval']):
                expected = expected_minutes.get(interval)
                if expected is None:
                    continue
                deltas = group.sort_values('datetime')['datetime'].diff().dt.total_seconds().div(60).dropna()
                if not deltas.empty:
                    cadence_parts.extend((deltas == expected).tolist())
            if cadence_parts:
                cadence_match_ratio = float(np.mean(cadence_parts))

        extreme_return_ratio = 0.0
        if {'ticker', 'close'}.issubset(price_data.columns):
            working = price_data.copy()
            working['close'] = pd.to_numeric(working['close'], errors='coerce')
            if 'datetime' in working.columns:
                working['_quality_datetime'] = pd.to_datetime(
                    working['datetime'],
                    errors='coerce',
                    utc=True,
                )
                working = working.sort_values(['ticker', '_quality_datetime'])
            returns = working.groupby('ticker', sort=False)['close'].pct_change(
                fill_method=None
            ).abs()
            finite_returns = returns.loc[returns.map(lambda value: math.isfinite(value) if pd.notna(value) else False)]
            if not finite_returns.empty:
                extreme_return_ratio = float((finite_returns > 0.50).mean())
        return {
            'cross_ticker_duplicate_ratio': cross_ticker_duplicate_ratio,
            'cadence_match_ratio': cadence_match_ratio,
            'extreme_return_ratio': extreme_return_ratio,
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
        if 'close' not in price_data.columns:
            return []

        prices = price_data['close']
        mean = prices.mean()
        std = prices.std()

        if std == 0:
            return []

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
