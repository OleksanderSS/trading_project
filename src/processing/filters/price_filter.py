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
        # 2 was meaningless: the quality assessment computes a standard
        # deviation, a cadence match ratio and a duplicate ratio, none of which
        # mean anything over two bars -- yet a 2-bar series passed the gate and
        # was then scored as if the numbers were informative. 30 is a
        # conservative statistical floor; the smallest series actually stored
        # here is 322 bars, so no real data is affected.
        self.min_candles = config.get('min_candles_per_timeframe', 30)
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

        # Every `continue` below removes an ENTIRE TIMEFRAME from the
        # pipeline. Each one recorded its reason in quality_report and said
        # nothing else, and nobody downstream reads that report -- so a
        # timeframe could be requested, collected, and then dropped here in
        # silence. That is the shape of the 2026-08-04 batch, which recorded
        # timeframes ['15m','1d','1h'] and delivered two: no 15m features, no
        # 15m targets, and 0 of 506 champions on 15m, with nothing in any log
        # to say why. The reasons are now logged where they are decided.
        for timeframe, tf_data in price_data.items():
            if not isinstance(tf_data, pd.DataFrame) or tf_data.empty:
                quality_report[timeframe] = {'status': 'empty', 'reason': 'no_data'}
                self.logger.error(
                    "Timeframe '%s' DROPPED: arrived empty. Nothing "
                    "downstream will exist for it.", timeframe,
                )
                continue

            # Integrity checks run on EVERY series, regardless of length.
            # Contamination is contamination on eight bars as much as on eight
            # hundred, and cross-ticker duplicate OHLCV is exactly the kind of
            # defect that shows up in a small slice. Only the STATISTICAL
            # quality score needs a usable sample, so the length gate sits
            # below these rather than above them -- putting it first meant
            # raising min_candles silently disabled the contamination guard.
            data_quality = self.assess_price_quality(tf_data)

            hard_failures = []
            if data_quality.get('cross_ticker_duplicate_ratio', 0.0) > self.max_cross_ticker_duplicate_ratio:
                hard_failures.append('cross_ticker_duplicate_ohlcv')
            cadence_ratio = data_quality.get('cadence_match_ratio')
            if cadence_ratio is not None and cadence_ratio < self.min_cadence_match_ratio:
                hard_failures.append('timeframe_cadence_mismatch')
            if data_quality.get('extreme_return_ratio', 0.0) > self.max_extreme_return_ratio:
                hard_failures.append('extreme_return_contamination')

            if hard_failures:
                quality_report[timeframe] = {
                    'status': 'low_quality',
                    'reason': ','.join(hard_failures),
                    'hard_failures': hard_failures,
                    **data_quality
                }
                self.logger.error(
                    "Timeframe '%s' DROPPED on %s (%d rows). cadence_match=%s, "
                    "extreme_return_ratio=%s. Thresholds: cadence>=%.2f, "
                    "extreme<=%.3f.",
                    timeframe, ','.join(hard_failures), len(tf_data),
                    data_quality.get('cadence_match_ratio'),
                    data_quality.get('extreme_return_ratio'),
                    self.min_cadence_match_ratio, self.max_extreme_return_ratio,
                )
                continue

            if len(tf_data) < self.min_candles:
                quality_report[timeframe] = {
                    'status': 'insufficient_data',
                    'reason': f'only_{len(tf_data)}_candles'
                }
                self.logger.error(
                    "Timeframe '%s' DROPPED: %d candle(s), minimum is %d.",
                    timeframe, len(tf_data), self.min_candles,
                )
                continue

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
                self.logger.error(
                    "Timeframe '%s' DROPPED: quality score %.2f below the "
                    "%.2f minimum (%d rows).",
                    timeframe, data_quality['overall_score'],
                    self.min_quality, len(tf_data),
                )
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
        """Detect anomalous BAR-TO-BAR MOVES, not unusual price levels.

        This used to z-score the close price against the mean of the whole
        series:

            z = (price - prices.mean()) / prices.std()

        For anything that trends, the series mean is a level the price passed
        through once, so |z| measures distance from that level rather than
        whether a bar is anomalous. Measured on real stored data across NVDA,
        KO, SPY and TSLA, injecting a single bad tick:

            +15% in one bar -> MISSED on all four
            +30% in one bar -> MISSED on all four
            +100%           -> caught

        A 30% single-day jump is an unmistakable data error and it went
        undetected. The same check simultaneously produced false positives:
        on untouched KO daily data it flagged the two highest closes (89.08
        and 88.49 in a 58-89 range) as "spikes" -- legitimate trend extremes.

        Scoring the RETURN catches all of the above cases and stops flagging
        trend extremes, because a normal bar at a new high is still a normal
        move.

        Anomalies are reported, not acted on -- filter_price_data counts them
        and drops nothing -- so this changes the quality report rather than
        what data flows onward.
        """
        if 'close' not in price_data.columns or len(price_data) < 3:
            return []

        prices = pd.to_numeric(price_data['close'], errors='coerce')
        returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        clean = returns.dropna()
        if len(clean) < 2:
            return []

        std = clean.std()
        if not np.isfinite(std) or std <= 1e-12:
            return []

        z_scores = (returns - clean.mean()) / std
        mask = z_scores.abs() > self.anomaly_threshold

        anomalies = []
        for idx in price_data.index[mask.fillna(False)]:
            z = float(z_scores.loc[idx])
            anomalies.append({
                'timestamp': idx,
                'value': float(prices.loc[idx]),
                'return_pct': float(returns.loc[idx]),
                'z_score': z,
                'type': 'spike' if z > 0 else 'dip',
            })
        return anomalies
