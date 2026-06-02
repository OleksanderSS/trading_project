"""
Missing Data Anomaly Detector - Detects and reports anomalies in filled data.
Monitors fill quality and identifies potential data integrity issues.
"""

from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("MissingDataAnomalyDetector")


class MissingDataAnomalyDetector:
    """
    Detects anomalies in filled missing data to ensure data quality.

    Key features:
    - Zero-fill detection
    - Sudden change detection after fills
    - Pattern analysis for fill quality
    - Statistical anomaly detection
    - Fill quality scoring
    """

    def __init__(self):
        self.anomaly_thresholds = {
            'zero_fill_z_score': 3.0,  # Z-score threshold for zero-fill detection
            'sudden_change_z_score': 4.0,  # Z-score threshold for sudden changes
            'pattern_deviation_threshold': 0.3,  # Deviation from expected patterns
            'min_fill_samples': 5,  # Minimum samples to consider for pattern analysis
        }

    def detect_fill_anomalies(self, original_df: pd.DataFrame, filled_df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Detect various types of fill anomalies.

        Args:
            original_df: DataFrame with original NaN values
            filled_df: DataFrame after filling

        Returns:
            List of anomaly dictionaries with details
        """
        anomalies = []

        for col in filled_df.columns:
            col_anomalies = self._detect_column_anomalies(
                original_df[col], filled_df[col], col
            )
            anomalies.extend(col_anomalies)

        # Global anomalies across columns
        global_anomalies = self._detect_global_anomalies(original_df, filled_df)
        anomalies.extend(global_anomalies)

        # Score overall fill quality
        quality_score = self._calculate_fill_quality_score(original_df, filled_df, anomalies)

        logger.info(f"MissingDataAnomalyDetector: Found {len(anomalies)} anomalies, quality score: {quality_score:.2f}")

        return anomalies

    def _detect_column_anomalies(self, original_series: pd.Series, filled_series: pd.Series, col_name: str) -> list[dict[str, Any]]:
        """Detect anomalies in a single column."""
        anomalies = []

        # Find where original was NaN but filled is not
        fill_mask = original_series.isna() & filled_series.notna()
        filled_values = filled_series[fill_mask]

        if len(filled_values) == 0:
            return anomalies

        # 1. Zero-fill anomalies
        zero_fill_anomalies = self._detect_zero_fill_anomalies(filled_values, col_name)
        anomalies.extend(zero_fill_anomalies)

        # 2. Sudden change anomalies
        sudden_change_anomalies = self._detect_sudden_change_anomalies(
            original_series, filled_series, fill_mask, col_name
        )
        anomalies.extend(sudden_change_anomalies)

        # 3. Pattern anomalies
        pattern_anomalies = self._detect_pattern_anomalies(filled_values, col_name)
        anomalies.extend(pattern_anomalies)

        # 4. Statistical anomalies
        stat_anomalies = self._detect_statistical_anomalies(filled_values, original_series, col_name)
        anomalies.extend(stat_anomalies)

        return anomalies

    def _detect_zero_fill_anomalies(self, filled_values: pd.Series, col_name: str) -> list[dict[str, Any]]:
        """Detect suspicious zero fills."""
        anomalies = []

        # Check for excessive zeros
        zero_count = (filled_values == 0).sum()
        total_filled = len(filled_values)
        zero_ratio = zero_count / total_filled if total_filled > 0 else 0

        # Flag if too many zeros (unless it's volume data)
        if zero_ratio > 0.5 and 'volume' not in col_name.lower():
            anomalies.append({
                'type': 'excessive_zero_fill',
                'column': col_name,
                'zero_ratio': zero_ratio,
                'zero_count': zero_count,
                'total_filled': total_filled,
                'severity': 'high' if zero_ratio > 0.8 else 'medium'
            })

        # Check for zero fills in inappropriate contexts
        if zero_count > 0:
            # Zeros in price data are suspicious
            if any(price in col_name.lower() for price in ['open', 'high', 'low', 'close']):
                anomalies.append({
                    'type': 'price_zero_fill',
                    'column': col_name,
                    'zero_count': zero_count,
                    'severity': 'high'
                })

            # Zeros in oscillating indicators (RSI, Stochastic) are suspicious
            if any(osc in col_name.lower() for osc in ['rsi', 'stoch', 'williams']):
                anomalies.append({
                    'type': 'oscillator_zero_fill',
                    'column': col_name,
                    'zero_count': zero_count,
                    'severity': 'medium'
                })

        return anomalies

    def _detect_sudden_change_anomalies(self, original_series: pd.Series, filled_series: pd.Series,
                                     fill_mask: pd.Series, col_name: str) -> list[dict[str, Any]]:
        """Detect sudden changes after fills."""
        anomalies = []

        # Find transitions from NaN to filled values, shifted by 1 to represent past transitions
        shifted_fill_mask = fill_mask.shift(1)
        fill_transitions = fill_mask & shifted_fill_mask.where(shifted_fill_mask.notna(), False)

        if not fill_transitions.any():
            return anomalies

        # Build a positional lookup for the index to find previous element safely
        index_list = list(filled_series.index)
        index_pos = {v: i for i, v in enumerate(index_list)}

        # Calculate changes at fill points
        for idx in filled_series.index[fill_transitions]:
            pos = index_pos.get(idx, 0)
            if pos == 0:  # Skip first point — no previous element
                continue

            # Get previous value by positional index (works for any index type)
            try:
                prev_idx = index_list[pos - 1]
                prev_val = filled_series.iloc[pos - 1]
            except (KeyError, IndexError):
                prev_val = None

            filled_val = filled_series.loc[idx]

            if prev_val is not None and filled_val is not None:
                change_magnitude = abs(filled_val - prev_val)

                # Calculate Z-score based on recent volatility excluding current point
                recent_values = filled_series.loc[:idx].shift(1).tail(20)
                if len(recent_values) > 1:
                    recent_std = recent_values.std()
                    if recent_std > 0:
                        z_score = change_magnitude / recent_std

                        if z_score > self.anomaly_thresholds['sudden_change_z_score']:
                            anomalies.append({
                                'type': 'sudden_change_after_fill',
                                'index': idx,
                                'magnitude': change_magnitude,
                                'z_score': z_score,
                                'prev_value': prev_val,
                                'filled_value': filled_val,
                                'severity': 'high' if z_score > 6 else 'medium'
                            })

        return anomalies

    def _detect_pattern_anomalies(self, filled_values: pd.Series, col_name: str) -> list[dict[str, Any]]:
        """Detect anomalies in fill patterns."""
        anomalies = []

        if len(filled_values) < self.anomaly_thresholds['min_fill_samples']:
            return anomalies

        # Check for repetitive patterns (indicative of lazy filling)
        unique_values = filled_values.nunique()
        total_values = len(filled_values)

        if unique_values == 1:  # All filled values are the same
            anomalies.append({
                'type': 'constant_fill_pattern',
                'column': col_name,
                'constant_value': filled_values.iloc[0],
                'count': total_values,
                'severity': 'high'
            })

        # Check for linear patterns (indicative of simple interpolation)
        if total_values > 10:
            # Check if values follow perfect linear progression
            values_array = filled_values.values
            is_linear = True
            expected_diff = values_array[1] - values_array[0]

            for i in range(2, len(values_array)):
                actual_diff = values_array[i] - values_array[i-1]
                if abs(actual_diff - expected_diff) > 1e-10:  # Allow tiny floating point differences
                    is_linear = False
                    break

            if is_linear:
                anomalies.append({
                    'type': 'linear_fill_pattern',
                    'column': col_name,
                    'pattern_length': total_values,
                    'severity': 'medium'
                })

        return anomalies

    def _detect_statistical_anomalies(self, filled_values: pd.Series, original_series: pd.Series,
                                   col_name: str) -> list[dict[str, Any]]:
        """Detect statistical anomalies in filled values."""
        anomalies = []

        if len(filled_values) < 10:
            return anomalies

        # Compare filled values distribution with original non-filled values
        original_non_null = original_series.dropna()

        if len(original_non_null) > 5:  # Only if we have enough original data
            # Statistical comparison
            filled_mean = filled_values.mean()
            filled_std = filled_values.std()
            original_mean = original_non_null.mean()
            original_std = original_non_null.std()

            # Check for distribution shift
            mean_shift = abs(filled_mean - original_mean)
            mean_shift_z = mean_shift / original_std if original_std > 0 else 0

            if mean_shift_z > 2.0:  # Significant shift in distribution
                anomalies.append({
                    'type': 'distribution_shift',
                    'column': col_name,
                    'original_mean': original_mean,
                    'filled_mean': filled_mean,
                    'shift_magnitude': mean_shift,
                    'shift_z_score': mean_shift_z,
                    'severity': 'high' if mean_shift_z > 4 else 'medium'
                })

            # Check for variance change
            if original_std > 0:
                variance_ratio = filled_std / original_std
                if variance_ratio > 3.0 or variance_ratio < 0.1:  # Significant variance change
                    anomalies.append({
                        'type': 'variance_anomaly',
                        'column': col_name,
                        'original_std': original_std,
                        'filled_std': filled_std,
                        'variance_ratio': variance_ratio,
                        'severity': 'medium'
                    })

        return anomalies

    def _detect_global_anomalies(self, original_df: pd.DataFrame, filled_df: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect global anomalies across all columns."""
        anomalies = []

        # Check for systematic filling issues
        fill_patterns = {}
        for col in filled_df.columns:
            fill_mask = original_df[col].isna() & filled_df[col].notna()
            if fill_mask.any():
                fill_patterns[col] = {
                    'fill_count': fill_mask.sum(),
                    'fill_timestamps': filled_df.index[fill_mask].tolist()
                }

        # Check if many columns were filled at the same time (systematic issue)
        if len(fill_patterns) > 1:
            timestamp_counts = {}
            for col, pattern in fill_patterns.items():
                for timestamp in pattern['fill_timestamps']:
                    timestamp_counts[timestamp] = timestamp_counts.get(timestamp, 0) + 1

            # Find timestamps with many simultaneous fills
            for timestamp, count in timestamp_counts.items():
                if count > len(filled_df.columns) * 0.3:  # More than 30% of columns filled at same time
                    anomalies.append({
                        'type': 'systematic_fill_anomaly',
                        'timestamp': timestamp,
                        'columns_filled': count,
                        'total_columns': len(filled_df.columns),
                        'severity': 'high'
                    })

        return anomalies

    def _calculate_fill_quality_score(self, original_df: pd.DataFrame, filled_df: pd.DataFrame,
                                 anomalies: list[dict[str, Any]]) -> float:
        """
        Calculate overall fill quality score (0-100).

        Higher score = better fill quality.
        """
        # Base score starts at 100
        score = 100.0

        # Deduct points for each anomaly
        for anomaly in anomalies:
            severity_multiplier = {
                'low': 5,
                'medium': 15,
                'high': 30
            }

            multiplier = severity_multiplier.get(anomaly.get('severity', 'medium'), 15)
            score -= multiplier

        # Bonus points for good practices
        total_missing = original_df.isna().sum().sum()
        total_filled = (~original_df.isna() & filled_df.notna()).sum().sum()

        if total_missing > 0:
            fill_rate = total_filled / total_missing
            if fill_rate > 0.95:  # Good fill rate
                score += 5

        # Ensure score stays in bounds
        score = max(0, min(100, score))

        return score

    def generate_anomaly_report(self, anomalies: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate comprehensive anomaly report."""
        if not anomalies:
            return {
                'total_anomalies': 0,
                'quality_score': 100,
                'severity_breakdown': {'low': 0, 'medium': 0, 'high': 0},
                'anomaly_types': {},
                'recommendations': ['No anomalies detected - fill quality is excellent']
            }

        # Categorize anomalies
        severity_counts = {'low': 0, 'medium': 0, 'high': 0}
        type_counts = {}

        for anomaly in anomalies:
            severity = anomaly.get('severity', 'medium')
            severity_counts[severity] += 1

            anomaly_type = anomaly.get('type', 'unknown')
            type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1

        # Generate recommendations
        recommendations = self._generate_recommendations(anomalies)

        return {
            'total_anomalies': len(anomalies),
            'quality_score': self._calculate_fill_quality_score(pd.DataFrame(), pd.DataFrame(), anomalies),
            'severity_breakdown': severity_counts,
            'anomaly_types': type_counts,
            'anomalies': anomalies,
            'recommendations': recommendations
        }

    def _generate_recommendations(self, anomalies: list[dict[str, Any]]) -> list[str]:
        """Generate recommendations based on detected anomalies."""
        recommendations = []

        anomaly_types = {a.get('type', '') for a in anomalies}

        if 'excessive_zero_fill' in anomaly_types:
            recommendations.append("Consider using forward-fill or interpolation instead of zero-filling")

        if 'price_zero_fill' in anomaly_types:
            recommendations.append("Price data should never be zero-filled - use last known price")

        if 'sudden_change_after_fill' in anomaly_types:
            recommendations.append("Review fill logic - sudden changes may indicate incorrect filling")

        if 'constant_fill_pattern' in anomaly_types:
            recommendations.append("Avoid constant value filling - use time-aware interpolation")

        if 'linear_fill_pattern' in anomaly_types:
            recommendations.append("Linear interpolation may be too simplistic - consider more sophisticated methods")

        if 'distribution_shift' in anomaly_types:
            recommendations.append("Filled values distribution differs significantly from original - review fill strategy")

        if 'systematic_fill_anomaly' in anomaly_types:
            recommendations.append("Systematic filling issues detected - check data pipeline integrity")

        if not recommendations:
            recommendations.append("Fill quality appears good - continue monitoring")

        return recommendations
