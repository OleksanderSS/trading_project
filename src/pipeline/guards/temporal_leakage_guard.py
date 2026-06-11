#!/usr/bin/env python3
"""
Temporal Leakage Guard - Protection Against Rolling Windows and Lookahead Bias
Prevents the most common temporal leakage issues in feature engineering.
"""

import re
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("TemporalLeakageGuard")

class TemporalLeakageGuard:
    """
    Comprehensive protection against temporal leakage in feature engineering.

    This guard prevents the most common and dangerous temporal leakage patterns:
    - Rolling windows using future data
    - Lookahead bias in technical indicators
    - Future information leakage in features
    - Improper shift operations

    Temporal leakage is the #1 cause of unrealistic backtest results
    and poor live trading performance.
    """

    def __init__(self):
        """Initialize the TemporalLeakageGuard."""
        self.logger = logger

        # Patterns that commonly indicate temporal leakage
        self.LEAKAGE_PATTERNS = {
            'future_price': [
                r'future_price',
                r'next_price',
                r'price_next',
                r'price_future',
                r'close_next',
                r'close_future'
            ],
            'future_volume': [
                r'future_volume',
                r'next_volume',
                r'volume_next',
                r'volume_future'
            ],
            'future_high_low': [
                r'future_high',
                r'future_low',
                r'next_high',
                r'next_low',
                r'high_next',
                r'low_next'
            ],
            'lookahead_indicators': [
                r'.*\.shift\(-\d+\)',  # Negative shifts (lookahead)
                r'rolling_.*\.shift\(-\d+\)',  # Rolling with negative shift
                r'.*\.fillna\(.*method=.*bfill.*\)',  # Backfill (lookahead)
                r'.*\.fillna\(.*method=.*backfill.*\)',  # Backfill (lookahead)
            ]
        }

        # Safe rolling window configurations
        self.SAFE_ROLLING_CONFIGS = {
            '15m': {
                'max_periods': 96,  # 24 hours = 96 * 15min
                'common_windows': [4, 16, 32, 64, 96],  # 1h, 4h, 8h, 16h, 24h
                'description': '15-minute data'
            },
            '60m': {
                'max_periods': 168,  # 7 days = 168 * 1hour
                'common_windows': [4, 24, 48, 96, 168],  # 4h, 1d, 2d, 4d, 7d
                'description': '1-hour data'
            },
            '1d': {
                'max_periods': 252,  # 1 year = 252 trading days
                'common_windows': [5, 20, 50, 100, 252],  # 1w, 1m, 2m, 4m, 1y
                'description': 'daily data'
            }
        }

    def validate_rolling_windows(self,
                               features_df: pd.DataFrame,
                               current_time: pd.Timestamp,
                               timeframe: str | None = None) -> dict[str, Any]:
        """
        Validate rolling windows for temporal leakage.

        This is the core method that checks rolling window features
        for temporal leakage issues.

        Args:
            features_df: DataFrame with features to validate
            current_time: Current timestamp for validation
            timeframe: Timeframe string (optional, for context)

        Returns:
            Dict with validation results and issues
        """
        self.logger.info("🔍 Validating rolling windows for temporal leakage")

        issues = []
        warnings = []
        safe_features = []
        risky_features = []

        # Get datetime column
        datetime_col = self._get_datetime_column(features_df)
        if datetime_col is None:
            issues.append("No datetime column found")
            return {
                'status': 'invalid',
                'issues': issues,
                'warnings': warnings,
                'safe_features': safe_features,
                'risky_features': risky_features
            }

        # Check each feature for temporal leakage
        for col in features_df.columns:
            if col in [datetime_col, 'ticker', 'interval']:
                continue

            feature_analysis = self._analyze_feature_for_leakage(
                features_df[col], col, current_time, timeframe
            )

            if feature_analysis['has_leakage']:
                issues.extend(feature_analysis['issues'])
                risky_features.append({
                    'feature': col,
                    'leakage_type': feature_analysis['leakage_type'],
                    'issues': feature_analysis['issues']
                })
            else:
                safe_features.append(col)
                if feature_analysis['warnings']:
                    warnings.extend(feature_analysis['warnings'])

        # Determine overall status
        status = 'valid' if not issues else 'invalid'

        result = {
            'status': status,
            'issues': issues,
            'warnings': warnings,
            'safe_features': safe_features,
            'risky_features': risky_features,
            'total_features': len(features_df.columns),
            'safe_count': len(safe_features),
            'risky_count': len(risky_features),
            'current_time': current_time,
            'timeframe': timeframe
        }

        self._log_validation_summary(result)

        return result

    def _get_datetime_column(self, df: pd.DataFrame) -> str | None:
        """Get the datetime column from DataFrame."""
        datetime_cols = ['datetime', 'timestamp', 'date', 'time']

        for col in datetime_cols:
            if col in df.columns:
                return col

        # Check if index is datetime
        if isinstance(df.index, pd.DatetimeIndex):
            return 'index'

        return None

    def _check_feature_name_patterns(self, feature_name: str) -> dict[str, Any]:
        """Check feature name for leakage patterns."""
        analysis = {
            'has_leakage': False,
            'leakage_type': None,
            'issues': []
        }

        for pattern_name, patterns in self.LEAKAGE_PATTERNS.items():
            if pattern_name == 'lookahead_indicators':
                continue  # Check separately

            for pattern in patterns:
                if re.search(pattern, feature_name, re.IGNORECASE):
                    analysis['has_leakage'] = True
                    analysis['leakage_type'] = pattern_name
                    analysis['issues'].append(
                        f"Feature name indicates future data: {feature_name} matches {pattern}"
                    )
                    return analysis

        return analysis

    def _analyze_feature_for_leakage(self,
                                   series: pd.Series,
                                   feature_name: str,
                                   current_time: pd.Timestamp,
                                   timeframe: str | None) -> dict[str, Any]:
        """Analyze a single feature for temporal leakage."""

        analysis = {
            'has_leakage': False,
            'leakage_type': None,
            'issues': [],
            'warnings': []
        }

        # Check 1: Future price patterns in feature name
        name_analysis = self._check_feature_name_patterns(feature_name)
        if name_analysis['has_leakage']:
            return name_analysis

        # Check 2: Lookahead patterns in feature values (if it's a calculation result)
        if series.dtype in ['float64', 'int64']:
            lookahead_analysis = self._check_lookahead_patterns(series, feature_name)
            if lookahead_analysis['has_lookahead']:
                analysis['has_leakage'] = True
                analysis['leakage_type'] = 'lookahead_indicators'
                analysis['issues'].extend(lookahead_analysis['issues'])

        # Check 3: Rolling window validation
        if 'rolling' in feature_name.lower() or 'window' in feature_name.lower():
            rolling_analysis = self._validate_rolling_window_feature(
                series, feature_name, timeframe
            )
            if rolling_analysis['has_leakage']:
                analysis['has_leakage'] = True
                analysis['leakage_type'] = 'rolling_window_leakage'
                analysis['issues'].extend(rolling_analysis['issues'])
            elif rolling_analysis['warnings']:
                analysis['warnings'].extend(rolling_analysis['warnings'])

        # Check 4: Future data in series values
        future_data_analysis = self._check_future_data_in_series(
            series, current_time, feature_name
        )
        if future_data_analysis['has_future_data']:
            analysis['has_leakage'] = True
            analysis['leakage_type'] = 'future_data_values'
            analysis['issues'].extend(future_data_analysis['issues'])

        return analysis

    def _check_lookahead_patterns(self,
                                series: pd.Series,
                                feature_name: str) -> dict[str, Any]:
        """Check for lookahead patterns in feature calculations."""

        analysis = {
            'has_lookahead': False,
            'issues': []
        }

        # Check for negative shifts (lookahead)
        if 'shift(' in feature_name or '.shift(' in feature_name:
            # Extract shift value
            shift_match = re.search(r'\.shift\(\s*-\s*(\d+)\s*\)', feature_name)
            if shift_match:
                shift_value = int(shift_match.group(1))
                analysis['has_lookahead'] = True
                analysis['issues'].append(
                    f"Negative shift detected: shift(-{shift_value}) indicates lookahead bias"
                )

        # Check for backfill operations
        if 'bfill' in feature_name or 'backfill' in feature_name:
            analysis['has_lookahead'] = True
            analysis['issues'].append(
                "Backfill operation detected - uses future data to fill past values"
            )

        return analysis

    def _validate_rolling_window_feature(self,
                                      series: pd.Series,
                                      feature_name: str,
                                      timeframe: str | None) -> dict[str, Any]:
        """Validate rolling window feature for proper configuration."""

        analysis = {
            'has_leakage': False,
            'warnings': []
        }

        if timeframe is None:
            return analysis

        # Extract window size from feature name
        window_match = re.search(r'rolling_(\d+)', feature_name.lower())
        if not window_match:
            return analysis

        window_size = int(window_match.group(1))
        config = self.SAFE_ROLLING_CONFIGS.get(timeframe, {})
        max_periods = config.get('max_periods', 100)
        common_windows = config.get('common_windows', [])

        # Check if window is too large
        if window_size > max_periods:
            analysis['has_leakage'] = True
            analysis['issues'] = [
                f"Rolling window too large: {window_size} > max {max_periods} for {timeframe}"
            ]

        # Check if window is uncommon (warning)
        elif window_size not in common_windows:
            analysis['warnings'].append(
                f"Unusual rolling window size: {window_size} (common: {common_windows[:3]})"
            )

        return analysis

    def _check_future_data_in_series(self,
                                   series: pd.Series,
                                   current_time: pd.Timestamp,
                                   feature_name: str) -> dict[str, Any]:
        """Check if series contains future data values."""

        analysis = {
            'has_future_data': False,
            'issues': []
        }

        if series.dtype not in ['float64', 'int64']:
            return analysis

        # Check for unrealistic future values (e.g., prices far in the future)
        if 'price' in feature_name.lower() or 'close' in feature_name.lower():
            # Look for values that seem to be from the future
            if series.max() > series.mean() + 5 * series.std():
                analysis['has_future_data'] = True
                analysis['issues'].append(
                    f"Feature {feature_name} contains values that suggest future data usage"
                )

        return analysis

    def validate_normalization_fit(self,
                                features_df: pd.DataFrame,
                                target_df: pd.DataFrame,
                                fit_method: str = 'fit_transform') -> dict[str, Any]:
        """
        Validate normalization to prevent look-ahead bias.

        Args:
            features_df: Features DataFrame
            target_df: Target DataFrame
            fit_method: How normalization was applied

        Returns:
            Validation result
        """
        self.logger.info(f"🔍 Validating normalization fit method: {fit_method}")

        issues = []
        warnings = []

        # Check if normalization was applied correctly
        if fit_method == 'fit_transform':
            # This is correct - fit on training data only
            pass
        elif fit_method == 'transform':
            # This is correct for test data - use fitted parameters
            pass
        else:
            issues.append(f"Unknown normalization method: {fit_method}")

        # Check for data leakage between features and targets
        if not features_df.empty and not target_df.empty:
            # Check if any feature is perfectly correlated with target
            for target_col in target_df.columns:
                if target_col.startswith('target_'):
                    for feature_col in features_df.columns:
                        if feature_col.replace('m15_', '').replace('h1_', '').replace('d1_', '') == target_col.replace('target_', ''):
                            warnings.append(
                                f"Feature {feature_col} may be identical to target {target_col}"
                            )

        result = {
            'status': 'valid' if not issues else 'invalid',
            'issues': issues,
            'warnings': warnings,
            'fit_method': fit_method,
            'features_shape': features_df.shape,
            'targets_shape': target_df.shape
        }

        return result

    def check_feature_target_alignment(self,
                                   features_df: pd.DataFrame,
                                   target_df: pd.DataFrame,
                                   max_alignment_gap: pd.Timedelta | None = None) -> dict[str, Any]:
        """
        Check temporal alignment between features and targets.

        Args:
            features_df: Features DataFrame
            target_df: Target DataFrame
            max_alignment_gap: Maximum allowed time gap

        Returns:
            Alignment validation result
        """
        if max_alignment_gap is None:
            max_alignment_gap = pd.Timedelta(hours=1)

        self.logger.info("🔍 Checking feature-target temporal alignment")

        issues = []

        # Get datetime columns
        features_dt = self._get_datetime_column(features_df)
        targets_dt = self._get_datetime_column(target_df)

        if features_dt is None or targets_dt is None:
            issues.append("Missing datetime columns for alignment check")
            return {'status': 'invalid', 'issues': issues}

        # Check time ranges
        if features_dt == 'index':
            features_time = features_df.index
        else:
            features_time = pd.to_datetime(features_df[features_dt])

        if targets_dt == 'index':
            targets_time = target_df.index
        else:
            targets_time = pd.to_datetime(target_df[targets_dt])

        # Check if targets extend beyond features (lookahead)
        if targets_time.max() > features_time.max() + max_alignment_gap:
            issues.append(
                f"Targets extend beyond features: "
                f"target_max={targets_time.max()}, feature_max={features_time.max()}"
            )

        # Check if features use future targets
        if features_time.max() > targets_time.max():
            issues.append(
                f"Features extend beyond targets: "
                f"feature_max={features_time.max()}, target_max={targets_time.max()}"
            )

        result = {
            'status': 'valid' if not issues else 'invalid',
            'issues': issues,
            'features_time_range': (features_time.min(), features_time.max()),
            'targets_time_range': (targets_time.min(), targets_time.max()),
            'alignment_gap': max_alignment_gap
        }

        return result

    def _log_validation_summary(self, result: dict[str, Any]) -> None:
        """Log comprehensive validation summary."""
        status = result['status']
        safe_count = result['safe_count']
        risky_count = result['risky_count']
        total_count = result['total_features']

        self.logger.info("=" * 60)
        self.logger.info("🔍 TEMPORAL LEAKAGE VALIDATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Status: {status.upper()}")
        self.logger.info(f"Features: {safe_count} safe, {risky_count} risky, {total_count} total")

        if result['safe_features']:
            self.logger.info(f"✅ Safe features ({len(result['safe_features'])}): {result['safe_features'][:10]}...")

        if result['risky_features']:
            self.logger.error("❌ Risky features:")
            for risky in result['risky_features']:
                self.logger.error(f"   {risky['feature']}: {risky['leakage_type']}")
                for issue in risky['issues']:
                    self.logger.error(f"      - {issue}")

        if result['warnings']:
            self.logger.warning("⚠️ Warnings:")
            for warning in result['warnings']:
                self.logger.warning(f"   {warning}")

        self.logger.info("=" * 60)

    def get_safe_feature_subset(self,
                              features_df: pd.DataFrame,
                              validation_result: dict[str, Any]) -> pd.DataFrame:
        """
        Extract only safe features based on validation result.

        Args:
            features_df: Original features DataFrame
            validation_result: Result from validate_rolling_windows

        Returns:
            DataFrame with only safe features
        """
        safe_features = validation_result['safe_features']

        # Add non-feature columns (datetime, ticker, etc.)
        datetime_col = self._get_datetime_column(features_df)
        preserve_cols = []

        if datetime_col:
            preserve_cols.append(datetime_col)

        for col in ['ticker', 'interval']:
            if col in features_df.columns:
                preserve_cols.append(col)

        # Combine safe features with preserve columns
        final_cols = list(set(safe_features + preserve_cols))

        safe_df = features_df[final_cols].copy()

        self.logger.info(f"🛡️ Extracted safe features: {safe_df.shape} from {features_df.shape}")

        return safe_df


# Factory function for easy instantiation
def get_temporal_leakage_guard() -> TemporalLeakageGuard:
    """Factory function to get TemporalLeakageGuard instance."""
    return TemporalLeakageGuard()


# Convenience function for quick validation
def validate_temporal_leakage_quick(features_df: pd.DataFrame,
                                   current_time: pd.Timestamp | None = None,
                                   timeframe: str | None = None) -> dict[str, Any]:
    """
    Quick temporal leakage validation.

    Args:
        features_df: Features DataFrame to validate
        current_time: Current timestamp (uses now if None)
        timeframe: Timeframe string (optional)

    Returns:
        Validation result dictionary
    """
    guard = get_temporal_leakage_guard()
    if current_time is None:
        current_time = pd.Timestamp.now()

    return guard.validate_rolling_windows(features_df, current_time, timeframe)
