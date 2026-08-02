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
            # NOTE: there used to be a 'lookahead_indicators' group here
            # holding r'.*\.shift\(-\d+\)', r'.*\.fillna\(.*bfill.*\)' and
            # friends, matched against FEATURE NAMES. A column is never named
            # "close.shift(-1)" -- those are Python expressions, and this is a
            # runtime check on a DataFrame's columns. Measured on the
            # 2026-08-02 export: 0 of 1,189 names contain "shift(", 0 contain
            # "bfill". The check could not fire, ever, by construction.
            #
            # Detecting a negative shift is a real and valuable check -- it
            # just belongs to a SOURCE scanner, where the expression actually
            # exists. It now lives in
            # tests/contracts/test_lookahead_operations.py, which reads src/
            # and honours the project's existing
            # "# audit-ignore: NEGATIVE_SHIFT_*" markers.
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

        # Check 2: lookback window size. The gate used to be
        # `if 'rolling' in name or 'window' in name`, and then the size was
        # read with r'rolling_(\d+)'. This project names windows
        # NAME_<periods>[_<timeframe>] -- SMA_200_60m, ATR_14_1d, AATR_14_1d.
        # Measured on the export: 36 of 1,189 names contain "rolling" or
        # "window", and ZERO match rolling_(\d+), so this never ran either.
        # 312 names carry a window under the real convention.
        rolling_analysis = self._validate_rolling_window_feature(
            series, feature_name, timeframe
        )
        analysis['warnings'].extend(rolling_analysis.get('warnings', []))

        # Check 4: Future data in series values
        future_data_analysis = self._check_future_data_in_series(
            series, current_time, feature_name
        )
        if future_data_analysis['has_future_data']:
            analysis['has_leakage'] = True
            analysis['leakage_type'] = 'future_data_values'
            analysis['issues'].extend(future_data_analysis['issues'])

        return analysis

    #: NAME_<periods> with an optional timeframe suffix -- the convention this
    #: project actually uses (SMA_200_60m, ATR_14_1d, AATR_14_1d).
    _WINDOW_IN_NAME = re.compile(
        r'_(\d+)(?:_(?:5m|15m|30m|60m|1h|1d|daily))?$', re.IGNORECASE
    )

    def _validate_rolling_window_feature(self,
                                      series: pd.Series,
                                      feature_name: str,
                                      timeframe: str | None) -> dict[str, Any]:
        """Flag a lookback longer than this timeframe's budget.

        WARNING ONLY, deliberately, and this is the substantive change.

        A long trailing window is NOT lookahead bias -- SMA_200 on hourly
        bars reads two hundred bars into the PAST and not one into the
        future. SAFE_ROLLING_CONFIGS encodes a modelling opinion ("do not
        look back more than a week on hourly data"), which is a reasonable
        thing to be told and an unreasonable thing to die for.

        That distinction was about to matter. FeatureGuards treats "Rolling
        window too large" as fatal and raises. Repairing the pattern without
        repairing the severity would have made Stage 3 abort on
        SMA_200_60m / EMA_200_60m -- 200 > the 168 budget -- four columns
        that are in the current export and contain no leakage whatsoever.
        """
        analysis: dict[str, Any] = {'has_leakage': False, 'warnings': []}

        if timeframe is None:
            return analysis

        match = self._WINDOW_IN_NAME.search(str(feature_name))
        if not match:
            return analysis

        window_size = int(match.group(1))
        config = self.SAFE_ROLLING_CONFIGS.get(timeframe, {})
        max_periods = config.get('max_periods')
        if not max_periods:
            # An unknown timeframe has no budget to exceed. Silence beats a
            # made-up default of 100, which would flag every 200-period
            # average on data whose cadence we could not identify.
            return analysis

        if window_size > max_periods:
            analysis['warnings'].append(
                f"Lookback longer than the {timeframe} budget: {feature_name} "
                f"spans {window_size} periods, budget {max_periods}. Not "
                f"leakage -- it reads only past bars -- but it reaches "
                f"further back than this timeframe is configured to trust."
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

        # Skip check for known safe features that can have high values
        safe_patterns = ['price_volume_trend', 'market_context_price_to_ma', 'volume_ratio', 'obv']
        if any(pattern in feature_name.lower() for pattern in safe_patterns):
            return analysis

        # Check for unrealistic future values (e.g., prices far in the future)
        if 'price' in feature_name.lower() or 'close' in feature_name.lower():
            # More conservative threshold: 10 standard deviations
            # Also check if there are extreme outliers (more than 1% of data)
            threshold = series.mean() + 10 * series.std()
            extreme_outliers = (series > threshold).sum()
            outlier_ratio = extreme_outliers / len(series)

            if series.max() > threshold and outlier_ratio > 0.01:
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
