#!/usr/bin/env python3
"""
Timeframe Alignment Guard - Critical Protection Against Temporal Leakage
Prevents the most common and dangerous error in multi-timeframe trading systems.
"""

from datetime import time
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("TimeframeAlignmentGuard")

class TimeframeAlignmentGuard:
    """
    Critical protection against temporal leakage in multi-timeframe systems.

    This is the MOST IMPORTANT guard for any trading system using multiple timeframes.
    90% of multi-timeframe systems have temporal leakage bugs that cause
    optimistic backtest results and poor live performance.

    Key protections:
    - Prevents using daily close before market close (4 PM EST)
    - Validates intraday data doesn't use future daily data
    - Ensures proper time alignment between timeframes
    - Provides safe timeframe selection for current time
    """

    # Market hours and close times (EST)
    MARKET_OPEN = time(9, 30)  # 9:30 AM EST
    MARKET_CLOSE = time(16, 0)   # 4:00 PM EST

    # Timeframe configurations
    TIMEFRAME_CONFIGS = {
        '15m': {
            'frequency': '15T',
            'market_hours_sensitive': True,
            'requires_daily_close': False,
            'max_future_lookahead': pd.Timedelta(minutes=15)
        },
        '60m': {
            'frequency': '1H',
            'market_hours_sensitive': True,
            'requires_daily_close': False,
            'max_future_lookahead': pd.Timedelta(hours=1)
        },
        '1d': {
            'frequency': '1D',
            'market_hours_sensitive': False,  # Daily close is fixed
            'requires_daily_close': True,
            'max_future_lookahead': pd.Timedelta(days=1)
        }
    }

    def __init__(self, strict_mode: bool = True):
        """Initialize the TimeframeAlignmentGuard.

        Args:
            strict_mode: If True, enforces strict market hour checks for live trading.
                        If False, allows data accumulation regardless of market hours.
        """
        self.logger = logger
        self.strict_mode = strict_mode

    def validate_timeframe_compatibility(self,
                                       features_by_tf: dict[str, pd.DataFrame],
                                       current_time: pd.Timestamp) -> dict[str, Any]:
        """
        Validate temporal compatibility of all timeframes.

        This is the core validation method that prevents temporal leakage.

        Args:
            features_by_tf: Dictionary of timeframes to DataFrames
            current_time: Current timestamp for validation

        Returns:
            Dict with validation results and issues
        """
        issues = []
        valid_timeframes = []
        warnings = []

        self.logger.info(f"🔍 Validating timeframe compatibility for {current_time}")

        for tf, df in features_by_tf.items():
            if tf not in self.TIMEFRAME_CONFIGS:
                issues.append(f"❌ {tf}: Unknown timeframe")
                continue

            if df.empty:
                warnings.append(f"⚠️ {tf}: Empty dataframe")
                continue

            # Ensure datetime column exists
            if 'datetime' not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                    df = df.rename(columns={'index': 'datetime'})
                else:
                    issues.append(f"❌ {tf}: No datetime column or index")
                    continue

            # Get latest timestamp for this timeframe
            latest_timestamp = pd.to_datetime(df['datetime'].max())

            # Validation 1: Check for future data usage
            if latest_timestamp > current_time:
                issues.append(
                    f"❌ {tf}: Uses future data "
                    f"(latest: {latest_timestamp}, current: {current_time})"
                )
                continue

            # Validation 2: Check daily close timing
            if tf == '1d':
                daily_close_issue = self._validate_daily_close_timing(latest_timestamp)
                if daily_close_issue:
                    issues.append(f"❌ {tf}: {daily_close_issue}")
                    continue

            # Validation 3: Check intraday vs daily compatibility
            if tf in ['15m', '60m']:
                intraday_issue = self._validate_intraday_daily_compatibility(
                    latest_timestamp, current_time
                )
                if intraday_issue:
                    issues.append(f"❌ {tf}: {intraday_issue}")
                    continue

            # Validation 4: Check data freshness
            freshness_issue = self._validate_data_freshness(tf, latest_timestamp, current_time)
            if freshness_issue:
                warnings.append(f"⚠️ {tf}: {freshness_issue}")

            valid_timeframes.append(tf)
            self.logger.info(f"✅ {tf}: Valid (latest: {latest_timestamp})")

        # Determine overall status
        status = 'valid' if not issues else 'invalid'

        result = {
            'status': status,
            'valid_timeframes': valid_timeframes,
            'issues': issues,
            'warnings': warnings,
            'current_time': current_time,
            'total_timeframes': len(features_by_tf),
            'valid_count': len(valid_timeframes)
        }

        if status == 'valid':
            self.logger.info(f"✅ All {len(valid_timeframes)} timeframes are temporally valid")
        else:
            self.logger.error(f"❌ Timeframe validation failed: {len(issues)} issues found")
            for issue in issues:
                self.logger.error(f"   {issue}")

        return result

    def _validate_daily_close_timing(self, daily_timestamp: pd.Timestamp) -> str | None:
        """Validate daily close timing."""
        # Daily close should be at or after 4:00 PM EST
        close_time = daily_timestamp.time()

        if close_time < self.MARKET_CLOSE:
            return f"Daily close before market close (time: {close_time}, should be >= {self.MARKET_CLOSE})"

        # Daily close should be on a trading day
        if daily_timestamp.weekday() >= 5:  # Saturday or Sunday
            return f"Daily close on weekend ({daily_timestamp.strftime('%A')})"

        return None

    def _validate_intraday_daily_compatibility(self,
                                           intraday_timestamp: pd.Timestamp,
                                           current_time: pd.Timestamp) -> str | None:
        """Validate intraday data doesn't use future daily close."""

        # In non-strict mode (data accumulation), skip market hour checks
        if not self.strict_mode:
            return None

        # Get today's market close time
        today_close = pd.Timestamp(
            current_time.year, current_time.month, current_time.day,
            self.MARKET_CLOSE.hour, self.MARKET_CLOSE.minute
        )

        # If current time is before market close, intraday shouldn't use daily data
        if current_time < today_close:
            return f"Using data before market close (current: {current_time}, close: {today_close})"

        return None

    def _validate_data_freshness(self,
                               tf: str,
                               latest_timestamp: pd.Timestamp,
                               current_time: pd.Timestamp) -> str | None:
        """Validate data freshness for each timeframe."""

        age = current_time - latest_timestamp
        self.TIMEFRAME_CONFIGS[tf]

        # Define maximum acceptable age for each timeframe
        max_ages = {
            '15m': pd.Timedelta(minutes=30),  # 30 minutes
            '60m': pd.Timedelta(hours=2),     # 2 hours
            '1d': pd.Timedelta(days=2)        # 2 days (weekend buffer)
        }

        max_age = max_ages.get(tf, pd.Timedelta(hours=1))

        if age > max_age:
            return f"Data is stale (age: {age}, max: {max_age})"

        return None

    def get_safe_timeframes_for_prediction(self, current_time: pd.Timestamp) -> list[str]:
        """
        Get safe timeframes for prediction at current time.

        This is crucial for live trading - determines which timeframes
        can be safely used for generating predictions.

        Args:
            current_time: Current timestamp

        Returns:
            List of safe timeframes for current time
        """
        safe_timeframes = []

        # Get today's market close time
        today_close = pd.Timestamp(
            current_time.year, current_time.month, current_time.day,
            self.MARKET_CLOSE.hour, self.MARKET_CLOSE.minute
        )

        # Check if market is open
        is_market_open = (
            current_time.weekday() < 5 and  # Monday-Friday
            today_close.time() >= self.MARKET_OPEN and
            current_time.time() < self.MARKET_CLOSE
        )

        if is_market_open:
            # During market hours: only intraday timeframes
            safe_timeframes = ['15m', '60m']
            self.logger.info(f"📈 Market open: Using intraday timeframes {safe_timeframes}")
        else:
            # After market close: all timeframes
            safe_timeframes = ['15m', '60m', '1d']
            self.logger.info(f"📊 Market closed: Using all timeframes {safe_timeframes}")

        return safe_timeframes

    def validate_feature_combination_safety(self,
                                         features_by_tf: dict[str, pd.DataFrame],
                                         current_time: pd.Timestamp) -> dict[str, Any]:
        """
        Validate safety of combining features from multiple timeframes.

        This is called before feature combination to ensure no temporal leakage.

        Args:
            features_by_tf: Dictionary of timeframes to feature DataFrames
            current_time: Current timestamp

        Returns:
            Dict with combination safety validation
        """
        # First run basic compatibility check
        compatibility_result = self.validate_timeframe_compatibility(
            features_by_tf, current_time
        )

        if compatibility_result['status'] == 'invalid':
            return {
                'status': 'unsafe',
                'reason': 'timeframe_incompatible',
                'issues': compatibility_result['issues']
            }

        # Additional combination-specific checks
        combination_issues = []

        # Check if we have both intraday and daily data
        has_intraday = any(tf in ['15m', '60m'] for tf in compatibility_result['valid_timeframes'])
        has_daily = '1d' in compatibility_result['valid_timeframes']

        if has_intraday and has_daily:
            # Special validation for mixed timeframe combinations
            daily_close_time = pd.Timestamp(
                current_time.year, current_time.month, current_time.day,
                self.MARKET_CLOSE.hour, self.MARKET_CLOSE.minute
            )

            if current_time < daily_close_time:
                combination_issues.append(
                    f"Cannot combine intraday with daily before market close "
                    f"(current: {current_time}, close: {daily_close_time})"
                )

        return {
            'status': 'safe' if not combination_issues else 'unsafe',
            'issues': combination_issues,
            'valid_timeframes': compatibility_result['valid_timeframes'],
            'can_combine': len(compatibility_result['valid_timeframes']) > 1
        }

    def get_timeframe_prefixes(self, timeframes: list[str]) -> dict[str, str]:
        """
        Get standardized prefixes for timeframe features.

        This ensures consistent naming when combining timeframes.

        Args:
            timeframes: List of timeframe strings

        Returns:
            Dict mapping original timeframe to prefix
        """
        prefixes = {}
        for tf in timeframes:
            if tf == '15m':
                prefixes[tf] = 'm15'
            elif tf == '60m':
                prefixes[tf] = 'h1'
            elif tf == '1d':
                prefixes[tf] = 'd1'
            else:
                prefixes[tf] = tf.replace('m', 'm').replace('h', 'h').replace('d', 'd')

        return prefixes

    def log_alignment_summary(self, validation_result: dict[str, Any]) -> None:
        """Log a comprehensive summary of alignment validation."""
        status = validation_result['status']
        valid_count = validation_result['valid_count']
        total_count = validation_result['total_timeframes']

        self.logger.info("=" * 60)
        self.logger.info("🔍 TIMEFRAME ALIGNMENT VALIDATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Status: {status.upper()}")
        self.logger.info(f"Valid timeframes: {valid_count}/{total_count}")
        self.logger.info(f"Current time: {validation_result['current_time']}")

        if validation_result['valid_timeframes']:
            self.logger.info(f"✅ Valid timeframes: {validation_result['valid_timeframes']}")

        if validation_result['issues']:
            self.logger.error("❌ Issues found:")
            for issue in validation_result['issues']:
                self.logger.error(f"   {issue}")

        if validation_result['warnings']:
            self.logger.warning("⚠️ Warnings:")
            for warning in validation_result['warnings']:
                self.logger.warning(f"   {warning}")

        self.logger.info("=" * 60)


# Factory function for easy instantiation
def get_timeframe_alignment_guard(strict_mode: bool = True) -> TimeframeAlignmentGuard:
    """Factory function to get TimeframeAlignmentGuard instance.

    Args:
        strict_mode: If True, enforces strict market hour checks for live trading.
                    If False, allows data accumulation regardless of market hours.
    """
    return TimeframeAlignmentGuard(strict_mode=strict_mode)


# Convenience function for quick validation
def validate_timeframes_quick(features_by_tf: dict[str, pd.DataFrame],
                            current_time: pd.Timestamp | None = None) -> dict[str, Any]:
    """
    Quick validation function for timeframe alignment.

    Args:
        features_by_tf: Dictionary of timeframes to DataFrames
        current_time: Current timestamp (uses now if None)

    Returns:
        Validation result dictionary
    """
    guard = get_timeframe_alignment_guard()
    if current_time is None:
        current_time = pd.Timestamp.now()

    return guard.validate_timeframe_compatibility(features_by_tf, current_time)
