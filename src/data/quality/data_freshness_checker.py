"""
Data Freshness Checker
Monitors data lag and ensures data is up-to-date.
"""

from datetime import datetime, timedelta

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("DataFreshnessChecker")


class DataFreshnessChecker:
    """
    Checks data freshness and alerts if data is stale.

    Audit Point: DATA LAYER → Data Freshness
    """

    def __init__(
        self,
        warning_threshold_hours: float = 1.0,
        error_threshold_hours: float = 24.0
    ):
        """
        Initialize freshness checker.

        Args:
            warning_threshold_hours: Warn if data older than this (hours)
            error_threshold_hours: Error if data older than this (hours)
        """
        self.warning_threshold = timedelta(hours=warning_threshold_hours)
        self.error_threshold = timedelta(hours=error_threshold_hours)
        self.metrics = {
            'checks_performed': 0,
            'warnings_issued': 0,
            'errors_issued': 0,
            'last_check_time': None,
            'last_data_time': None,
            'last_lag_hours': None
        }

    def check_freshness(
        self,
        df: pd.DataFrame,
        timestamp_column: str = 'timestamp'
    ) -> dict[str, any]:
        """
        Check data freshness and return metrics.

        Args:
            df: DataFrame with timestamp column
            timestamp_column: Name of timestamp column

        Returns:
            Dict with freshness metrics and status
        """
        self.metrics['checks_performed'] += 1
        self.metrics['last_check_time'] = datetime.now()

        if df.empty:
            logger.error("❌ Cannot check freshness: DataFrame is empty")
            return {
                'status': 'ERROR',
                'message': 'Empty DataFrame',
                'lag_hours': None
            }

        if timestamp_column not in df.columns:
            logger.error(f"❌ Timestamp column '{timestamp_column}' not found in DataFrame")
            return {
                'status': 'ERROR',
                'message': f'Column {timestamp_column} not found',
                'lag_hours': None
            }

        # Get latest timestamp
        try:
            latest_timestamp = pd.to_datetime(df[timestamp_column]).max()
            # Remove timezone info if present to avoid comparison issues
            if latest_timestamp.tz is not None:
                latest_timestamp = latest_timestamp.tz_localize(None)
            self.metrics['last_data_time'] = latest_timestamp
        except Exception as e:
            logger.error(f"❌ Error parsing timestamps: {e}")
            return {
                'status': 'ERROR',
                'message': f'Timestamp parsing error: {e}',
                'lag_hours': None
            }

        # Calculate lag
        now = pd.Timestamp.now()  # This is tz-naive by default
        lag = now - latest_timestamp
        lag_hours = lag.total_seconds() / 3600
        self.metrics['last_lag_hours'] = lag_hours

        # Determine status
        if lag > self.error_threshold:
            self.metrics['errors_issued'] += 1
            logger.error(
                f"❌ DATA TOO OLD: {lag_hours:.1f} hours old "
                f"(threshold: {self.error_threshold.total_seconds() / 3600:.1f}h)"
            )
            logger.error(f"   Latest data: {latest_timestamp}")
            logger.error(f"   Current time: {now}")
            return {
                'status': 'ERROR',
                'message': f'Data is {lag_hours:.1f} hours old',
                'lag_hours': lag_hours,
                'latest_timestamp': latest_timestamp,
                'current_time': now
            }

        elif lag > self.warning_threshold:
            self.metrics['warnings_issued'] += 1
            logger.warning(
                f"⚠️ DATA AGING: {lag_hours:.1f} hours old "
                f"(threshold: {self.warning_threshold.total_seconds() / 3600:.1f}h)"
            )
            logger.warning(f"   Latest data: {latest_timestamp}")
            return {
                'status': 'WARNING',
                'message': f'Data is {lag_hours:.1f} hours old',
                'lag_hours': lag_hours,
                'latest_timestamp': latest_timestamp,
                'current_time': now
            }

        else:
            logger.info(f"✅ Data is fresh: {lag_hours:.2f} hours old")
            return {
                'status': 'OK',
                'message': f'Data is fresh ({lag_hours:.2f} hours old)',
                'lag_hours': lag_hours,
                'latest_timestamp': latest_timestamp,
                'current_time': now
            }

    def get_metrics(self) -> dict[str, any]:
        """Get freshness checker metrics."""
        return self.metrics.copy()

    def reset_metrics(self):
        """Reset metrics counters."""
        self.metrics = {
            'checks_performed': 0,
            'warnings_issued': 0,
            'errors_issued': 0,
            'last_check_time': None,
            'last_data_time': None,
            'last_lag_hours': None
        }
        logger.info("Metrics reset")


def check_data_freshness(
    df: pd.DataFrame,
    timestamp_column: str = 'timestamp',
    warning_threshold_hours: float = 1.0,
    error_threshold_hours: float = 24.0
) -> dict[str, any]:
    """
    Quick function to check data freshness.

    Args:
        df: DataFrame with timestamp column
        timestamp_column: Name of timestamp column
        warning_threshold_hours: Warn if older than this
        error_threshold_hours: Error if older than this

    Returns:
        Freshness check result
    """
    checker = DataFreshnessChecker(
        warning_threshold_hours=warning_threshold_hours,
        error_threshold_hours=error_threshold_hours
    )
    return checker.check_freshness(df, timestamp_column)
