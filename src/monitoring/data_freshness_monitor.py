#!/usr/bin/env python3
"""
Data Freshness Monitor - Real-time Monitoring of Data Freshness
Monitors all data sources and alerts on stale data.
"""

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("DataFreshnessMonitor")

class DataFreshnessMonitor:
    """
    Real-time monitoring of data freshness across all sources.

    This monitor continuously checks the freshness of all data sources
    and generates alerts when data becomes stale. This is critical
    for trading systems where stale data can cause significant
    performance degradation.

    Key features:
    - Monitors all data sources (prices, news, macro, features)
    - Configurable freshness thresholds per data type
    - Multiple alert channels (Telegram, email, logs)
    - Historical tracking of freshness metrics
    - Automatic recovery notifications
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the DataFreshnessMonitor.

        Args:
            config: Configuration dictionary for monitoring settings
        """
        self.logger = logger
        self.config = config or {}

        # Freshness thresholds for different data types
        self.FRESHNESS_THRESHOLDS = {
            'prices_15m': {
                'max_age_hours': 0.5,      # 30 minutes
                'warning_age_hours': 0.25,    # 15 minutes
                'critical_age_hours': 1.0,     # 1 hour
                'description': '15-minute price data'
            },
            'prices_60m': {
                'max_age_hours': 2.0,       # 2 hours
                'warning_age_hours': 1.0,     # 1 hour
                'critical_age_hours': 4.0,     # 4 hours
                'description': '1-hour price data'
            },
            'prices_1d': {
                'max_age_hours': 24.0,      # 24 hours
                'warning_age_hours': 12.0,    # 12 hours
                'critical_age_hours': 48.0,    # 48 hours
                'description': 'daily price data'
            },
            'news': {
                'max_age_hours': 2.0,       # 2 hours
                'warning_age_hours': 1.0,     # 1 hour
                'critical_age_hours': 6.0,     # 6 hours
                'description': 'news data'
            },
            'macro': {
                'max_age_hours': 48.0,      # 48 hours
                'warning_age_hours': 24.0,    # 24 hours
                'critical_age_hours': 72.0,    # 72 hours
                'description': 'macroeconomic data'
            },
            'features': {
                'max_age_hours': 4.0,       # 4 hours
                'warning_age_hours': 2.0,     # 2 hours
                'critical_age_hours': 8.0,     # 8 hours
                'description': 'engineered features'
            },
            'models': {
                'max_age_hours': 24.0,      # 24 hours
                'warning_age_hours': 12.0,    # 12 hours
                'critical_age_hours': 48.0,    # 48 hours
                'description': 'trained models'
            }
        }

        self.alert_history = []
        self.freshness_history = []

        # Data source paths (can be configured)
        self.data_paths = {
            'prices_15m': self.config.get('prices_15m_path', 'data/processed/prices/15m'),
            'prices_60m': self.config.get('prices_60m_path', 'data/processed/prices/60m'),
            'prices_1d': self.config.get('prices_1d_path', 'data/processed/prices/1d'),
            'news': self.config.get('news_path', 'data/processed/news'),
            'macro': self.config.get('macro_path', 'data/processed/macro'),
            'features': self.config.get('features_path', 'data/processed/features'),
            'models': self.config.get('models_path', 'models')
        }

        # Alert settings
        self.alert_settings = {
            'telegram_enabled': self.config.get('telegram_enabled', True),
            'email_enabled': self.config.get('email_enabled', False),
            'log_enabled': self.config.get('log_enabled', True),
            'alert_cooldown_minutes': self.config.get('alert_cooldown_minutes', 30)
        }

        self.last_alerts = {}

    async def check_all_data_sources(self,
                                  current_time: pd.Timestamp | None = None) -> dict[str, Any]:
        """
        Check freshness of all configured data sources.

        Args:
            current_time: Current timestamp (uses now if None)

        Returns:
            Dict with freshness status for all sources
        """
        if current_time is None:
            current_time = pd.Timestamp.now()

        self.logger.info(f"🔍 Checking data freshness for {current_time}")

        results = {}
        overall_status = 'fresh'

        # Check each data source
        for source_name, _source_config in self.FRESHNESS_THRESHOLDS.items():
            try:
                source_result = await self._check_data_source_freshness(
                    source_name, current_time
                )
                results[source_name] = source_result

                # Update overall status
                if source_result['status'] == 'critical':
                    overall_status = 'critical'
                elif source_result['status'] == 'stale' and overall_status != 'critical':
                    overall_status = 'stale'
                elif source_result['status'] == 'warning' and overall_status == 'fresh':
                    overall_status = 'warning'

            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.exception(f"Error checking {source_name}: {e}")
                results[source_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                overall_status = 'error'

        # Create summary
        summary = {
            'overall_status': overall_status,
            'current_time': current_time,
            'sources': results,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Store in history
        self.freshness_history.append(summary)

        # Keep only last 1000 entries
        if len(self.freshness_history) > 1000:
            self.freshness_history = self.freshness_history[-1000:]

        # Generate alerts if needed
        await self._generate_alerts(summary)

        # Log summary
        self._log_freshness_summary(summary)

        return summary

    async def _check_data_source_freshness(self,
                                        source_name: str,
                                        current_time: pd.Timestamp) -> dict[str, Any]:
        """Check freshness of a single data source."""

        # Get latest timestamp for this source
        latest_timestamp = await self._get_latest_timestamp(source_name)

        if latest_timestamp is None:
            return {
                'status': 'error',
                'message': f'Could not determine latest timestamp for {source_name}'
            }

        # Calculate data age
        data_age = current_time - latest_timestamp
        data_age_hours = data_age.total_seconds() / 3600

        # Get thresholds for this source
        thresholds = self.FRESHNESS_THRESHOLDS[source_name]

        # Determine status
        if data_age_hours >= thresholds['critical_age_hours']:
            status = 'critical'
        elif data_age_hours >= thresholds['max_age_hours']:
            status = 'stale'
        elif data_age_hours >= thresholds['warning_age_hours']:
            status = 'warning'
        else:
            status = 'fresh'

        result = {
            'status': status,
            'latest_timestamp': latest_timestamp,
            'data_age_hours': data_age_hours,
            'data_age': data_age,
            'thresholds': thresholds,
            'source_name': source_name,
            'description': thresholds['description']
        }

        return result

    def _get_latest_timestamp_from_parquet(self, path_obj: Path) -> pd.Timestamp | None:
        """Get latest timestamp from parquet files in a directory."""
        parquet_files = list(path_obj.glob('*.parquet'))
        if not parquet_files:
            return None

        latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_parquet(latest_file)

        datetime_col = self._find_datetime_column(df)
        if datetime_col:
            return pd.to_datetime(df[datetime_col].max())
        return None

    def _get_latest_timestamp_from_models(self, path_obj: Path) -> pd.Timestamp | None:
        """Get latest timestamp from model files."""
        model_files = list(path_obj.rglob('*.pkl'))
        if not model_files:
            return None

        latest_file = max(model_files, key=lambda x: x.stat().st_mtime)
        return pd.Timestamp.fromtimestamp(latest_file.stat().st_mtime)

    async def _get_latest_timestamp(self, source_name: str) -> pd.Timestamp | None:
        """Get the latest timestamp for a data source."""
        data_path = self.data_paths.get(source_name)
        if data_path is None:
            return None

        path_obj = Path(data_path)

        if not path_obj.exists():
            return None

        try:
            if source_name.startswith('prices') or source_name in ['news', 'macro', 'features']:
                return self._get_latest_timestamp_from_parquet(path_obj)
            elif source_name == 'models':
                return self._get_latest_timestamp_from_models(path_obj)

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f"Error getting latest timestamp for {source_name}: {e}")

        return None

    def _find_datetime_column(self, df: pd.DataFrame) -> str | None:
        """Find datetime column in DataFrame."""
        datetime_cols = ['datetime', 'timestamp', 'date', 'time']

        for col in datetime_cols:
            if col in df.columns:
                return col

        # Check if index is datetime
        if isinstance(df.index, pd.DatetimeIndex):
            return 'index'

        return None

    async def _generate_alerts(self, freshness_summary: dict[str, Any]) -> None:
        """Generate alerts for stale data sources."""

        overall_status = freshness_summary['overall_status']
        current_time = freshness_summary['current_time']

        # Only alert for warning, stale, or critical
        if overall_status == 'fresh':
            return

        # Check each source for alerts
        for source_name, source_result in freshness_summary['sources'].items():
            if source_result['status'] in ['warning', 'stale', 'critical']:
                await self._send_alert(source_name, source_result, current_time)

    async def _send_alert(self,
                        source_name: str,
                        source_result: dict[str, Any],
                        current_time: pd.Timestamp) -> None:
        """Send alert for a specific data source."""

        # Check cooldown to avoid spam
        if not self._should_send_alert(source_name, current_time):
            return

        alert_message = self._create_alert_message(source_name, source_result, current_time)

        # Send to different channels
        if self.alert_settings['telegram_enabled']:
            await self._send_telegram_alert(alert_message)

        if self.alert_settings['email_enabled']:
            await self._send_email_alert(alert_message)

        if self.alert_settings['log_enabled']:
            self.logger.warning(f"🚨 DATA FRESHNESS ALERT: {alert_message}")

        # Record alert
        self.alert_history.append({
            'source_name': source_name,
            'status': source_result['status'],
            'message': alert_message,
            'timestamp': current_time.isoformat()
        })

        # Update last alert time
        self.last_alerts[source_name] = current_time

    def _should_send_alert(self, source_name: str, current_time: pd.Timestamp) -> bool:
        """Check if alert should be sent (cooldown logic)."""

        last_alert = self.last_alerts.get(source_name)
        if last_alert is None:
            return True

        cooldown_minutes = self.alert_settings['alert_cooldown_minutes']
        time_since_last = current_time - last_alert

        return time_since_last.total_seconds() / 60 > cooldown_minutes

    def _create_alert_message(self,
                           source_name: str,
                           source_result: dict[str, Any],
                           current_time: pd.Timestamp) -> str:
        """Create alert message."""

        status = source_result['status']
        data_age_hours = source_result['data_age_hours']
        description = source_result['description']
        latest_timestamp = source_result['latest_timestamp']

        # Create status emoji
        status_emojis = {
            'warning': '⚠️',
            'stale': '🔴',
            'critical': '🚨'
        }

        emoji = status_emojis.get(status, '❓')

        message = (
            f"{emoji} DATA FRESHNESS ALERT\n"
            f"Source: {source_name} ({description})\n"
            f"Status: {status.upper()}\n"
            f"Data Age: {data_age_hours:.1f} hours\n"
            f"Latest Data: {latest_timestamp}\n"
            f"Current Time: {current_time}\n"
            f"Action: Check data collection pipeline"
        )

        return message

    async def _send_telegram_alert(self, message: str) -> None:
        """Send alert to Telegram."""
        try:
            # This would integrate with existing Telegram notifier
            from src.utils.universal_notifier import UniversalNotifier

            notifier = UniversalNotifier()
            await notifier.send_message(message)

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f"Failed to send Telegram alert: {e}")

    async def _send_email_alert(self, message: str) -> None:
        """Send alert via email."""
        try:
            # Email integration would go here
            # For now, just log
            self.logger.info(f"Email alert (not implemented): {message}")

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f"Failed to send email alert: {e}")

    def _log_freshness_summary(self, summary: dict[str, Any]) -> None:
        """Log comprehensive freshness summary."""

        overall_status = summary['overall_status']
        current_time = summary['current_time']

        self.logger.info("=" * 60)
        self.logger.info("🔍 DATA FRESHNESS MONITORING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Overall Status: {overall_status.upper()}")
        self.logger.info(f"Check Time: {current_time}")

        # Log status for each source
        for source_name, source_result in summary['sources'].items():
            status = source_result['status']
            data_age_hours = source_result.get('data_age_hours', 0)
            description = source_result.get('description', source_name)

            status_emojis = {
                'fresh': '✅',
                'warning': '⚠️',
                'stale': '🔴',
                'critical': '🚨',
                'error': '❌'
            }

            emoji = status_emojis.get(status, '❓')
            self.logger.info(f"{emoji} {source_name}: {status} ({data_age_hours:.1f}h old) - {description}")

        self.logger.info("=" * 60)

    def get_freshness_history(self,
                           hours: int = 24,
                           source_name: str | None = None) -> list[dict[str, Any]]:
        """Get freshness history for analysis."""

        cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=hours)

        filtered_history = []
        for entry in self.freshness_history:
            entry_time = pd.Timestamp(entry['timestamp'])

            if entry_time >= cutoff_time:
                if source_name is None or source_name in entry['sources']:
                    filtered_history.append(entry)

        return filtered_history

    def get_freshness_metrics(self,
                            hours: int = 24,
                            source_name: str | None = None) -> dict[str, Any]:
        """Calculate freshness metrics over time period."""

        history = self.get_freshness_history(hours, source_name)

        if not history:
            return {'error': 'No history available'}

        metrics = {
            'total_checks': len(history),
            'period_hours': hours,
            'source_metrics': {},
            'overall_availability': 0
        }

        fresh_counts = self._aggregate_fresh_counts(history, source_name)
        self._calculate_availability_metrics(metrics, fresh_counts)

        return metrics

    def _aggregate_fresh_counts(self, history: list[dict[str, Any]], source_name: str | None) -> dict[str, dict[str, int]]:
        """Aggregate fresh counts by source."""
        fresh_counts = {}
        for entry in history:
            for src_name, src_result in entry['sources'].items():
                if source_name is None or src_name == source_name:
                    if src_name not in fresh_counts:
                        fresh_counts[src_name] = {'fresh': 0, 'total': 0}
                    fresh_counts[src_name]['total'] += 1
                    if src_result['status'] == 'fresh':
                        fresh_counts[src_name]['fresh'] += 1
        return fresh_counts

    def _calculate_availability_metrics(self, metrics: dict[str, Any], fresh_counts: dict[str, dict[str, int]]) -> None:
        """Calculate and update availability metrics."""
        total_fresh = 0
        total_checks = 0

        for src, counts in fresh_counts.items():
            availability = (counts['fresh'] / counts['total']) * 100
            metrics['source_metrics'][src] = {
                'availability_pct': availability,
                'fresh_checks': counts['fresh'],
                'total_checks': counts['total']
            }
            total_fresh += counts['fresh']
            total_checks += counts['total']

        if total_checks > 0:
            metrics['overall_availability'] = (total_fresh / total_checks) * 100


    async def start_monitoring(self,
                            check_interval_minutes: int = 15) -> None:
        """Start continuous monitoring."""

        self.logger.info(f"🔄 Starting data freshness monitoring (check every {check_interval_minutes} minutes)")

        while True:
            try:
                await self.check_all_data_sources()
                await asyncio.sleep(check_interval_minutes * 60)

            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.exception(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error


# Factory function for easy instantiation
def get_data_freshness_monitor(config: dict[str, Any] | None = None) -> DataFreshnessMonitor:
    """Factory function to get DataFreshnessMonitor instance."""
    return DataFreshnessMonitor(config)


# Convenience function for quick check
async def check_freshness_quick(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Quick freshness check of all data sources.

    Args:
        config: Configuration dictionary

    Returns:
        Freshness summary dictionary
    """
    monitor = get_data_freshness_monitor(config)
    return await monitor.check_all_data_sources()
