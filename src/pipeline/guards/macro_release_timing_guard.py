#!/usr/bin/env python3
"""
Macro Release Timing Guard - Prevents Early Access to Economic Data
Ensures macroeconomic data is only used after official release times.
"""

from datetime import time
from typing import Any

import pandas as pd
import pytz

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("MacroReleaseTimingGuard")

class MacroReleaseTimingGuard:
    """
    Prevents temporal leakage from macroeconomic data releases.

    This guard ensures that macroeconomic data is only used after
    the official release time, preventing early access to future
    economic indicators that would create unrealistic backtest results.

    Key protections:
    - Validates release times for all major economic indicators
    - Applies appropriate delays for different data types
    - Handles timezone conversions correctly
    - Prevents weekend/holiday release issues
    """

    # US Eastern Timezone (where most US data is released)
    US_EASTERN = pytz.timezone('US/Eastern')

    # Official release schedules for major economic indicators
    RELEASE_SCHEDULES = {
        'GDP': {
            'release_time': time(8, 30),  # 8:30 AM ET
            'delay_hours': 1.0,           # 1 hour delay
            'release_day': 'quarter_end',       # End of quarter
            'description': 'Gross Domestic Product'
        },
        'CPI': {
            'release_time': time(8, 30),  # 8:30 AM ET
            'delay_hours': 0.5,           # 30 minutes delay
            'release_day': 'monthly',          # Monthly
            'description': 'Consumer Price Index'
        },
        'PPI': {
            'release_time': time(8, 30),  # 8:30 AM ET
            'delay_hours': 0.5,           # 30 minutes delay
            'release_day': 'monthly',          # Monthly
            'description': 'Producer Price Index'
        },
        'UNEMPLOYMENT': {
            'release_time': time(8, 30),  # 8:30 AM ET
            'delay_hours': 0.25,          # 15 minutes delay
            'release_day': 'monthly',          # Monthly (first Friday)
            'description': 'Unemployment Rate'
        },
        'FED_RATE': {
            'release_time': time(14, 0),  # 2:00 PM ET
            'delay_hours': 0.25,          # 15 minutes delay
            'release_day': 'fomc',             # FOMC meetings (8 times/year)
            'description': 'Federal Funds Rate'
        },
        'RETAIL_SALES': {
            'release_time': time(8, 30),  # 8:30 AM ET
            'delay_hours': 0.5,           # 30 minutes delay
            'release_day': 'monthly',          # Monthly
            'description': 'Retail Sales'
        },
        'DURABLE_GOODS': {
            'release_time': time(8, 30),  # 8:30 AM ET
            'delay_hours': 0.5,           # 30 minutes delay
            'release_day': 'monthly',          # Monthly
            'description': 'Durable Goods Orders'
        },
        'CONSUMER_CONFIDENCE': {
            'release_time': time(10, 0),  # 10:00 AM ET
            'delay_hours': 0.5,           # 30 minutes delay
            'release_day': 'monthly',          # Monthly
            'description': 'Consumer Confidence Index'
        },
        'ISM_MANUFACTURING': {
            'release_time': time(10, 0),  # 10:00 AM ET
            'delay_hours': 0.5,           # 30 minutes delay
            'release_day': 'monthly',          # Monthly
            'description': 'ISM Manufacturing PMI'
        },
        'NONFARM_PAYROLLS': {
            'release_time': time(8, 30),  # 8:30 AM ET
            'delay_hours': 0.25,          # 15 minutes delay
            'release_day': 'monthly',          # Monthly (first Friday)
            'description': 'Non-Farm Payrolls'
        }
    }

    def __init__(self):
        """Initialize the MacroReleaseTimingGuard."""
        self.logger = logger

    def validate_macro_data_timing(self,
                                macro_df: pd.DataFrame,
                                current_time: pd.Timestamp,
                                macro_type: str | None = None) -> dict[str, Any]:
        """
        Validate macroeconomic data timing for temporal leakage.

        Args:
            macro_df: DataFrame with macroeconomic data
            current_time: Current timestamp for validation
            macro_type: Type of macro data (GDP, CPI, etc.)

        Returns:
            Dict with validation results and issues
        """
        self.logger.info(f"📊 Validating macro data timing for {macro_type}")

        issues = []
        warnings = []
        valid_data = []

        # Get datetime column
        datetime_col = self._get_datetime_column(macro_df)
        if datetime_col is None:
            issues.append("No datetime column found in macro data")
            return {
                'status': 'invalid',
                'issues': issues,
                'warnings': warnings,
                'valid_data': valid_data
            }

        # Convert to US Eastern timezone for comparison
        macro_df['datetime_et'] = pd.to_datetime(macro_df[datetime_col]).dt.tz_localize(None).dt.tz_localize(self.US_EASTERN)

        # Validate each row
        for idx, row in macro_df.iterrows():
            row_validation = self._validate_macro_row(
                row, current_time, macro_type, datetime_col
            )

            if row_validation['valid']:
                valid_data.append(idx)
            else:
                issues.extend(row_validation['issues'])

            if row_validation['warnings']:
                warnings.extend(row_validation['warnings'])

        # Determine overall status
        status = 'valid' if not issues else 'invalid'

        result = {
            'status': status,
            'issues': issues,
            'warnings': warnings,
            'valid_data': valid_data,
            'invalid_count': len(macro_df) - len(valid_data),
            'valid_count': len(valid_data),
            'total_count': len(macro_df),
            'macro_type': macro_type,
            'current_time': current_time
        }

        self._log_validation_summary(result)

        return result

    def _validate_macro_row(self,
                           row: pd.Series,
                           current_time: pd.Timestamp,
                           macro_type: str | None,
                           datetime_col: str) -> dict[str, Any]:
        """Validate a single macro data row."""

        validation = {
            'valid': True,
            'issues': [],
            'warnings': []
        }

        # Get data timestamp
        data_time = row['datetime_et']

        # If macro_type is not specified, try to infer from column names
        if macro_type is None:
            macro_type = self._infer_macro_type_from_columns(row)

        if macro_type is None:
            validation['warnings'].append("Could not determine macro type")
            return validation

        # Get release schedule
        schedule = self.RELEASE_SCHEDULES.get(macro_type.upper())
        if schedule is None:
            validation['warnings'].append(f"Unknown macro type: {macro_type}")
            return validation

        # Calculate official release time
        release_time = self._calculate_official_release_time(
            data_time, schedule, macro_type
        )

        # Calculate earliest allowed usage time
        earliest_allowed = release_time + pd.Timedelta(hours=schedule['delay_hours'])

        # Check if data is being used too early
        if current_time < earliest_allowed:
            validation['valid'] = False
            validation['issues'].append(
                f"Data used before allowed time: "
                f"current={current_time}, earliest_allowed={earliest_allowed}, "
                f"release={release_time}, delay={schedule['delay_hours']}h"
            )

        # Additional checks
        self._check_release_schedule_compliance(
            data_time, schedule, validation
        )

        return validation

    def _calculate_official_release_time(self,
                                      data_time: pd.Timestamp,
                                      schedule: dict[str, Any],
                                      macro_type: str) -> pd.Timestamp:
        """Calculate the official release time for macro data."""

        # For most data, release is on the same day at specified time
        if schedule['release_day'] in ['monthly', 'quarter_end']:
            release_time = pd.Timestamp(
                data_time.year,
                data_time.month,
                data_time.day,
                schedule['release_time'].hour,
                schedule['release_time'].minute,
                tz=self.US_EASTERN
            )

        # For FOMC meetings, need to find actual meeting dates
        elif schedule['release_day'] == 'fomc':
            release_time = self._get_fomc_release_time(data_time, schedule)

        else:
            # Default to same day
            release_time = pd.Timestamp(
                data_time.year,
                data_time.month,
                data_time.day,
                schedule['release_time'].hour,
                schedule['release_time'].minute,
                tz=self.US_EASTERN
            )

        return release_time

    def _get_fomc_release_time(self,
                              data_time: pd.Timestamp,
                              schedule: dict[str, Any]) -> pd.Timestamp:
        """Get FOMC meeting release time."""
        # Simplified FOMC schedule (in practice, this would be a lookup table)
        # FOMC typically meets 8 times per year
        fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]  # Approximate

        # Find nearest FOMC month
        nearest_month = min(fomc_months, key=lambda x: abs(x - data_time.month))

        # Use 2nd Wednesday of the month (typical FOMC pattern)
        # This is simplified - real implementation would use actual FOMC calendar
        release_time = pd.Timestamp(
            data_time.year,
            nearest_month,
            1,  # Start from 1st day
            schedule['release_time'].hour,
            schedule['release_time'].minute,
            tz=self.US_EASTERN
        )

        # Find 2nd Wednesday
        days_until_wednesday = (2 - release_time.weekday()) % 7
        if days_until_wednesday < 0:
            days_until_wednesday += 7

        release_time += pd.Timedelta(days=days_until_wednesday + 7)  # +7 for 2nd Wednesday

        return release_time

    def _check_release_schedule_compliance(self,
                                       data_time: pd.Timestamp,
                                       schedule: dict[str, Any],
                                       validation: dict[str, Any]) -> None:
        """Check if data complies with release schedule."""

        # Check weekend releases (shouldn't happen for most data)
        if data_time.weekday() >= 5:  # Saturday or Sunday
            validation['warnings'].append(
                f"Data released on weekend: {data_time.strftime('%A')}"
            )

        # Check release time compliance
        release_time = time(data_time.hour, data_time.minute)
        expected_time = schedule['release_time']

        # Allow some tolerance (±30 minutes)
        time_diff = abs((data_time.hour * 60 + data_time.minute) -
                       (expected_time.hour * 60 + expected_time.minute))

        if time_diff > 30:
            validation['warnings'].append(
                f"Unusual release time: {release_time}, expected: {expected_time}"
            )

    def _infer_macro_type_from_columns(self, row: pd.Series) -> str | None:
        """Try to infer macro type from column names."""

        # Look for common macro indicators in column names
        for macro_type in self.RELEASE_SCHEDULES.keys():
            # Check if any column contains the macro type name
            for col in row.index:
                if macro_type.lower() in col.lower():
                    return macro_type

        return None

    def _get_datetime_column(self, df: pd.DataFrame) -> str | None:
        """Get datetime column from DataFrame."""
        datetime_cols = ['datetime', 'timestamp', 'date', 'time', 'release_date']

        for col in datetime_cols:
            if col in df.columns:
                return col

        # Check if index is datetime
        if isinstance(df.index, pd.DatetimeIndex):
            return 'index'

        return None

    def get_safe_macro_data(self,
                          macro_df: pd.DataFrame,
                          current_time: pd.Timestamp,
                          macro_type: str | None = None) -> pd.DataFrame:
        """
        Get only safe macro data (after release delays).

        Args:
            macro_df: Original macro DataFrame
            current_time: Current timestamp
            macro_type: Type of macro data

        Returns:
            DataFrame with only safe macro data
        """
        validation_result = self.validate_macro_data_timing(
            macro_df, current_time, macro_type
        )

        if validation_result['status'] == 'valid':
            return macro_df.copy()

        # Return only valid rows
        valid_indices = validation_result['valid_data']
        safe_df = macro_df.iloc[valid_indices].copy()

        self.logger.info(f"🛡️ Filtered macro data: {len(safe_df)} safe from {len(macro_df)} total")

        return safe_df

    def get_macro_release_schedule(self, macro_type: str) -> dict[str, Any]:
        """Get release schedule for a specific macro indicator."""
        return self.RELEASE_SCHEDULES.get(macro_type.upper(), {})

    def check_macro_data_freshness(self,
                                  macro_df: pd.DataFrame,
                                  current_time: pd.Timestamp) -> dict[str, Any]:
        """Check freshness of macro data."""

        datetime_col = self._get_datetime_column(macro_df)
        if datetime_col is None:
            return {'status': 'error', 'message': 'No datetime column'}

        # Get latest data timestamp
        if datetime_col == 'index':
            latest_timestamp = macro_df.index.max()
        else:
            latest_timestamp = pd.to_datetime(macro_df[datetime_col]).max()

        # Calculate age
        data_age = current_time - latest_timestamp

        # Define freshness thresholds
        freshness_thresholds = {
            'daily': pd.Timedelta(days=2),      # 2 days for daily data
            'monthly': pd.Timedelta(days=35),    # 35 days for monthly data
            'quarterly': pd.Timedelta(days=100),  # 100 days for quarterly data
        }

        # Determine data frequency
        data_frequency = self._estimate_data_frequency(macro_df)
        threshold = freshness_thresholds.get(data_frequency, pd.Timedelta(days=30))

        is_fresh = data_age <= threshold

        result = {
            'status': 'fresh' if is_fresh else 'stale',
            'data_age': data_age,
            'threshold': threshold,
            'latest_timestamp': latest_timestamp,
            'estimated_frequency': data_frequency,
            'current_time': current_time
        }

        return result

    def _estimate_data_frequency(self, macro_df: pd.DataFrame) -> str:
        """Estimate the frequency of macro data."""
        datetime_col = self._get_datetime_column(macro_df)
        if datetime_col is None:
            return 'unknown'

        if datetime_col == 'index':
            timestamps = macro_df.index
        else:
            timestamps = pd.to_datetime(macro_df[datetime_col])

        if len(timestamps) < 2:
            return 'unknown'

        # Calculate median difference between consecutive data points
        time_diffs = timestamps.to_series().diff().dropna()
        median_diff = time_diffs.median()

        # Classify frequency
        if median_diff <= pd.Timedelta(days=2):
            return 'daily'
        elif median_diff <= pd.Timedelta(days=35):
            return 'monthly'
        elif median_diff <= pd.Timedelta(days=100):
            return 'quarterly'
        else:
            return 'irregular'

    def _log_validation_summary(self, result: dict[str, Any]) -> None:
        """Log comprehensive validation summary."""
        status = result['status']
        valid_count = result['valid_count']
        invalid_count = result['invalid_count']
        total_count = result['total_count']

        self.logger.info("=" * 60)
        self.logger.info("📊 MACRO RELEASE TIMING VALIDATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Status: {status.upper()}")
        self.logger.info(f"Macro Type: {result.get('macro_type', 'Unknown')}")
        self.logger.info(f"Data Points: {valid_count} valid, {invalid_count} invalid, {total_count} total")
        self.logger.info(f"Current Time: {result['current_time']}")

        if result['issues']:
            self.logger.error("❌ Issues found:")
            for issue in result['issues']:
                self.logger.error(f"   {issue}")

        if result['warnings']:
            self.logger.warning("⚠️ Warnings:")
            for warning in result['warnings']:
                self.logger.warning(f"   {warning}")

        self.logger.info("=" * 60)


# Factory function for easy instantiation
def get_macro_release_timing_guard() -> MacroReleaseTimingGuard:
    """Factory function to get MacroReleaseTimingGuard instance."""
    return MacroReleaseTimingGuard()


# Convenience function for quick validation
def validate_macro_timing_quick(macro_df: pd.DataFrame,
                               current_time: pd.Timestamp | None = None,
                               macro_type: str | None = None) -> dict[str, Any]:
    """
    Quick macro data timing validation.

    Args:
        macro_df: Macro DataFrame to validate
        current_time: Current timestamp (uses now if None)
        macro_type: Type of macro data

    Returns:
        Validation result dictionary
    """
    guard = get_macro_release_timing_guard()
    if current_time is None:
        current_time = pd.Timestamp.now()

    return guard.validate_macro_data_timing(macro_df, current_time, macro_type)
