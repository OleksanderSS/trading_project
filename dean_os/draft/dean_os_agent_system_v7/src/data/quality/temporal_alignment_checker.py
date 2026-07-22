"""
Temporal Alignment Checker
Prevents temporal leakage by ensuring proper timestamp alignment.
"""
import logging

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("TemporalAlignmentChecker")


class TemporalAlignmentChecker:
    """
    Checks temporal alignment to prevent data leakage.

    Audit Points:
    - News timestamp alignment
    - Macro release timing
    - Timeframe alignment
    """

    def __init__(self):
        self.violations = []
        self.checks_performed = 0

    def check_news_alignment(
        self,
        market_df: pd.DataFrame,
        news_df: pd.DataFrame,
        market_timestamp_col: str = 'timestamp',
        news_timestamp_col: str = 'published_at'
    ) -> dict[str, any]:
        """
        Check that news is not used before it was published.

        Args:
            market_df: Market data with timestamps
            news_df: News data with published_at timestamps
            market_timestamp_col: Market timestamp column name
            news_timestamp_col: News timestamp column name

        Returns:
            Dict with check results
        """
        self.checks_performed += 1

        if market_df.empty or news_df.empty:
            return {'status': 'SKIP', 'message': 'Empty DataFrame'}

        # Ensure timestamps are datetime
        market_df[market_timestamp_col] = pd.to_datetime(market_df[market_timestamp_col])
        news_df[news_timestamp_col] = pd.to_datetime(news_df[news_timestamp_col])

        violations = []

        # Check each market timestamp
        for _idx, row in market_df.iterrows():
            market_time = row[market_timestamp_col]

            # Find news that would be used at this market time
            # News should only be used AFTER published_at
            future_news = news_df[news_df[news_timestamp_col] > market_time]

            if len(future_news) > 0:
                # Check if any features reference future news
                for col in market_df.columns:
                    if 'news' in col.lower() and not pd.isna(row[col]):
                        violation = {
                            'market_time': market_time,
                            'feature': col,
                            'future_news_count': len(future_news),
                            'earliest_future_news': future_news[news_timestamp_col].min()
                        }
                        violations.append(violation)
                        logger.warning(
                            f"⚠️ TEMPORAL LEAKAGE: Feature '{col}' at {market_time} "
                            f"may use {len(future_news)} future news items"
                        )

        if violations:
            self.violations.extend(violations)
            return {
                'status': 'VIOLATION',
                'message': f'Found {len(violations)} potential leakage violations',
                'violations': violations
            }
        else:
            logger.info("✅ News alignment check passed: No temporal leakage detected")
            return {
                'status': 'OK',
                'message': 'No temporal leakage detected',
                'violations': []
            }

    def filter_future_news(
        self,
        current_timestamp: pd.Timestamp,
        news_df: pd.DataFrame,
        news_timestamp_col: str = 'published_at'
    ) -> pd.DataFrame:
        """
        Filter out news published after current_timestamp.

        Args:
            current_timestamp: Current market timestamp
            news_df: News DataFrame
            news_timestamp_col: News timestamp column

        Returns:
            Filtered news DataFrame (only past news)
        """
        news_df[news_timestamp_col] = pd.to_datetime(news_df[news_timestamp_col])
        past_news = news_df[news_df[news_timestamp_col] <= current_timestamp].copy()

        filtered_count = len(news_df) - len(past_news)
        if filtered_count > 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Filtered {filtered_count} future news items "
                    f"(current time: {current_timestamp})"
                )

        return past_news

    def check_macro_release_timing(
        self,
        macro_df: pd.DataFrame,
        date_col: str = 'date',
        release_time_col: str = 'release_time'
    ) -> dict[str, any]:
        """
        Check that macro data includes release time, not just date.

        Args:
            macro_df: Macro data DataFrame
            date_col: Date column name
            release_time_col: Release time column name

        Returns:
            Dict with check results
        """
        self.checks_performed += 1

        if macro_df.empty:
            return {'status': 'SKIP', 'message': 'Empty DataFrame'}

        if release_time_col not in macro_df.columns:
            logger.error(
                f"❌ CRITICAL: Macro data missing '{release_time_col}' column. "
                f"Using calendar date only causes temporal leakage!"
            )
            return {
                'status': 'ERROR',
                'message': f'Missing {release_time_col} column',
                'recommendation': 'Add release_time to macro data to prevent leakage'
            }

        # Check if release_time is properly formatted
        try:
            macro_df[release_time_col] = pd.to_datetime(macro_df[release_time_col])
            logger.info("✅ Macro data has proper release_time column")
            return {
                'status': 'OK',
                'message': 'Macro release timing properly configured'
            }
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"❌ Error parsing release_time: {e}")
            return {
                'status': 'ERROR',
                'message': f'Invalid release_time format: {e}'
            }

    def check_timeframe_alignment(
        self,
        df_15m: pd.DataFrame | None = None,
        df_1h: pd.DataFrame | None = None,
        df_1d: pd.DataFrame | None = None,
        timestamp_col: str = 'timestamp'
    ) -> dict[str, any]:
        """
        Check that different timeframes are properly aligned.

        Args:
            df_15m: 15-minute data
            df_1h: 1-hour data
            df_1d: 1-day data
            timestamp_col: Timestamp column name

        Returns:
            Dict with check results
        """
        self.checks_performed += 1

        results = {
            'status': 'OK',
            'checks': []
        }

        # Check 1h alignment with 15m
        if df_15m is not None and df_1h is not None:
            # 1h close should align with 15m close at hour boundaries
            df_15m[timestamp_col] = pd.to_datetime(df_15m[timestamp_col])
            df_1h[timestamp_col] = pd.to_datetime(df_1h[timestamp_col])

            # Get hour boundaries from 15m data
            hour_boundaries = df_15m[df_15m[timestamp_col].dt.minute == 0]

            # Check if 1h timestamps match
            misaligned = []
            for ts in df_1h[timestamp_col]:
                if ts not in hour_boundaries[timestamp_col].values:
                    misaligned.append(ts)

            if misaligned:
                logger.warning(
                    f"⚠️ Found {len(misaligned)} misaligned 1h timestamps with 15m data"
                )
                results['checks'].append({
                    'timeframes': '1h vs 15m',
                    'status': 'WARNING',
                    'misaligned_count': len(misaligned)
                })
            else:
                logger.info("✅ 1h and 15m timeframes properly aligned")
                results['checks'].append({
                    'timeframes': '1h vs 15m',
                    'status': 'OK'
                })

        # Check 1d alignment
        if df_1d is not None:
            df_1d[timestamp_col] = pd.to_datetime(df_1d[timestamp_col])

            # 1d close should be at market close time (16:00 EST typically)
            close_times = df_1d[timestamp_col].dt.hour.unique()

            if 16 not in close_times and 21 not in close_times:  # 16:00 EST or 21:00 UTC
                logger.warning(
                    f"⚠️ 1d close times may not align with market close: {close_times}"
                )
                results['checks'].append({
                    'timeframe': '1d',
                    'status': 'WARNING',
                    'message': 'Close times may not match market close'
                })
            else:
                logger.info("✅ 1d timeframe aligned with market close")
                results['checks'].append({
                    'timeframe': '1d',
                    'status': 'OK'
                })

        return results

    def get_violations(self) -> list[dict]:
        """Get all detected violations."""
        return self.violations.copy()

    def reset(self):
        """Reset checker state."""
        self.violations = []
        self.checks_performed = 0


def check_temporal_alignment(
    market_df: pd.DataFrame,
    news_df: pd.DataFrame | None = None,
    macro_df: pd.DataFrame | None = None
) -> dict[str, any]:
    """
    Quick function to check temporal alignment.

    Args:
        market_df: Market data
        news_df: News data (optional)
        macro_df: Macro data (optional)

    Returns:
        Combined check results
    """
    checker = TemporalAlignmentChecker()
    results = {'checks': []}

    if news_df is not None:
        news_result = checker.check_news_alignment(market_df, news_df)
        results['checks'].append(('news_alignment', news_result))

    if macro_df is not None:
        macro_result = checker.check_macro_release_timing(macro_df)
        results['checks'].append(('macro_timing', macro_result))

    # Overall status
    statuses = [r[1]['status'] for r in results['checks']]
    if 'VIOLATION' in statuses or 'ERROR' in statuses:
        results['overall_status'] = 'FAIL'
    elif 'WARNING' in statuses:
        results['overall_status'] = 'WARNING'
    else:
        results['overall_status'] = 'PASS'

    return results
