
import logging
from typing import Any

import pandas as pd

from src.features.enrichers.base import BaseEnricher

logger = logging.getLogger(__name__)

class SignificanceFeaturesEnricher(BaseEnricher):
    """
    An analyzer that filters, balances, or engineers features based on the
    significance of events. It identifies important data points to focus the
    modeling process.
    """

    @property
    def name(self) -> str:
        return "significance_features"

    @property
    def priority(self) -> int:
        return 70

    def __init__(self, significance_col: str = 'is_significant', min_events_per_ticker: int = 10, mode: str = 'feature_engineering'):
        """
        Initializes the SignificanceAnalyzer.

        Args:
            significance_col (str): The name of the boolean column indicating event significance.
            min_events_per_ticker (int): The minimum number of significant events required for a ticker to be included in 'filter' mode.
            mode (str): The operational mode. Can be 'filter', 'balance', or 'feature_engineering'.
        """
        super().__init__()  # Initialize BaseEnricher (sets up self.logger)

        # Ensure significance_col is a string, not a dict
        if isinstance(significance_col, dict):
            significance_col = significance_col.get('name', 'is_significant')

        self.significance_col = significance_col
        self.min_events_per_ticker = min_events_per_ticker
        self.mode = mode
        logger.info(f"SignificanceAnalyzer initialized in '{self.mode}' mode.")

    def _enrich_impl(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Analyzes data to filter, balance, or create features based on event significance.

        Args:
            df (pd.DataFrame): The input data with a significance column.
            **kwargs: Not used in this implementation.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        # Ensure significance_col is a string for column check
        col_name = self.significance_col
        if isinstance(col_name, dict):
            col_name = col_name.get('name', 'is_significant')

        # ✅ If column is missing, create it based on volatility or other metrics
        if col_name not in df.columns:
            logger.info(f"Significance column '{col_name}' not found. Creating it based on volatility...")
            df = self._create_significance_column(df, col_name)
            if col_name not in df.columns:
                logger.warning(f"Failed to create significance column '{col_name}'. Skipping analysis.")
                return df

        if self.mode == 'filter':
            return self._filter_significant_events(df)
        elif self.mode == 'feature_engineering':
            return self._create_significance_features(df)
        elif self.mode == 'balance':
            logger.warning("Mode 'balance' is not implemented. A dedicated preprocessor should be used.")
            return df
        else:
            logger.warning(f"Unknown mode '{self.mode}'. Returning original data.")
            return df

    def _filter_significant_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the DataFrame, keeping only significant events and tickers with enough data.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Original size for filtering: {len(df)}")

        filtered_df = df[df[self.significance_col]].copy()

        if 'ticker' in filtered_df.columns:
            ticker_counts = filtered_df['ticker'].value_counts()
            valid_tickers = ticker_counts[ticker_counts >= self.min_events_per_ticker].index

            if len(valid_tickers) < len(ticker_counts.index):
                removed_tickers = set(ticker_counts.index) - set(valid_tickers)
                logger.info(f"Removing tickers with insufficient significant events: {removed_tickers}")
                filtered_df = filtered_df[filtered_df['ticker'].isin(valid_tickers)]

        logger.info(f"Filtered from {len(df)} to {len(filtered_df)} rows based on significance.")
        return filtered_df

    def _create_significance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates new features based on the significance column.
        """
        df_out = df.copy()

        # Ensure there's a 'ticker' column for correct grouping
        if 'ticker' not in df_out.columns:
            logger.warning("No 'ticker' column found. Treating data as a single entity.")
            df_out['ticker'] = 'default'

        # Rolling count of significant events
        # ✅ FIX: use transform to avoid duplicate index issues with groupby+rolling+reset_index
        df_out['significant_events_7d'] = (
            df_out.groupby('ticker')[self.significance_col]
            .transform(lambda s: s.rolling(window=7, min_periods=1).sum())
        )
        df_out['significant_events_30d'] = (
            df_out.groupby('ticker')[self.significance_col]
            .transform(lambda s: s.rolling(window=30, min_periods=1).sum())
        )

        # Days since the last significant event
        # ✅ FIX: simple cumcount without nested groupby to avoid duplicate index issues
        def _days_since_last_event(series: pd.Series) -> pd.Series:
            result = pd.Series(0, index=series.index, dtype=float)
            counter = 0
            for i, val in enumerate(series):
                if val:
                    counter = 0
                else:
                    counter += 1
                result.iloc[i] = counter
            return result

        df_out['days_since_last_significant'] = (
            df_out.groupby('ticker')[self.significance_col]
            .transform(_days_since_last_event)
        )

        # Significance intensity
        df_out['significance_intensity_7d'] = df_out['significant_events_7d'] / 7.0

        logger.info("Created new features based on event significance.")
        # If a default ticker was added, remove it before returning
        if 'ticker' in df_out.columns and df_out['ticker'].nunique() == 1 and df_out['ticker'].iloc[0] == 'default':
            df_out = df_out.drop(columns=['ticker'])

        return df_out

    def _create_significance_column(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """
        Creates is_significant column based on volatility or price movements.
        An event is significant if it's in the top 20% of volatility or price changes.
        """
        df_out = df.copy()

        # Try to calculate significance based on available columns
        if 'returns' in df_out.columns:
            # Use returns volatility
            threshold = df_out['returns'].abs().quantile(0.80)  # Top 20%
            df_out[col_name] = df_out['returns'].abs() >= threshold
            logger.info(f"Created '{col_name}' based on returns (threshold: {threshold:.4f})")
        elif 'close' in df_out.columns:
            # Calculate returns from close price
            df_out['_temp_returns'] = (
                df_out['close']
                .pct_change(fill_method=None)
                .replace([float('inf'), float('-inf')], float('nan'))
            )
            threshold = df_out['_temp_returns'].abs().quantile(0.80)
            df_out[col_name] = df_out['_temp_returns'].abs() >= threshold
            df_out = df_out.drop(columns=['_temp_returns'])
            logger.info(f"Created '{col_name}' based on price changes (threshold: {threshold:.4f})")
        elif 'VOLATILITY_20' in df_out.columns:
            # Use existing volatility feature
            threshold = df_out['VOLATILITY_20'].quantile(0.80)
            df_out[col_name] = df_out['VOLATILITY_20'] >= threshold
            logger.info(f"Created '{col_name}' based on VOLATILITY_20 (threshold: {threshold:.4f})")
        else:
            logger.warning("Cannot create significance column: no suitable columns found (returns, close, or VOLATILITY_20)")

        return df_out
