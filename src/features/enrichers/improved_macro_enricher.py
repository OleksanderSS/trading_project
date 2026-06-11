"""
Improved Macro Features Enricher with better null handling for continue mode.
"""


import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class ImprovedMacroEnricher:
    """Improved macro enricher that handles missing data better."""

    def __init__(self):
        self.macro_cache = {}  # Cache last known macro values

    def _handle_missing_macro_data(self, df: pd.DataFrame, macro_data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing macro data with smart interpolation instead of zeros.
        """
        if macro_data.empty:
            logger.warning("No macro data available, using cached values")
            return self._fill_with_cached_values(df)

        # Check for missing dates
        missing_dates = set(df.index) - set(macro_data.index)
        if missing_dates:
            logger.info(f"Found {len(missing_dates)} missing macro dates, interpolating...")

            # Use forward fill with interpolation instead of zeros
            macro_filled = macro_data.reindex(df.index, method='ffill')

            # For remaining NaN values, use interpolation
            macro_filled = macro_filled.interpolate(method='linear', limit=7)

            # For any remaining NaN values, use median of recent values
            for col in macro_filled.columns:
                if macro_filled[col].isna().any():
                    recent_median = macro_filled[col].dropna().tail(30).median()
                    macro_filled[col] = macro_filled[col].fillna(recent_median)

            # Update cache
            self._update_macro_cache(macro_filled)

            return self._merge_macro_data(df, macro_filled)

        return self._merge_macro_data(df, macro_data)

    def _fill_with_cached_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill DataFrame with cached macro values."""
        for date in df.index:
            if date in self.macro_cache:
                for col, value in self.macro_cache[date].items():
                    df.loc[date, f'FRED_{col}'] = value
            else:
                # Use reasonable defaults instead of zeros
                default_values = {
                    'GDP': 2.0,  # 2% GDP growth
                    'UNRATE': 5.0,  # 5% unemployment
                    'CPIAUCSL': 2.5,  # 2.5% inflation
                    'FEDFUNDS': 2.0,  # 2% Fed funds rate
                }
                for col, default in default_values.items():
                    df.loc[date, f'FRED_{col}'] = default

        return df

    def _update_macro_cache(self, macro_data: pd.DataFrame) -> None:
        """Update macro cache with latest values."""
        for date in macro_data.index:
            if date not in self.macro_cache:
                self.macro_cache[date] = {}

            for col in macro_data.columns:
                if not pd.isna(macro_data.loc[date, col]):
                    self.macro_cache[date][col] = macro_data.loc[date, col]

    def _merge_macro_data(self, df: pd.DataFrame, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Merge macro data with main DataFrame."""
        for col in macro_data.columns:
            df[f'FRED_{col}'] = macro_data[col].reindex(df.index, method='ffill')

        return df
