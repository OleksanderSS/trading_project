# src/processing/price_preprocessor.py

import pandas as pd
from typing import List, Optional
from src.core.logging.logger import ProjectLogger

# Default metrics, can be overridden via configuration
DEFAULT_PRICE_METRICS = ['open', 'high', 'low', 'close', 'volume']

logger = ProjectLogger.get_logger("PricePreprocessor")

class PricePreprocessor:
    def normalize_price_df(self, df: pd.DataFrame, price_metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalizes a price DataFrame into a consistent schema: [datetime, ticker, ohlcv, interval].
        Handles raw data from collectors or already partially processed data.
        CRITICAL: Preserves 'interval' column to prevent timeframe mixing.
        """
        metrics = price_metrics or DEFAULT_PRICE_METRICS
        target_columns = ["datetime", "ticker"] + metrics
        preserve_columns = ["interval", "hash"]

        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to price preprocessor.")
            return pd.DataFrame(columns=target_columns)

        df = df.copy()

        if self._is_already_normalized(df, target_columns):
            processed_df = self._preserve_metadata_columns(df, target_columns, preserve_columns)
        else:
            processed_df = self._normalize_structure(df, metrics, target_columns)

        processed_df = self._finalize_dataframe(processed_df, metrics, target_columns, preserve_columns)
        return processed_df

    def _is_already_normalized(self, df: pd.DataFrame, target_columns: List[str]) -> bool:
        """Check if DataFrame is already in normalized format."""
        current_cols = set(df.columns)
        return all(col in current_cols for col in target_columns)

    def _preserve_metadata_columns(self, df: pd.DataFrame, target_columns: List[str], preserve_columns: List[str]) -> pd.DataFrame:
        """Preserve interval and other metadata columns."""
        logger.debug("DataFrame is already in normalized format. Validating metrics...")
        columns_to_keep = target_columns.copy()
        current_cols = set(df.columns)
        
        for col in preserve_columns:
            if col in current_cols and col not in columns_to_keep:
                columns_to_keep.append(col)
        
        return df[columns_to_keep]

    def _normalize_structure(self, df: pd.DataFrame, metrics: List[str], target_columns: List[str]) -> pd.DataFrame:
        """Normalize DataFrame structure by flattening and pivoting."""
        # Handle MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            logger.info("Flattening MultiIndex columns...")
            df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]

        try:
            tidy = self._melt_dataframe(df, metrics)
            processed_df = self._pivot_to_format(tidy)
            return processed_df
        except Exception as e:
            logger.error(f"Failed to normalize price DataFrame structure: {e}")
            return pd.DataFrame(columns=target_columns)

    def _melt_dataframe(self, df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """Melt DataFrame to identify metrics and tickers."""
        value_cols = [c for c in df.columns if any(m in c.lower() for m in metrics)]
        id_cols = [c for c in df.columns if c not in value_cols]
        
        return df.melt(id_vars=id_cols, value_vars=value_cols,
                       var_name="raw_metric", value_name="value")

    def _pivot_to_format(self, tidy: pd.DataFrame) -> pd.DataFrame:
        """Pivot melted DataFrame back to semi-wide format."""
        def split_metric_ticker(name):
            parts = name.lower().split('_')
            found_metric = next((m for m in DEFAULT_PRICE_METRICS if m in parts), None)
            found_ticker = next((p for p in parts if p != found_metric), "unknown").upper()
            return found_metric, found_ticker

        tidy[['metric', 'ticker']] = tidy['raw_metric'].apply(
            lambda x: pd.Series(split_metric_ticker(x))
        )

        return tidy.pivot_table(
            index=["datetime", "ticker"],
            columns="metric", 
            values="value"
        ).reset_index()

    def _finalize_dataframe(self, processed_df: pd.DataFrame, metrics: List[str], target_columns: List[str], preserve_columns: List[str]) -> pd.DataFrame:
        """Apply final validation and timestamp conversion."""
        # Convert datetime
        processed_df["datetime"] = pd.to_datetime(processed_df["datetime"], errors="coerce")
        
        # Ensure all required metrics exist
        for m in metrics:
            if m not in processed_df.columns:
                logger.warning(f"Missing metric column '{m}' in processed data. Adding as NaN.")
                processed_df[m] = pd.NA

        # Preserve metadata columns
        columns_to_keep = target_columns.copy()
        
        for col in preserve_columns:
            if col in processed_df.columns and col not in columns_to_keep:
                columns_to_keep.append(col)
        
        final_df = processed_df[columns_to_keep].dropna(subset=['datetime', 'ticker'])
        
        tickers_count = final_df["ticker"].nunique()
        if 'interval' in final_df.columns:
            logger.info(f"Price normalization complete: {len(final_df)} rows, {tickers_count} tickers, interval column PRESERVED.")
        else:
            logger.warning(f"Price normalization complete: {len(final_df)} rows, {tickers_count} tickers, interval column MISSING.")
        
        return final_df