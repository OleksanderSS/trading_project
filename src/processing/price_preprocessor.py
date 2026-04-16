# src/processing/price_preprocessor.py

import pandas as pd
from typing import List, Optional
from src.core.logging.logger import ProjectLogger

# Default metrics, can be overridden via configuration
DEFAULT_PRICE_METRICS = ['open', 'high', 'low', 'close', 'volume']

logger = ProjectLogger.get_logger("PricePreprocessor")

def normalize_price_df(df: pd.DataFrame, price_metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalizes a price DataFrame into a consistent schema: [datetime, ticker, ohlcv, interval].
    Handles raw data from collectors or already partially processed data.
    CRITICAL: Preserves 'interval' column to prevent timeframe mixing.
    """
    metrics = price_metrics or DEFAULT_PRICE_METRICS
    target_columns = ["datetime", "ticker"] + metrics
    
    # CRITICAL: Preserve interval column if it exists
    preserve_columns = ["interval", "hash"]  # Columns to preserve if they exist

    if df is None or df.empty:
        logger.warning("Empty DataFrame provided to price preprocessor.")
        return pd.DataFrame(columns=target_columns)

    df = df.copy()

    # 1. Check if the DataFrame is already in the correct format
    current_cols = set(df.columns)
    if all(col in current_cols for col in target_columns):
        logger.debug("DataFrame is already in normalized format. Validating metrics...")
        # Preserve interval and other metadata columns
        columns_to_keep = target_columns.copy()
        for col in preserve_columns:
            if col in current_cols and col not in columns_to_keep:
                columns_to_keep.append(col)
        processed_df = df[columns_to_keep]
    else:
        # 2. Handle MultiIndex columns (flattening) if present
        if isinstance(df.columns, pd.MultiIndex):
            logger.info("Flattening MultiIndex columns...")
            df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]

        # 3. Detect metric and ticker columns using melting
        # We assume columns are named like 'close_AAPL' or 'AAPL_close'
        value_cols = [c for c in df.columns if any(m in c.lower() for m in metrics)]
        id_cols = [c for c in df.columns if c not in value_cols]

        try:
            tidy = df.melt(id_vars=id_cols, value_vars=value_cols,
                           var_name="raw_metric", value_name="value")
            
            # Determine ticker and metric from the column name
            # Heuristic: split by '_' and identify which part matches a metric
            def split_metric_ticker(name):
                parts = name.lower().split('_')
                found_metric = next((m for m in metrics if m in parts), None)
                found_ticker = next((p for p in parts if p != found_metric), "unknown").upper()
                return found_metric, found_ticker

            tidy[['metric', 'ticker']] = tidy['raw_metric'].apply(
                lambda x: pd.Series(split_metric_ticker(x))
            )

            # 4. Pivot back to semi-wide format
            processed_df = tidy.pivot_table(
                index=["datetime", "ticker"],
                columns="metric", 
                values="value"
            ).reset_index()
            
        except Exception as e:
            logger.error(f"Failed to normalize price DataFrame structure: {e}")
            return pd.DataFrame(columns=target_columns)

    # 5. Final validation and timestamp conversion
    processed_df["datetime"] = pd.to_datetime(processed_df["datetime"], errors="coerce")
    
    # Ensure all required metrics exist (fill missing with NaN)
    for m in metrics:
        if m not in processed_df.columns:
            logger.warning(f"Missing metric column '{m}' in processed data. Adding as NaN.")
            processed_df[m] = pd.NA

    # Enforce strict schema and order, but preserve interval if it exists
    columns_to_keep = target_columns.copy()
    for col in preserve_columns:
        if col in processed_df.columns and col not in columns_to_keep:
            columns_to_keep.append(col)
    
    final_df = processed_df[columns_to_keep].dropna(subset=['datetime', 'ticker'])
    
    tickers_count = final_df["ticker"].nunique()
    if 'interval' in final_df.columns:
        logger.info(f"Price normalization complete: {len(final_df)} rows, {tickers_count} tickers, interval column PRESERVED.")
    else:
        logger.warning(f"Price normalization complete: {len(final_df)} rows, {tickers_count} tickers, interval column MISSING!")

    return final_df