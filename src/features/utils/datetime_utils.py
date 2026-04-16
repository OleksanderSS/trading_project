"""
Utility functions for consistent datetime and index handling across pipeline stages.

This module provides standardized functions to:
1. Normalize datetime columns (column vs index)
2. Handle timezone conversion
3. Support multiple datetime column names (datetime, published_at, timestamp, etc.)
"""

import pandas as pd
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def ensure_datetime_column(df: pd.DataFrame, raise_on_missing: bool = False) -> pd.DataFrame:
    """
    Ensures a DataFrame has a proper 'datetime' column.
    
    Handles:
    - Restores datetime from index if it's a DatetimeIndex
    - Renames published_at/timestamp/date to datetime if datetime missing
    - Removes timezone info to avoid comparison errors
    - Ensures column exists for groupby/merge operations
    
    Args:
        df: Input DataFrame
        raise_on_missing: If True, raises ValueError if datetime cannot be found/created
    
    Returns:
        DataFrame with datetime column properly set
        
    Raises:
        ValueError: If raise_on_missing=True and no datetime column found
    """
    df = df.copy()
    
    # Check if datetime is already in columns
    if 'datetime' in df.columns:
        # Ensure it's datetime type and remove timezone
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df['datetime'] = df['datetime'].dt.tz_localize(None)
        return df
    
    # Check if datetime is in index
    if df.index.name == 'datetime' and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df['datetime'] = df['datetime'].dt.tz_localize(None)
        return df
    
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if 'index' in df.columns:
            df = df.rename(columns={'index': 'datetime'})
        elif df.columns[0] not in df.columns[1:]:  # Check if first col is already named 'datetime'
            df = df.rename(columns={df.columns[0]: 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['datetime'] = df['datetime'].dt.tz_localize(None)
        return df
    
    # Try alternative datetime column names
    for alt_name in ['published_at', 'timestamp', 'date', 'created_at', 'updated_at', 'time']:
        if alt_name in df.columns:
            logger.info(f"✅ Using '{alt_name}' column as datetime")
            df['datetime'] = pd.to_datetime(df[alt_name], utc=True)
            df['datetime'] = df['datetime'].dt.tz_localize(None)
            return df
    
    # If datetime not found
    if raise_on_missing:
        available_cols = df.columns.tolist()[:10]
        raise ValueError(
            f"Cannot find datetime column. Available columns: {available_cols}"
        )
    
    logger.warning(f"⚠️ No datetime column found in DataFrame")
    logger.warning(f"   Available columns: {df.columns.tolist()[:10]}")
    
    # Create a default datetime if needed
    df['datetime'] = pd.Timestamp.now()
    return df


def ensure_ticker_column(df: pd.DataFrame, default_ticker: str = 'UNKNOWN') -> pd.DataFrame:
    """
    Ensures a DataFrame has a 'ticker' column.
    
    Args:
        df: Input DataFrame
        default_ticker: Default value if ticker column missing
    
    Returns:
        DataFrame with ticker column
    """
    df = df.copy()
    if 'ticker' not in df.columns:
        df['ticker'] = default_ticker
    return df


def normalize_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes metadata columns across pipeline stages.
    
    Ensures:
    - datetime column exists and is proper type
    - ticker column exists
    - datetime has no timezone (UTC localization removed)
    - Both columns are accessible as DataFrame columns (not index)
    
    Args:
        df: Input DataFrame
    
    Returns:
        Normalized DataFrame with proper metadata columns
    """
    df = ensure_datetime_column(df, raise_on_missing=False)
    df = ensure_ticker_column(df)
    return df


def split_datetime_ticker(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into features (without datetime/ticker) and metadata.
    
    Useful for keeping features clean for model input while preserving metadata.
    
    Args:
        df: Input DataFrame
    
    Returns:
        (features_df, metadata_df) where:
        - features_df contains all columns except datetime, ticker
        - metadata_df contains datetime and ticker columns
    """
    df = ensure_datetime_column(df, raise_on_missing=True)
    df = ensure_ticker_column(df)
    
    metadata_cols = ['datetime', 'ticker']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    return df[feature_cols], df[metadata_cols]


def roundtrip_datetime_ticker(features_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines features with metadata while preserving index alignment.
    
    Ensures that:
    - datetime and ticker are properly restored to features
    - No index misalignment occurs
    
    Args:
        features_df: Feature DataFrame (may lack datetime/ticker)
        metadata_df: Metadata DataFrame with datetime and ticker
    
    Returns:
        Combined DataFrame with datetime and ticker columns
    """
    result = features_df.copy()
    
    # Reset index if needed
    if result.index.name == 'datetime' or isinstance(result.index, pd.DatetimeIndex):
        result = result.reset_index()
    
    # Add metadata
    if len(metadata_df) == len(result):
        result['datetime'] = metadata_df['datetime'].values
        result['ticker'] = metadata_df['ticker'].values
    else:
        logger.warning(
            f"⚠️ Length mismatch: features={len(result)}, metadata={len(metadata_df)}"
        )
    
    return result


def deduplicate_on_metadata(df: pd.DataFrame, keep: str = 'first') -> pd.DataFrame:
    """
    Deduplicates DataFrame on datetime and ticker columns.
    
    Args:
        df: Input DataFrame
        keep: 'first', 'last', or False (drop all duplicates)
    
    Returns:
        Deduplicated DataFrame
    """
    df = normalize_metadata_columns(df)
    
    dedup_cols = ['datetime', 'ticker']
    # Filter to only existing columns
    dedup_cols = [c for c in dedup_cols if c in df.columns]
    
    if not dedup_cols:
        logger.warning("⚠️ No duplicate keys available (missing datetime/ticker)")
        return df
    
    return df.drop_duplicates(subset=dedup_cols, keep=keep).reset_index(drop=True)


def ensure_datetime_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures DataFrame is sorted by datetime (and ticker if present).
    
    Args:
        df: Input DataFrame
    
    Returns:
        Sorted DataFrame
    """
    df = ensure_datetime_column(df, raise_on_missing=True)
    
    sort_cols = ['datetime']
    if 'ticker' in df.columns:
        sort_cols.append('ticker')
    
    return df.sort_values(sort_cols).reset_index(drop=True)
