import logging
import pandas as pd
import numpy as np
import hashlib
import re
from typing import List, Dict, Optional, Union
from src.core.logging.logger import ProjectLogger
logger = ProjectLogger.get_logger('DataCleaner')


class DataCleaner:
    """
    A utility class providing static methods for data sanitization and cleaning 
    of market and news data before feature engineering.
    """

    @staticmethod
    def remove_outliers_zscore(df: pd.DataFrame, columns: Union[str, List[
        str]]='close', threshold: float=3.0) ->pd.DataFrame:
        """
        Removes outliers from specified columns based on Z-score calculated on rolling log returns.
        
        Args:
            df: Input DataFrame.
            columns: Column name or list of columns to analyze.
            threshold: Z-score threshold for outlier detection (default 3.0).
            
        Returns:
            DataFrame with rows containing outliers in specified columns removed.
        """
        if df is None or df.empty:
            return df
        if isinstance(columns, str):
            columns = [columns]
        df_out = df.copy()
        try:
            total_mask = pd.Series([False] * len(df_out), index=df_out.index)
            for col in columns:
                if col not in df_out.columns:
                    continue
                if 'ticker' in df_out.columns:
                    log_returns = df_out.groupby('ticker', group_keys=False
                        )[col].apply(lambda s: np.log(s / s.shift(1)))
                    rolling_mean = log_returns.groupby(df_out['ticker'],
                        group_keys=False).apply(lambda s: s.rolling(20,
                        min_periods=1).mean())
                    rolling_std = log_returns.groupby(df_out['ticker'],
                        group_keys=False).apply(lambda s: s.rolling(20,
                        min_periods=1).std())
                else:
                    log_returns = np.log(df_out[col] / df_out[col].shift(1))
                    rolling_mean = log_returns.rolling(20, min_periods=1).mean()
                    rolling_std = log_returns.rolling(20, min_periods=1).std()
                z_scores = (log_returns - rolling_mean) / rolling_std
                col_mask = (z_scores.abs() > threshold).where(z_scores.notna(), False)
                total_mask |= col_mask
            outlier_count = total_mask.sum()
            if outlier_count > 0:
                df_out = df_out[~total_mask]
                logger.info(
                    f'Removed {outlier_count} rows containing outliers in columns {columns} (Threshold: {threshold})'
                    )
            return df_out
        except Exception as e:
            logger.error(
                f'Error during outlier removal for columns {columns}: {e}')
            return df

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, method: str='ffill'
        ) ->pd.DataFrame:
        """
        Handles missing values in the DataFrame using specified method.
        """
        if df is None or df.empty:
            return df
        df_out = df.copy()
        nan_count = df_out.isna().sum().sum()
        if nan_count > 0:
            data_cols = [col for col in df_out.columns if col != 'ticker']
            if method == 'ffill':
                if 'ticker' in df_out.columns:
                    df_out[data_cols] = df_out.groupby('ticker')[data_cols].ffill()
                else:
                    df_out = df_out.ffill()
            elif method == 'bfill':
                logger.warning(
                    "Backfill is disabled for causal time series cleaning; using forward-fill instead."
                    )
                if 'ticker' in df_out.columns:
                    df_out[data_cols] = df_out.groupby('ticker')[data_cols].ffill()
                else:
                    df_out = df_out.ffill()
            logger.info(f'Handled {nan_count} missing values using {method}.')
        return df_out

    @staticmethod
    def validate_schema(df: pd.DataFrame, required_cols: List[str]) ->bool:
        """
        Validates if the DataFrame contains all required columns.
        """
        if df is None:
            logger.error('Schema validation failed: DataFrame is None')
            return False
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.warning(
                f'Schema validation: Missing required columns: {missing}')
            return False
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f'Schema validation successful for columns: {required_cols}')
        return True


def harmonize_dataframe(df: pd.DataFrame, dropna_cols: bool=False
    ) ->pd.DataFrame:
    """
    Standardizes DataFrame structure, types and removes duplicates.
    """
    if df is None or df.empty:
        logger.warning('Harmonize received an empty or None DataFrame.')
        return df
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.convert_dtypes()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).replace('nan', '')
    if dropna_cols:
        empty_cols = [c for c in df.columns if df[c].dropna().empty]
        if empty_cols:
            df = df.drop(columns=empty_cols)
            logger.info(f'Dropped empty columns: {empty_cols}')
    return df


def safe_fill(df: pd.DataFrame, zero_fill_cols: Optional[List[str]]=None,
    unknown_fill_val: str='unknown') ->pd.DataFrame:
    """
    Safely fills NaN values based on column types and specific column requirements.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    if zero_fill_cols:
        for col in zero_fill_cols:
            if col in df.columns:
                df[col] = df[col].where(df[col].notna(), 0)
    num_cols = df.select_dtypes(include=['number']).columns
    if 'ticker' in df.columns:
        df[num_cols] = df.groupby('ticker')[num_cols].ffill()
    else:
        df[num_cols] = df[num_cols].ffill()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(unknown_fill_val)
    return df


def _sanitize_index_timezone(df: pd.DataFrame, label: str) ->pd.DataFrame:
    """Sanitize DatetimeIndex timezone."""
    try:
        df.index = df.index.tz_localize(None)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'[{label}] Converted DatetimeIndex to timezone-naive.')
    except Exception as e:
        logger.error(f'Виникла помилка: {e}', exc_info=True)
        logger.warning(f'[{label}] Failed to sanitize index timezone: {e}')
        raise
    return df


def _sanitize_column_timezone(df: pd.DataFrame, col: str, label: str) ->None:
    """Sanitize a single datetime column timezone."""
    if df[col].dt.tz is not None:
        try:
            df[col] = df[col].dt.tz_localize(None)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"[{label}] Converted column '{col}' to timezone-naive.")
        except Exception as e:
            logger.error(f'Виникла помилка: {e}', exc_info=True)
            logger.warning(
                f"[{label}] Failed to sanitize column '{col}' timezone: {e}")
            raise


def sanitize_dataframe_timezone(df: pd.DataFrame, label: str='sanitize_df'
    ) ->pd.DataFrame:
    """
    Ensures that a DataFrame's DatetimeIndex and datetime columns are timezone-naive.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = _sanitize_index_timezone(df, label)
    datetime_cols = [col for col in df.columns if pd.api.types.
        is_datetime64_any_dtype(df[col])]
    for col in datetime_cols:
        _sanitize_column_timezone(df, col, label)
    return df


def generate_content_hash(df: pd.DataFrame, cols_to_hash: Optional[List[str]]=None) ->pd.Series:
    """
    Generates hashes for rows based on content. Optimized for performance.
    """
    if cols_to_hash is None:
        cols_to_hash = ['title', 'description', 'published_at']
    if df.empty:
        return pd.Series([], dtype=str)
    content = pd.Series('', index=df.index)
    for col in cols_to_hash:
        if col in df.columns:
            content += df[col].fillna('').astype(str) + ' '
    content = content.str.lower().str.replace('[^\\w\\s]', '', regex=True
        ).str.replace('\\s+', ' ', regex=True).str.strip()

    def _sha256(val):
        return hashlib.sha256(val.encode('utf-8')).hexdigest()
    hashes = content.apply(_sha256)
    logger.info(f'Generated {len(hashes)} content hashes for deduplication.')
    return hashes


def _apply_column_mapping(normalized: pd.DataFrame, column_mapping: Dict[
    str, str]) ->None:
    """Apply column mapping to normalized DataFrame."""
    for old_col, new_col in column_mapping.items():
        if old_col in normalized.columns and new_col not in normalized.columns:
            normalized.rename(columns={old_col: new_col}, inplace=True)


def _add_missing_columns(normalized: pd.DataFrame, source_type: str,
    required_columns: List[str]) ->None:
    """Add missing required columns with defaults."""
    for col in required_columns:
        if col not in normalized.columns:
            if col == 'id':
                normalized[col] = range(len(normalized))
            elif col == 'source_type':
                normalized[col] = source_type
            elif 'score' in col.lower() or 'count' in col.lower():
                normalized[col] = 0.0
            else:
                normalized[col] = ''


def _process_published_at(normalized: pd.DataFrame) ->None:
    """Process published_at column."""
    if 'published_at' in normalized.columns:
        normalized['published_at'] = pd.to_datetime(normalized[
            'published_at'], errors='coerce')


def normalize_to_unified_schema(df: pd.DataFrame, source_type: str,
    required_columns: List[str], column_mapping: Dict[str, str]
    ) ->pd.DataFrame:
    """
    Normalizes a DataFrame to a standard project-wide schema using dynamic mapping.
    """
    if df.empty:
        return df
    normalized = df.copy()
    _apply_column_mapping(normalized, column_mapping)
    _add_missing_columns(normalized, source_type, required_columns)
    _process_published_at(normalized)
    normalized['hash'] = generate_content_hash(normalized)
    logger.info(
        f"Normalized {len(normalized)} records from source '{source_type}' to unified schema."
        )
    return normalized[required_columns]


def merge_and_deduplicate(dataframes: List[pd.DataFrame], hash_col: str='hash'
    ) ->pd.DataFrame:
    """
    Merges DataFrames and removes duplicates based on a hash column.
    """
    if not dataframes:
        return pd.DataFrame()
    merged = pd.concat(dataframes, ignore_index=True)
    if merged.empty:
        return merged
    if hash_col not in merged.columns:
        logger.warning(f"Deduplication skipped: Column '{hash_col}' missing.")
        return merged
    before = len(merged)
    merged.drop_duplicates(subset=[hash_col], keep='last', inplace=True)
    logger.info(
        f"Removed {before - len(merged)} duplicates based on '{hash_col}'. Remaining: {len(merged)}"
        )
    return merged


def filter_by_terms(df: pd.DataFrame, terms: List[str], search_col: str=
    'description') ->pd.DataFrame:
    """
    Filters a DataFrame based on search terms in a specific column.
    """
    if df.empty or not terms or search_col not in df.columns:
        return df
    pattern = '\\b(?:' + '|'.join(map(re.escape, terms)) + ')\\b'
    mask = df[search_col].str.contains(pattern, case=False, regex=True, na=
        False)
    filtered_df = df[mask]
    logger.info(
        f"Filtered records: {len(df)} -> {len(filtered_df)} based on terms in '{search_col}'."
        )
    return filtered_df
