
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

def _get_adjusted_threshold(indicator: str, config: dict, ticker: str | None = None, timeframe: str | None = None) -> float:
    """
    Returns an adjusted threshold for a given indicator, applying ticker and timeframe specifics.
    """
    base_threshold = config.get('thresholds', {}).get(indicator, 0.05) # Default to 5%

    if ticker and ticker in config.get('adjustments', {}).get('tickers', {}):
        return float(config['adjustments']['tickers'][ticker].get(indicator, base_threshold))

    if timeframe and timeframe in config.get('adjustments', {}).get('timeframes', {}):
        return float(config['adjustments']['timeframes'][timeframe].get(indicator, base_threshold))

    return float(base_threshold)

def _is_change_significant(
    current_value: float,
    previous_value: float,
    indicator: str,
    config: dict,
    ticker: str | None = None,
    timeframe: str | None = None
) -> bool:
    """Checks if a percentage change is significant based on a dynamic threshold."""
    if pd.isna(previous_value) or pd.isna(current_value) or previous_value == 0:
        return False

    pct_change = abs((current_value - previous_value) / previous_value)
    threshold = _get_adjusted_threshold(indicator, config, ticker, timeframe)

    return pct_change > threshold

def _extract_from_col(col_name: str, known_items: list[str]) -> str | None:
    """Helper to extract a known item from a column name."""
    col_lower = col_name.lower()
    for item in known_items:
        if item.lower() in col_lower:
            return item
    return None

def _detect_macro_indicator_type(col_name: str) -> str:
    """Determines the macro indicator type from the column name."""
    col_lower = col_name.lower()
    if 'vix' in col_lower:
        return 'vix_change'
    elif 'bond' in col_lower or 'yield' in col_lower:
        return 'bond_yield_change'
    else:
        return 'price_change'

def detect_significant_events(
    data: pd.DataFrame,
    config: dict[str, Any],
    price_cols: list[str] | None = None,
    volume_cols: list[str] | None = None,
    sentiment_cols: list[str] | None = None,
    macro_cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Detects significant events in a DataFrame based on the provided configuration.
    """
    if data.empty:
        logger.warning("Input DataFrame is empty.")
        return data

    df = _prepare_dataframe(data)
    all_sig_cols = []

    # Process regular columns
    col_map = {
        'price_change': price_cols or [],
        'volume_change': volume_cols or [],
        'sentiment_change': sentiment_cols or [],
    }

    all_sig_cols.extend(_process_regular_columns(df, col_map, config))

    # Process macro columns
    if macro_cols:
        all_sig_cols.extend(_process_macro_columns(df, macro_cols, config))

    # Set significance flag
    _set_significance_flag(df, all_sig_cols)

    # Clean up temporary ticker column
    df = _cleanup_dataframe(df)

    return df

def _prepare_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for significance detection"""
    df = data.copy()

    if 'ticker' not in df.columns:
        df['ticker'] = 'default_ticker'

    sort_keys = ['ticker', 'date'] if 'date' in df.columns else ['ticker']
    df = df.sort_values(by=sort_keys).reset_index(drop=True)

    return df

def _process_regular_columns(df: pd.DataFrame, col_map: dict[str, list[str]],
                           config: dict[str, Any]) -> list[str]:
    """Process regular columns for significance detection"""
    all_sig_cols = []
    known_tickers = config.get('known_tickers', [])
    known_timeframes = config.get('known_timeframes', [])

    for indicator, columns in col_map.items():
        for col in columns:
            if col not in df.columns:
                continue

            sig_col = _create_significance_column(df, col, indicator, config, known_tickers, known_timeframes)
            if sig_col:
                all_sig_cols.append(sig_col)

    return all_sig_cols

def _process_macro_columns(df: pd.DataFrame, macro_cols: list[str],
                         config: dict[str, Any]) -> list[str]:
    """Process macro columns for significance detection"""
    all_sig_cols = []

    for col in macro_cols:
        if col not in df.columns:
            continue

        indicator = _detect_macro_indicator_type(col)
        sig_col = _create_macro_significance_column(df, col, indicator, config)
        if sig_col:
            all_sig_cols.append(sig_col)

    return all_sig_cols

def _create_significance_column(df: pd.DataFrame, col: str, indicator: str,
                               config: dict[str, Any], known_tickers: list[str],
                               known_timeframes: list[str]) -> str | None:
    """Create significance column for regular indicators"""
    ticker = _extract_from_col(col, known_tickers)
    timeframe = _extract_from_col(col, known_timeframes)

    prev_col = df.groupby('ticker')[col].shift(1)
    sig_col_name = f"sig_{col}"

    # Use helper function to avoid lambda closure issues
    def check_significance(row, col_name=col, prev_col_data=prev_col,
                          ind=indicator, cfg=config, tk=ticker, tf=timeframe):
        return _is_change_significant(
            row[col_name], prev_col_data.loc[row.name], ind, cfg, tk, tf
        )

    df[sig_col_name] = df.apply(check_significance, axis=1)
    return sig_col_name

def _create_macro_significance_column(df: pd.DataFrame, col: str,
                                    indicator: str, config: dict[str, Any]) -> str | None:
    """Create significance column for macro indicators"""
    prev_col = df[col].shift(1)
    sig_col_name = f"sig_{col}"

    # Use helper function to avoid lambda closure issues
    def check_macro_significance(row, col_name=col, prev_col_data=prev_col,
                               ind=indicator, cfg=config):
        return _is_change_significant(
            row[col_name], prev_col_data.loc[row.name], ind, cfg
        )

    df[sig_col_name] = df.apply(check_macro_significance, axis=1)
    return sig_col_name

def _set_significance_flag(df: pd.DataFrame, all_sig_cols: list[str]) -> None:
    """Set the overall significance flag"""
    if all_sig_cols:
        df['is_significant'] = df[all_sig_cols].any(axis=1)
    else:
        df['is_significant'] = False

def _cleanup_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up temporary columns"""
    if 'default_ticker' in df['ticker'].values:
        df = df.drop(columns=['ticker'])
    return df

def get_significance_summary(df: pd.DataFrame) -> dict:
    """
    Returns a summary of significant events from an analyzed DataFrame.
    """
    if 'is_significant' not in df.columns:
        return {}

    total_records = len(df)
    significant_records = df['is_significant'].sum()

    summary: dict[str, Any] = {
        'total_records': total_records,
        'significant_records': int(significant_records),
        'significance_ratio': float(significant_records / total_records if total_records > 0 else 0),
        'ticker_stats': {}
    }

    if 'ticker' in df.columns:
        ticker_groups = df.groupby('ticker')['is_significant']
        summary['ticker_stats'] = {
            ticker: {
                'significant': int(group.sum()),
                'total': len(group),
                'ratio': group.mean()
            } for ticker, group in ticker_groups
        }

    return summary
