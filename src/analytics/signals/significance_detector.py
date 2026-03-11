
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

def _get_adjusted_threshold(indicator: str, config: Dict, ticker: Optional[str] = None, timeframe: Optional[str] = None) -> float:
    """
    Returns an adjusted threshold for a given indicator, applying ticker and timeframe specifics.
    """
    base_threshold = config.get('thresholds', {}).get(indicator, 0.05) # Default to 5%
    
    if ticker and ticker in config.get('adjustments', {}).get('tickers', {}):
        return config['adjustments']['tickers'][ticker].get(indicator, base_threshold)
        
    if timeframe and timeframe in config.get('adjustments', {}).get('timeframes', {}):
        return config['adjustments']['timeframes'][timeframe].get(indicator, base_threshold)
        
    return base_threshold

def _is_change_significant(
    current_value: float, 
    previous_value: float, 
    indicator: str, 
    config: Dict, 
    ticker: Optional[str] = None, 
    timeframe: Optional[str] = None
) -> bool:
    """Checks if a percentage change is significant based on a dynamic threshold."""
    if pd.isna(previous_value) or pd.isna(current_value) or previous_value == 0:
        return False
        
    pct_change = abs((current_value - previous_value) / previous_value)
    threshold = _get_adjusted_threshold(indicator, config, ticker, timeframe)
    
    return pct_change > threshold

def _extract_from_col(col_name: str, known_items: List[str]) -> Optional[str]:
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
    config: Dict[str, Any],
    price_cols: Optional[List[str]] = None,
    volume_cols: Optional[List[str]] = None,
    sentiment_cols: Optional[List[str]] = None,
    macro_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Detects significant events in a DataFrame based on the provided configuration.
    """
    if data.empty:
        logger.warning("Input DataFrame is empty.")
        return data

    df = data.copy()
    
    if 'ticker' not in df.columns:
        df['ticker'] = 'default_ticker'

    sort_keys = ['ticker', 'date'] if 'date' in df.columns else ['ticker']
    df = df.sort_values(by=sort_keys).reset_index(drop=True)
    
    all_sig_cols = []
    
    col_map = {
        'price_change': price_cols or [],
        'volume_change': volume_cols or [],
        'sentiment_change': sentiment_cols or [],
    }

    known_tickers = config.get('known_tickers', [])
    known_timeframes = config.get('known_timeframes', [])

    for indicator, columns in col_map.items():
        for col in columns:
            if col not in df.columns: continue

            ticker = _extract_from_col(col, known_tickers)
            timeframe = _extract_from_col(col, known_timeframes)
            
            prev_col = df.groupby('ticker')[col].shift(1)
            sig_col_name = f"sig_{col}"
            
            df[sig_col_name] = df.apply(
                lambda row: _is_change_significant(
                    row[col], prev_col.loc[row.name], indicator, config, ticker, timeframe
                ), 
                axis=1
            )
            all_sig_cols.append(sig_col_name)

    if macro_cols:
        for col in macro_cols:
            if col not in df.columns: continue
            indicator = _detect_macro_indicator_type(col)
            prev_col = df[col].shift(1)
            sig_col_name = f"sig_{col}"
            df[sig_col_name] = df.apply(
                lambda row: _is_change_significant(
                    row[col], prev_col.loc[row.name], indicator, config
                ),
                axis=1
            )
            all_sig_cols.append(sig_col_name)

    if all_sig_cols:
        df['is_significant'] = df[all_sig_cols].any(axis=1)
    else:
        df['is_significant'] = False

    if 'default_ticker' in df['ticker'].values:
        df = df.drop(columns=['ticker'])
        
    return df

def get_significance_summary(df: pd.DataFrame) -> Dict:
    """
    Returns a summary of significant events from an analyzed DataFrame.
    """
    if 'is_significant' not in df.columns:
        return {}
    
    total_records = len(df)
    significant_records = df['is_significant'].sum()
    
    summary = {
        'total_records': total_records,
        'significant_records': int(significant_records),
        'significance_ratio': significant_records / total_records if total_records > 0 else 0,
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
