# utils/data_storage.py

import os
import pandas as pd
from threading import Thread
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("TradingProjectLogger")

def _remove_timezone_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove timezone from all datetime columns."""
    for col in df.columns:
        if pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = df[col].dt.tz_convert(None)  # ВИПРАВЛЕНО
    return df

def save_to_storage(df: pd.DataFrame, filepath: str, async_save: bool = True, save_index: bool = False):
    """Save DataFrame to Parquet (synchronously for critical data)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_to_save = _remove_timezone_from_df(df.copy())

    def _save():
        try:
            df_to_save.to_parquet(filepath, index=save_index)
            logger.info(f"[Storage] Saved {len(df_to_save)} rows, {len(df_to_save.columns)} columns to {filepath}")
        except Exception as e:
            logger.error(f"[Storage] Error saving {filepath}: {e}")

    # Use synchronous save for critical data files
    if 'accumulated_data.parquet' in filepath or not async_save:
        _save()
    else:
        Thread(target=_save, daemon=True).start()

def load_from_storage(filepath: str) -> pd.DataFrame:
    """Load DataFrame from Parquet."""
    if not os.path.exists(filepath):
        logger.warning(f"[Storage] File not found: {filepath}")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(filepath)
        logger.info(f"[Storage] Loaded {len(df)} rows, {len(df.columns)} columns from {filepath}")
        return df
    except Exception as e:
        logger.error(f"[Storage] Error loading {filepath}: {e}")
        return pd.DataFrame()
