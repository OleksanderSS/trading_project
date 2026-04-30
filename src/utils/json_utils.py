# utils/json_utils.py

import pandas as pd
import numpy as np
from datetime import datetime
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

def sanitize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Converts all datetime columns to ISO format for serialization."""
    df_out = df.copy()
    for col in df_out.columns:
        if pd.api.types.is_datetime64_any_dtype(df_out[col]):
            df_out[col] = df_out[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)
            logger.debug(f"[json_utils] Column '{col}' converted to ISO format")
    return df_out

def _convert_timestamp(value):
    """Convert timestamp to ISO format."""
    return value.isoformat()

def _convert_nan():
    """Convert NaN to None."""
    return None

def _convert_numpy_numeric(value):
    """Convert numpy numeric types to standard Python types."""
    return value.item()

def _convert_dict(value):
    """Recursively convert dictionary."""
    return sanitize_record_for_json(value)

def _convert_list(value, convert_func):
    """Convert list elements."""
    return [convert_func(v) for v in value]

def _convert_value(value):
    """Convert a single value to JSON-compatible format."""
    if isinstance(value, (pd.Timestamp, datetime)):
        return _convert_timestamp(value)
    if isinstance(value, float) and np.isnan(value):
        return _convert_nan()
    if isinstance(value, (np.integer, np.floating)):
        return _convert_numpy_numeric(value)
    if isinstance(value, dict):
        return _convert_dict(value)
    if isinstance(value, list):
        return _convert_list(value, _convert_value)
    return value

def sanitize_record_for_json(record: dict) -> dict:
    """Recursively converts a dictionary to a JSON-compatible format."""
    sanitized = {k: _convert_value(v) for k, v in record.items()}
    logger.debug(f"[json_utils] Dictionary sanitized for JSON: keys={list(sanitized.keys())}")
    return sanitized