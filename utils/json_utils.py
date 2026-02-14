# utils/json_utils.py

import pandas as pd
import numpy as np
from datetime import datetime
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("TradingProjectLogger")

def sanitize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Конвертує all datetime-колонки у ISO-формат for серandалandforцandї."""
    df_out = df.copy()
    for col in df_out.columns:
        if pd.api.types.is_datetime64_any_dtype(df_out[col]):
            df_out[col] = df_out[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)
            logger.debug(f"[json_utils] Колонка '{col}' конвертована у ISO-формат")
    return df_out

def sanitize_record_for_json(record: dict) -> dict:
    """Рекурсивно конвертує словник у формат, сумandсний with JSON."""
    def convert(value):
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        elif isinstance(value, float) and np.isnan(value):
            return None
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()  # перетворює numpy-типи у сandндартнand int/float
        elif isinstance(value, dict):
            return sanitize_record_for_json(value)
        elif isinstance(value, list):
            return [convert(v) for v in value]
        else:
            return value

    sanitized = {k: convert(v) for k, v in record.items()}
    logger.debug(f"[json_utils] Словник очищено for JSON: ключand={list(sanitized.keys())}")
    return sanitized