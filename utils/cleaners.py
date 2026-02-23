# utils/cleaners.py

import pandas as pd
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("TradingProjectLogger")

def check_datetime_types(df: pd.DataFrame, label: str):
    if df.empty:
        logger.debug(f"[tz-check] {label}: empty")
        return
    if isinstance(df.index, pd.DatetimeIndex):
        logger.debug(f"[tz-check] {label} index dtype: {df.index.dtype}")
        if df.index.tz is not None:
            logger.warning(f"[tz-check] {label} index tz-aware: {df.index.tz}")
    for col in df.columns:
        dtype = df[col].dtype
        if "datetime64" in str(dtype):
            logger.debug(f"[tz-check] {label} column '{col}' dtype: {dtype}")
            if "tz" in str(dtype):
                logger.warning(f"[tz-check] {label} column '{col}' is tz-aware")

def assert_tz_naive(df: pd.DataFrame, label: str, strict: bool = True):
    if df.empty:
        return
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        msg = f"[assert_tz_naive] {label} index still tz-aware: {df.index.tz}"
        if strict:
            raise ValueError(msg)
        else:
            logger.error(msg)
    for col in df.columns:
        dtype = df[col].dtype
        if "datetime64" in str(dtype) and "tz" in str(dtype):
            msg = f"[assert_tz_naive] {label} tz-aware column: {col}"
            if strict:
                raise ValueError(msg)
            else:
                logger.error(msg)

def sanitize_dataframe(df: pd.DataFrame, label: str = "sanitize_dataframe") -> pd.DataFrame:
    if df is None or df.empty:
        logger.debug(f"[sanitize_dataframe] {label}: empty or None")
        return df
    try:
        check_datetime_types(df, f"{label} [before]")

        # --- Очистка tz with andнwhereксу ---
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            try:
                df.index = df.index.tz_convert(None)
                logger.debug(f"[sanitize_dataframe] {label}: tz_convert on index")
            except Exception:
                df.index = df.index.tz_convert(None)
                logger.debug(f"[sanitize_dataframe] {label}: tz_convert on index")

        # --- Очистка tz with колонок ---
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = df[col].dt.tz_convert(None)
                    logger.debug(f"[sanitize_dataframe] {label}: tz_convert on column '{col}'")
                except Exception:
                    try:
                        df[col] = df[col].dt.tz_convert(None)
                        logger.debug(f"[sanitize_dataframe] {label}: tz_convert on column '{col}'")
                    except Exception as e:
                        logger.warning(f"[sanitize_dataframe] {label}: cannot clean tz from '{col}': {e}")

        check_datetime_types(df, f"{label} [after]")
        assert_tz_naive(df, f"{label} [after]", strict=False)

    except Exception as e:
        logger.error(f"[sanitize_dataframe] {label} error: {e}")
    return df

def read_csv_safe(path: str, sep: str = ",", **kwargs) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1251"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, sep=sep, **kwargs)
            logger.info(f"[read_csv_safe] Successfully read {path} with encoding={enc}")
            return sanitize_dataframe(df, label=f"read_csv_safe[{enc}]")
        except Exception as e:
            last_err = e
            continue
    logger.error(f"[read_csv_safe] Cannot read CSV {path}: {last_err}")
    return pd.DataFrame()