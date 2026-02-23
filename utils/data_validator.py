# utils/data_validator.py

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Any, Dict
from datetime import datetime
from utils.trading_calendar import TradingCalendar
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("TradingProjectLogger")


class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1,
    check_nulls: bool = True,
    check_duplicates: bool = True,
    context: str = "data"
) -> pd.DataFrame:
    """Validate DataFrame integrity"""
    if df is None:
        raise DataValidationError(f"{context}: DataFrame is None")
    
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"{context}: Expected DataFrame, got {type(df)}")
    
    if df.empty:
        raise DataValidationError(f"{context}: DataFrame is empty")
    
    if len(df) < min_rows:
        raise DataValidationError(f"{context}: DataFrame has {len(df)} rows, minimum {min_rows} required")
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise DataValidationError(f"{context}: Missing required columns: {missing_cols}")
    
    # Check for null values
    if check_nulls:
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0].to_dict()
        if null_cols:
            logger.warning(f"{context}: Null values found: {null_cols}")
    
    # Check for duplicates
    if check_duplicates and df.duplicated().any():
        dup_count = df.duplicated().sum()
        logger.warning(f"{context}: Found {dup_count} duplicate rows")
    
    return df


def safe_data_operation(
    operation: callable,
    data: Any,
    context: str = "operation",
    default_return: Any = None
) -> Any:
    """Safely perform operation on data with validation"""
    try:
        # Validate input data
        if isinstance(data, pd.DataFrame):
            validate_dataframe(data, context=context)
        elif isinstance(data, pd.Series):
            if data.empty:
                raise DataValidationError(f"{context}: Series is empty")
        
        # Perform operation
        result = operation(data)
        
        # Validate result
        if isinstance(result, pd.DataFrame):
            validate_dataframe(result, context=f"{context}_result")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in {context}: {e}")
        return default_return


def clean_and_validate(
    X,
    y,
    drop_empty_rows: bool = True,
    min_nonzero_ratio: float = 0.01,
    calendar: Optional[TradingCalendar] = None,
    fill_strategy: str = "ffill"
):
    """
    Очистка and валandдацandя X and y перед тренуванням моwhereлей.
    """

    if X is None or y is None:
        raise ValueError("[VALIDATOR] X or y дорandвнює None")
    if len(X) == 0 or len(y) == 0:
        raise ValueError("[VALIDATOR] X or y порожнand")

    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y_ser = y if isinstance(y, pd.Series) else pd.Series(y)
    logger.debug(f"[VALIDATOR] Початковand роwithмandри X: {X_df.shape}, y: {y_ser.shape}")

    # forмandна Inf  NaN
    X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_ser.replace([np.inf, -np.inf], np.nan, inplace=True)

    # forповnotння пропускandв
    if fill_strategy == "ffill":
        X_df = X_df.fillna(method="ffill").fillna(method="bfill").fillna(0)
        y_ser = y_ser.fillna(method="ffill").fillna(method="bfill").fillna(0)
    elif fill_strategy == "zero":
        X_df = X_df.fillna(0)
        y_ser = y_ser.fillna(0)

    # видалення notторгових днandв
    if calendar and "date" in X_df.columns:
        try:
            X_df["date"] = pd.to_datetime(X_df["date"], errors="coerce").dt.date
            mask = X_df["date"].apply(calendar.is_trading_day)
            removed = (~mask).sum()
            if removed > 0:
                logger.warning(f"[VALIDATOR] Видалено {removed} notторгових рядкandв")
            X_df = X_df.loc[mask]
            y_ser = y_ser.loc[mask]
        except Exception as e:
            logger.warning(f"[VALIDATOR] Не вдалося перевandрити торговand днand: {e}")

    # видалення порожнandх рядкandв
    if drop_empty_rows:
        mask = ~((X_df == 0).all(axis=1) | X_df.isna().all(axis=1))
        removed_rows = len(X_df) - mask.sum()
        if removed_rows > 0:
            logger.warning(f"[VALIDATOR] Видалено {removed_rows} повнandстю порожнandх рядкandв")
        X_df = X_df.loc[mask]
        y_ser = y_ser.loc[mask]

    # видалення слабких колонок
    nonzero_ratio = (X_df != 0).sum(axis=0) / max(1, len(X_df))
    weak_cols = nonzero_ratio[nonzero_ratio < min_nonzero_ratio].index
    if len(weak_cols) > 0:
        logger.warning(f"[VALIDATOR] Видалено {len(weak_cols)} слабких колонок: {list(weak_cols)}")
        X_df.drop(columns=weak_cols, inplace=True)

    # фandнальнand перевandрки
    if X_df.empty:
        raise ValueError("[VALIDATOR] Пandсля очистки X сandв порожнandм")
    if y_ser.empty:
        raise ValueError("[VALIDATOR] Пandсля очистки y сandв порожнandм")
    if X_df.isna().sum().sum() > 0 or y_ser.isna().sum() > 0:
        raise ValueError("[VALIDATOR] Пandсля очистки forлишились NaN")

    logger.info(f"[VALIDATOR] Пandсля очистки X: {X_df.shape}, y: {y_ser.shape}")
    return X_df.values, y_ser.values


def validate_ohlcv_data(
    df: pd.DataFrame,
    price_columns: List[str] = ["open", "high", "low", "close"],
    volume_column: str = "volume",
    datetime_column: str = "datetime",
    strict_mode: bool = True
) -> pd.DataFrame:
    """
    Validate OHLCV candlestick data for trading
    
    Args:
        df: DataFrame with OHLCV data
        price_columns: List of price column names
        volume_column: Volume column name
        datetime_column: Datetime column name
        strict_mode: If True, raise exceptions; if False, log warnings
        
    Returns:
        Validated DataFrame
        
    Raises:
        DataValidationError: If validation fails in strict mode
    """
    
    context = "OHLCV validation"
    
    # Basic DataFrame validation
    validate_dataframe(
        df, 
        required_columns=price_columns + [volume_column, datetime_column],
        context=context
    )
    
    errors = []
    warnings = []
    
    # 1. Check for negative prices
    for col in price_columns:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                error_msg = f"{context}: Found {negative_count} negative values in {col}"
                if strict_mode:
                    errors.append(error_msg)
                else:
                    warnings.append(error_msg)
    
    # 2. Check for negative volume
    if volume_column in df.columns:
        negative_volume = (df[volume_column] < 0).sum()
        if negative_volume > 0:
            error_msg = f"{context}: Found {negative_volume} negative volume values"
            if strict_mode:
                errors.append(error_msg)
            else:
                warnings.append(error_msg)
    
    # 3. Check OHLC logic: high >= max(open, close) and low <= min(open, close)
    if all(col in df.columns for col in ["open", "high", "low", "close"]):
        # High should be >= max(open, close)
        high_invalid = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
        if high_invalid > 0:
            error_msg = f"{context}: Found {high_invalid} rows where high < max(open, close)"
            if strict_mode:
                errors.append(error_msg)
            else:
                warnings.append(error_msg)
        
        # Low should be <= min(open, close)
        low_invalid = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
        if low_invalid > 0:
            error_msg = f"{context}: Found {low_invalid} rows where low > min(open, close)"
            if strict_mode:
                errors.append(error_msg)
            else:
                warnings.append(error_msg)
        
        # High should be >= low
        high_low_invalid = (df["high"] < df["low"]).sum()
        if high_low_invalid > 0:
            error_msg = f"{context}: Found {high_low_invalid} rows where high < low"
            if strict_mode:
                errors.append(error_msg)
            else:
                warnings.append(error_msg)
    
    # 4. Check for zero prices (possible data issues)
    for col in price_columns:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                warning_msg = f"{context}: Found {zero_count} zero values in {col} (possible data quality issue)"
                warnings.append(warning_msg)
    
    # 5. Check datetime continuity and gaps
    if datetime_column in df.columns:
        try:
            df[datetime_column] = pd.to_datetime(df[datetime_column])
            df_sorted = df.sort_values(datetime_column)
            
            # Check for duplicate timestamps
            duplicate_times = df_sorted[datetime_column].duplicated().sum()
            if duplicate_times > 0:
                warning_msg = f"{context}: Found {duplicate_times} duplicate timestamps"
                warnings.append(warning_msg)
            
            # Check for large time gaps (possible missing data)
            if len(df_sorted) > 1:
                time_diffs = df_sorted[datetime_column].diff().dropna()
                # Flag gaps larger than 2x median time difference
                median_gap = time_diffs.median()
                large_gaps = (time_diffs > median_gap * 2).sum()
                if large_gaps > 0:
                    warning_msg = f"{context}: Found {large_gaps} large time gaps (possible missing data)"
                    warnings.append(warning_msg)
                    
        except Exception as e:
            warning_msg = f"{context}: Could not validate datetime continuity: {e}"
            warnings.append(warning_msg)
    
    # Log warnings
    for warning in warnings:
        logger.warning(warning)
    
    # Raise errors if strict mode
    if errors and strict_mode:
        raise DataValidationError(f"{context}: Validation failed - {'; '.join(errors)}")
    
    # Log summary
    logger.info(f"[OHLCV] Validation completed - {len(warnings)} warnings, {len(errors)} errors")
    
    return df


def check_price_data_quality(
    df: pd.DataFrame,
    price_columns: List[str] = ["open", "high", "low", "close"],
    threshold_zscore: float = 3.0
) -> Dict[str, Any]:
    """
    Check price data quality for outliers and anomalies
    
    Args:
        df: DataFrame with price data
        price_columns: List of price columns to check
        threshold_zscore: Z-score threshold for outlier detection
        
    Returns:
        Dictionary with quality metrics
    """
    
    quality_report = {
        "total_rows": len(df),
        "columns_checked": price_columns,
        "outliers": {},
        "statistics": {},
        "quality_score": 0.0
    }
    
    for col in price_columns:
        if col not in df.columns:
            continue
            
        series = df[col].dropna()
        
        # Basic statistics
        stats = {
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "median": series.median()
        }
        quality_report["statistics"][col] = stats
        
        # Outlier detection using Z-score
        z_scores = np.abs((series - stats["mean"]) / stats["std"])
        outliers = series[z_scores > threshold_zscore]
        
        quality_report["outliers"][col] = {
            "count": len(outliers),
            "percentage": len(outliers) / len(series) * 100,
            "values": outliers.tolist()[:10]  # First 10 outliers
        }
    
    # Calculate overall quality score (0-100)
    total_outliers = sum(info["count"] for info in quality_report["outliers"].values())
    total_values = sum(len(df[col].dropna()) for col in price_columns if col in df.columns)
    
    if total_values > 0:
        outlier_percentage = (total_outliers / total_values) * 100
        quality_report["quality_score"] = max(0, 100 - outlier_percentage * 10)  # Penalize outliers
    
    return quality_report


def validate_trading_days(
    df: pd.DataFrame,
    datetime_column: str = "datetime",
    calendar: Optional[TradingCalendar] = None
) -> pd.DataFrame:
    """
    Validate that data only includes trading days
    
    Args:
        df: DataFrame with datetime column
        datetime_column: Name of datetime column
        calendar: TradingCalendar instance (optional)
        
    Returns:
        DataFrame with only trading days
    """
    
    if datetime_column not in df.columns:
        logger.warning(f"[TRADING_DAYS] Datetime column '{datetime_column}' not found")
        return df
    
    if calendar is None:
        logger.warning("[TRADING_DAYS] No trading calendar provided, skipping validation")
        return df
    
    try:
        df_copy = df.copy()
        df_copy[datetime_column] = pd.to_datetime(df_copy[datetime_column])
        
        # Filter to only trading days
        is_trading_day = df_copy[datetime_column].dt.date.apply(calendar.is_trading_day)
        non_trading_days = (~is_trading_day).sum()
        
        if non_trading_days > 0:
            logger.info(f"[TRADING_DAYS] Filtered out {non_trading_days} non-trading days")
            df_copy = df_copy[is_trading_day]
        
        return df_copy
        
    except Exception as e:
        logger.error(f"[TRADING_DAYS] Error validating trading days: {e}")
        return df