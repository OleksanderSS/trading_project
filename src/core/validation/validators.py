#!/usr/bin/env python3
"""
Centralized validation module for the trading system, utilizing Pydantic for record-level validation
and optimized vector operations in Pandas for large DataFrame validation.
"""

import re
from datetime import date, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, root_validator, validator

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("DataValidator")


class DataValidationError(Exception):
    """Specialized exception for data validation errors, especially for DataFrames."""
    pass

# --- Pydantic models for record-level validation ---

class TradingAction(str, Enum):
    """Allowed trading actions"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class Timeframe(str, Enum):
    """Allowed timeframes"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"

class OrderType(str, Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class SignalStrength(str, Enum):
    """Signal strength levels"""
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"

class TickerValidator(BaseModel):
    """Validator for ticker symbols"""
    symbol: str = Field(..., min_length=1, max_length=10)

    @validator('symbol')
    def validate_ticker_format(cls, v):
        if not v or not re.match(r'^[A-Za-z0-9.-]+$', v):
            raise ValueError("Ticker must contain only letters, numbers, '.' or '-'")
        return v.upper()

class MarketDataSchema(BaseModel):
    """
    Pydantic schema for market data validation (OHLCV).
    Ensures integrity for every individual record.
    """
    timestamp: datetime
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)

    @root_validator
    def validate_ohlc_logic(cls, values):
        high, low = values.get('high'), values.get('low')
        open_val, close = values.get('open'), values.get('close')

        if high is not None and low is not None and high < low:
            raise ValueError(f"High ({high}) cannot be lower than Low ({low})")

        if high is not None:
            if open_val is not None and high < open_val:
                raise ValueError(f"High ({high}) cannot be lower than Open ({open_val})")
            if close is not None and high < close:
                raise ValueError(f"High ({high}) cannot be lower than Close ({close})")

        if low is not None:
            if open_val is not None and low > open_val:
                raise ValueError(f"Low ({low}) cannot be higher than Open ({open_val})")
            if close is not None and low > close:
                raise ValueError(f"Low ({low}) cannot be higher than Close ({close})")

        return values

class TradingSignal(BaseModel):
    """Validator for trading signals"""
    ticker: str
    action: TradingAction
    price: float | None = Field(None, gt=0)
    confidence: float = Field(..., ge=0, le=1)
    timestamp: datetime = Field(default_factory=datetime.now)
    strength: SignalStrength | None = None
    strategy: str | None = None

    @validator('ticker')
    def validate_ticker(cls, v):
        return TickerValidator(symbol=v).symbol


class TradeOrder(BaseModel):
    """Validator for trading orders"""
    ticker: str
    action: TradingAction
    order_type: OrderType = OrderType.MARKET
    quantity: int = Field(..., gt=0)
    price: float | None = None
    stop_price: float | None = None
    timestamp: datetime = Field(default_factory=datetime.now)

    @validator('ticker')
    def validate_ticker(cls, v):
        return TickerValidator(symbol=v).symbol

    @validator('price')
    def validate_price(cls, v, values):
        if values.get('order_type') in [OrderType.LIMIT, OrderType.STOP_LIMIT] and v is None:
            raise ValueError("Price is required for LIMIT and STOP_LIMIT orders")
        if v is not None and v <= 0:
            raise ValueError("Price must be positive")
        return v


class BacktestRequest(BaseModel):
    """Validator for backtesting requests"""

    tickers: list[str] = Field(..., min_items=1, max_items=100)
    timeframes: list[Timeframe] = Field(..., min_items=1)
    start_date: date
    end_date: date
    initial_capital: float = Field(..., gt=0, le=10**9)
    strategies: list[str] = Field(..., min_items=1)

    @validator('tickers')
    def validate_tickers(cls, v):
        validated = [TickerValidator(symbol=t).symbol for t in v]
        if len(set(validated)) != len(validated):
            raise ValueError("Duplicate tickers found")
        return validated

    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError("End date must be after start date")
        return v


class DataValidator:
    """
    Main validator class, combining Pydantic models for individual record validation
    and optimized vector methods for DataFrame validation.
    """

    @staticmethod
    def validate_df(df: pd.DataFrame, context: str = "Market Data") -> pd.DataFrame:
        """
        Vectorized DataFrame validation for high performance.
        Checks OHLCV logic and basic constraints.
        """
        if df.empty:
            logger.warning(f"Validation skipped: {context} DataFrame is empty.")
            return df

        initial_len = len(df)

        # 1. Required columns check
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataValidationError(f"{context}: Missing required columns: {missing}")

        # 2. Positive values and candle logic check (Vectorized)
        invalid_mask = (
            (df['open'] <= 0) | (df['high'] <= 0) | (df['low'] <= 0) | (df['close'] <= 0) |
            (df['volume'] < 0) |
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )

        error_count = invalid_mask.sum()
        if error_count > 0:
            logger.error(f"{context}: Found {error_count} rows with invalid OHLCV logic.")
            # Return DataFrame without invalid rows
            df = df[~invalid_mask].copy()
            logger.info(f"{context}: Removed {error_count} invalid rows. Remaining: {len(df)}/{initial_len}")
        else:
            logger.info(f"Successfully validated all {len(df)} rows of {context} using vector checks.")

        return df

    @staticmethod
    def detect_leakage(
        df: pd.DataFrame, target_cols: list[str], threshold: float = 0.99
    ) -> list[str]:
        """
        Detects features that have suspiciously high correlation with the target.
        Prevents Data Leakage (future-to-past leakage).
        """
        leaking_features = []

        # Exclude metadata and the targets themselves from feature checking
        exclude = set(target_cols) | {'datetime', 'ticker', 'timestamp'}
        features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

        for target in target_cols:
            if target not in df.columns:
                continue

            leaks = df[features].corrwith(df[target]).abs()
            leaks = leaks[leaks > threshold]
            if not leaks.empty:
                for feat, corr in leaks.items():
                    logger.warning(
                        f"POTENTIAL DATA LEAKAGE: Feature '{feat}' has correlation {corr:.4f} with target '{target}'"
                    )
                    leaking_features.append(feat)

        return list(set(leaking_features))

    @staticmethod
    def validate_prediction_input(df: pd.DataFrame, model_features: list[str]) -> bool:
        """
        Validates input data before prediction: existence of all features and absence of NaNs.
        """
        missing_features = [f for f in model_features if f not in df.columns]
        if missing_features:
            logger.error(f"Prediction Input Error: Missing features in DataFrame: {missing_features}")
            return False

        nan_counts = df[model_features].isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0]

        if not cols_with_nan.empty:
            logger.warning(
                f"Prediction Input Warning: NaN values found in features: {cols_with_nan.to_dict()}"
            )
            # If the last row (the one we predict for) has NaNs - it's critical
            if df[model_features].iloc[-1].isna().any():
                logger.error("Prediction Input Error: Critical NaN in the latest data row.")
                return False

        return True

    # --- Methods for object validation via Pydantic ---

    @staticmethod
    def validate_request(model: BaseModel, data: dict[str, Any]) -> BaseModel:
        """Universal validation method for any Pydantic model."""
        try:
            return model(**data)
        except ValidationError as e:
            # Reformat errors for better readability
            error_messages = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
            raise DataValidationError(f"Validation error: {'; '.join(error_messages)}") from e

    # --- Static methods for DataFrame validation ---

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        required_columns: list[str] | None = None,
        min_rows: int = 1,
        check_nulls: bool = True,
        check_duplicates: bool = True,
        context: str = "DataFrame",
    ) -> pd.DataFrame:
        """Performs basic DataFrame integrity checks."""
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(f"{context}: Expected DataFrame, got {type(df)}")

        if df.empty:
            if min_rows > 0:
                raise DataValidationError(f"{context}: DataFrame is empty")
            else:
                return df # Empty DataFrame is allowed if min_rows=0

        if len(df) < min_rows:
            raise DataValidationError(f"{context}: DataFrame has {len(df)} rows, expected at least {min_rows}")

        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise DataValidationError(f"{context}: Missing required columns: {missing_cols}")

        if check_nulls and df.isnull().values.any():
            null_counts = df.isnull().sum()
            null_cols = null_counts[null_counts > 0].to_dict()
            logger.warning(f"{context}: Found NULL values: {null_cols}")

        if check_duplicates and df.duplicated().any():
            dup_count = df.duplicated().sum()
            logger.warning(f"{context}: Found {dup_count} duplicated rows")

        return df

    @staticmethod
    def validate_ohlcv_dataframe(
        df: pd.DataFrame,
        strict_mode: bool = True
    ) -> pd.DataFrame:
        """Performs detailed validation on OHLCV data DataFrames."""
        price_columns = ["open", "high", "low", "close"]
        required_cols = price_columns + ["volume", "timestamp"]
        DataValidator.validate_dataframe(df, required_columns=required_cols, context="OHLCV")

        errors = []
        warnings = []

        # Negative values check
        if (df[price_columns] < 0).any().any():
            errors.append("Negative prices found")
        if (df["volume"] < 0).any():
            errors.append("Negative volume found")

        # Candle logic check
        if (df["high"] < df[["open", "close"]].max(axis=1)).any():
            errors.append("Rows found where High < max(Open, Close)")
        if (df["low"] > df[["open", "close"]].min(axis=1)).any():
            errors.append("Rows found where Low > min(Open, Close)")
        if (df["high"] < df["low"]).any():
            errors.append("Rows found where High < Low")

        # Zero prices check (usually a warning)
        if (df[price_columns] == 0).any().any():
            warnings.append("Zero prices found, which may indicate data quality issues")

        for w in warnings:
            logger.warning(f"OHLCV Validation: {w}")

        if errors:
            error_message = f"OHLCV Validation errors: {'; '.join(errors)}"
            if strict_mode:
                raise DataValidationError(error_message)
            else:
                logger.warning(error_message)

        return df

    @staticmethod
    def check_data_quality(
        df: pd.DataFrame,
        price_columns: list[str] | None = None,
        z_score_threshold: float = 3.0,
    ) -> dict[str, Any]:
        """Analyzes a DataFrame for outliers and calculates a quality score."""
        if price_columns is None:
            price_columns = ["open", "high", "low", "close"]

        report: dict[str, Any] = {
            "outliers": {},
            "statistics": {},
            "quality_score": 100.0,
        }
        df_numeric = df[price_columns].apply(pd.to_numeric, errors="coerce").dropna()

        total_values = len(df_numeric)
        if total_values == 0:
            return report

        total_outliers = 0
        for col in price_columns:
            series = df_numeric[col]
            mean, std = float(series.mean()), float(series.std())

            stats: dict[str, float] = {"mean": mean, "std": std}
            report["statistics"][col] = stats

            if std > 0:
                z_scores = np.abs((series - mean) / std)
                outliers = z_scores > z_score_threshold
                outlier_count = int(outliers.sum())
                if outlier_count > 0:
                    report["outliers"][col] = outlier_count
                    total_outliers += outlier_count

        # Quality score: 100% - percentage of outliers multiplied by 10
        outlier_percentage = (total_outliers / (total_values * len(price_columns))) * 100
        report["quality_score"] = max(0, 100 - (outlier_percentage * 10))

        return report
