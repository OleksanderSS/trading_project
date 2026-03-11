#!/usr/bin/env python3
"""
Централізований модуль валідації для торгової системи, що використовує Pydantic для валідації на рівні записів
та оптимізовані векторні операції Pandas для валідації великих DataFrame.
"""

import logging
import re
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, validator, root_validator

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("DataValidator")


class DataValidationError(Exception):
    """Спеціальний виняток для помилок валідації даних, особливо для DataFrame."""
    pass

# --- Pydantic моделі для валідації на рівні записів ---

class TradingAction(str, Enum):
    """Допустимі торгові дії"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class Timeframe(str, Enum):
    """Допустимі таймфрейми"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"

class OrderType(str, Enum):
    """Типи ордерів"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class SignalStrength(str, Enum):
    """Сила сигналу"""
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"

class TickerValidator(BaseModel):
    """Валідатор для тікерів"""
    symbol: str = Field(..., min_length=1, max_length=10)

    @validator('symbol')
    def validate_ticker_format(cls, v):
        if not v or not re.match(r'^[A-Za-z0-9.-]+$', v):
            raise ValueError("Тікер повинен містити тільки літери, цифри, . або -")
        return v.upper()

class MarketDataSchema(BaseModel):
    """
    Pydantic схема для валідації ринкових даних (OHLCV).
    Гарантує цілісність кожного окремого запису.
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
    """Валідатор для торгових сигналів"""
    ticker: str
    action: TradingAction
    price: Optional[float] = Field(None, gt=0)
    confidence: float = Field(..., ge=0, le=1)
    timestamp: datetime = Field(default_factory=datetime.now)
    strength: Optional[SignalStrength] = None
    strategy: Optional[str] = None

    @validator('ticker')
    def validate_ticker(cls, v):
        return TickerValidator(symbol=v).symbol


class TradeOrder(BaseModel):
    """Валідатор для торгових ордерів"""
    ticker: str
    action: TradingAction
    order_type: OrderType = OrderType.MARKET
    quantity: int = Field(..., gt=0)
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    @validator('ticker')
    def validate_ticker(cls, v):
        return TickerValidator(symbol=v).symbol

    @validator('price')
    def validate_price(cls, v, values):
        if values.get('order_type') in [OrderType.LIMIT, OrderType.STOP_LIMIT] and v is None:
            raise ValueError("Ціна є обов'язковою для ордерів LIMIT та STOP_LIMIT")
        if v is not None and v <= 0:
            raise ValueError("Ціна повинна бути позитивною")
        return v


class BacktestRequest(BaseModel):
    """Валідатор для запитів бектестингу"""
    tickers: List[str] = Field(..., min_items=1, max_items=100)
    timeframes: List[Timeframe] = Field(..., min_items=1)
    start_date: date
    end_date: date
    initial_capital: float = Field(..., gt=0, le=10**9)
    strategies: List[str] = Field(..., min_items=1)

    @validator('tickers')
    def validate_tickers(cls, v):
        validated = [TickerValidator(symbol=t).symbol for t in v]
        if len(set(validated)) != len(validated):
            raise ValueError("Знайдено дублікати тікерів")
        return validated

    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError("Дата закінчення повинна бути після дати початку")
        return v


class DataValidator:
    """
    Основний клас валідації, що об'єднує Pydantic моделі для валідації записів
    та оптимізовані векторні методи для валідації DataFrame.
    """

    @staticmethod
    def validate_df(df: pd.DataFrame, context: str = "Market Data") -> pd.DataFrame:
        """
        Векторна валідація DataFrame для високої продуктивності.
        Перевіряє логіку OHLCV та базові обмеження.
        """
        if df.empty:
            logger.warning(f"Validation skipped: {context} DataFrame is empty.")
            return df

        initial_len = len(df)
        
        # 1. Перевірка обов'язкових колонок
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataValidationError(f"{context}: Missing required columns: {missing}")

        # 2. Перевірка позитивних значень та логіки свічок (Векторно)
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
            # Повертаємо DataFrame без помилкових рядків
            df = df[~invalid_mask].copy()
            logger.info(f"{context}: Removed {error_count} invalid rows. Remaining: {len(df)}/{initial_len}")
        else:
            logger.info(f"Successfully validated all {len(df)} rows of {context} using vector checks.")
            
        return df

    @staticmethod
    def detect_leakage(df: pd.DataFrame, target_cols: List[str], threshold: float = 0.99) -> List[str]:
        """
        Виявляє ознаки (features), які мають підозріло високу кореляцію з таргетом.
        Запобігає Data Leakage (витоку майбутнього в минуле).
        """
        leaking_features = []
        
        # Виключаємо метадані та самі таргети з перевірки ознак
        exclude = set(target_cols) | {'datetime', 'ticker', 'timestamp'}
        features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        
        for target in target_cols:
            if target not in df.columns:
                continue
                
            correlations = df[features].corrwith(df[target]).abs()
            leaks = correlations[correlations >= threshold]
            
            if not leaks.empty:
                for feat, corr in leaks.items():
                    logger.warning(f"POTENTIAL DATA LEAKAGE: Feature '{feat}' has correlation {corr:.4f} with target '{target}'")
                    leaking_features.append(feat)
                    
        return list(set(leaking_features))

    @staticmethod
    def validate_prediction_input(df: pd.DataFrame, model_features: List[str]) -> bool:
        """
        Перевіряє вхідні дані перед прогнозом: наявність всіх ознак та відсутність NaN.
        """
        missing_features = [f for f in model_features if f not in df.columns]
        if missing_features:
            logger.error(f"Prediction Input Error: Missing features in DataFrame: {missing_features}")
            return False
            
        nan_counts = df[model_features].isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        
        if not cols_with_nan.empty:
            logger.warning(f"Prediction Input Warning: NaN values found in features: {cols_with_nan.to_dict()}")
            # Якщо останній рядок (для якого робимо прогноз) має NaN - це критично
            if df[model_features].iloc[-1].isna().any():
                logger.error("Prediction Input Error: Critical NaN in the latest data row.")
                return False
                
        return True

    # --- Методи для валідації об'єктів через Pydantic ---

    @staticmethod
    def validate_request(model: BaseModel, data: Dict[str, Any]) -> BaseModel:
        """Універсальний метод валідації для будь-якої Pydantic моделі."""
        try:
            return model(**data)
        except ValidationError as e:
            # Переформатування помилок для кращої читабельності
            error_messages = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
            raise DataValidationError(f"Помилка валідації: {'; '.join(error_messages)}")

    # --- Статичні методи для валідації DataFrame ---

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        min_rows: int = 1,
        check_nulls: bool = True,
        check_duplicates: bool = True,
        context: str = "DataFrame"
    ) -> pd.DataFrame:
        """Виконує базову перевірку цілісності DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(f"{context}: Очікувався DataFrame, отримано {type(df)}")

        if df.empty:
            if min_rows > 0:
                raise DataValidationError(f"{context}: DataFrame порожній")
            else:
                return df # Порожній DataFrame є допустимим, якщо min_rows=0

        if len(df) < min_rows:
            raise DataValidationError(f"{context}: DataFrame має {len(df)} рядків, потрібно щонайменше {min_rows}")

        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise DataValidationError(f"{context}: Відсутні обов'язкові колонки: {missing_cols}")

        if check_nulls and df.isnull().values.any():
            null_counts = df.isnull().sum()
            null_cols = null_counts[null_counts > 0].to_dict()
            logger.warning(f"{context}: Знайдено NULL-значення: {null_cols}")

        if check_duplicates and df.duplicated().any():
            dup_count = df.duplicated().sum()
            logger.warning(f"{context}: Знайдено {dup_count} дубльованих рядків")

        return df

    @staticmethod
    def validate_ohlcv_dataframe(
        df: pd.DataFrame,
        strict_mode: bool = True
    ) -> pd.DataFrame:
        """Виконує детальну валідацію DataFrame з OHLCV даними."""
        price_columns = ["open", "high", "low", "close"]
        required_cols = price_columns + ["volume", "timestamp"]
        DataValidator.validate_dataframe(df, required_columns=required_cols, context="OHLCV")

        errors = []
        warnings = []

        # Перевірка на негативні значення
        if (df[price_columns] < 0).any().any():
            errors.append("Знайдено негативні ціни")
        if (df["volume"] < 0).any():
            errors.append("Знайдено негативний об'єм")

        # Перевірка логіки свічок
        if (df["high"] < df[["open", "close"]].max(axis=1)).any():
            errors.append("Знайдено рядки, де High < max(Open, Close)")
        if (df["low"] > df[["open", "close"]].min(axis=1)).any():
            errors.append("Знайдено рядки, де Low > min(Open, Close)")
        if (df["high"] < df["low"]).any():
            errors.append("Знайдено рядки, де High < Low")

        # Перевірка на нульові ціни (зазвичай це попередження)
        if (df[price_columns] == 0).any().any():
            warnings.append("Знайдено нульові ціни, що може свідчити про проблеми з якістю даних")

        for w in warnings:
            logger.warning(f"OHLCV Validation: {w}")

        if errors:
            error_message = f"Помилки валідації OHLCV: {'; '.join(errors)}"
            if strict_mode:
                raise DataValidationError(error_message)
            else:
                logger.warning(error_message)

        return df

    @staticmethod
    def check_data_quality(
        df: pd.DataFrame,
        price_columns: List[str] = ["open", "high", "low", "close"],
        z_score_threshold: float = 3.0
    ) -> Dict[str, Any]:
        """Аналізує DataFrame на предмет викидів та розраховує оцінку якості."""
        report = {"outliers": {}, "statistics": {}, "quality_score": 100.0}
        df_numeric = df[price_columns].apply(pd.to_numeric, errors='coerce').dropna()

        total_values = len(df_numeric)
        if total_values == 0:
            return report

        total_outliers = 0
        for col in price_columns:
            series = df_numeric[col]
            mean, std = series.mean(), series.std()
            report["statistics"][col] = {"mean": mean, "std": std}

            if std > 0:
                z_scores = np.abs((series - mean) / std)
                outliers = z_scores > z_score_threshold
                outlier_count = outliers.sum()
                if outlier_count > 0:
                    report["outliers"][col] = outlier_count
                    total_outliers += outlier_count

        # Оцінка якості: 100% - відсоток викидів, помножений на 10
        outlier_percentage = (total_outliers / (total_values * len(price_columns))) * 100
        report["quality_score"] = max(0, 100 - (outlier_percentage * 10))

        return report