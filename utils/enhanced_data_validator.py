#!/usr/bin/env python3
"""
Модуль валідації вхідних data для trading системи
Заwithoutпечує перевірку та санітизацію всіх вхідних data
"""

from typing import Any, Dict, List, Optional, Union, Literal
from datetime import datetime, date
from decimal import Decimal
import re
import pandas as pd
import numpy as np
from pydantic import BaseModel, validator, Field, ValidationError
from enum import Enum


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
        """Валідація формату тікера"""
        if not v:
            raise ValueError("Ticker cannot be empty")
        
        # Перевірка на літери та цифри
        if not re.match(r'^[A-Za-z0-9]+$', v):
            raise ValueError("Ticker must contain only letters and numbers")
        
        # Перевірка на довжину
        if len(v) > 10:
            raise ValueError("Ticker too long (max 10 characters)")
        
        return v.upper()


class PriceValidator(BaseModel):
    """Валідатор для цін"""
    price: float = Field(..., gt=0)
    timestamp: Optional[datetime] = None
    
    @validator('price')
    def validate_price_range(cls, v):
        """Валідація діапазону ціни"""
        if v <= 0:
            raise ValueError("Price must be positive")
        
        if v > 1000000:  # 1 мільйон за акцію
            raise ValueError("Price seems too high")
        
        # Перевірка на розумну кількість знаків після коми
        if len(str(v).split('.')[-1]) > 6:
            raise ValueError("Too many decimal places in price")
        
        return round(v, 6)


class VolumeValidator(BaseModel):
    """Валідатор для обсягів"""
    volume: int = Field(..., ge=0)
    
    @validator('volume')
    def validate_volume_range(cls, v):
        """Валідація діапазону обсягу"""
        if v < 0:
            raise ValueError("Volume cannot be negative")
        
        if v > 10**12:  # 1 трильйон акцій
            raise ValueError("Volume seems too high")
        
        return v


class TradingSignal(BaseModel):
    """Валідатор для торгових сигналів"""
    ticker: str
    action: TradingAction
    price: Optional[float] = None
    confidence: float = Field(ge=0, le=1)
    timestamp: datetime
    strength: Optional[SignalStrength] = None
    strategy: Optional[str] = None
    
    @validator('ticker')
    def validate_ticker(cls, v):
        """Валідація тікера"""
        ticker_validator = TickerValidator(symbol=v)
        return ticker_validator.symbol
    
    @validator('price')
    def validate_price(cls, v):
        """Валідація ціни"""
        if v is not None:
            price_validator = PriceValidator(price=v)
            return price_validator.price
        return v
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Валідація впевненості"""
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return round(v, 3)


class TradeOrder(BaseModel):
    """Валідатор для торгових ордерів"""
    ticker: str
    action: TradingAction
    order_type: OrderType = OrderType.MARKET
    quantity: int = Field(..., gt=0)
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime
    
    @validator('ticker')
    def validate_ticker(cls, v):
        """Валідація тікера"""
        ticker_validator = TickerValidator(symbol=v)
        return ticker_validator.symbol
    
    @validator('quantity')
    def validate_quantity(cls, v):
        """Валідація кількості"""
        if v <= 0:
            raise ValueError("Quantity must be positive")
        
        if v > 10**9:  # 1 мільярд акцій
            raise ValueError("Quantity seems too high")
        
        return v
    
    @validator('price')
    def validate_price(cls, v, values):
        """Валідація ціни"""
        if v is not None:
            price_validator = PriceValidator(price=v)
            return price_validator.price
        
        # Для маркет ордерів ціна не обов'язкова
        if values.get('order_type') == OrderType.MARKET:
            return None
        
        raise ValueError("Price is required for non-market orders")
    
    @validator('stop_price')
    def validate_stop_price(cls, v):
        """Валідація стоп-ціни"""
        if v is not None:
            price_validator = PriceValidator(price=v)
            return price_validator.price
        return v


class BacktestRequest(BaseModel):
    """Валідатор для запитів бектестингу"""
    tickers: List[str] = Field(..., min_items=1, max_items=100)
    timeframes: List[Timeframe] = Field(..., min_items=1)
    start_date: date
    end_date: date
    initial_capital: float = Field(..., gt=0)
    strategies: List[str] = Field(..., min_items=1)
    
    @validator('tickers')
    def validate_tickers(cls, v):
        """Валідація списку тікерів"""
        validated_tickers = []
        for ticker in v:
            ticker_validator = TickerValidator(symbol=ticker)
            validated_tickers.append(ticker_validator.symbol)
        
        # Перевірка на дублікати
        if len(set(validated_tickers)) != len(validated_tickers):
            raise ValueError("Duplicate tickers found")
        
        return validated_tickers
    
    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        """Валідація дат"""
        if v > date.today():
            raise ValueError("Date cannot be in the future")
        
        # Перевірка на розумний діапазон
        min_date = date(2000, 1, 1)
        if v < min_date:
            raise ValueError("Date too early (minimum 2000-01-01)")
        
        return v
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        """Валідація діапазону дат"""
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError("End date must be after start date")
        
        # Перевірка на максимальну тривалість
        if 'start_date' in values:
            max_duration = pd.Timedelta(days=365 * 5)  # 5 років
            if pd.to_datetime(v) - pd.to_datetime(values['start_date']) > max_duration:
                raise ValueError("Backtest period too long (maximum 5 years)")
        
        return v
    
    @validator('initial_capital')
    def validate_capital(cls, v):
        """Валідація початкового капіталу"""
        if v <= 0:
            raise ValueError("Initial capital must be positive")
        
        if v > 10**12:  # 1 трильйон
            raise ValueError("Initial capital seems too high")
        
        return v
    
    @validator('strategies')
    def validate_strategies(cls, v):
        """Валідація стратегій"""
        allowed_strategies = {
            'momentum', 'mean_reversion', 'sentiment', 'volatility',
            'rsi_mean_reversion', 'bollinger_bands', 'macd_crossover',
            'volume_price', 'atr_breakout', 'stochastic_oscillator',
            'williams_r', 'cci_strategy', 'multi_timeframe',
            'sector_rotation', 'pairs_trading', 'volatility_mean_reversion'
        }
        
        for strategy in v:
            if strategy not in allowed_strategies:
                raise ValueError(f"Unknown strategy: {strategy}")
        
        # Перевірка на дублікати
        if len(set(v)) != len(v):
            raise ValueError("Duplicate strategies found")
        
        return v


class OHLCVData(BaseModel):
    """Валідатор для OHLCV data"""
    ticker: str
    timestamp: datetime
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: int = Field(..., ge=0)
    
    @validator('ticker')
    def validate_ticker(cls, v):
        """Валідація тікера"""
        ticker_validator = TickerValidator(symbol=v)
        return ticker_validator.symbol
    
    @validator('open', 'high', 'low', 'close')
    def validate_prices(cls, v):
        """Валідація цін"""
        if v <= 0:
            raise ValueError("Price must be positive")
        return round(v, 6)
    
    @validator('high')
    def validate_high_price(cls, v, values):
        """Валідація високої ціни"""
        if 'low' in values and v < values['low']:
            raise ValueError("High price cannot be lower than low price")
        if 'open' in values and v < values['open']:
            raise ValueError("High price cannot be lower than open price")
        if 'close' in values and v < values['close']:
            raise ValueError("High price cannot be lower than close price")
        return v
    
    @validator('low')
    def validate_low_price(cls, v, values):
        """Валідація низької ціни"""
        if 'high' in values and v > values['high']:
            raise ValueError("Low price cannot be higher than high price")
        if 'open' in values and v > values['open']:
            raise ValueError("Low price cannot be higher than open price")
        if 'close' in values and v > values['close']:
            raise ValueError("Low price cannot be higher than close price")
        return v
    
    @validator('volume')
    def validate_volume(cls, v):
        """Валідація обсягу"""
        if v < 0:
            raise ValueError("Volume cannot be negative")
        return v


class PortfolioPosition(BaseModel):
    """Валідатор для позицій портфеля"""
    ticker: str
    quantity: int = Field(..., ge=0)
    entry_price: float = Field(..., gt=0)
    entry_date: datetime
    current_price: Optional[float] = None
    
    @validator('ticker')
    def validate_ticker(cls, v):
        """Валідація тікера"""
        ticker_validator = TickerValidator(symbol=v)
        return ticker_validator.symbol
    
    @validator('quantity')
    def validate_quantity(cls, v):
        """Валідація кількості"""
        if v < 0:
            raise ValueError("Quantity cannot be negative")
        return v
    
    @validator('entry_price', 'current_price')
    def validate_prices(cls, v):
        """Валідація цін"""
        if v is not None and v <= 0:
            raise ValueError("Price must be positive")
        return round(v, 6) if v is not None else v


class RiskMetrics(BaseModel):
    """Валідатор для ризик-метрик"""
    total_return: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    volatility: Optional[float] = None
    win_rate: Optional[float] = Field(None, ge=0, le=1)
    
    @validator('total_return')
    def validate_total_return(cls, v):
        """Валідація загальної дохідності"""
        if abs(v) > 100:  # Більше 10000% зміни
            raise ValueError("Total return seems unrealistic")
        return round(v, 6)
    
    @validator('sharpe_ratio')
    def validate_sharpe_ratio(cls, v):
        """Валідація Sharpe ratio"""
        if v is not None and abs(v) > 10:
            raise ValueError("Sharpe ratio seems unrealistic")
        return round(v, 3) if v is not None else v
    
    @validator('max_drawdown')
    def validate_max_drawdown(cls, v):
        """Валідація максимальної просадки"""
        if v is not None:
            if v > 0:
                raise ValueError("Max drawdown cannot be positive")
            if v < -1:
                raise ValueError("Max drawdown cannot be less than -100%")
        return round(v, 4) if v is not None else v
    
    @validator('volatility')
    def validate_volatility(cls, v):
        """Валідація волатильності"""
        if v is not None and (v <= 0 or v > 5):
            raise ValueError("Volatility should be between 0 and 500%")
        return round(v, 4) if v is not None else v


class DataValidator:
    """Основний клас валідації data"""
    
    @staticmethod
    def validate_trading_signal(data: Dict[str, Any]) -> TradingSignal:
        """Валідація торгового сигналу"""
        try:
            return TradingSignal(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid trading signal: {e}")
    
    @staticmethod
    def validate_trade_order(data: Dict[str, Any]) -> TradeOrder:
        """Валідація торгового ордера"""
        try:
            return TradeOrder(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid trade order: {e}")
    
    @staticmethod
    def validate_backtest_request(data: Dict[str, Any]) -> BacktestRequest:
        """Валідація запиту бектестингу"""
        try:
            return BacktestRequest(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid backtest request: {e}")
    
    @staticmethod
    def validate_ohlcv_data(data: Dict[str, Any]) -> OHLCVData:
        """Валідація OHLCV data"""
        try:
            return OHLCVData(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid OHLCV data: {e}")
    
    @staticmethod
    def validate_portfolio_position(data: Dict[str, Any]) -> PortfolioPosition:
        """Валідація позиції портфеля"""
        try:
            return PortfolioPosition(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid portfolio position: {e}")
    
    @staticmethod
    def validate_risk_metrics(data: Dict[str, Any]) -> RiskMetrics:
        """Валідація ризик-метрик"""
        try:
            return RiskMetrics(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid risk metrics: {e}")
    
    @staticmethod
    def sanitize_dataframe(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """Санітизація DataFrame"""
        # Перевірка наявності колонок
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Видалення дублікатів
        df = df.drop_duplicates()
        
        # Очищення від inf та NaN значень
        for col in required_columns:
            if col in df.columns:
                # Заміна inf на NaN
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                # Видалення рядків з NaN
                df = df.dropna(subset=[col])
        
        # Перевірка на NaN значення після очищення
        if df[required_columns].isnull().any().any():
            raise ValueError("DataFrame contains NaN values in required columns")
        
        # Перевірка типів data
        for col in required_columns:
            if df[col].dtype == object:
                # Спроба конвертувати в числовий тип
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    raise ValueError(f"Column {col} contains non-numeric values")
        
        return df
    
    @staticmethod
    def validate_time_series_data(df: pd.DataFrame, date_column: str = 'timestamp') -> pd.DataFrame:
        """Валідація часових рядів"""
        # Перевірка сортування по даті
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)
            
            # Перевірка на пропуски в часі
            if df[date_column].duplicated().any():
                raise ValueError("Time series contains duplicate timestamps")
        
        return df


# Декоратор для валідації функцій
def validate_input(validator_class: type, field_name: str = 'data'):
    """Декоратор для валідації вхідних data функцій"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Знаходимо аргумент для валідації
            if field_name in kwargs:
                data = kwargs[field_name]
            elif len(args) > 0:
                data = args[0]
            else:
                raise ValueError(f"Field '{field_name}' not found")
            
            # Валідація
            if isinstance(data, dict):
                validator_instance = validator_class()
                if hasattr(validator_instance, f'validate_{field_name}'):
                    validated_data = getattr(validator_instance, f'validate_{field_name}')(data)
                    kwargs[field_name] = validated_data
                else:
                    raise ValueError(f"Validator for '{field_name}' not found")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Приклади використання
if __name__ == "__main__":
    # Тест валідації
    validator = DataValidator()
    
    # Валідація сигналу
    signal_data = {
        'ticker': 'AAPL',
        'action': 'BUY',
        'confidence': 0.85,
        'timestamp': datetime.now(),
        'strength': 'STRONG'
    }
    
    try:
        signal = validator.validate_trading_signal(signal_data)
        print(f"Valid signal: {signal}")
    except ValueError as e:
        print(f"Invalid signal: {e}")
    
    # Валідація запиту бектестингу
    backtest_data = {
        'tickers': ['AAPL', 'GOOGL', 'MSFT'],
        'timeframes': ['1d', '1h'],
        'start_date': date(2023, 1, 1),
        'end_date': date(2024, 12, 31),
        'initial_capital': 100000,
        'strategies': ['momentum', 'mean_reversion']
    }
    
    try:
        backtest = validator.validate_backtest_request(backtest_data)
        print(f"Valid backtest request: {backtest}")
    except ValueError as e:
        print(f"Invalid backtest request: {e}")
