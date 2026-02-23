#!/usr/bin/env python3
"""
🚨 CUSTOM EXCEPTIONS - Project-Specific Exception Classes
Визначає custom exceptions для кращої обробки помилок
"""

import logging

logger = logging.getLogger(__name__)

class TradingSystemError(Exception):
    """Base exception for trading system"""
    pass

class DataCollectionError(TradingSystemError):
    """Error in data collection"""
    def __init__(self, message: str, collector: str = None, ticker: str = None):
        super().__init__(message)
        self.collector = collector
        self.ticker = ticker
        logger.error(f"DataCollectionError: {message} (collector={collector}, ticker={ticker})")

class ModelTrainingError(TradingSystemError):
    """Error in model training"""
    def __init__(self, message: str, model: str = None, ticker: str = None):
        super().__init__(message)
        self.model = model
        self.ticker = ticker
        logger.error(f"ModelTrainingError: {message} (model={model}, ticker={ticker})")

class FeatureEngineeringError(TradingSystemError):
    """Error in feature engineering"""
    def __init__(self, message: str, feature: str = None, ticker: str = None):
        super().__init__(message)
        self.feature = feature
        self.ticker = ticker
        logger.error(f"FeatureEngineeringError: {message} (feature={feature}, ticker={ticker})")

class SignalGenerationError(TradingSystemError):
    """Error in signal generation"""
    def __init__(self, message: str, signal_type: str = None, ticker: str = None):
        super().__init__(message)
        self.signal_type = signal_type
        self.ticker = ticker
        logger.error(f"SignalGenerationError: {message} (signal_type={signal_type}, ticker={ticker})")

class RiskManagementError(TradingSystemError):
    """Error in risk management"""
    def __init__(self, message: str, position: str = None, risk_level: str = None):
        super().__init__(message)
        self.position = position
        self.risk_level = risk_level
        logger.error(f"RiskManagementError: {message} (position={position}, risk_level={risk_level})")

class ConfigurationError(TradingSystemError):
    """Error in configuration"""
    def __init__(self, message: str, config_key: str = None):
        super().__init__(message)
        self.config_key = config_key
        logger.error(f"ConfigurationError: {message} (config_key={config_key})")

class SecurityError(TradingSystemError):
    """Error in security operations"""
    def __init__(self, message: str, operation: str = None):
        super().__init__(message)
        self.operation = operation
        logger.error(f"SecurityError: {message} (operation={operation})")

class ValidationError(TradingSystemError):
    """Error in data validation"""
    def __init__(self, message: str, field: str = None, value=None):
        super().__init__(message)
        self.field = field
        self.value = value
        logger.error(f"ValidationError: {message} (field={field}, value={value})")

class ExternalAPIError(TradingSystemError):
    """Error in external API calls"""
    def __init__(self, message: str, api: str = None, status_code: int = None):
        super().__init__(message)
        self.api = api
        self.status_code = status_code
        logger.error(f"ExternalAPIError: {message} (api={api}, status_code={status_code})")

# Utility functions for error handling
def safe_execute(func, *args, **kwargs):
    """
    Безпечне виконання функції з обробкою помилок
    """
    try:
        return func(*args, **kwargs)
    except TradingSystemError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in {func.__name__}: {e}")
        raise TradingSystemError(f"Unexpected error in {func.__name__}: {e}")

def log_and_raise(exception_class, message: str, **kwargs):
    """
    Логує та піднімає exception
    """
    logger.error(message)
    raise exception_class(message, **kwargs)

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    Декоратор для retry логіки
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                    import time
                    time.sleep(delay)
            return None
        return wrapper
    return decorator
