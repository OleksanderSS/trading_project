# utils/error_handler.py

"""
Enhanced error handling for trading project with retry mechanisms and graceful degradation
"""

import logging
import traceback
import time
from typing import Optional, Any, Callable, Dict, List
from functools import wraps
from datetime import datetime
from config.pipeline_config import LOGGING_CONFIG

logger = logging.getLogger(__name__)


class TradingError(Exception):
    """Base exception for trading project"""
    pass


class DataValidationError(TradingError):
    """Error in data validation"""
    pass


class ModelTrainingError(TradingError):
    """Error in model training"""
    pass


class ConfigurationError(TradingError):
    """Error in configuration"""
    pass


class PerformanceError(TradingError):
    """Error related to performance issues"""
    pass


class ErrorHandler:
    """
    Enhanced error handler with retry mechanisms and graceful degradation
    """
    
    def __init__(self):
        self.error_counts = {}
        self.error_history = []
        self.logger = logging.getLogger(__name__)
    
    def handle_error(self, error: Exception, context: str = "", severity: str = "error") -> Dict[str, Any]:
        """
        Обробка помилки with whereandльним loggingм
        
        Args:
            error: Exception object
            context: Context where error occurred
            severity: Error severity (error, warning, critical)
        
        Returns:
            Dict with error information
        """
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'severity': severity,
            'traceback': traceback.format_exc() if LOGGING_CONFIG.get('enable_file_logging', True) else None
        }
        
        # Логування помилки
        if severity == "critical":
            self.logger.critical(f" CRITICAL ERROR in {context}: {error}")
        elif severity == "error":
            self.logger.error(f"[ERROR] ERROR in {context}: {error}")
        elif severity == "warning":
            self.logger.warning(f"[WARN] WARNING in {context}: {error}")
        
        # Збandр сandтистики
        error_key = f"{context}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.error_history.append(error_info)
        
        # Обмеження andсторandї errors
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]
        
        return error_info
    
    def retry_with_backoff(self, max_retries: int = 3, backoff_factor: float = 1.0, exceptions: tuple = (Exception,)):
        """
        Декоратор for retry with експоnotнцandйним backoff
        
        Args:
            max_retries: Максимальна кandлькandсть спроб
            backoff_factor: Фактор backoff
            exceptions: Типи виняткandв for retry
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if attempt == max_retries:
                            self.handle_error(e, f"{func.__name__} (final attempt)", "error")
                            raise
                        
                        wait_time = backoff_factor * (2 ** attempt)
                        self.handle_error(e, f"{func.__name__} (attempt {attempt + 1}/{max_retries + 1})", "warning")
                        self.logger.info(f" Retrying {func.__name__} in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                
                raise last_exception
            
            return wrapper
        return decorator
    
    def graceful_degradation(self, fallback_value: Any = None, context: str = ""):
        """
        Декоратор for graceful degradation
        
        Args:
            fallback_value: Значення for forмовчуванням при помилцand
            context: Контекст виконання
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.handle_error(e, f"{func.__name__} in {context}", "warning")
                    self.logger.info(f"[REFRESH] Using fallback value for {func.__name__}")
                    return fallback_value
            
            return wrapper
        return decorator
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Отримати withвandт про помилки"""
        total_errors = sum(self.error_counts.values())
        most_common = sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_errors': total_errors,
            'unique_error_types': len(self.error_counts),
            'most_common_errors': most_common,
            'recent_errors': self.error_history[-10:],
            'error_rate_by_context': self._calculate_error_rates()
        }
    
    def _calculate_error_rates(self) -> Dict[str, float]:
        """Роwithрахувати сandвки errors по контексandх"""
        context_errors = {}
        for error_key, count in self.error_counts.items():
            context = error_key.split(':')[0]
            context_errors[context] = context_errors.get(context, 0) + count
        
        return context_errors
    
    def reset_error_counts(self) -> None:
        """Скинути лandчильники errors"""
        self.error_counts.clear()
        self.error_history.clear()
        self.logger.info("[REFRESH] Error counters reset")


# Глобальний обробник errors
_error_handler = None

def get_error_handler() -> ErrorHandler:
    """Отримати глобальний обробник errors"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler

def handle_error(error: Exception, context: str = "", severity: str = "error") -> Dict[str, Any]:
    """Обробити помилку"""
    handler = get_error_handler()
    return handler.handle_error(error, context, severity)

def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 1.0, exceptions: tuple = (Exception,)):
    """Декоратор for retry"""
    handler = get_error_handler()
    return handler.retry_with_backoff(max_retries, backoff_factor, exceptions)

def graceful_degradation(fallback_value: Any = None, context: str = ""):
    """Декоратор for graceful degradation"""
    handler = get_error_handler()
    return handler.graceful_degradation(fallback_value, context)

def log_error_summary() -> None:
    """Залогувати withвandт про помилки"""
    handler = get_error_handler()
    summary = handler.get_error_summary()
    
    logger.info("[DATA] ERROR SUMMARY")
    logger.info(f" Total errors: {summary['total_errors']}")
    logger.info(f"[SEARCH] Unique error types: {summary['unique_error_types']}")
    
    if summary['most_common_errors']:
        logger.info("[UP] Most common errors:")
        for error_type, count in summary['most_common_errors']:
            logger.info(f"   {error_type}: {count}")


class DataError(TradingError):
    """Data-related errors"""
    pass


class ModelError(TradingError):
    """Model-related errors"""
    pass


class CollectorError(TradingError):
    """Data collection errors"""
    pass


class ValidationError(TradingError):
    """Data validation errors"""
    pass


def handle_errors(
    error_type: type = TradingError,
    default_return: Any = None,
    log_level: str = "ERROR",
    reraise: bool = False
):
    """Decorator for standardized error handling
    
    Args:
        error_type: Exception type to catch
        default_return: Value to return on error
        log_level: Logging level for errors
        reraise: Whether to re-raise exception
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_type as e:
                # Log the error with context
                error_msg = f"Error in {func.__name__}: {str(e)}"
                getattr(logger, log_level.lower())(error_msg)
                
                # Log full traceback for debugging
                logger.debug(f"Traceback: {traceback.format_exc()}")
                
                if reraise:
                    raise
                return default_return
            except Exception as e:
                # Handle unexpected errors
                error_msg = f"Unexpected error in {func.__name__}: {str(e)}"
                logger.error(error_msg)
                logger.debug(f"Traceback: {traceback.format_exc()}")
                
                if reraise:
                    raise
                return default_return
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    default_return: Any = None,
    error_type: type = TradingError
) -> Any:
    """Safely execute a function with error handling
    
    Args:
        func: Function to execute
        args: Function arguments
        kwargs: Function keyword arguments
        default_return: Value to return on error
        error_type: Expected error type
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **(kwargs or {}))
    except error_type as e:
        logger.error(f"Error executing {func.__name__}: {str(e)}")
        return default_return
    except Exception as e:
        logger.error(f"Unexpected error executing {func.__name__}: {str(e)}")
        return default_return


class ErrorContext:
    """Context manager for error handling"""
    
    def __init__(
        self,
        operation: str,
        error_type: type = TradingError,
        default_return: Any = None,
        log_level: str = "ERROR"
    ):
        self.operation = operation
        self.error_type = error_type
        self.default_return = default_return
        self.log_level = log_level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            duration = datetime.now() - self.start_time
            logger.info(f"Operation '{self.operation}' completed successfully in {duration}")
            return True
        
        if issubclass(exc_type, self.error_type):
            error_msg = f"Operation '{self.operation}' failed: {str(exc_val)}"
            getattr(logger, self.log_level.lower())(error_msg)
            logger.debug(f"Traceback: {traceback.format_exception(exc_type, exc_val, exc_tb)}")
            return True  # Suppress exception
        
        # For unexpected errors, log but don't suppress
        logger.error(f"Unexpected error in operation '{self.operation}': {str(exc_val)}")
        return False


def validate_input(data: Any, validator: Callable[[Any], bool], error_msg: str = "Invalid input"):
    """Validate input data and raise ValidationError if invalid
    
    Args:
        data: Data to validate
        validator: Validation function that returns bool
        error_msg: Error message for validation failure
        
    Raises:
        ValidationError: If validation fails
    """
    if not validator(data):
        raise ValidationError(error_msg)


def log_performance(func: Callable) -> Callable:
    """Decorator to log function performance
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with performance logging
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            duration = datetime.now() - start_time
            logger.info(f"{func.__name__} completed in {duration}")
            return result
        except Exception as e:
            duration = datetime.now() - start_time
            logger.error(f"{func.__name__} failed after {duration}: {str(e)}")
            raise
    
    return wrapper
