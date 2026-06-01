import logging
import time
import traceback
from functools import wraps
from typing import Dict, Any, Type, Callable, Optional, List
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json
from src.core.logging.logger import ProjectLogger
from src.core.logging.notifier import UniversalNotifier
from src.config.unified_config_manager import UnifiedConfigManager


class TradingSystemError(Exception):
    """Base exception class for trading system errors."""
    pass


class PipelineError(TradingSystemError):
    """Base exception for pipeline-specific failures."""
    pass


class StageError(PipelineError):
    """Base exception for individual pipeline stage failures."""
    pass


class StageExecutionError(StageError):
    """Exception raised when a stage fails during execution."""
    pass


class ModelLoadingError(StageError):
    """Exception raised when loading a model fails."""
    pass


class ConfigurationError(TradingSystemError):
    """Exception raised when a system or training configuration is invalid."""
    pass


class IErrorHandler(ABC):
    """Interface for error handling implementations."""

    @abstractmethod
    def handle_error(self, error: Exception, context: str='General',
        severity: str='error', should_raise: bool=False) ->Dict[str, Any]:
        """Handles an error with specified context and severity."""
        pass


logger = ProjectLogger.get_logger('ErrorHandler')


def log_and_raise(exception_class: Type[Exception], message: str, **kwargs):
    """
    Logs an error message and then raises the specified exception.

    Args:
        exception_class: The class of the exception to raise.
        message: The error message to log and include in the exception.
        **kwargs: Additional arguments to pass to the exception's constructor.
    """
    logger.error(message, extra=kwargs)
    raise exception_class(message, **kwargs)


def safe_execute(func: Callable, *args, **kwargs):
    """
    Safely executes a function, wrapping unexpected errors in a TradingSystemError.

    Args:
        func: The function to execute.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        The result of the function call.

    Raises:
        TradingSystemError: If the function raises an unexpected exception.
    """
    try:
        return func(*args, **kwargs)
    except TradingSystemError:
        raise
    except Exception as e:
        logger.error(f'Unexpected error in {func.__name__}: {e}', exc_info=True
            )
        raise TradingSystemError(
            f'An unexpected error occurred in {func.__name__}: {e}') from e


class ErrorHandler(IErrorHandler):
    """Provides centralized error handling, including retries and error tracking."""

    def __init__(self, config_manager: Optional[UnifiedConfigManager]=None,
        logger_name: str='ErrorHandler'):
        """Initializes the ErrorHandler with a logger name and notifier."""
        from src.config.unified_config_manager import get_current_config
        self.logger = ProjectLogger.get_logger(logger_name)
        self.config_manager = config_manager or get_current_config()
        self.notifier = UniversalNotifier(self.config_manager
            ) if self.config_manager else None
        self.error_counts: Dict[str, int] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.notification_cooldowns: Dict[str, datetime] = {}
        self.cooldown_minutes = 10

    def _format_error_message(self, error_info: Dict[str, Any]) ->str:
        """Formats error dictionary into a string for notification."""
        context = error_info.get('context', 'N/A')
        if isinstance(context, dict):
            context_str = json.dumps(context)
        else:
            context_str = str(context)
        message = f"""<b>🚨 {error_info.get('severity', 'ERROR').upper()} 🚨</b>
<b>Time:</b> {error_info.get('timestamp')}
<b>Type:</b> {error_info.get('error_type')}
<b>Context:</b> <code>{context_str}</code>
<b>Message:</b> <pre>{error_info.get('error_message')}</pre>
"""
        return message

    def handle_error(self, error: Exception, context: Any='General',
        severity: str='error', should_raise: bool=False) ->Dict[str, Any]:
        """
        Logs and counts an error, with an option to re-raise it. Sends notification for critical/error severity.
        """
        error_type = type(error).__name__
        error_message = str(error)
        error_info = {'timestamp': datetime.now().isoformat(), 'error_type':
            error_type, 'error_message': error_message, 'context': context,
            'severity': severity, 'traceback': traceback.format_exc()}
        log_message = f'Error in {context}: {error_message}'
        if severity == 'critical':
            self.logger.critical(
                f'Critical failure in {context}: {error_message}', exc_info
                =True)
        elif severity == 'error':
            self.logger.error(log_message, exc_info=True)
        elif severity == 'warning':
            self.logger.warning(log_message)
        error_key = f'{str(context)}:{error_type}'
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.error_history.append(error_info)
        if severity in ['critical', 'error'] and self.notifier:
            now = datetime.now()
            last_notified = self.notification_cooldowns.get(error_key)
            if not last_notified or now - last_notified > timedelta(minutes
                =self.cooldown_minutes):
                notification_message = self._format_error_message(error_info)
                self.notifier.sync_send(notification_message, level=
                    severity.upper())
                self.notification_cooldowns[error_key] = now
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]
        if should_raise:
            raise error
        return error_info

    def get_error_summary(self) ->Dict[str, Any]:
        """Returns a summary of all error counts."""
        total_errors = sum(self.error_counts.values())
        most_common = sorted(self.error_counts.items(), key=lambda x: x[1],
            reverse=True)[:5]
        return {'total_errors': total_errors, 'unique_error_types': len(
            self.error_counts), 'most_common_errors': most_common,
            'recent_errors': self.error_history[-10:],
            'error_rate_by_context': self._calculate_error_rates()}

    def _calculate_error_rates(self) ->Dict[str, float]:
        """Calculates error rates by context."""
        context_errors = {}
        for error_key, count in self.error_counts.items():
            context = error_key.split(':')[0]
            context_errors[context] = context_errors.get(context, 0) + count
        return context_errors

    def reset_error_counts(self) ->None:
        """Resets error counters."""
        self.error_counts.clear()
        self.error_history.clear()
        self.notification_cooldowns.clear()
        self.logger.info('Error counters reset')

    def retry(self, max_retries: int=3, delay: float=1.0, backoff: float=
        2.0, exceptions: tuple=(Exception,)):
        """
        A decorator for retrying a function with exponential backoff.
        """

        def decorator(func: Callable):

            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_retries:
                            current_delay = delay * backoff ** attempt
                            self.logger.warning(
                                f"Attempt {attempt + 1}/{max_retries + 1} for '{func.__name__}' failed: {e}. Retrying in {current_delay:.2f} seconds..."
                                )
                            time.sleep(current_delay)
                        else:
                            self.handle_error(e,
                                f'{func.__name__} (final attempt)', 'error')
                raise TradingSystemError(
                    f"Function '{func.__name__}' failed after {max_retries + 1} attempts."
                    ) from last_exception
            return wrapper
        return decorator

    def graceful_degradation(self, fallback_value: Any=None, context: str=''):
        """
        Decorator for graceful degradation.
        """

        def decorator(func: Callable) ->Callable:

            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                    self.handle_error(e, f'{func.__name__} in {context}',
                        'warning')
                    self.logger.info(
                        f'Using fallback value for {func.__name__}')
                    return fallback_value
            return wrapper
        return decorator

    def log_performance(self, func: Callable) ->Callable:
        """
        Decorator to log function performance.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = datetime.now() - start_time
                self.logger.info(f'{func.__name__} completed in {duration}')
                return result
            except Exception as e:
                duration = datetime.now() - start_time
                self.logger.error(
                    f'{func.__name__} failed after {duration}: {str(e)}')
                raise
        return wrapper


_error_handler = None


def get_error_handler(config_manager: Optional[UnifiedConfigManager]=None
    ) ->ErrorHandler:
    """Gets the global error handler."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler(config_manager=config_manager)
    return _error_handler


def handle_error(error: Exception, context: str='', severity: str='error'
    ) ->Dict[str, Any]:
    """Handles an error globally."""
    handler = get_error_handler()
    return handler.handle_error(error, context, severity)


def retry(max_retries: int=3, delay: float=1.0, backoff: float=2.0,
    exceptions: tuple=(Exception,)):
    """Retry decorator using the global handler."""
    handler = get_error_handler()
    return handler.retry(max_retries, delay, backoff, exceptions)


def graceful_degradation(fallback_value: Any=None, context: str=''):
    """Graceful degradation decorator using the global handler."""
    handler = get_error_handler()
    return handler.graceful_degradation(fallback_value, context)


def log_performance(func: Callable) ->Callable:
    """Performance logging decorator using the global handler."""
    handler = get_error_handler()
    return handler.log_performance(func)


def log_error_summary() ->None:
    """Logs the global error summary."""
    handler = get_error_handler()
    summary = handler.get_error_summary()
    logger.info('ERROR SUMMARY')
    logger.info(f" Total errors: {summary['total_errors']}")
    logger.info(f" Unique error types: {summary['unique_error_types']}")
    if summary['most_common_errors']:
        logger.info(' Most common errors:')
        for error_type, count in summary['most_common_errors']:
            logger.info(f'   {error_type}: {count}')


class ErrorContext:
    """Context manager for error handling."""

    def __init__(self, operation: str, error_type: type=TradingSystemError,
        log_level: str='ERROR'):
        """Initializes the ErrorContext."""
        self.operation = operation
        self.error_type = error_type
        self.log_level = log_level
        self.start_time = None

    def __enter__(self):
        """Enters the error context."""
        self.start_time = datetime.now()
        logger.info(f'Starting operation: {self.operation}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exits the error context, handling any exceptions."""
        if exc_type is None:
            duration = datetime.now() - self.start_time
            logger.info(
                f"Operation '{self.operation}' completed successfully in {duration}"
                )
            return True
        if issubclass(exc_type, self.error_type):
            error_msg = f"Operation '{self.operation}' failed: {str(exc_val)}"
            getattr(logger, self.log_level.lower())(error_msg)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f'Traceback: {traceback.format_exception(exc_type, exc_val, exc_tb)}'
                    )
            return True
        logger.error(
            f"Unexpected error in operation '{self.operation}': {str(exc_val)}"
            )
        return False


def validate_input(data: Any, validator: Callable[[Any], bool], error_msg:
    str='Invalid input'):
    """
    Validate input data and raise ValidationError if invalid.

    Raises:
        ValidationError: If validation fails.
    """
    from src.core.validation.validators import DataValidationError as ValidationError
    if not validator(data):
        raise ValidationError(error_msg)
