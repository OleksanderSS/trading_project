import functools
import logging
from collections.abc import Callable


def log_and_raise(logger: logging.Logger):
    """
    Decorator that logs any exception raised by the decorated function
    with full stack trace and then re-raises the exception.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.error(f"❌ Exception in {func.__name__}: {e}", exc_info=True)
                raise

        return wrapper

    return decorator
