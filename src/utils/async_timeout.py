# Async timeout wrapper for collectors
import asyncio
import inspect
from functools import wraps

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


def async_timeout(timeout_seconds: int = 300):
    """Decorator to add timeout to async functions."""

    def decorator(func):
        if not inspect.iscoroutinefunction(func):
            raise TypeError(f"Function {func.__name__} must be a coroutine function.")

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds") from None

        return async_wrapper

    return decorator


# Usage:
@async_timeout(timeout_seconds=120)
async def collect_data():
    # Your collection logic here
    pass
