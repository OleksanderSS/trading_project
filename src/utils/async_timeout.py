
# Async timeout wrapper for collectors
import asyncio
from functools import wraps
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


def async_timeout(timeout_seconds: int = 300):
    """Decorator to add timeout to async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
                return None
        return wrapper
    return decorator

# Usage:
@async_timeout(timeout_seconds=120)
async def collect_data():
    # Your collection logic here
    pass
