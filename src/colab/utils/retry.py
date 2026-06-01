"""Retry decorator for timeout handling"""

import time
from functools import wraps
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


def retry_on_timeout(max_retries=3, wait_seconds=5):
    """Декоратор для повтору при timeout"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (TimeoutError, ConnectionError, RuntimeError) as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"⚠️ Attempt {attempt + 1} failed: "
                            f"{str(e)[:100]}\n"
                            f"   Retrying in {wait_seconds} seconds...")
                        time.sleep(wait_seconds)
                    else:
                        logger.error(f"❌ Failed after {max_retries} attempts")
                        raise
        return wrapper
    return decorator
