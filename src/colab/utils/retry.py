"""Retry decorator for timeout handling"""

import time
from functools import wraps


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
                        print(
                            f"⚠️ Attempt {attempt + 1} failed: "
                            f"{str(e)[:100]}")
                        print(f"   Retrying in {wait_seconds} seconds...")
                        time.sleep(wait_seconds)
                    else:
                        print(f"❌ Failed after {max_retries} attempts")
                        raise
        return wrapper
    return decorator
