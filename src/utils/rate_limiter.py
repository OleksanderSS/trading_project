import logging
# src/utils/rate_limiter.py

import asyncio
import time
import threading
from typing import Optional

from src.core.logging.logger import ProjectLogger

# Initialize logger for the module
logger = ProjectLogger.get_logger("RateLimiter")

class RateLimiter:
    """
    Implements a "Token Bucket" rate limiting algorithm with smooth token replenishment.

    This class is safe for use in both synchronous and asynchronous code
    due to the use of `threading.Lock` and `asyncio.Lock`.

    In the project architecture, this limiter is centrally integrated into `HttpClientFactory`,
    ensuring that all HTTP requests to external APIs automatically adhere to
    established rate limits, enhancing system reliability.
    """

    def __init__(self, rate_limit: int = 10, per_seconds: float = 1.0):
        """
        Initializes the rate limiter.

        Args:
            rate_limit: Maximum number of requests (tokens) in the bucket.
            per_seconds: Time window in seconds over which a full bucket of tokens is replenished.
        """
        if rate_limit <= 0 or per_seconds <= 0:
            raise ValueError("Rate limit and period must be positive numbers.")
        
        self.rate_limit = rate_limit
        self.per_seconds = per_seconds
        self.allowance = float(rate_limit)  # Current number of available tokens
        self.last_check_time = time.monotonic()
        
        # Locks for thread and async safety
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        
        logger.info(f"RateLimiter initialized: {rate_limit} requests per {per_seconds} seconds.")

    def _update_allowance(self) -> None:
        """Replenishes the token bucket based on elapsed time."""
        current_time = time.monotonic()
        time_passed = current_time - self.last_check_time
        self.last_check_time = current_time
        
        # Add new tokens. Quantity is proportional to elapsed time.
        replenishment = time_passed * (self.rate_limit / self.per_seconds)
        self.allowance += replenishment
        
        # Cap tokens at maximum value
        if self.allowance > self.rate_limit:
            self.allowance = float(self.rate_limit)

    def acquire(self) -> None:
        """Synchronously waits until a token is available."""
        with self._lock:
            self._update_allowance()
            if self.allowance < 1.0:
                # Calculate precise wait time until the next token appears
                sleep_duration = (1.0 - self.allowance) * (self.per_seconds / self.rate_limit)
                # Only log if wait time is significant (>50ms) to reduce log noise
                if sleep_duration > 0.05:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Rate limit reached. Waiting: {sleep_duration:.4f} sec.")
                time.sleep(sleep_duration)
                self._update_allowance() # Re-update after waiting
            
            self.allowance -= 1.0
        # Reduce log noise - only log on significant waits

    def try_acquire(self) -> bool:
        """Attempts to acquire a token without blocking (synchronously)."""
        with self._lock:
            self._update_allowance()
            if self.allowance >= 1.0:
                self.allowance -= 1.0
                return True
        
        return False

    async def acquire_async(self) -> None:
        """Asynchronously waits until a token is available."""
        async with self._async_lock:
            self._update_allowance()
            if self.allowance < 1.0:
                sleep_duration = (1.0 - self.allowance) * (self.per_seconds / self.rate_limit)
                # Only log if wait time is significant (>50ms) to reduce log noise
                if sleep_duration > 0.05:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Rate limit reached. Async waiting: {sleep_duration:.4f} sec.")
                await asyncio.sleep(sleep_duration)
                self._update_allowance() # Re-update after waiting
            
            self.allowance -= 1.0
        # Reduce log noise - only log on significant waits
