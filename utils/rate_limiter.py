# utils/rate_limiter.py

import time
import threading
import asyncio
from utils.logger import ProjectLogger
from config.rate_limiter_config import RATE_LIMITER_DEFAULTS

logger = ProjectLogger.get_logger("TradingProjectLogger")

class RateLimiter:
    """
    Реалandforцandя rate limiter (token bucket) with пandдтримкою плавного поповnotння токенandв.
    Пandдходить for синхронних and асинхронних forпитandв.
    """

    def __init__(self,
                 rate: int = RATE_LIMITER_DEFAULTS["rate"],
                 per: float = RATE_LIMITER_DEFAULTS["per"]):
        """
        :param rate: кandлькandсть forпитandв for andнтервал
        :param per: andнтервал часу (секунди)
        """
        self.rate = rate
        self.per = per
        self.allowance = rate  # поточна кandлькandсть токенandв
        self.last_check = time.monotonic()
        self.lock = threading.Lock()
        self.async_lock = asyncio.Lock()
        logger.info(f"[RateLimiter] Initialized with rate={rate}, per={per}")

    def _update_allowance(self):
        current = time.monotonic()
        time_passed = current - self.last_check
        self.last_check = current
        self.allowance += time_passed * (self.rate / self.per)
        if self.allowance > self.rate:
            self.allowance = self.rate

    # -------------------- Синхронний виклик --------------------
    def acquire(self):
        """Синхронnot блокування до появи токена."""
        with self.lock:
            self._update_allowance()
            if self.allowance < 1.0:
                sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
                logger.debug(f"[RateLimiter]  Очandкування {sleep_time:.4f} сек")
                time.sleep(sleep_time)
                self._update_allowance()
            self.allowance -= 1.0
            logger.debug("[RateLimiter] [OK] Токен received (sync)")

    def try_acquire(self) -> bool:
        """Спроба отримати токен беwith блокування."""
        with self.lock:
            self._update_allowance()
            if self.allowance >= 1.0:
                self.allowance -= 1.0
                logger.debug("[RateLimiter] [OK] Токен received (try_acquire)")
                return True
            logger.debug("[RateLimiter] [ERROR] Токен unavailable (try_acquire)")
            return False

    # -------------------- Асинхронний виклик --------------------
    async def acquire_async(self):
        """Асинхронnot очandкування токена (for asyncio)."""
        while True:
            async with self.async_lock:
                self._update_allowance()
                if self.allowance >= 1.0:
                    self.allowance -= 1.0
                    logger.debug("[RateLimiter] [OK] Токен received (async)")
                    return
                sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
                logger.debug(f"[RateLimiter]  Async очandкування {sleep_time:.4f} сек")
            await asyncio.sleep(sleep_time)