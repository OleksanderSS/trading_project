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
    Реалізує алгоритм обмеження швидкості "Token Bucket" з плавним поповненням токенів.

    Цей клас є безпечним для використання як у синхронному, так і в асинхронному коді
    завдяки використанню `threading.Lock` та `asyncio.Lock`.

    В архітектурі проєкту цей лімітер централізовано інтегрований у `HttpClientFactory`,
    що гарантує, що всі HTTP-запити до зовнішніх API автоматично дотримуються
    встановлених лімітів швидкості, підвищуючи надійність системи.
    """

    def __init__(self, rate_limit: int = 10, per_seconds: float = 1.0):
        """
        Ініціалізує лімітер швидкості.

        Args:
            rate_limit: Максимальна кількість запитів (токенів) у "відрі".
            per_seconds: Часове вікно в секундах, за яке поповнюється повне "відро" токенів.
        """
        if rate_limit <= 0 or per_seconds <= 0:
            raise ValueError("Ліміт швидкості та період мають бути позитивними числами.")
        
        self.rate_limit = rate_limit
        self.per_seconds = per_seconds
        self.allowance = float(rate_limit)  # Поточна кількість доступних токенів
        self.last_check_time = time.monotonic()
        
        # Блокування для потоко- та асинхронної безпеки
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        
        logger.info(f"RateLimiter ініціалізовано: {rate_limit} запитів за {per_seconds} секунд.")

    def _update_allowance(self) -> None:
        """Поповнює "відро" токенів на основі часу, що минув."""
        current_time = time.monotonic()
        time_passed = current_time - self.last_check_time
        self.last_check_time = current_time
        
        # Додаємо нові токени. Кількість пропорційна часу, що минув.
        replenishment = time_passed * (self.rate_limit / self.per_seconds)
        self.allowance += replenishment
        
        # Обмежуємо кількість токенів максимальним значенням
        if self.allowance > self.rate_limit:
            self.allowance = float(self.rate_limit)

    def acquire(self) -> None:
        """Синхронно очікує, доки не з'явиться вільний токен."""
        with self._lock:
            self._update_allowance()
            if self.allowance < 1.0:
                # Обчислюємо точний час очікування до появи наступного токена
                sleep_duration = (1.0 - self.allowance) * (self.per_seconds / self.rate_limit)
                logger.debug(f"Ліміт швидкості досягнуто. Очікування: {sleep_duration:.4f} сек.")
                time.sleep(sleep_duration)
                self._update_allowance() # Повторне оновлення після очікування
            
            self.allowance -= 1.0
        logger.debug("Токен отримано (синхронно).")

    def try_acquire(self) -> bool:
        """Спроба отримати токен без блокування (синхронно)."""
        with self._lock:
            self._update_allowance()
            if self.allowance >= 1.0:
                self.allowance -= 1.0
                logger.debug("Токен отримано (try_acquire).")
                return True
        
        logger.debug("Немає доступних токенів (try_acquire).")
        return False

    async def acquire_async(self) -> None:
        """Асинхронно очікує, доки не з'явиться вільний токен."""
        async with self._async_lock:
            self._update_allowance()
            if self.allowance < 1.0:
                sleep_duration = (1.0 - self.allowance) * (self.per_seconds / self.rate_limit)
                logger.debug(f"Ліміт швидкості досягнуто. Асинхронне очікування: {sleep_duration:.4f} сек.")
                await asyncio.sleep(sleep_duration)
                self._update_allowance() # Повторне оновлення після очікування
            
            self.allowance -= 1.0
        logger.debug("Токен отримано (асинхронно).")
