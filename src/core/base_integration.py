from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from src.core.error_handling.error_handler import get_error_handler


class BaseIntegration(ABC):
    """
    Abstract base class for all external system integrations.
    Provides a unified interface for status checking and identification.
    """

    def __init__(self):
        self.error_handler = get_error_handler()

    def fetch_with_retry(self, func, *args, **kwargs):
        """Standardized retry logic for API calls."""
        return self.error_handler.retry(max_retries=3, delay=1.0)(func)(*args, **kwargs)

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique identifier for the integration."""

    @abstractmethod
    def ping(self) -> bool:
        """Verifies if the external service or resource is reachable."""

    def get_status(self) -> dict[str, Any]:
        """Returns a standardized status dictionary."""
        is_alive = False
        error = None
        try:
            is_alive = self.ping()
        except Exception as e:
            self.logger.error(f"Виникла помилка: {e}", exc_info=True)
            error = str(e)
            raise
        return {
            "integration_name": self.name,
            "status": "online" if is_alive else "offline",
            "reachable": is_alive,
            "last_check": datetime.now().isoformat(),
            "error": error,
        }
