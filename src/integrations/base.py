from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class BaseIntegration(ABC):
    """
    Abstract base class for all external system integrations.
    Provides a unified interface for status checking and identification.
    """

    def __init__(self):
        from src.core.logging.logger import ProjectLogger
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)

    @property
    @abstractmethod
    def name(self) ->str:
        """
        Returns the unique identifier for the integration.
        """
        pass

    @abstractmethod
    def ping(self) ->bool:
        """
        Verifies if the external service or resource is reachable.

        Returns:
            bool: True if reachable, False otherwise.
        """
        pass

    def get_status(self) ->dict[str, Any]:
        """
        Returns a standardized dictionary containing the integration's status.

        Returns:
            Dict[str, Any]: Status metadata including connectivity and timestamp.
        """
        is_alive = False
        error = None
        try:
            is_alive = self.ping()
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            error = str(e)
            raise
        return {'integration_name': self.name, 'status': 'online' if
            is_alive else 'offline', 'reachable': is_alive, 'last_check':
            datetime.now().isoformat(), 'error': error}
