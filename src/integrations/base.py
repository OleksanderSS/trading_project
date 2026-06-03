from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime


class BaseIntegration(ABC):
    """
    Abstract base class for all external system integrations.
    Provides a unified interface for status checking and identification.
    """

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

    def get_status(self) ->Dict[str, Any]:
        """
        Returns a standardized dictionary containing the integration's status.
        
        Returns:
            Dict[str, Any]: Status metadata including connectivity and timestamp.
        """
        is_alive = False
        error = None
        try:
            is_alive = self.ping()
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            error = str(e)
            raise
        return {'integration_name': self.name, 'status': 'online' if
            is_alive else 'offline', 'reachable': is_alive, 'last_check':
            datetime.now().isoformat(), 'error': error}
