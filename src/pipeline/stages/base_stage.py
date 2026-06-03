# src/pipeline/stages/base_stage.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.error_handling.error_handler import ErrorHandler

class BaseStage(ABC):
    """Abstract base class for all pipeline stages."""
    def __init__(self, 
                 config_manager: UnifiedConfigManager, 
                 error_handler: ErrorHandler, 
                 http_client_factory: Optional[HttpClientFactory] = None, 
                 **kwargs: Any):
        self.config_manager = config_manager
        self.error_handler = error_handler
        self.http_client_factory = http_client_factory or HttpClientFactory(config_manager=self.config_manager, error_handler=self.error_handler)
        # Allow for other dependencies to be passed, but they won't be standard attributes
        self.brain = kwargs.get('brain', {}) # Optional brain object for state sharing

    def handle_stage_error(self, error: Exception, context: str = "", severity: str = "error", should_raise: bool = False) -> Dict[str, Any]:
        """Standardized stage error handling wrapper."""
        full_context = f"{self.__class__.__name__}:{context}" if context else self.__class__.__name__
        if self.error_handler:
            return self.error_handler.handle_error(error, full_context, severity, should_raise)
        raise error

    @abstractmethod
    async def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Each stage must implement a run method."""
        pass
