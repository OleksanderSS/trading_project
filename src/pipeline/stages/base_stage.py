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

    @abstractmethod
    async def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Each stage must implement a run method."""
        pass
