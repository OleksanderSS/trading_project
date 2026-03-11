from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseMonitor(ABC):
    """
    Abstract base class for all system and data monitors.
    Defines a consistent interface for metric collection and health assessment.
    """

    @abstractmethod
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Abstract method to collect and return specific metrics.
        Returns:
            Dict[str, Any]: A dictionary containing metric names and their current values.
        """
        pass

    @property
    @abstractmethod
    def monitor_name(self) -> str:
        """
        Unique identifier for the monitor instance.
        """
        pass

    def is_healthy(self) -> bool:
        """
        Default implementation to determine health status based on thresholds.
        Can be overridden by subclasses for more complex logic.
        
        Returns:
            bool: True if metrics are within acceptable limits, False otherwise.
        """
        metrics = self.collect_metrics()
        # Basic implementation: checks for 'status' or 'error' keys in metrics
        if 'error' in metrics or metrics.get('status') == 'critical':
            return False
        return True