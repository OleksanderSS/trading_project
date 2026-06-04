from abc import ABC, abstractmethod
from typing import Any


class IAnalyzer(ABC):
    """
    Base interface for all analyzers in the system.
    Provides a unified contract for the AnalyticsEngine.
    """

    @abstractmethod
    def analyze(self, data: Any, **kwargs) -> dict[str, Any] | Any:
        """
        Executes the analysis logic on the provided data.

        Args:
            data: The input data for analysis (e.g., DataFrame, list of news, etc.).
            **kwargs: Flexible parameters for specific analyzer implementations.

        Returns:
            The result of the analysis, typically a dictionary of metrics or signals.
        """
        pass
