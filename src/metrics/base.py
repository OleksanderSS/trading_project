from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseMetricCalculator(ABC):
    """
    Abstract base class for all metric calculators in the system.
    Provides a unified interface for ML, financial, and system performance metrics.
    """

    @abstractmethod
    def calculate(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Performs the metric calculations on the provided data.

        Args:
            data: The input data (e.g., DataFrame, array of predictions, equity curve).
            **kwargs: Additional parameters for specific calculations.

        Returns:
            Dict[str, Any]: A dictionary containing the calculated metrics.
        """
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """
        Returns the category of the metrics (e.g., 'ml', 'financial', 'system').
        """
        pass

    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """
        Checks if the provided data is suitable for the specific metrics.

        Args:
            data: The input data to validate.

        Returns:
            bool: True if data is valid, False otherwise.
        """
        pass

    def get_summary(self, metrics: Dict[str, Any]) -> str:
        """
        Generates a human-readable summary of the calculated metrics.
        """
        summary_lines = [f"--- {self.category.upper()} Metrics Summary ---"]
        for key, value in metrics.items():
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            summary_lines.append(f"{key}: {formatted_value}")
        return "\n".join(summary_lines)