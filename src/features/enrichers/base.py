# src/features/enrichers/base.py

from abc import ABC, abstractmethod
import pandas as pd

class BaseEnricher(ABC):
    """
    Abstract base class for all enrichers that work with the main DataFrame.
    Defines a unified interface for adding features and a unique identifier.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the enricher, used for configuration and logging."""
        pass

    @property
    @abstractmethod
    def priority(self) -> int:
        """
        Determines the execution order in the FeatureOrchestrator.
        Lower values are executed first (e.g., 0 runs before 100).
        """
        pass

    @abstractmethod
    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Abstract method for enriching a DataFrame.

        Args:
            df: The input DataFrame to enrich.
            **kwargs: Additional keyword arguments for specific implementations.

        Returns:
            A DataFrame with added features.
        """
        pass