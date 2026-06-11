# src/features/enrichers/base.py

import logging
from abc import ABC, abstractmethod

import pandas as pd


class EnricherError(Exception):
    """Custom exception for enricher-specific errors."""
    pass


class BaseEnricher(ABC):
    """
    Abstract base class for all enrichers that work with the main DataFrame.
    Defines a unified interface for adding features and a unique identifier.

    ✅ Phase 4 Quality: Standardized error handling with template method pattern.
    All enrichers now follow consistent error handling, logging, and fallback behavior.
    """

    def __init__(self, strict: bool = False):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.strict = strict

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the enricher, used for configuration and logging."""
        pass

    @property
    @abstractmethod
    def priority(self) -> int:
        """Determines execution order."""
        pass

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Template method for DataFrame enrichment.
        If strict=True, errors propagate instead of returning original DF.
        """
        try:
            self.logger.debug(f"🔄 Starting enrichment with {self.__class__.__name__}")
            result = self._enrich_impl(df, **kwargs)

            if not isinstance(result, pd.DataFrame):
                raise ValueError(f"Enricher must return DataFrame, got {type(result)}")

            if len(result) == 0:
                raise ValueError("Enricher cannot return empty DataFrame")

            return result

        except Exception as e:
            if self.strict:
                self.logger.error(f"❌ {self.__class__.__name__} failed in STRICT mode: {e}")
                raise EnricherError(f"Strict Enricher {self.__class__.__name__} failed: {e}") from e
            else:
                self.logger.warning(f"⚠️ {self.__class__.__name__} error (non-strict): {e}")
                return df

    @abstractmethod
    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Abstract method for enricher-specific implementation.

        Subclasses should implement this method without error handling -
        the base class template method handles all errors.

        Args:
            df: The input DataFrame to enrich.
            **kwargs: Additional keyword arguments.

        Returns:
            A DataFrame with added features.
        """
        pass
