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

    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

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

    def choose_text_column(
        self, news_df: pd.DataFrame, candidates: list[str]
    ) -> str | None:
        """Pick the column that carries text, not the first one that exists.

        This database stores blanks as '' rather than NaN, so "the column is
        present" and "`notna()` is true" are both satisfied by 15,000 empty
        strings. Three enrichers were fooled by it in turn, each reporting
        success on nothing:

            sentiment      15,274 items scored in 2.8s, every one "neutral"
            keyword_entity 32s per timeframe -> "Avg keywords: 0.0"
            news_impact    "Successfully calculated" -> range [0.000, 0.000]

        The news_impact case shows why an empty read is not a harmless zero:
        FinBERT labels an empty string "neutral", and enrichment.yaml weights
        neutral at 0.0 — correctly, for news that really is neutral. So the
        whole column became zero, a value that reads exactly like "no news
        mattered today". The weight is right; labelling blanks as neutral is
        what was wrong.

        Lifted here from KeywordEntityEnricher, whose own comment had already
        drawn the conclusion: wherever this news frame is read, "the column
        exists" has to mean "the column has something in it".

        Returns None when nothing has content, so a caller can refuse rather
        than proceed on blanks.
        """
        filled: dict[str, int] = {}
        for col in candidates:
            if col not in news_df.columns:
                continue
            values = news_df[col].fillna('').astype(str).str.strip()
            filled[col] = int((values != '').sum())

        best = max(filled, key=filled.get) if filled else None
        if best is None or filled[best] == 0:
            self.logger.error(
                "No usable text in news data (non-empty counts: %s). "
                "%s is skipped rather than run on blanks.",
                filled or news_df.columns.tolist()[:10], self.name,
            )
            return None

        if filled[best] < len(news_df):
            self.logger.info(
                "%s reading '%s': %d of %d items carry text.",
                self.name, best, filled[best], len(news_df),
            )
        return best

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Template method for DataFrame enrichment with standardized error handling.

        Process:
        1. Log enrichment start
        2. Call subclass implementation (_enrich_impl)
        3. Validate result
        4. Handle errors with appropriate logging and fallback
        5. Log completion

        Args:
            df: The input DataFrame to enrich.
            **kwargs: Additional keyword arguments for specific implementations.

        Returns:
            A DataFrame with added features, or original df on error.

        Raises:
            EnricherError: For unexpected errors that should propagate.
        """
        try:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"🔄 Starting enrichment with {self.__class__.__name__}")

            # Call subclass implementation
            result = self._enrich_impl(df, **kwargs)

            # Validate result
            if not isinstance(result, pd.DataFrame):
                raise ValueError(f"Enricher must return DataFrame, got {type(result)}")

            if len(result) == 0:
                raise ValueError("Enricher cannot return empty DataFrame")

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"✅ {self.__class__.__name__} completed: {result.shape[1] - df.shape[1]} features added")
            return result

        except KeyError as e:
            # Missing column - log warning and return original
            self.logger.warning(f"⚠️ {self.__class__.__name__} missing required column: {e}")
            return df

        except ValueError as e:
            # Data validation error - log warning and return original
            self.logger.warning(f"⚠️ {self.__class__.__name__} validation error: {e}")
            return df

        except (TypeError, AttributeError, ZeroDivisionError) as e:
            # Unexpected error - log error and raise EnricherError
            self.logger.error(f"❌ {self.__class__.__name__} unexpected error: {e}", exc_info=True)
            raise EnricherError(f"Enricher {self.__class__.__name__} failed: {e}") from e

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
