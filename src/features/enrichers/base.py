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

    @staticmethod
    def bar_window(bar_time: "pd.Series") -> "pd.Timedelta":
        """How far apart these bars are, read off the bars themselves.

        The bound a backward `merge_asof` needs. Without a `tolerance` the join
        carries the last reading forward for as long as the series runs, so a
        flag built on `value.notna()` answers "have we ever had news" rather
        than "is there news on this bar". Measured on the v17 batch, and the
        agreement is exact to four decimal places on every timeframe:

            bars inside the collected news era   15m 1.0000  60m 0.2153  1d 0.2037
            sentiment_available                  15m 1.0000  60m 0.2153  1d 0.2037
            news_coverage                        15m 1.0000  60m 0.2153  1d 0.2037

        Three flags reduced to a copy of a fourth. `news_coverage` is supposed
        to mark the era; these are supposed to mark the bar.

        Inferred rather than configured because these frames hold many tickers
        at the same timestamps, so counting rows would give a spacing 22 times
        too small.
        """
        stamps = pd.to_datetime(pd.Series(bar_time), errors="coerce").dropna()
        unique = pd.Series(stamps.unique())
        if len(unique) < 2:
            return pd.Timedelta(days=1)
        gaps = unique.sort_values().diff().dropna()
        gaps = gaps[gaps > pd.Timedelta(0)]
        return pd.Timedelta(gaps.median()) if len(gaps) else pd.Timedelta(days=1)

    @staticmethod
    def parse_money(values: "pd.Series") -> "pd.Series":
        """Read a number that a scraper stored the way a page displayed it.

        `pd.to_numeric` is the reflex and it is silently wrong here: it returns
        NaN for every one of these, and NaN sums to zero, so the feature is a
        confident 0.0 rather than a missing value anybody would notice.

        Measured on `insider_trades` -- 1,395 rows, all three numeric columns
        stored as text:

            value     '-$4,962,488'   '+$10,681,309'
            price     '$522.37'
            quantity  '-9,500'        '+637,200'

        1,395 of 1,395 became NaN, which is why `insider_net_value_30d` is the
        constant 0.0 on all three timeframes. The enricher even had a fallback
        to `price * quantity` -- and that failed the same way, because both
        halves are the same kind of string.

        Parentheses are read as negative, the accountant's convention some
        sources use instead of a leading minus.
        """
        text = (
            values.astype(str)
            .str.strip()
            .str.replace(r"[,\s$€£¥]", "", regex=True)
        )
        negative = text.str.startswith("(") & text.str.endswith(")")
        text = text.str.strip("()")
        parsed = pd.to_numeric(text, errors="coerce")
        return parsed.where(~negative, -parsed)

    @staticmethod
    def normalise_news_ticker(values: "pd.Series") -> "pd.Series":
        """Fold a news frame's ticker onto 'general' or a real symbol.

        `fillna('general')` is not enough, because this database writes blanks
        as '' and not as NaN. On the 2026-08-14 batch every one of the 15,661
        cleaned news rows carried ticker == '' -- not one NaN -- so the fill
        did nothing, the hourly aggregation grouped everything under '', and
        the merge then looked for 'general' and for 'AAPL' and matched
        neither. Both branches empty means every bar group is passed through
        without counts, which is why keyword_count and entity_count were
        absent from the feature set for four rebuilds running.

        The same '' -- not NaN -- distinction had already fooled three
        enrichers on the TEXT column (see choose_text_column). This is the
        same mistake on the ticker column.

        NlpFeaturesEnricher already had the right predicate inline
        (`isin({'general', 'nan', ''})`); this is that rule, named, so the
        next caller inherits it instead of rediscovering it.

        Case-folded, because the merge compares these against bar tickers and
        a case difference costs every count just as silently.
        """
        folded = values.astype(str).str.strip().str.lower()
        blank = folded.isin({'', 'nan', 'none', 'null', 'general'})
        return folded.mask(blank, 'general')

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
