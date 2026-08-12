
import numpy as np
import pandas as pd

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.features.enrichers.base import BaseEnricher

logger = ProjectLogger.get_logger("SentimentFeaturesEnricher")

# Constants to avoid duplication
DATETIME64_NS = "datetime64[ns]"

class SentimentFeaturesEnricher(BaseEnricher):
    """
    Enriches the DataFrame with advanced sentiment features derived from news scores.
    Calculates rolling statistics, momentum, intensity, and decay-weighted sentiment.
    """

    @property
    def name(self) -> str:
        return "sentiment_features"

    @property
    def priority(self) -> int:
        """
        Determines the execution order in the FeatureOrchestrator.
        Set to 40 to run after NLPFeaturesEnricher (30).
        """
        return 40

    def __init__(self):
        """
        Initializes the enricher by loading settings from the unified configuration.
        """
        super().__init__()  # Initialize BaseEnricher (sets up self.logger)
        config_manager = get_current_config()
        self.sentiment_config = config_manager.get('enrichment.sentiment', {})

        # Default windows if not provided in config
        self.windows = self.sentiment_config.get('windows', [5, 20, 50])
        self.decay_factor = self.sentiment_config.get('decay_factor', 0.95)
        self.enabled_features = self.sentiment_config.get('enabled_features', [
            'rolling_mean', 'rolling_std', 'velocity', 'intensity', 'decay_weighted'
        ])

        logger.info(f"SentimentFeaturesEnricher initialized with windows: {self.windows}")

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Adds sentiment-based features to the input DataFrame.
        Uses news data from kwargs if available.

        Args:
            df: DataFrame containing at least 'datetime', 'ticker'.
            **kwargs: May contain 'news' DataFrame with sentiment scores.

        Returns:
            DataFrame with additional sentiment features.
        """
        if not self._validate_input(df):
            return df

        working_df = df.copy()
        sentiment_col = self._find_sentiment_column(working_df)
        if sentiment_col is None:
            working_df, sentiment_col = self._merge_news_sentiment(working_df, **kwargs)

        if sentiment_col is None:
            logger.warning("No sentiment data available. Skipping sentiment enrichment.")
            return df

        df_enriched = self._prepare_dataframe(working_df, sentiment_col)
        final_df = self._process_ticker_groups(df_enriched, sentiment_col)

        logger.info(f"Sentiment enrichment complete. Added {len(final_df.columns) - len(df.columns)} features.")
        return final_df

    def _validate_input(self, df: pd.DataFrame) -> bool:
        """Validates input DataFrame requirements."""
        if df.empty:
            logger.warning("Received an empty DataFrame for sentiment enrichment.")
            return False

        if 'ticker' not in df.columns:
            logger.error("Required column 'ticker' missing for sentiment enrichment.")
            return False

        return True

    def _find_sentiment_column(self, df: pd.DataFrame) -> str | None:
        """Finds existing sentiment column in DataFrame."""
        for col in ['nlp_sentiment_score', 'sentiment_score', 'sentiment']:
            if col in df.columns:
                return col
        return None

    def _merge_news_sentiment(self, df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, str | None]:
        """Attempts to merge sentiment from news data."""
        news_df = kwargs.get('news')
        if not self._validate_news_data(news_df):
            return df, None

        news_count = len(news_df) if news_df is not None else 0
        logger.info(f"Attempting to merge sentiment from news data ({news_count} rows)")

        # Debug: покажемо всі колонки в news даних
        if news_df is not None:
            logger.info(f"News data columns: {news_df.columns.tolist()}")
            # Покажемо приклади sentiment колонок
            sentiment_cols = ['sentiment', 'nlp_sentiment', 'sentiment_score', 'title_sentiment']
            for col in sentiment_cols:
                if col in news_df.columns:
                    unique_vals = news_df[col].unique()
                    non_null = news_df[col].notna().sum()
                    logger.info(f"News sentiment column '{col}': {non_null}/{len(news_df)} non-null, values: {unique_vals[:5]}")

        time_col = self._find_time_column(news_df)
        news_sentiment_col = self._find_news_sentiment_column(news_df)

        if not (time_col and news_sentiment_col):
            logger.warning(f"Missing required columns: news_sentiment_col={news_sentiment_col}, time_col={time_col}")
            # Return original df without sentiment enrichment
            return df, None

        sentiment_agg = self._aggregate_news_sentiment(news_df, time_col, news_sentiment_col)
        df = self._merge_sentiment_with_main_df(df, sentiment_agg)

        if 'nlp_sentiment_score' not in df.columns:
            return df, None

        return df, 'nlp_sentiment_score'

    def _validate_news_data(self, news_df: pd.DataFrame | None) -> bool:
        """Validates news DataFrame."""
        return (news_df is not None and
                isinstance(news_df, pd.DataFrame) and
                not news_df.empty)

    def _find_time_column(self, news_df: pd.DataFrame) -> str | None:
        """Finds time column in news DataFrame."""
        possible_time_cols = ['published_at', 'publishedAt', 'published_date', 'date', 'timestamp', 'datetime']
        for col in possible_time_cols:
            if col in news_df.columns:
                return col
        return None

    def _find_news_sentiment_column(self, news_df: pd.DataFrame) -> str | None:
        """Finds sentiment column in news DataFrame."""
        # Check for various possible sentiment column names
        possible_cols = ['sentiment_score', 'sentiment', 'finbert_score', 'news_sentiment', 'nlp_sentiment']
        for col in possible_cols:
            if col in news_df.columns:
                logger.info(f"Found sentiment column '{col}' in news data")
                return col
        return None

    def _normalize_time_column(self, news_df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Normalize timezone and convert to datetime64[ns]."""
        news_df = news_df.copy()
        news_df[time_col] = pd.to_datetime(news_df[time_col], errors='coerce', utc=True)
        if news_df[time_col].dt.tz is not None:
            news_df[time_col] = news_df[time_col].dt.tz_localize(None)
        news_df[time_col] = news_df[time_col].astype(DATETIME64_NS)
        return news_df

    def _aggregate_ticker_news(self, ticker_news: pd.DataFrame, time_col: str, sentiment_col: str) -> pd.DataFrame:
        """Aggregate ticker-specific news sentiment."""
        # Convert sentiment to numeric, handling empty strings and non-numeric values
        ticker_news[sentiment_col] = pd.to_numeric(ticker_news[sentiment_col], errors='coerce')
        return ticker_news.groupby(['ticker', pd.Grouper(key=time_col, freq='1h')])[sentiment_col].mean().reset_index()

    def _aggregate_general_news(self, general_news: pd.DataFrame, time_col: str, sentiment_col: str) -> pd.DataFrame:
        """Aggregate general news sentiment."""
        # Convert sentiment to numeric, handling empty strings and non-numeric values
        general_news[sentiment_col] = pd.to_numeric(general_news[sentiment_col], errors='coerce')
        general_sentiment = general_news.groupby(pd.Grouper(key=time_col, freq='1h'))[sentiment_col].mean().reset_index()
        general_sentiment['ticker'] = 'general'
        return general_sentiment

    def _aggregate_by_type_column(self, news_df: pd.DataFrame, time_col: str, sentiment_col: str) -> list[pd.DataFrame]:
        """Aggregate news by type column (general or ticker)."""
        general_news = news_df[news_df['type'] == 'general']
        ticker_news = news_df[news_df['type'] != 'general']
        sentiment_parts = []

        if not general_news.empty:
            sentiment_parts.append(self._aggregate_general_news(general_news, time_col, sentiment_col))

        if not ticker_news.empty:
            # Convert sentiment to numeric before aggregation
            ticker_news[sentiment_col] = pd.to_numeric(ticker_news[sentiment_col], errors='coerce')
            ticker_sentiment = ticker_news.groupby(['type', pd.Grouper(key=time_col, freq='1h')])[sentiment_col].mean().reset_index()
            ticker_sentiment = ticker_sentiment.rename(columns={'type': 'ticker'})
            sentiment_parts.append(ticker_sentiment)

        return sentiment_parts

    def _aggregate_by_ticker_column(self, news_df: pd.DataFrame, time_col: str, sentiment_col: str) -> list[pd.DataFrame]:
        """Aggregate news by ticker column."""
        ticker_news = news_df[news_df['ticker'].notna()]
        general_news = news_df[news_df['ticker'].isna()]
        sentiment_parts = []

        if not general_news.empty:
            sentiment_parts.append(self._aggregate_general_news(general_news, time_col, sentiment_col))

        if not ticker_news.empty:
            sentiment_parts.append(self._aggregate_ticker_news(ticker_news, time_col, sentiment_col))

        return sentiment_parts

    def _aggregate_news_sentiment(self, news_df: pd.DataFrame, time_col: str, sentiment_col: str) -> pd.DataFrame:
        """Aggregates news sentiment by time and ticker."""
        news_df = self._normalize_time_column(news_df, time_col)

        logger.info(f"Found time column '{time_col}' with {len(news_df[news_df[time_col].notna()])} valid timestamps")

        sentiment_parts = []

        if 'ticker' in news_df.columns:
            sentiment_parts.extend(self._aggregate_by_ticker_column(news_df, time_col, sentiment_col))
        elif 'type' in news_df.columns:
            sentiment_parts.extend(self._aggregate_by_type_column(news_df, time_col, sentiment_col))
        else:
            # Convert sentiment to numeric before aggregation
            news_df[sentiment_col] = pd.to_numeric(news_df[sentiment_col], errors='coerce')
            global_sentiment = news_df.groupby(pd.Grouper(key=time_col, freq='1h'))[sentiment_col].mean().reset_index()
            global_sentiment['ticker'] = 'general'
            sentiment_parts.append(global_sentiment)

        if sentiment_parts:
            sentiment_agg = pd.concat(sentiment_parts, ignore_index=True)
        else:
            sentiment_agg = pd.DataFrame(columns=['ticker', 'datetime', 'nlp_sentiment_score'])

        # Both names are known: `time_col` and `sentiment_col` were resolved
        # by the caller. Renaming them explicitly replaces a loop that took
        # "the first column that is not ticker or datetime" and called it the
        # sentiment.
        #
        # The aggregations above return [ticker, <time_col>, <sentiment_col>]
        # and the time column is `published_at`, not `datetime` -- so the loop
        # renamed the TIMESTAMP to nlp_sentiment_score and left the sentiment
        # untouched under its own name. Measured on 2026-08-12:
        #
        #   {'ticker': 'AAPL',
        #    'nlp_sentiment_score': Timestamp('2026-03-27 00:00:00'),
        #    'sentiment': 0.0377...}
        #
        # `pd.to_numeric` on that timestamp yields ~1.7e18 nanoseconds, which
        # is never null -- so `sentiment_available` read 1.0 on every bar in
        # the batch, including 9,070 daily bars predating any news, and all
        # thirteen sentiment features (rolling mean, std, velocity, decay
        # weighting, news intensity) were computed from epoch nanoseconds.
        # The actual sentiment reached nothing.
        renames = {}
        if time_col in sentiment_agg.columns and 'datetime' not in sentiment_agg.columns:
            renames[time_col] = 'datetime'
        if sentiment_col in sentiment_agg.columns:
            renames[sentiment_col] = 'nlp_sentiment_score'
        if renames:
            sentiment_agg = sentiment_agg.rename(columns=renames)

        self._normalize_datetime_column(sentiment_agg, 'datetime')
        if 'ticker' not in sentiment_agg.columns:
            sentiment_agg['ticker'] = 'general'

        if 'nlp_sentiment_score' not in sentiment_agg.columns:
            # Say it rather than guess. A guessed column is what produced
            # timestamps-as-sentiment for months.
            logger.error(
                "News aggregation produced no sentiment column (had %s, "
                "expected '%s'); sentiment features will be skipped.",
                list(sentiment_agg.columns), sentiment_col,
            )

        return sentiment_agg

    def _normalize_datetime_column(self, df: pd.DataFrame, col: str) -> None:
        """Normalizes datetime column to timezone-naive datetime64[ns]."""
        if col in df.columns:
            # Ensure it is a datetime object first
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')

            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if hasattr(df[col].dtype, 'tz') and df[col].dt.tz is not None:
                    df[col] = df[col].dt.tz_localize(None)
                # Convert to ns precision
                if df[col].dtype != DATETIME64_NS:
                    df[col] = df[col].astype(DATETIME64_NS)

    def _merge_sentiment_with_main_df(self, df: pd.DataFrame, sentiment_agg: pd.DataFrame) -> pd.DataFrame:
        """Merges aggregated sentiment with main DataFrame."""
        try:
            has_datetime = ('datetime' in df.columns or
                           df.index.name == 'datetime' or
                           isinstance(df.index, pd.DatetimeIndex))

            if not has_datetime:
                logger.warning("No datetime column found in main DataFrame for sentiment merge")
                return df

            if 'datetime' not in df.columns:
                df = df.reset_index(names='datetime' if not df.index.name else None)

            # Normalize timezone + precision in df
            self._normalize_datetime_column(df, 'datetime')

            # Normalize timezone + precision in sentiment_agg
            if 'datetime' not in sentiment_agg.columns:
                if hasattr(sentiment_agg.index, 'name') and sentiment_agg.index.name:
                    sentiment_agg = sentiment_agg.reset_index()
                else:
                    sentiment_agg = sentiment_agg.reset_index(names=['datetime'])

            self._normalize_datetime_column(sentiment_agg, 'datetime')

            merge_keys = ['ticker', 'datetime'] if 'ticker' in sentiment_agg.columns else ['datetime']
            df = df.merge(sentiment_agg, on=merge_keys, how='left')

            # Set datetime as index if it exists
            if 'datetime' in df.columns:
                df = df.set_index('datetime')

            logger.info("Merged sentiment from news data")

            return df

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Error merging sentiment with main DataFrame: {e}")
            return df

    def _prepare_dataframe(self, df: pd.DataFrame, sentiment_col: str) -> pd.DataFrame:
        """Prepares DataFrame for feature calculation."""
        df_enriched = df.copy()

        # Sort by ticker and datetime to ensure rolling windows work correctly
        if 'datetime' in df_enriched.columns:
            df_enriched = df_enriched.sort_values(['ticker', 'datetime'])

        # Fill sentiment strictly within each ticker, then use neutral for tickers with no signal.
        sentiment_values = pd.to_numeric(df_enriched[sentiment_col], errors='coerce')
        carried_sentiment = sentiment_values.groupby(df_enriched['ticker']).ffill()

        # The flag reports a reading on THIS bar, not the presence of a
        # value after forward-fill.
        #
        # It was computed from carried_sentiment, i.e. after the ffill --
        # so one reading anywhere in a ticker's history made every later row
        # "available", forever. Measured on the 2026-08-06 export it is the
        # constant 1.0 on all three timeframes, including 5,757 daily rows
        # predating any sentiment source in the database.
        #
        # A constant column cannot inform a model, and this one does worse
        # than that: it asserts a reading exists for rows carrying a value
        # stale by days. Distinguishing a fresh sentiment from a stale one
        # is the entire reason to have the flag.
        df_enriched['sentiment_available'] = sentiment_values.notna().astype(int)
        df_enriched[sentiment_col] = carried_sentiment.where(carried_sentiment.notna(), 0.0)

        return df_enriched

    def _process_ticker_groups(self, df_enriched: pd.DataFrame, sentiment_col: str) -> pd.DataFrame:
        """Processes sentiment features for each ticker group."""
        results = []
        for _ticker, ticker_group in df_enriched.groupby('ticker'):
            ticker_group = ticker_group.copy()

            self._add_rolling_statistics(ticker_group, sentiment_col)
            self._add_sentiment_velocity(ticker_group, sentiment_col)
            self._add_news_intensity(ticker_group, sentiment_col)
            self._add_decay_weighted_sentiment(ticker_group, sentiment_col)

            results.append(ticker_group)

        # Recombine groups
        return pd.concat(results).sort_index()

    def _add_rolling_statistics(self, ticker_group: pd.DataFrame, sentiment_col: str) -> None:
        """Adds rolling statistics (mean, std, EMA) features."""
        if not ('rolling_mean' in self.enabled_features or 'rolling_std' in self.enabled_features):
            return

        for window in self.windows:
            if 'rolling_mean' in self.enabled_features:
                ticker_group[f'sentiment_sma_{window}'] = (ticker_group[sentiment_col]
                                                           .rolling(window=window, min_periods=1)
                                                           .mean())
            if 'rolling_std' in self.enabled_features:
                sentiment_std = (ticker_group[sentiment_col]
                                 .rolling(window=window, min_periods=1)
                                 .std())
                ticker_group[f'sentiment_std_{window}'] = sentiment_std.where(sentiment_std.notna(), 0)

        ticker_group['sentiment_ema'] = (ticker_group[sentiment_col]
                                        .ewm(span=self.windows[0], adjust=False)
                                        .mean())

    def _add_sentiment_velocity(self, ticker_group: pd.DataFrame, sentiment_col: str) -> None:
        """Adds sentiment velocity (momentum) feature."""
        if 'velocity' not in self.enabled_features:
            return

        # Change over the last 3 intervals
        sentiment_velocity = ticker_group[sentiment_col].diff(periods=3)
        ticker_group['sentiment_velocity'] = sentiment_velocity.where(sentiment_velocity.notna(), 0)

    def _add_news_intensity(self, ticker_group: pd.DataFrame, sentiment_col: str) -> None:
        """Adds news intensity feature."""
        if 'intensity' not in self.enabled_features:
            return

        has_news = (ticker_group[sentiment_col] != 0).astype(int)
        ticker_group['news_intensity'] = (has_news
                                          .rolling(window=self.windows[0], min_periods=1)
                                          .sum())

    def _add_decay_weighted_sentiment(self, ticker_group: pd.DataFrame, sentiment_col: str) -> None:
        """Adds decay-weighted sentiment feature."""
        if 'decay_weighted' not in self.enabled_features:
            return

        ticker_group['sentiment_decay_weighted'] = self._calculate_decay_weights(ticker_group[sentiment_col])

    def _calculate_decay_weights(self, series: pd.Series) -> pd.Series:
        """
        Applies an exponential decay to historical sentiment scores.
        Recent scores have higher influence.
        """
        def apply_decay(x):
            weights = self.decay_factor ** np.arange(len(x))[::-1]
            return np.sum(x * weights) / np.sum(weights)

        # Apply using a rolling window equal to the shortest SMA window
        window_size = self.windows[0]
        return series.rolling(window=window_size, min_periods=1).apply(apply_decay, raw=True)
