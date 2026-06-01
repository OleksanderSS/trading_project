
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
            return df

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

    def _aggregate_news_sentiment(self, news_df: pd.DataFrame, time_col: str, sentiment_col: str) -> pd.DataFrame:
        """Aggregates news sentiment by time and ticker."""
        # Normalize timezone and convert to datetime64[ns]
        news_df[time_col] = pd.to_datetime(news_df[time_col], errors='coerce', utc=True)
        if news_df[time_col].dt.tz is not None:
            news_df[time_col] = news_df[time_col].dt.tz_localize(None)
        news_df[time_col] = news_df[time_col].astype(DATETIME64_NS)

        logger.info(f"Found time column '{time_col}' with {len(news_df[news_df[time_col].notna()])} valid timestamps")

        # Aggregate sentiment by date and ticker (if available)
        sentiment_parts = []

        if 'ticker' in news_df.columns:
            # Агрегація по тікерах + загальні новини без тікера
            ticker_news = news_df[news_df['ticker'].notna()]
            general_news = news_df[news_df['ticker'].isna()]

            if not general_news.empty:
                # Обробка загальних новин
                general_sentiment = general_news.groupby(pd.Grouper(key=time_col, freq='1h'))[sentiment_col].mean().reset_index()
                general_sentiment['ticker'] = 'general'  # Додаємо загальний тікер
                sentiment_parts.append(general_sentiment)

            if not ticker_news.empty:
                # Обробка новин по тікерах
                ticker_sentiment = ticker_news.groupby(['ticker', pd.Grouper(key=time_col, freq='1h')])[sentiment_col].mean().reset_index()
                sentiment_parts.append(ticker_sentiment)

        elif 'type' in news_df.columns:
            # Агрегація по типу новин (general або ticker)
            general_news = news_df[news_df['type'] == 'general']
            ticker_news = news_df[news_df['type'] != 'general']

            if not general_news.empty:
                # Обробка загальних новин
                general_sentiment = general_news.groupby(pd.Grouper(key=time_col, freq='1h'))[sentiment_col].mean().reset_index()
                general_sentiment['ticker'] = 'general'  # Додаємо загальний тікер
                sentiment_parts.append(general_sentiment)

            if not ticker_news.empty:
                # Обробка новин по тікерах
                ticker_sentiment = ticker_news.groupby(['type', pd.Grouper(key=time_col, freq='1h')])[sentiment_col].mean().reset_index()
                ticker_sentiment = ticker_sentiment.rename(columns={'type': 'ticker'})
                sentiment_parts.append(ticker_sentiment)
        else:
            # If there is no ticker in news, aggregate only by date (global sentiment)
            global_sentiment = news_df.groupby(pd.Grouper(key=time_col, freq='1h'))[sentiment_col].mean().reset_index()
            global_sentiment['ticker'] = 'general'  # Додаємо загальний тікер для консистентності
            sentiment_parts.append(global_sentiment)

        # Об'єднуємо всі частини
        if sentiment_parts:
            sentiment_agg = pd.concat(sentiment_parts, ignore_index=True)
        else:
            sentiment_agg = pd.DataFrame(columns=['ticker', 'datetime', 'nlp_sentiment_score'])

        # Normalize timezone + precision in sentiment_agg
        self._normalize_datetime_column(sentiment_agg, 'datetime')

        # Переконуємось що колонки мають правильні назви
        if 'ticker' not in sentiment_agg.columns:
            sentiment_agg['ticker'] = 'general'
        if 'nlp_sentiment_score' not in sentiment_agg.columns:
            # Якщо колонка з sentiment має іншу назву, перейменовуємо
            for col in sentiment_agg.columns:
                if col not in ['ticker', 'datetime']:
                    sentiment_agg = sentiment_agg.rename(columns={col: 'nlp_sentiment_score'})
                    break

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

        except Exception as e:
            logger.error(f"Error merging sentiment with main DataFrame: {e}")
            return df

    def _prepare_dataframe(self, df: pd.DataFrame, sentiment_col: str) -> pd.DataFrame:
        """Prepares DataFrame for feature calculation."""
        df_enriched = df.copy()

        # Sort by ticker and datetime to ensure rolling windows work correctly
        if 'datetime' in df_enriched.columns:
            df_enriched = df_enriched.sort_values(['ticker', 'datetime'])

        # Fill sentiment strictly within each ticker, then use neutral for tickers with no signal.
        sentiment_values = pd.to_numeric(df_enriched[sentiment_col], errors='coerce')
        df_enriched[sentiment_col] = (
            sentiment_values.groupby(df_enriched['ticker']).ffill()
            .fillna(0.0)
        )

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
                ticker_group[f'sentiment_std_{window}'] = (ticker_group[sentiment_col]
                                                           .rolling(window=window, min_periods=1)
                                                           .std().fillna(0))

        ticker_group['sentiment_ema'] = (ticker_group[sentiment_col]
                                        .ewm(span=self.windows[0], adjust=False)
                                        .mean())

    def _add_sentiment_velocity(self, ticker_group: pd.DataFrame, sentiment_col: str) -> None:
        """Adds sentiment velocity (momentum) feature."""
        if 'velocity' not in self.enabled_features:
            return

        # Change over the last 3 intervals
        ticker_group['sentiment_velocity'] = (ticker_group[sentiment_col]
                                              .diff(periods=3)
                                              .fillna(0))

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
