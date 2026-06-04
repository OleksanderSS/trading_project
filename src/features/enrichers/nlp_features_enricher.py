# src/features/enrichers/nlp_features_enricher.py


import pandas as pd

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.features.enrichers.base import BaseEnricher
from src.features.nlp.processors.news_analyzer import NewsAnalyzer

logger = ProjectLogger.get_logger("NLPFeaturesEnricher")

# Constants to avoid duplication
DATETIME64_NS = "datetime64[ns]"

class NLPFeaturesEnricher(BaseEnricher):
    """
    Enriches the main DataFrame with NLP-based features derived from news.
    Uses NewsAnalyzer to process raw news text into sentiment and cluster scores.
    """

    def __init__(self):
        super().__init__()  # Initialize BaseEnricher (sets up self.logger)
        self.config = get_current_config().get('enrichment.nlp_features', {})
        self.analyzer = NewsAnalyzer(
            n_clusters=self.config.get('n_clusters', 5),
            max_features=self.config.get('max_features', 1000)
        )
        logger.info(f"NLPFeaturesEnricher initialized with {self.config.get('n_clusters', 5)} clusters.")

    @property
    def name(self) -> str:
        return "nlp_features"

    @property
    def priority(self) -> int:
        """
        Determines the execution order in the FeatureOrchestrator.
        Set to 30 to run after TechnicalAnalysis (20) but before SentimentFeatures (40).
        """
        return 30

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Processes news data and merges NLP features into the main price DataFrame.

        Args:
            df: The main price DataFrame (must have DatetimeIndex and 'ticker' column).
            **kwargs: Should contain 'news' (pd.DataFrame).

        Returns:
            DataFrame with 'nlp_' prefixed sentiment and clustering features.
        """
        news_df = kwargs.get('news')

        if not self._validate_inputs(df, news_df):
            return df

        logger.info(f"Starting NLP analysis for {len(news_df) if news_df is not None else 0} news items...")
        logger.info(f"News columns: {news_df.columns.tolist() if news_df is not None else []}")

        try:
            analyzed_news = self._analyze_news(news_df)
            if analyzed_news.empty:
                return df

            features_to_merge = self._prepare_features_for_merge(analyzed_news)
            if features_to_merge.empty:
                return df

            df_enriched = self._prepare_main_dataframe(df)
            final_df = self._merge_features_by_ticker(df_enriched, features_to_merge)

            return final_df

        except Exception as e:
            logger.error(f"Error during NLP feature enrichment: {e}", exc_info=True)
            return df

    def _validate_inputs(self, df: pd.DataFrame, news_df: pd.DataFrame | None) -> bool:
        """Validate input dataframes."""
        if news_df is None or news_df.empty:
            logger.warning("No news data provided for NLP enrichment. Skipping.")
            return False

        if 'ticker' not in df.columns:
            logger.error("Main DataFrame missing 'ticker' column. NLP enrichment aborted.")
            return False

        return True

    def _analyze_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Perform news analysis using NewsAnalyzer."""
        analyzed_news = self.analyzer.cluster_news(
            news_df,
            text_column=self.config.get('text_column', 'title'),
            date_column=self.config.get('date_column', 'published_at')
        )

        if analyzed_news.empty:
            logger.warning("NewsAnalyzer returned empty results.")

        return analyzed_news

    def _prepare_features_for_merge(self, analyzed_news: pd.DataFrame) -> pd.DataFrame:
        """Prepare features dataframe for merging with main data."""
        nlp_cols = ['sentiment_score', 'subjectivity_score', 'cluster']
        available_cols = [c for c in nlp_cols if c in analyzed_news.columns]

        features_to_merge = analyzed_news.copy()

        # Create datetime column
        features_to_merge = self._create_datetime_column(features_to_merge)
        if features_to_merge.empty:
            return features_to_merge

        # Select and rename columns
        keep_cols = ['datetime'] + available_cols + (['ticker'] if 'ticker' in features_to_merge.columns else [])
        features_to_merge = features_to_merge[keep_cols]

        rename_map = {col: f"nlp_{col}" for col in available_cols}
        features_to_merge = features_to_merge.rename(columns=rename_map)

        # Normalize timezone and precision
        features_to_merge = self._normalize_datetime_column(features_to_merge, 'datetime')

        return features_to_merge

    def _create_datetime_column(self, features_to_merge: pd.DataFrame) -> pd.DataFrame:
        """Create datetime column from index or find existing date column."""
        if isinstance(features_to_merge.index, pd.DatetimeIndex):
            features_to_merge['datetime'] = features_to_merge.index
            features_to_merge = features_to_merge.reset_index(drop=True)
            features_to_merge['datetime'] = pd.to_datetime(features_to_merge['datetime']).dt.tz_localize(None)
            features_to_merge['datetime'] = features_to_merge['datetime'].astype(DATETIME64_NS)
            logger.info("Created 'datetime' column from DatetimeIndex (tz-naive, ns precision)")
        elif 'datetime' not in features_to_merge.columns:
            date_col = self._find_date_column(features_to_merge)
            if date_col:
                features_to_merge['datetime'] = pd.to_datetime(features_to_merge[date_col])
                features_to_merge = self._normalize_datetime_column(features_to_merge, 'datetime')
                logger.info(f"Created 'datetime' column from '{date_col}' (tz-naive, ns precision)")
            else:
                logger.error(f"No datetime column found. Columns: {features_to_merge.columns.tolist()}")
                return pd.DataFrame()

        if 'datetime' not in features_to_merge.columns:
            logger.error(f"No 'datetime' column after processing. Columns: {features_to_merge.columns.tolist()}")
            return pd.DataFrame()

        return features_to_merge

    def _find_date_column(self, df: pd.DataFrame) -> str | None:
        """Find date column in dataframe."""
        possible_date_cols = ['published_at', 'publishedAt', 'published_date', 'date', 'timestamp']
        for col in possible_date_cols:
            if col in df.columns:
                return col
        return None

    def _normalize_datetime_column(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Normalize timezone and precision of datetime column."""
        if col_name in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col_name]):
                if hasattr(df[col_name].dtype, 'tz') and df[col_name].dt.tz is not None:
                    df[col_name] = df[col_name].dt.tz_localize(None)
                    logger.info(f"Removed timezone from {col_name} for merge compatibility")

                if df[col_name].dtype != DATETIME64_NS:
                    df[col_name] = df[col_name].astype(DATETIME64_NS)
                    logger.info(f"Converted {col_name} to ns precision")

        return df

    def _prepare_main_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare main dataframe for merging."""
        df_enriched = df.copy()

        if not isinstance(df_enriched.index, pd.DatetimeIndex):
            if 'datetime' in df_enriched.columns:
                df_enriched = df_enriched.set_index('datetime')
            else:
                df_enriched.index = pd.to_datetime(df_enriched.index)

        # Normalize timezone and precision
        if df_enriched.index.tz is not None:
            df_enriched.index = df_enriched.index.tz_localize(None)
            logger.info("Removed timezone from df index for merge compatibility")

        if df_enriched.index.dtype != DATETIME64_NS:
            df_enriched.index = df_enriched.index.astype(DATETIME64_NS)
            logger.info("Converted df index to ns precision")

        return df_enriched.sort_index()

    def _merge_features_by_ticker(self, df_enriched: pd.DataFrame, features_to_merge: pd.DataFrame) -> pd.DataFrame:
        """Merge features with main dataframe by ticker groups."""
        features_to_merge = features_to_merge.sort_values('datetime')
        result_dfs = []

        for ticker, group in df_enriched.groupby('ticker'):
            ticker_features = self._get_ticker_features(features_to_merge, ticker)

            if ticker_features.empty:
                result_dfs.append(group)
                continue

            merged_group = self._merge_group_features(group, ticker_features)
            result_dfs.append(merged_group)

        final_df = pd.concat(result_dfs).sort_index()

        # Log added features
        nlp_cols = [col for col in final_df.columns if col.startswith('nlp_')]
        logger.info(f"NLP enrichment complete. Added features: {nlp_cols}")

        return final_df

    def _get_ticker_features(self, features_to_merge: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Get features for specific ticker."""
        if 'ticker' in features_to_merge.columns:
            ticker_features = features_to_merge[features_to_merge['ticker'] == ticker]
        else:
            # Global news applies to all tickers
            ticker_features = features_to_merge.copy()

        # Prevent merge_asof Duplicate Key ValueErrors
        return ticker_features.drop_duplicates(subset=['datetime'], keep='last')

    def _merge_group_features(self, group: pd.DataFrame, ticker_features: pd.DataFrame) -> pd.DataFrame:
        """Merge features for a single ticker group."""
        drop_cols = ['ticker'] if 'ticker' in ticker_features.columns else []
        merged_group = pd.merge_asof(
            group,
            ticker_features.drop(columns=drop_cols) if drop_cols else ticker_features,
            left_index=True,
            right_on='datetime',
            direction='backward'
        )
        # Restore the original index
        merged_group.set_index('datetime', inplace=True)
        return merged_group
