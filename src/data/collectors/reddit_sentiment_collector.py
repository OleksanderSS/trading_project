# src/data/collectors/reddit_sentiment_collector.py

import hashlib
import random
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector


class RedditSentimentCollector(BaseCollector):
    """Optional collector for aggregate Reddit sentiment."""
    collector_type = "reddit_sentiment"
    data_type = "alternative"
    collector_name = "reddit_sentiment"

    def __init__(self, configs: dict[str, Any], http_client_factory: HttpClientFactory,
                 db_manager: DataManager, cache_manager: CacheManager | None = None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.enabled = self.configs.get('enabled', False)
        self.timeout = self.configs.get('timeout', 30)
        self.table_name = self.configs.get('table_name', "reddit_sentiment_data")
        self.hash_keys = self.configs.get('hash_keys', ["date", "sentiment_score", "viral_posts"])
        self.subreddits = self.configs.get('subreddits', ["wallstreetbets", "stocks", "investing"])
        self.use_synthetic_data = self.configs.get('use_synthetic_data', False)
        self.logger.info(
            "RedditSentimentCollector initialized. "
            f"Enabled: {self.enabled}; synthetic fallback: {self.use_synthetic_data}"
        )

    def _generate_hash(self, row: pd.Series) -> str:
        """Generates a stable hash for a record."""
        hash_string = "|".join(str(row.get(key, "")) for key in self.hash_keys)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    async def run(self, **kwargs) -> pd.DataFrame | None:
        """Fetches Reddit Sentiment data and returns DataFrame."""
        if not self.enabled:
            self.logger.warning("RedditSentimentCollector is disabled")
            return None

        try:
            self.logger.info("Fetching Reddit sentiment data")

            # Fetch data
            data = await self._fetch_reddit_sentiment_data()
            if not data:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)

            if df.empty:
                self.logger.warning("No Reddit Sentiment data received")
                return None

            # Standardize columns
            df = self._standardize_columns(df)

            # Add metadata
            df['collector_type'] = self.collector_type
            df['collector_name'] = self.collector_name
            df['data_type'] = self.data_type
            df['collected_at'] = datetime.now()

            # Generate hashes for deduplication
            df['record_hash'] = df.apply(self._generate_hash, axis=1)

            self.logger.info(f"Successfully fetched {len(df)} Reddit Sentiment records")
            return df

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error in RedditSentimentCollector: {e}")
            raise RuntimeError("Reddit sentiment collection failed") from e

    async def _fetch_reddit_sentiment_data(self) -> list[dict[str, Any]]:
        """
        Fetches Reddit Sentiment data.

        Real Reddit ingestion is intentionally disabled for now. If enabled later,
        it should be treated as a low-weight aggregate sentiment signal, not as a
        high-trust event source.
        """
        if self.use_synthetic_data:
            self.logger.warning("Using synthetic Reddit sentiment data for development only")
            return await self._generate_synthetic_reddit_data()

        self.logger.info(
            "Reddit sentiment integration is disabled until a vetted aggregate sentiment adapter is configured"
        )
        return []

    def _get_subreddit_params(self, subreddit: str) -> tuple[float, int, int]:
        """Get base sentiment, viral posts, and mentions for a subreddit."""
        if subreddit == "wallstreetbets":
            return random.uniform(-0.3, 0.3), random.randint(5, 25), random.randint(500, 2000)
        elif subreddit == "stocks":
            return random.uniform(-0.1, 0.2), random.randint(2, 15), random.randint(200, 800)
        elif subreddit == "investing":
            return random.uniform(0.0, 0.3), random.randint(1, 8), random.randint(100, 400)
        return 0.0, 0, 0

    def _classify_sentiment(self, sentiment_score: float) -> str:
        """Classify sentiment score into categories."""
        if sentiment_score > 0.3:
            return "Very Bullish"
        elif sentiment_score > 0.1:
            return "Bullish"
        elif sentiment_score > -0.1:
            return "Neutral"
        elif sentiment_score > -0.3:
            return "Bearish"
        return "Very Bearish"

    async def _generate_synthetic_reddit_data(self) -> list[dict[str, Any]]:
        """Generates realistic-looking synthetic Reddit sentiment data for testing."""
        self.logger.info("Generating synthetic Reddit sentiment data")

        try:
            base_date = datetime.now() - timedelta(days=60)
            data: list[dict[str, Any]] = []

            # Simulate data for each subreddit
            for subreddit in self.subreddits:
                for i in range(30):  # 30 days per subreddit
                    date_obj = base_date + timedelta(days=i)

                    base_sentiment, viral_posts, mentions = self._get_subreddit_params(subreddit)
                    if viral_posts == 0 and mentions == 0:
                        continue

                    # Add daily variation
                    daily_variation = random.uniform(-0.1, 0.1)
                    sentiment_score = max(-1.0, min(1.0, base_sentiment + daily_variation))
                    sentiment_classification = self._classify_sentiment(sentiment_score)

                    data.append({
                        'date': date_obj.strftime('%Y-%m-%d'),
                        'subreddit': subreddit,
                        'sentiment_score': sentiment_score,
                        'sentiment_classification': sentiment_classification,
                        'mentions': mentions,
                        'viral_posts': viral_posts,
                        'engagement_score': (viral_posts * 0.7) + (mentions * 0.3),
                        'extreme_sentiment': 1 if abs(sentiment_score) > 0.5 else 0,
                        'is_synthetic': True,
                        'data_source': 'synthetic_reddit_simulation',
                        'timestamp': date_obj
                    })

            return data

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error fetching Reddit Sentiment data: {e}")
            raise RuntimeError("Failed to generate synthetic Reddit sentiment data") from e

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names and data types."""
        try:
            # Ensure required columns exist
            if 'date' not in df.columns:
                df['date'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')

            required_cols = ['sentiment_score', 'sentiment_classification', 'mentions']
            for col in required_cols:
                if col not in df.columns:
                    self.logger.error(f"Reddit Sentiment data missing '{col}' column")
                    return pd.DataFrame()

            # Convert date column
            df['date'] = pd.to_datetime(df['date'])

            # Ensure numeric types
            numeric_cols = ['sentiment_score', 'mentions', 'viral_posts', 'engagement_score', 'extreme_sentiment']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)

            # Add derived features
            df['sentiment_sma'] = df.groupby('subreddit')['sentiment_score'].transform(lambda x: x.rolling(7).mean().shift(1))
            df['sentiment_volatility'] = df.groupby('subreddit')['sentiment_score'].transform(lambda x: x.rolling(7).std().shift(1))
            df['viral_spike'] = df.groupby('subreddit')['viral_posts'].transform(lambda x: (x > x.quantile(0.9)).astype(int))

            return df
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error standardizing Reddit Sentiment columns: {e}")
            return pd.DataFrame()

    async def collect_data(self, **kwargs) -> list[dict[str, Any]] | None:
        """
        UNIFIED data collection - retrieval only, without database storage.
        """
        df = await self.run(**kwargs)
        return df.to_dict('records') if df is not None else None
