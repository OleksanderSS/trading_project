# src/data/collectors/reddit_sentiment_collector.py

import pandas as pd
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import random

from .base_collector import BaseCollector
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager
from src.core.cache.cache_manager import CacheManager

class RedditSentimentCollector(BaseCollector):
    """Optional collector for aggregate Reddit sentiment."""
    collector_type = "reddit_sentiment"
    data_type = "alternative"
    collector_name = "reddit_sentiment"

    def __init__(self, configs: Dict[str, Any], http_client_factory: HttpClientFactory, 
                 db_manager: DataManager, cache_manager: Optional[CacheManager] = None, **kwargs):
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

    async def run(self, **kwargs) -> Optional[pd.DataFrame]:
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

        except Exception as e:
            self.logger.error(f"Error in RedditSentimentCollector: {e}")
            return None

    async def _fetch_reddit_sentiment_data(self) -> List[Dict[str, Any]]:
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
    
    async def _generate_synthetic_reddit_data(self) -> List[Dict[str, Any]]:
        """Generates realistic-looking synthetic Reddit sentiment data for testing."""
        self.logger.info("Generating synthetic Reddit sentiment data")

        try:
            base_date = datetime.now() - timedelta(days=60)
            data: List[Dict[str, Any]] = []

            # Simulate data for each subreddit
            for subreddit in self.subreddits:
                for i in range(30):  # 30 days per subreddit
                    date_obj = base_date + timedelta(days=i)

                    # Simulate realistic Reddit sentiment
                    if subreddit == "wallstreetbets":
                        # High volatility, extreme sentiments
                        base_sentiment = random.uniform(-0.3, 0.3)
                        viral_posts = random.randint(5, 25)
                        mentions = random.randint(500, 2000)
                    elif subreddit == "stocks":
                        # Moderate sentiment, more balanced
                        base_sentiment = random.uniform(-0.1, 0.2)
                        viral_posts = random.randint(2, 15)
                        mentions = random.randint(200, 800)
                    elif subreddit == "investing":
                        # Generally positive, lower volatility
                        base_sentiment = random.uniform(0.0, 0.3)
                        viral_posts = random.randint(1, 8)
                        mentions = random.randint(100, 400)
                    else:
                        continue

                    # Add daily variation
                    daily_variation = random.uniform(-0.1, 0.1)
                    sentiment_score = max(-1.0, min(1.0, base_sentiment + daily_variation))

                    # Classify sentiment
                    if sentiment_score > 0.3:
                        sentiment_classification = "Very Bullish"
                    elif sentiment_score > 0.1:
                        sentiment_classification = "Bullish"
                    elif sentiment_score > -0.1:
                        sentiment_classification = "Neutral"
                    elif sentiment_score > -0.3:
                        sentiment_classification = "Bearish"
                    else:
                        sentiment_classification = "Very Bearish"

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

        except Exception as e:
            self.logger.error(f"Error fetching Reddit Sentiment data: {e}")
            return []

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
        except Exception as e:
            self.logger.error(f"Error standardizing Reddit Sentiment columns: {e}")
            return pd.DataFrame()

    async def collect_data(self, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """
        UNIFIED data collection - retrieval only, without database storage.
        """
        df = await self.run(**kwargs)
        return df.to_dict('records') if df is not None else None
