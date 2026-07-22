# src/data/collectors/reddit_sentiment_collector.py

import hashlib
import re
import feedparser
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector


class RedditSentimentCollector(BaseCollector):
    """Collector for aggregate Reddit sentiment via free RSS feeds."""
    collector_type = "reddit_sentiment"
    data_type = "alternative"
    collector_name = "reddit_sentiment"

    def __init__(self, configs: dict[str, Any], http_client_factory: HttpClientFactory,
                 db_manager: DataManager, cache_manager: CacheManager | None = None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.enabled = self.configs.get('enabled', False)
        self.timeout = self.configs.get('timeout', 30)
        self.table_name = self.configs.get('table_name', "sociological_sentiment_data") # Змінено цільову таблицю
        self.hash_keys = self.configs.get('hash_keys', ["date", "subreddit", "post_id"])
        self.subreddits = self.configs.get('subreddits', ["wallstreetbets", "stocks", "investing", "economics"])
        
        self.logger.info(
            f"RedditSentimentCollector (RSS Mode) initialized. "
            f"Enabled: {self.enabled}; Subreddits: {self.subreddits}"
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
            self.logger.info(f"Fetching Reddit RSS data for {len(self.subreddits)} subreddits")

            # Fetch data
            data = await self._fetch_reddit_rss_data()
            if not data:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)

            if df.empty:
                self.logger.warning("No Reddit Sentiment data received")
                return None

            # Add metadata
            df['collector_type'] = self.collector_type
            df['collector_name'] = self.collector_name
            df['data_type'] = self.data_type
            df['collected_at'] = datetime.now()

            # Generate hashes for deduplication
            df['record_hash'] = df.apply(self._generate_hash, axis=1)
            
            # Фільтруємо існуючі в базі записи
            new_df = self.db_manager.filter_new_records(self.table_name, df, unique_cols=["record_hash"])
            if new_df.empty:
                self.logger.info("No novel Reddit posts identified against historical database.")
                return None

            # Зберігаємо нові
            self.db_manager.upsert(self.table_name, new_df, unique_on=["record_hash"])

            self.logger.info(f"Successfully fetched and saved {len(new_df)} new Reddit posts.")
            return new_df

        except Exception as e:
            self.logger.error(f"Error in RedditSentimentCollector: {e}")
            raise RuntimeError("Reddit sentiment collection failed") from e

    async def _fetch_reddit_rss_data(self) -> list[dict[str, Any]]:
        """
        Fetches data from Reddit RSS feeds.
        No API key required, perfectly legal, avoids PRAW limits.
        """
        all_posts = []
        
        client = await self.http_client_factory.get_http_client()
        headers = {"User-Agent": "DEAN_OS_Agent research@example.com"} # SEC style user-agent
        
        async with client:
            for subreddit in self.subreddits:
                url = f"https://www.reddit.com/r/{subreddit}/.rss"
                try:
                    response = await client.get(url, headers=headers, timeout=self.timeout)
                    response.raise_for_status()
                    
                    feed = feedparser.parse(response.text)
                    for entry in feed.entries:
                        # Extract basic sentiment from title (дуже базовий аналіз для сумісності)
                        title = entry.title.lower()
                        bull_words = ['call', 'moon', 'bull', 'buy', 'long', 'gain', 'beat']
                        bear_words = ['put', 'crash', 'bear', 'sell', 'short', 'loss', 'miss']
                        
                        bull_score = sum(1 for w in bull_words if re.search(r'\b' + w + r'\b', title))
                        bear_score = sum(1 for w in bear_words if re.search(r'\b' + w + r'\b', title))
                        
                        sentiment_score = 0.0
                        if bull_score > bear_score:
                            sentiment_score = 0.5
                        elif bear_score > bull_score:
                            sentiment_score = -0.5
                            
                        # Extract post ID from link
                        post_id = entry.link.split('/comments/')[1].split('/')[0] if '/comments/' in entry.link else entry.link
                        
                        try:
                            # RFC 3339 format parsing
                            published_dt = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%S%z")
                        except ValueError:
                            published_dt = datetime.now(timezone.utc)
                            
                        all_posts.append({
                            'date': published_dt.strftime('%Y-%m-%d'),
                            'subreddit': subreddit,
                            'post_id': post_id,
                            'title': entry.title,
                            'link': entry.link,
                            'sentiment_score': sentiment_score, # For backward compatibility
                            'mentions': 1, # Base
                            'viral_posts': 0, # Cannot know from RSS reliably
                        })
                except Exception as e:
                    self.logger.warning(f"Failed to fetch RSS for r/{subreddit}: {e}")
                    
        return all_posts

    async def collect_data(self, **kwargs) -> list[dict[str, Any]] | None:
        """
        UNIFIED data collection - retrieval only, without database storage.
        """
        df = await self.run(**kwargs)
        return df.to_dict('records') if df is not None else None
