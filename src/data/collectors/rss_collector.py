
import asyncio
import pandas as pd
import feedparser
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

from .base_collector import BaseCollector
from src.core.logging.logger import ProjectLogger
from src.features.nlp.deduplication_service import DeduplicationService
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager # Додаємо імпорт

class RSSCollector(BaseCollector):
    """A collector for fetching data from RSS feeds."""
    collector_type = "rss"

    def __init__(self, configs: Dict[str, Any], http_client_factory: HttpClientFactory, db_manager: DataManager, **kwargs):
        # ВИПРАВЛЕНО: `db_manager` тепер правильно передається в `super()`
        super().__init__(configs, http_client_factory, db_manager, **kwargs)
        self.logger = ProjectLogger.get_logger(__name__)
        
        deduplication_configs = self.configs.get('deduplication', {})
        self.deduplication_service = DeduplicationService(
            n_clusters=deduplication_configs.get('n_clusters', 10),
            max_features=deduplication_configs.get('max_features', 500)
        )
        self.period_days = self._parse_period_to_days()
        self.logger.info(f"RSSCollector initialized successfully. News older than {self.period_days} days will be filtered.")

    def _parse_period_to_days(self) -> int:
        period_str = self.configs.get('params', {}).get('period', '60d')
        if 'd' in period_str:
            return int(period_str.replace('d', ''))
        return 60

    async def run(self, tickers: Optional[List[str]] = None, keywords: Optional[List[str]] = None, **kwargs) -> Optional[pd.DataFrame]:
        """Fetches, filters by date, and deduplicates news from the configured RSS feeds."""
        self.logger.info("Starting data collection for rss...")
        
        # Потребує UnifiedConfigManager для доступу до rss_feeds, тому ми створюємо його тут
        config_manager = UnifiedConfigManager()
        knowledge_base = config_manager.get_config('knowledge_base')
        feeds = knowledge_base.get('rss_feeds', [])
        if not feeds:
            self.logger.warning("No RSS feeds found in knowledge_base.yaml. Skipping collection.")
            return None

        self.logger.info(f"Loaded {len(feeds)} RSS feeds.")

        tasks = [self._fetch_feed(feed['name'], feed['url']) for feed in feeds]
        all_articles = await asyncio.gather(*tasks)
        
        flat_articles = [article for sublist in all_articles if sublist for article in sublist]

        if not flat_articles:
            self.logger.info("No new articles found in any RSS feed.")
            return None
            
        self.logger.info(f"Collected {len(flat_articles)} raw articles from all RSS feeds.")

        raw_df = pd.DataFrame(flat_articles)
        
        self.logger.info("Applying deduplication to raw news data...")
        deduplicated_df = self.deduplication_service.deduplicate(raw_df, text_column='content')

        return deduplicated_df if not deduplicated_df.empty else None

    async def _fetch_feed(self, name: str, url: str) -> Optional[List[Dict]]:
        """Fetches and parses a single RSS feed, filtering by date."""
        self.logger.debug(f"Fetching feed '{name}' from {url}")
        try:
            client = self.http_client_factory.get_http_client()
            response = await client.get(url, timeout=self.configs.get('timeout', 20))
            response.raise_for_status()
            
            feed_data = feedparser.parse(response.text)
            if feed_data.bozo:
                self.logger.warning(f"Feed '{name}' ({url}) is malformed. Bozo reason: {feed_data.bozo_exception}")
            
            limit = self.configs.get("params", {}).get("limit_per_feed", 50)
            
            cutoff_date = datetime.now().astimezone() - timedelta(days=self.period_days)
            
            articles = []
            for entry in feed_data.entries[:limit]:
                processed_entry = self._process_entry(entry, name)
                if processed_entry and processed_entry['published_date'] >= cutoff_date:
                    articles.append(processed_entry)
            
            return articles
        except Exception as e:
            self.logger.error(f"Error fetching or parsing feed '{name}' from {url}: {e}")
            return None

    def _process_entry(self, entry: Dict, source_name: str) -> Optional[Dict]:
        """Extracts relevant information from a single RSS entry."""
        published_str = entry.get('published')
        if not published_str:
            return None
            
        try:
            published_date = pd.to_datetime(published_str).astimezone()
        except (ValueError, TypeError):
            self.logger.warning(f"Could not parse date: '{published_str}' for feed '{source_name}'. Skipping entry.")
            return None

        return {
            "title": entry.get('title'),
            "link": entry.get('link'),
            "published_date": published_date,
            "source": source_name,
            "content": entry.get('summary')
        }
