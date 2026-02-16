# collectors/rss_collector.py

from collectors.base_collector import BaseCollector
import feedparser
import pandas as pd
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime, timezone
from dateutil import parser as date_parser

from enrichment.keyword_extractor import KeywordExtractor
from config.config import RSS_FEEDS

logger = logging.getLogger(__name__)

class RSSCollector(BaseCollector):
    def __init__(
        self, 
        db_path: str = ":memory:", 
        table_name: str = "rss_data", 
        feeds: Optional[Dict[str, Dict[str, Any]]] = None, 
        keyword_dict: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ):
        super().__init__(db_path=db_path, table_name=table_name, **kwargs)
        self.feeds = feeds or RSS_FEEDS
        self.keyword_extractor = KeywordExtractor(keyword_dict=keyword_dict)
        logger.info(f"[RSSCollector] Initialized with {self.keyword_extractor.count()} relevant keywords")

    def _parse_date(self, date_string: str) -> Optional[datetime]:
        """Robustly parse date strings into timezone-aware datetime objects."""
        if not date_string:
            return None
        try:
            # Use dateutil.parser for its flexibility
            parsed_date = date_parser.parse(date_string)
            # If the parsed date is naive, assume it's UTC
            if parsed_date.tzinfo is None:
                return parsed_date.replace(tzinfo=timezone.utc)
            # Otherwise, convert it to UTC
            return parsed_date.astimezone(timezone.utc)
        except (ValueError, TypeError) as e:
            logger.warning(f"[RSSCollector] Could not parse date: '{date_string}'. Error: {e}")
            return None

    def fetch_feed(self, name: str, url: str, relevance_threshold: float = 0.5) -> pd.DataFrame:
        try:
            feed = feedparser.parse(url)
            if feed.bozo:
                raise Exception(f"Bozo feed detected: {feed.bozo_exception}")

            entries = []
            for entry in feed.entries:
                relevance = self.keyword_extractor.calculate_relevance(
                    entry.get("title", "") + " " + entry.get("summary", "")
                )
                if relevance['score'] >= relevance_threshold:
                    entries.append({
                        "source": name,
                        "title": entry.get("title"),
                        "link": entry.get("link"),
                        "published_at": self._parse_date(entry.get("published") or entry.get("updated")),
                        "summary": entry.get("summary"),
                        "relevance_score": relevance['score'],
                        "matched_keywords": ",".join(relevance['keywords'])
                    })
            
            logger.info(f"[RSSCollector] {name}: Found {len(entries)} relevant articles out of {len(feed.entries)}.")
            return pd.DataFrame(entries)
        except Exception as e:
            logger.error(f"[RSSCollector] Failed to fetch or parse feed '{name}' from {url}: {e}")
            return pd.DataFrame()

    def fetch(self) -> pd.DataFrame:
        all_articles = []
        for name, feed_info in self.feeds.items():
            df = self.fetch_feed(name, feed_info['url'], feed_info.get('relevance_threshold', 0.5))
            if not df.empty:
                all_articles.append(df)
        
        if not all_articles:
            logger.warning("[RSSCollector] No relevant articles found across all feeds.")
            return pd.DataFrame()

        df_all = pd.concat(all_articles, ignore_index=True)
        df_all.dropna(subset=['published_at'], inplace=True) # Drop articles where date parsing failed
        
        logger.info(f"[RSSCollector] Total relevant records remaining: {len(df_all)}")
        return df_all

    def collect(self) -> pd.DataFrame:
        df = self.fetch()
        if not df.empty:
            # Convert datetime to naive for DB storage if needed
            df_to_save = df.copy()
            df_to_save['published_at'] = df_to_save['published_at'].dt.tz_localize(None)
            self.save(df_to_save.to_dict(orient='records'))
        return df
