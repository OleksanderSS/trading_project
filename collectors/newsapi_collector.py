#!/usr/bin/env python3
"""NewsAPI Collector"""

import logging
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Any
import yaml
import os
import requests

# Внутрішні імпорти
from collectors.base_collector import BaseCollector
from utils.news_utils import expand_news_to_all_tickers

# Видалено NEWS_API_CONFIG з імпорту
from config.config import TICKERS, PATHS

logger = logging.getLogger(__name__)

class NewsAPICollector(BaseCollector):
    """Збирач новин NewsAPI"""

    def __init__(self, api_key: str, queries: List[str], start_date: str, end_date: str, keyword_dict: dict, cache_path: str):
        super().__init__(db_path=os.path.join(cache_path, "newsapi.db"), table_name="news")
        self.api_key = api_key
        self.queries = queries
        self.start_date = start_date
        self.end_date = end_date
        self.keyword_dict = keyword_dict

    def fetch(self) -> pd.DataFrame:
        """Public method to fetch data, which calls the internal collect method."""
        return self.collect()

    def collect(self) -> pd.DataFrame:
        """Collects news from NewsAPI."""
        all_articles = []
        for query in self.queries:
            try:
                response = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": query,
                        "from": self.start_date,
                        "to": self.end_date,
                        "apiKey": self.api_key,
                        "language": "en",
                        "sortBy": "publishedAt"
                    }
                )
                response.raise_for_status()
                articles = response.json().get("articles", [])
                all_articles.extend(articles)
                # Sleep to avoid hitting rate limits
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching news from NewsAPI for query '{query}': {e}")

        if not all_articles:
            return pd.DataFrame()

        df = pd.DataFrame(all_articles)
        df['published_at'] = pd.to_datetime(df['publishedAt'])
        df['source'] = df['source'].apply(lambda x: x['name'] if isinstance(x, dict) else x)
        df = df[["published_at", "title", "description", "source", "url"]]

        # Expand news to all tickers based on keywords
        expanded_df = expand_news_to_all_tickers(df, self.keyword_dict)
        return expanded_df