from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("NewsFilter")

class NewsFilter:
    """Specialized filter for news data with deduplication and quality checks."""

    def __init__(self, config: dict[str, Any]):
        self.min_title_len = config.get('news_title_min_len', 10)
        self.min_content_len = config.get('news_content_min_len', 50)

    def filter_news_data(self, news_data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Intelligently filters news articles."""
        if not isinstance(news_data, pd.DataFrame) or news_data.empty:
            return pd.DataFrame(), {'status': 'empty', 'articles': 0}

        initial_count = len(news_data)

        # 1. Basic length filters
        if 'title' in news_data.columns:
            news_data = news_data[news_data['title'].str.len() >= self.min_title_len]

        # 2. Deduplication
        if 'title' in news_data.columns:
            news_data = news_data.drop_duplicates(subset=['title'])

        return news_data, {
            'status': 'accepted',
            'initial_articles': initial_count,
            'final_articles': len(news_data),
            'removed': initial_count - len(news_data)
        }
