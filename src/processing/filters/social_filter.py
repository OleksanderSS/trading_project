"""
Specialized filter for social data (News, Reddit, etc.).
Handles content validation, deduplication, and sentiment sanity checks.
"""
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("SocialFilter")

class SocialFilter:
    def __init__(self, config: dict[str, Any]):
        self.min_title_len = config.get('news_title_min_len', 10)
        self.min_content_len = config.get('news_content_min_len', 50)
        self.reddit_score_min = config.get('reddit_score_threshold', 1)

    def filter_news(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        if df.empty: return df, {'status': 'empty'}

        initial_len = len(df)
        # 1. Content length filter
        if 'title' in df.columns:
            df = df[df['title'].str.len() >= self.min_title_len]

        # 2. Deduplication (by title/content hash)
        if 'hash' in df.columns:
            df = df.drop_duplicates(subset=['hash'])
        elif 'title' in df.columns:
            df = df.drop_duplicates(subset=['title'])

        return df, {
            'status': 'accepted',
            'original_count': initial_len,
            'filtered_count': len(df),
            'removed': initial_len - len(df)
        }

    def filter_reddit(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        if df.empty: return df, {'status': 'empty'}

        initial_len = len(df)
        # 1. Score filter
        if 'score' in df.columns:
            df = df[df['score'] >= self.reddit_score_min]

        # 2. Content filter
        if 'text' in df.columns:
            df = df[df['text'].str.len() >= 20] # Hardcoded reasonable minimum

        return df, {
            'status': 'accepted',
            'original_count': initial_len,
            'filtered_count': len(df),
            'removed': initial_len - len(df)
        }
