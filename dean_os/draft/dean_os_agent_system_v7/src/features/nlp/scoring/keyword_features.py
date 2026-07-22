# src/feature_engineering/nlp/keyword_features.py

from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("TradingProjectLogger")

class KeywordExtractor:
    """
    Extracts keywords from text based on a provided dictionary or list.
    """
    def __init__(self, keyword_config: Any):
        if isinstance(keyword_config, dict):
            self.keywords = [kw.lower() for sublist in keyword_config.values() for kw in sublist]
        elif isinstance(keyword_config, list):
            self.keywords = [kw.lower() for kw in keyword_config]
        else:
            self.keywords = []

        self.keyword_set = set(self.keywords)

    def extract_keywords(self, text: str) -> list[str]:
        """
        Finds all unique keywords from the config that are present in the text.
        """
        if not isinstance(text, str) or not self.keywords:
            return []

        text_lower = text.lower()
        found_keywords = {kw for kw in self.keyword_set if kw in text_lower}
        return list(found_keywords)

def build_keyword_features(
    df_news: pd.DataFrame,
    keyword_list: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates features based on keyword matches in news:
    - match_count: number of matches in each row
    - keyword_match_count: aggregated number of mentions per day
    - keyword_density: normalized intensity of mentions

    Returns:
    - df_news with match_count column
    - df_daily with keyword_match_count, news_count, keyword_density columns
    """
    df_news = df_news.copy()

    if "published_at" not in df_news.columns:
        raise ValueError("[KeywordFeatures] [ERROR] df_news does not have 'published_at' column")
    if "description" not in df_news.columns:
        raise ValueError("[KeywordFeatures] [ERROR] df_news does not have 'description' column")

    df_news["published_at"] = pd.to_datetime(df_news["published_at"], errors="coerce")
    df_news = df_news.dropna(subset=["published_at"])
    df_news["date"] = df_news["published_at"].dt.date

    # [SEARCH] Find matches using a simple count
    def count_matches(text: str) -> int:
        if not isinstance(text, str):
            return 0
        return sum(kw.lower() in text.lower() for kw in keyword_list)

    df_news["match_count"] = df_news["description"].apply(count_matches)

    # [DATA] Aggregate by day
    df_daily = df_news.groupby("date").agg({
        "match_count": "sum",
        "description": "count"
    }).rename(columns={
        "match_count": "keyword_match_count",
        "description": "news_count"
    }).reset_index()

    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.set_index("date")

    # Add keyword_density
    df_daily["keyword_density"] = df_daily["keyword_match_count"] / df_daily["news_count"].replace(0, 1)

    logger.info(f"[KeywordFeatures] [OK] Built features: {df_daily.shape[0]} days, {df_news.shape[0]} news")

    return df_news, df_daily
