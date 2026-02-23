# utils/keyword_features.py

import pandas as pd
from typing import List, Tuple
from utils.logger import ProjectLogger


logger = ProjectLogger.get_logger("TradingProjectLogger")

def build_keyword_features(
    df_news: pd.DataFrame,
    keyword_list: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Створює фandчand на основand withбandгandв keywords у новинах:
    - match_count: кandлькandсть withбandгandв у кожному рядку
    - keyword_match_count: агрегована кandлькandсть withгадок на whereнь
    - keyword_density: нормалandwithована andнтенсивнandсть withгадок

    Поверandє:
    - df_news with колонкою match_count
    - df_daily with колонками keyword_match_count, news_count, keyword_density
    """

    df_news = df_news.copy()

    if "published_at" not in df_news.columns:
        raise ValueError("[KeywordFeatures] [ERROR] df_news not має колонки 'published_at'")
    if "description" not in df_news.columns:
        raise ValueError("[KeywordFeatures] [ERROR] df_news not має колонки 'description'")

    df_news["published_at"] = pd.to_datetime(df_news["published_at"], errors="coerce")
    df_news = df_news.dropna(subset=["published_at"])
    df_news["date"] = df_news["published_at"].dt.date

    # [SEARCH] Пошук withбandгandв по словнику
    def count_matches(text: str) -> int:
        if not isinstance(text, str):
            return 0
        return sum(kw.lower() in text.lower() for kw in keyword_list)

    df_news["match_count"] = df_news["description"].apply(count_matches)

    # [DATA] Агрегування по днях
    df_daily = df_news.groupby("date").agg({
        "match_count": "sum",
        "description": "count"
    }).rename(columns={
        "match_count": "keyword_match_count",
        "description": "news_count"
    }).reset_index()

    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.set_index("date")

    #  Додати keyword_density
    df_daily["keyword_density"] = df_daily["keyword_match_count"] / df_daily["news_count"].replace(0, 1)

    logger.info(f"[KeywordFeatures] [OK] Побудовано оwithнаки: {df_daily.shape[0]} днandв, {df_news.shape[0]} новин")

    return df_news, df_daily