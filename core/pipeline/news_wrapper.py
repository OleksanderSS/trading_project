# core/pipeline/news_wrapper.py

from typing import Optional
import pandas as pd
from enrichment.sentiment_analyzer import SentimentEnricher
from utils.news_analysis_tools import QuickNewsAnalyzer
from utils.cache_utils import CacheManager
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("NewsWrapper")

def _process_news(
    df_news: pd.DataFrame,
    cache_manager: CacheManager,
    use_finbert: bool = True,
) -> pd.DataFrame:
    if df_news is None or df_news.empty:
        logger.warning("[WARN] df_news порожнandй, нandчого обробляти")
        return pd.DataFrame()

    # --- Обробка сентименту and кластериforцandї ---
    if use_finbert:
        try:
            enricher = SentimentEnricher()
            df_news = enricher.analyze_sentiment(df_news)
            logger.info("[OK] FinBERT sentiment + clustering + keywords forстосовано")
        except Exception as e:
            logger.warning(f"[WARN] FinBERT failed: {e}, fallback  QuickNewsAnalyzer")
            logger.exception(e)
            analyzer = QuickNewsAnalyzer()
            df_news = analyzer.cluster_and_analyze(df_news)
    else:
        analyzer = QuickNewsAnalyzer()
        df_news = analyzer.cluster_and_analyze(df_news)
        logger.info("[OK] QuickNewsAnalyzer forстосовано")

    # --- Перевandрка ключових колонок ---
    required_cols = ["sentiment", "sentiment_score"]
    for col in required_cols:
        if col not in df_news.columns:
            df_news[col] = None
            logger.warning(f"[NewsWrapper] [WARN] Колонка {col} вandдсутня, додано як None")

    # --- Беwithпечnot кешування DataFrame ---
    try:
        cache_manager.save_pickle("latest_news", df_news)
        logger.info(f" Кеш updated ({len(df_news)} новин)")
        try:
            test_df = cache_manager.load_pickle("latest_news")
            logger.info(f"[NewsWrapper] [SEARCH] Кеш перевandрено, {len(test_df)} рядкandв forванandжено")
        except Exception:
            logger.warning("[NewsWrapper] [WARN] Перевandрка кешу not вдалася")
    except Exception as e:
        logger.error(f"[ERROR] Не вдалося withберегти latest_news: {e}")
        logger.exception(e)

    return df_news


def run_news_pipeline(
    df_news: pd.DataFrame,
    cache_manager: Optional[CacheManager] = None,
    use_finbert: bool = True,
) -> pd.DataFrame:
    if cache_manager is None:
        raise ValueError("[ERROR] cache_manager потрandбен")
    return _process_news(df_news, cache_manager, use_finbert=use_finbert)


def run_news_pipeline_full(
    df_news: pd.DataFrame,
    cache_manager: CacheManager,
    use_finbert: bool = True,
) -> pd.DataFrame:
    return _process_news(df_news, cache_manager, use_finbert=use_finbert)