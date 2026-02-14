# core/pipeline/news_pipeline.py

import pandas as pd
import time
from typing import Optional, Dict, List, Tuple, Any
from collectors.news_collector import NewsCollector
from enrichment.sentiment_analyzer import SentimentEnricher
from enrichment.keyword_extractor import KeywordExtractor
from enrichment.summarizer import Summarizer
from utils.cache_utils import CacheManager
from utils.logger import ProjectLogger
from utils.sentiment.news_score import compute_news_score
from utils.sentiment.thresholds import compute_adaptive_sentiment_thresholds, SENTIMENT_THRESHOLDS
from utils.news_processing import unify_news
from utils.keyword_features import build_keyword_features
from config.config_loader import load_yaml_config
from config.config import PATHS
from datetime import timedelta, datetime


logger = ProjectLogger.get_logger("TradingProjectLogger")


def _collect_news_sources(
        collector: NewsCollector,
        tickers: List[str] = [],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        rss_feeds: Optional[Dict] = None,
        api_data: Optional[List] = None,
        gdelt_data: Optional[List] = None,
        batch_seconds: float = 1.0
) -> pd.DataFrame:
    """Внутрandшня функцandя for withбору новин with рandwithних джерел andwith дроселюванням."""
    all_dfs = []

    fetchers: List[Tuple[Any, str]] = []
    if rss_feeds:
        fetchers.append((lambda: collector.collect(tickers=tickers, start_date=start_date, end_date=end_date), "RSS"))
    if api_data:
        fetchers.append((lambda: collector.collect_api(api_data,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date),
            "API"))
    if gdelt_data:
        fetchers.append((lambda: collector.collect_gdelt(gdelt_data,
            start_date=start_date,
            end_date=end_date),
            "GDELT"))

    if not fetchers:
        logger.warning("[NewsPipeline] [WARN] Жодnot джерело новин not активоваnot (порожнand rss/api/gdelt data).")
        return pd.DataFrame()

    for func, name in fetchers:
        try:
            df = func()
            if df is not None and not df.empty:
                logger.info(f"[News] [OK] {name}  {len(df)} новин withandбрано")
                all_dfs.append(df)
            else:
                logger.warning(f"[News] [WARN] {name} not повернуло новин")
        except Exception:
            logger.exception(f"[News] [ERROR] {name} fetch crashed")

        if batch_seconds > 0:
            time.sleep(batch_seconds)

    if not all_dfs:
        logger.warning("[News] [WARN] Жодnot джерело not повернуло новин пandсля дроселювання")

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def fetch_and_process_news(
        rss_feeds: Optional[Dict] = None,
        api_data: Optional[List] = None,
        gdelt_data: Optional[List] = None,
        cache_manager: Optional[CacheManager] = None,
        sentiment_enricher: Optional[SentimentEnricher] = None,
        keyword_extractor: Optional[KeywordExtractor] = None,
        summarizer: Optional[Summarizer] = None,
        use_cache: bool = True,
        force_refresh: bool = False,
        sentiment_batch_size: int = 16,
        max_news: int = 5000,
        skip_cluster: bool = False,
        keyword_dict: Optional[Dict[str, List[str]]] = None,
        tickers: List[str] = [],
        start_date: Any = None,
        end_date: Any = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Dict[str, float]]:
    #  Конверandцandя дат
    start_dt = pd.to_datetime(start_date, errors="coerce") if start_date else None
    end_dt = pd.to_datetime(end_date, errors="coerce") if end_date else None

    # --- 1. Інandцandалandforцandя ---
    cache_manager = cache_manager or CacheManager()
    news_config_path = PATHS.get("news_config", "config/news_sources.yaml")
    keyword_dict = keyword_dict or load_yaml_config(news_config_path).get("keywords", {})
    flat_keywords = [kw for group in keyword_dict.values() for kw in group]

    keyword_extractor = keyword_extractor or KeywordExtractor(flat_keywords)
    sentiment_enricher = sentiment_enricher or SentimentEnricher(cache_manager=cache_manager, keyword_dict=keyword_dict)
    summarizer = summarizer or Summarizer()

    collector = NewsCollector(
        db_path=PATHS.get("news_db", "db/news.db"),
        table_name="news",
        strict=False,
        keyword_dict=keyword_dict,
        rss_feeds=rss_feeds,
        cache_manager=cache_manager
    )

    # --- 2. Збandр новин ---
    cached_news = cache_manager.get_df("news.parquet") if use_cache and not force_refresh else pd.DataFrame()
    new_news = _collect_news_sources(collector, rss_feeds, api_data, gdelt_data, tickers, start_dt, end_dt)
    combined_news = pd.concat([cached_news, new_news], ignore_index=True) if not cached_news.empty else new_news
    combined_news = unify_news(combined_news)

    if combined_news.empty:
        logger.warning("[NewsPipeline] [ERROR] Новини not withandбранand or вandдсутнand.")
        return pd.DataFrame(), pd.DataFrame(), {}, SENTIMENT_THRESHOLDS.copy()

    # --- 3. Гарантandя ключових колонок ---
    for col in ['link', 'title', 'published_at', 'content', 'summary']:
        if col not in combined_news.columns:
            combined_news[col] = ''

    combined_news = combined_news.drop_duplicates(subset=['link', 'title', 'published_at'], keep='first')
    combined_news['published_at'] = pd.to_datetime(combined_news['published_at'],
        errors='coerce').fillna(pd.to_datetime('now'))
    combined_news.sort_values('published_at', ascending=False, inplace=True)
    combined_news = combined_news.head(max_news)

    # --- 4. Фandльтрацandя по даandх ---
    if start_dt:
        combined_news = combined_news[combined_news["published_at"] >= start_dt]
    if end_dt:
        combined_news = combined_news[combined_news["published_at"] <= end_dt]
    if not start_dt:
        one_year_ago = pd.to_datetime('now') - timedelta(days=365)
        combined_news = combined_news[combined_news["published_at"] >= one_year_ago]

    # --- 5. Keywords ---
    combined_news, _ = build_keyword_features(combined_news, flat_keywords + keyword_extractor.tickers)
    combined_news = combined_news[combined_news["match_count"] > 0]

    cache_manager.set_df("news.parquet", combined_news)

    # --- 6. Сентимент ---
    combined_news['keywords'] = combined_news['content'].fillna('').apply(keyword_extractor.extract_keywords)
    combined_news = sentiment_enricher.analyze_sentiment(df=combined_news)

    sentiment_series = combined_news.get('sentiment',
        pd.Series(['neutral'] * len(combined_news),
        index=combined_news.index))
    combined_news['score'] = combined_news.get('sentiment_score', 0.0)

    combined_news['positive'] = (sentiment_series == 'positive').astype(float)
    combined_news['negative'] = (sentiment_series == 'negative').astype(float)
    combined_news['neutral'] = (sentiment_series == 'neutral').astype(float)

    combined_news['news_score'] = combined_news.apply(
        lambda row: compute_news_score(
            sentiment={'positive': row['positive'], 'negative': row['negative'], 'neutral': row['neutral']},
            keywords=row.get('keywords', [])
        ),
        axis=1
    )

    # --- 7. Агрегування ---
    combined_news['date'] = combined_news['published_at'].dt.normalize().dt.tz_convert(None)
    news_daily = combined_news.groupby('date').agg(
        news_score=('news_score', 'mean'),
        sentiment_score=('score', 'mean'),
        avg_news_lag=('published_at', lambda x: (x.max() - x.min()).total_seconds() / 3600 if len(x) > 1 else 0),
        news_count=('title', 'count'),
        positive_count=('positive', 'sum'),
        negative_count=('negative', 'sum'),
    ).reset_index()

    # --- 8. Середня тональнandсть and пороги ---
    avg_sentiment = {k: combined_news[k].mean() for k in ['positive', 'negative', 'neutral'] if k in combined_news}
    total = sum(avg_sentiment.values())
    avg_sentiment = {k: v / total for k, v in avg_sentiment.items()} if total > 0 else {'positive': 0.0,
                                                                                        'negative': 0.0, 'neutral': 1.0}

    try:
        sentiment_thresholds = compute_adaptive_sentiment_thresholds(combined_news)
    except Exception:
        sentiment_thresholds = SENTIMENT_THRESHOLDS.copy()
        logger.exception("[News] [ERROR] Thresholds calc failed, використовуються сandндартнand.")

    logger.info(f"[NewsPipeline] [OK] Завершено processing {len(combined_news)} новин")
    return combined_news, news_daily, avg_sentiment, sentiment_thresholds