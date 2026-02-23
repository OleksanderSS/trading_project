#!/usr/bin/env python3
"""
Stage 1: Data Collection Layer

This stage is responsible for collecting raw data from various sources.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# --- Local Imports ---
from collectors.fred_collector import FREDCollector
from collectors.yf_collector import YFCollector
from collectors.rss_collector import RSSCollector
from collectors.google_news_collector import GoogleNewsCollector

from utils.config_manager import get_secret
from utils.cache_utils import CacheManager
from config.config_loader import load_yaml_config
from config.source_quality import load_source_quality_config, get_google_news_keywords
from config.tickers_config import TICKERS, KEYWORDS
from config.config import (
    PATHS, USE_MEMORY_DB, TIME_FRAMES, YF_MAX_PERIODS, DATA_INTERVALS,
    START_FINANCIAL, END_FINANCIAL, START_NEWS_INTRADAY, END_NEWS_INTRADAY
)

logger = logging.getLogger(__name__)

def _normalize_date_column(df: pd.DataFrame, col: str = "published_at") -> pd.DataFrame:
    """Ensures a column is a timezone-naive datetime object."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        if hasattr(df[col].dt, "tz") and df[col].dt.tz is not None:
            df[col] = df[col].dt.tz_localize(None)
    return df

def _extract_ticker_from_text(text: str, tickers: list) -> str or None:
    """Extracts the first ticker found in the text."""
    if not isinstance(text, str):
        return None
    for ticker in tickers:
        # Use word boundaries to avoid matching parts of words
        if pd.Series(text).str.contains(f'\\b{ticker}\\b', case=False, regex=True).any():
            return ticker
    return None

def _collect_price_data(tickers: list) -> pd.DataFrame:
    """Collects raw OHLCV price data."""
    logger.info(f"Collecting price data for tickers: {tickers}")
    db_path = PATHS["db"] if not USE_MEMORY_DB else ":memory:"

    timeframes_with_dates = {}
    for tf in TIME_FRAMES:
        period = YF_MAX_PERIODS.get(tf)
        interval = DATA_INTERVALS.get(tf)
        start_date = (datetime.now() - timedelta(days=60)) if tf in ["15m", "60m"] else START_FINANCIAL
        end_date = datetime.now() if tf in ["15m", "60m"] else END_FINANCIAL
        timeframes_with_dates[tf] = {"period": period, "interval": interval, "start_date": start_date, "end_date": end_date}

    yf_collector = YFCollector(tickers=tickers, timeframes=timeframes_with_dates, db_path=db_path)
    price_df = yf_collector.fetch()

    if price_df.empty:
        return pd.DataFrame()

    price_df.ffill(inplace=True)
    price_df.reset_index(drop=True, inplace=True)
    
    if "date" in price_df.columns and "datetime" not in price_df.columns:
        price_df.rename(columns={"date": "datetime"}, inplace=True)

    price_df = _normalize_date_column(price_df, "datetime")
    logger.info(f"Successfully collected price data. Shape: {price_df.shape}")
    return price_df

def _collect_news_data(tickers: list, keyword_dict: dict, rss_feeds: dict, cache_path: str) -> pd.DataFrame:
    """Collects and aggregates news from RSS and Google News."""
    logger.info("Collecting news data...")
    cache_manager = CacheManager(base_path=cache_path)
    
    rss_collector = RSSCollector(rss_feeds=rss_feeds, keyword_dict=keyword_dict, cache_manager=cache_manager)
    
    source_quality_config = load_source_quality_config()
    google_keywords_config = get_google_news_keywords()
    all_google_keywords = [kw for kws in google_keywords_config.values() for kw in kws]

    google_news_collector = GoogleNewsCollector(keywords=all_google_keywords, source_quality_config=source_quality_config, cache_manager=cache_manager, days_back=60)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_rss = executor.submit(rss_collector.fetch, start_date=START_NEWS_INTRADAY, end_date=END_NEWS_INTRADAY)
        future_google = executor.submit(google_news_collector.fetch)
        news_sources = [future_rss.result(), future_google.result()]

    valid_news_dfs = [df for df in news_sources if isinstance(df, pd.DataFrame) and not df.empty]
    if not valid_news_dfs:
        logger.warning("No news data collected.")
        return pd.DataFrame()

    all_news = pd.concat(valid_news_dfs, ignore_index=True)
    all_news = _normalize_date_column(all_news, "published_at")

    # --- Ticker Extraction --- 
    all_news['ticker'] = all_news['title'].apply(lambda x: _extract_ticker_from_text(x, tickers))
    logger.info(f"Assigned tickers to {all_news['ticker'].notna().sum()} news articles.")

    if 'title' in all_news.columns:
        all_news.sort_values('published_at', ascending=True, inplace=True)
        all_news.drop_duplicates(subset=['title'], keep='first', inplace=True)

    all_news['description'] = all_news.get('description', pd.Series(dtype=str)).fillna("")

    logger.info(f"Deduplicated news. Final shape: {all_news.shape}")
    return all_news

def _collect_macro_data(cache_path: str) -> pd.DataFrame:
    """Collects macroeconomic data from FRED."""
    logger.info("Collecting FRED data...")
    fred_api_key = get_secret("FRED_API_KEY", "demo")
    fred_collector = FREDCollector(api_key=fred_api_key, start_date=START_FINANCIAL, end_date=END_FINANCIAL, cache_path=cache_path)
    macro_df = fred_collector.fetch_all()
    if not macro_df.empty:
        macro_df = macro_df.ffill().bfill().infer_objects(copy=False)
    logger.info(f"Collected macro data. Shape: {macro_df.shape}")
    return macro_df

def run_stage_1(config_path: str = "config/news_sources.yaml") -> dict:
    """Executes the data collection stage."""
    logger.info("--- Starting Stage 1: Data Collection ---")
    cache_path = "./data/cache/stage_1"
    config = load_yaml_config(config_path)
    tickers = list(TICKERS.keys())
    rss_feeds = {k: v for key in ["rss", "rss_main", "rss_alt"] if key in config and isinstance(config[key], dict) for k, v in config[key].items()}

    if not tickers:
        logger.error("No tickers found. Aborting.")
        return {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_prices = executor.submit(_collect_price_data, tickers)
        future_news = executor.submit(_collect_news_data, tickers, KEYWORDS, rss_feeds, cache_path)
        future_macro = executor.submit(_collect_macro_data, cache_path)
        prices_df = future_prices.result()
        news_df = future_news.result()
        macro_df = future_macro.result()

    logger.info("--- Stage 1: Finished ---")
    return {"prices": prices_df, "news": news_df, "macro": macro_df}
