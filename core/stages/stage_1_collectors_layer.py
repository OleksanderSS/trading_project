#!/usr/bin/env python3
"""
Stage 1: Data Collection Layer

This stage's sole responsibility is to produce a clean, deduplicated list of
financially-relevant news articles. It filters out junk and semantic duplicates.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import re
from typing import List
import yaml 
from sentence_transformers import SentenceTransformer, util

# --- Local Imports ---
from collectors.fred_collector import FREDCollector
from collectors.yf_collector import YFCollector
from collectors.rss_collector import RSSCollector
from collectors.google_news_collector import GoogleNewsCollector

from utils.config_manager import get_secret
from utils.cache_utils import CacheManager
from config.config_loader import load_yaml_config
from config.config import START_FINANCIAL, END_FINANCIAL

logger = logging.getLogger(__name__)

# --- Semantic Deduplication Setup ---
try:
    dedup_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("SentenceTransformer model for deduplication loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model: {e}")
    dedup_model = None

def _normalize_date_column(df: pd.DataFrame, col: str = "published_at") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        if hasattr(df[col].dt, "tz") and df[col].dt.tz is not None:
            df[col] = df[col].dt.tz_localize(None)
    return df

def _is_news_relevant(text: str, relevance_keywords: List[str]) -> bool:
    if not isinstance(text, str):
        return False
    pattern = r'\b(' + '|'.join(re.escape(k) for k in relevance_keywords) + r')\b'
    return bool(re.search(pattern, text, re.IGNORECASE))

def _deduplicate_news_semantically(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Removes semantically similar news titles using vector embeddings."""
    if df.empty or 'title' not in df.columns:
        logger.warning("Deduplication skipped: DataFrame is empty or 'title' column missing.")
        return df

    if dedup_model is None:
        logger.warning("Semantic model not loaded. Falling back to title-based deduplication.")
        deduplicated_df = df.drop_duplicates(subset=['title'], keep='first')
        logger.info(f"Fallback deduplication complete. Kept {len(deduplicated_df)} of {len(df)} articles.")
        return deduplicated_df.reset_index(drop=True)

    logger.info(f"Starting semantic deduplication on {len(df)} articles with threshold {threshold}...")
    titles = df['title'].tolist()
    embeddings = dedup_model.encode(titles, convert_to_tensor=True, show_progress_bar=False)
    cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
    df['cluster_id'] = -1
    cluster_id_counter = 0
    for i in range(len(titles)):
        if df.at[i, 'cluster_id'] == -1:
            similar_indices = np.where(cosine_scores[i] > threshold)[0]
            df.loc[similar_indices, 'cluster_id'] = cluster_id_counter
            cluster_id_counter += 1
    deduplicated_df = df.drop_duplicates(subset=['cluster_id'], keep='first')
    logger.info(f"Semantic deduplication complete. Kept {len(deduplicated_df)} unique articles.")
    return deduplicated_df.drop(columns=['cluster_id']).reset_index(drop=True)

def _collect_price_data() -> pd.DataFrame:
    logger.info("Collecting price data for all configured tickers.")
    yf_collector = YFCollector()
    price_df = yf_collector.fetch()
    if price_df.empty:
        logger.error("Price data collection failed critically.")
        return pd.DataFrame()
    logger.info(f"Successfully collected price data. Shape: {price_df.shape}")
    return price_df

def _collect_news_data(cache_path: str, config_path: str) -> pd.DataFrame:
    logger.info("Collecting and processing news data...")
    cache_manager = CacheManager(base_path=cache_path)
    config = load_yaml_config(config_path)

    # --- Gemini's Unified Keyword Loading ---
    unified_keywords = []
    if 'keywords' in config and isinstance(config['keywords'], dict):
        for category, terms in config['keywords'].items():
            if isinstance(terms, list):
                unified_keywords.extend(terms)
            elif isinstance(terms, dict):
                for ticker, ticker_kws in terms.items():
                    unified_keywords.extend(ticker_kws)
    unified_keywords = list(set(map(str, unified_keywords)))
    logger.info(f"Loaded {len(unified_keywords)} unique keywords from YAML for all collectors.")
    # --- End of Unified Keyword Loading ---

    rss_feeds = {k: v for key in ["rss", "rss_main", "rss_alt"] if key in config and isinstance(config[key], dict) for k, v in config[key].items()}
    rss_collector = RSSCollector(rss_feeds=rss_feeds, keywords=unified_keywords)
    
    # Modified by Gemini: Use the same unified keywords for Google News
    google_news_collector = GoogleNewsCollector(keywords=unified_keywords, source_quality_config={}, cache_manager=cache_manager, days_back=60)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_rss = executor.submit(rss_collector.fetch)
        future_google = executor.submit(google_news_collector.fetch)
        news_sources = [future_rss.result(), future_google.result()]

    valid_news_dfs = [df for df in news_sources if isinstance(df, pd.DataFrame) and not df.empty]
    if not valid_news_dfs:
        logger.warning("No news data collected from any source.")
        return pd.DataFrame()

    all_news = pd.concat(valid_news_dfs, ignore_index=True)
    all_news.dropna(subset=["title"], inplace=True)
    all_news = _normalize_date_column(all_news, "published_at")

    # The relevance filter is now implicit in the collectors, but we can keep it as a safeguard.
    all_news['is_relevant'] = all_news.apply(lambda row: _is_news_relevant(f"{row['title']} {row.get('description', '')}", unified_keywords), axis=1)
    relevant_news = all_news[all_news['is_relevant']].copy()
    if len(relevant_news) < len(all_news):
         logger.info(f"{len(relevant_news)} of {len(all_news)} news articles passed the safeguard relevance filter.")

    if relevant_news.empty:
        return pd.DataFrame()

    final_news = _deduplicate_news_semantically(relevant_news)
    logger.info(f"Stage 1 news processing complete. Produced {len(final_news)} final articles.")
    
    required_columns = ['title', 'link', 'published_at', 'source', 'description']
    available_columns = [col for col in required_columns if col in final_news.columns]
    
    if len(available_columns) < len(required_columns):
        missing = set(required_columns) - set(available_columns)
        logger.warning(f"Final news DataFrame is missing expected columns: {missing}.")

    return final_news[available_columns]

def _collect_macro_data(cache_path: str) -> pd.DataFrame:
    logger.info("Collecting FRED data...")
    fred_api_key = get_secret("FRED_API_KEY", "demo")
    fred_collector = FREDCollector(api_key=fred_api_key, start_date=START_FINANCIAL, end_date=END_FINANCIAL, cache_path=cache_path)
    macro_df = fred_collector.fetch_all()
    if not macro_df.empty:
        macro_df = macro_df.ffill().bfill().infer_objects(copy=False)
    logger.info(f"Collected macro data. Shape: {macro_df.shape}")
    return macro_df

def run_stage_1(config_path: str = "config/news_sources.yaml") -> dict:
    logger.info("--- Starting Stage 1: Data Collection & Filtering ---")
    cache_path = "./data/cache/stage_1"

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_prices = executor.submit(_collect_price_data)
        future_news = executor.submit(_collect_news_data, cache_path, config_path)
        future_macro = executor.submit(_collect_macro_data, cache_path)
        
        prices_df = future_prices.result()
        news_df = future_news.result()
        macro_df = future_macro.result()

    logger.info("--- Stage 1: Finished ---")
    return {"prices": prices_df, "news": news_df, "macro": macro_df}
