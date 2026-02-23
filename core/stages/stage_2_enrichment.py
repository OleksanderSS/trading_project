#!/usr/bin/env python3
"""Оптимізований Stage 2 Enrichment з кешуванням та покращеною логікою"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

sys.path.append('c:/trading_project')

# Імпортуємо існуючі модулі
from config.config import TICKERS, TIME_FRAMES, PATHS
from utils.advanced_features import add_pre_post_features, add_event_features, add_context_features

# 🎯 НОВІ ІМПОРТИ - рефакторинг логіки
from utils.feature_engineering import create_feature_engineering_utils
from utils.data_versioning import create_data_versioning

# core/stages/stage_2_enrichment.py - Multi-Timeframe RSI Context Processing

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Імпортуємо новий модуль для мульти-таймфрейм RSI
from core.stages.stage_2_multi_tf_rsi import create_multi_tf_rsi_dataset, validate_rsi_coverage

from enrichment.roberta_sentiment import RobertaSentimentAnalyzer
from utils.trading_days import get_previous_trading_days, get_previous_trading_sessions, get_us_holidays

logger = logging.getLogger(__name__)

# Структура директорій
CACHE_DIR = "data/cache"
PROCESSED_DIR = "data/processed"

def setup_directories():
    """Створює структуру директорій"""
    for directory in [CACHE_DIR, PROCESSED_DIR]:
        os.makedirs(directory, exist_ok=True)
        os.makedirs(os.path.join(directory, "news"), exist_ok=True)
        os.makedirs(os.path.join(directory, "prices"), exist_ok=True)
        os.makedirs(os.path.join(directory, "macro"), exist_ok=True)
        os.makedirs(os.path.join(directory, "insider"), exist_ok=True)

def get_cache_path(data_type, date_str):
    """Повертає шлях до кешованого файлу"""
    return os.path.join(CACHE_DIR, f"{data_type}_{date_str}.parquet")

def get_processed_path(data_type, date_str):
    """Повертає шлях до обробленого файлу"""
    return os.path.join(PROCESSED_DIR, f"{data_type}_{date_str}.parquet")

def is_cache_valid(cache_path, max_age_hours=24):
    """Перевіряє, чи кеш дійсний"""
    if not os.path.exists(cache_path):
        return False
    
    file_age = time.time() - os.path.getmtime(cache_path)
    return file_age < (max_age_hours * 3600)

def run_stage_2_enrich_optimized(stage1_data, keyword_dict, tickers=TICKERS, time_frames=TIME_FRAMES, mode="train", min_gap_abs_percent=0.5):
    logger.info("[Stage2] 🚀 Starting optimized Stage 2 enrichment with refactored logic...")
    
    feature_utils = create_feature_engineering_utils()
    data_versioning = create_data_versioning()
    
    logger.info("[Stage2] 📦 Unpacking and validating Stage 1 data...")
    raw_news = stage1_data.get("all_news", pd.DataFrame())
    price_df = stage1_data.get("prices", pd.DataFrame())
    macro_df = stage1_data.get("macro", pd.DataFrame())
    insider_df = stage1_data.get("insider", pd.DataFrame())
    
    if raw_news.empty:
        logger.warning("[Stage2] ⚠️ No news data available!")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    if price_df.empty:
        logger.error("[Stage2] ❌ No price data available!")
        return raw_news, pd.DataFrame(), {}
        
    logger.info("[Stage2] 🔧 Creating features and targets using utility...")
    
    try:
        merged_price_data = _create_merged_price_data(price_df, tickers.keys(), time_frames.keys())
        
        enhanced_data = feature_utils.create_all_features_and_targets(
            df=merged_price_data,
            price_col='close',
            tickers=tickers.keys(),
            timeframes=time_frames.keys(),
            include_technical=True,
            include_targets=True
        )
        
        validation_results = feature_utils.validate_features(enhanced_data)
        logger.info(f"[Stage2] ✅ Feature validation: {validation_results}")
        
    except Exception as e:
        logger.error(f"[Stage2] ❌ Error creating features: {e}")
        return raw_news, price_df, {}

    logger.info("[Stage2] 🔍 Checking data versioning...")
    
    try:
        version_info = data_versioning.register_file(
            file_path="data/stages/enhanced_features.parquet",
            data_type="technical_indicators",
            description="Enhanced features and targets",
            metadata=validation_results
        )
        
        is_fresh, fresh_info = data_versioning.is_file_fresh(
            "data/stages/enhanced_features.parquet",
            "technical_indicators"
        )
        
        if not is_fresh:
            logger.warning(f"[Stage2] ⚠️ Data not fresh: {fresh_info['reason']}")
        else:
            logger.info("[Stage2] ✅ Data is fresh")
            
    except Exception as e:
        logger.warning(f"[Stage2] ⚠️ Versioning check failed: {e}")

    logger.info("[Stage2] 📊 Preparing standardized results...")
    
    final_result = {
        'enhanced_features': enhanced_data,
        'validation_results': validation_results,
        'version_info': version_info if 'version_info' in locals() else {},
        'data_quality': {
            'news_rows': len(raw_news),
            'price_rows': len(price_df),
            'feature_rows': len(enhanced_data),
            'feature_columns': len(enhanced_data.columns),
            'target_columns': len(validation_results.get('target_columns', [])),
            'technical_columns': len(validation_results.get('technical_columns', []))
        }
    }
    
    logger.info(f"[Stage2] ✅ Stage 2 completed: {enhanced_data.shape}")
    
    return raw_news, enhanced_data, final_result

def _create_merged_price_data(price_df: pd.DataFrame, tickers: List[str], time_frames: List[str]) -> pd.DataFrame:
    logger.info("[Stage2] 🔗 Creating merged price data...")
    
    all_ticker_data = []
    for ticker in tickers:
        ticker_data_tf = []
        for timeframe in time_frames:
            mask = (price_df['ticker'] == ticker) & (price_df['interval'] == timeframe)
            tf_data = price_df[mask].set_index('date').copy()
            tf_data = tf_data.add_suffix(f'_{timeframe}')
            ticker_data_tf.append(tf_data)
        
        if ticker_data_tf:
            # Resample and merge all timeframes for a single ticker
            resampled_data = [d.resample('min').asfreq() for d in ticker_data_tf]
            merged_ticker_data = pd.concat(resampled_data, axis=1)
            merged_ticker_data = merged_ticker_data.add_prefix(f'{ticker}_')
            all_ticker_data.append(merged_ticker_data)

    if all_ticker_data:
        # Combine all tickers
        result = pd.concat(all_ticker_data, axis=1)
        # Forward fill to propagate values
        result = result.ffill()
        logger.info(f"[Stage2] ✅ Merged data created: {result.shape}")
        return result.reset_index()
    else:
        logger.warning("[Stage2] ⚠️ No data to merge")
        return pd.DataFrame()

def run_enrichment_pipeline(stage1_data, tickers, time_frames):
    raw_news = stage1_data.get("all_news", pd.DataFrame())
    price_df = stage1_data.get("prices", pd.DataFrame())
    macro_df = stage1_data.get("macro", pd.DataFrame())
    insider_df = stage1_data.get("insider", pd.DataFrame())

    if raw_news.empty or price_df.empty:
        logger.error("[Stage2] ❌ No news or price data available!")
        return pd.DataFrame(), {}

    logger.info("[Stage2] 🚀 Starting Enrichment Pipeline...")

    # 1. Create Multi-TF RSI Dataset
    multi_tf_df = create_multi_tf_rsi_dataset(raw_news, price_df, warmup_periods=30)
    if multi_tf_df.empty:
        return raw_news, {}

    # 2. Add Trading Date
    try:
        multi_tf_df['trade_date'] = get_previous_trading_days(multi_tf_df['published_at'])
    except Exception as e:
        logger.error(f"[Stage2] ❌ Error adding trade_date: {e}")
        multi_tf_df['trade_date'] = pd.NaT

    # 3. Sentiment Analysis
    logger.info("[Stage2] 🤖 Step 3: Simple sentiment analysis...")
    try:
        unique_titles = multi_tf_df['title'].fillna(multi_tf_df['description'].fillna('')).drop_duplicates().tolist()
        if unique_titles:
            from enrichment.simple_sentiment import SimpleSentimentAnalyzer
            analyzer = SimpleSentimentAnalyzer()
            sentiment_results = analyzer.analyze_batch(unique_titles)
            sentiment_map = dict(zip(unique_titles, sentiment_results))
            multi_tf_df['sentiment_score'] = multi_tf_df['title'].fillna(multi_tf_df['description'].fillna('')).map(sentiment_map)
            logger.info(f"[Stage2] ✅ Simple sentiment analysis completed for {len(sentiment_results)} titles")
        else:
            logger.warning("[Stage2] ⚠️ No titles for sentiment analysis")
            multi_tf_df['sentiment_score'] = 0.0
    except Exception as e:
        logger.error(f"[Stage2] ❌ Error in sentiment analysis: {e}")
        multi_tf_df['sentiment_score'] = 0.0

    # 4. Merge Macro and Insider Data
    try:
        if not macro_df.empty:
            multi_tf_df['date'] = pd.to_datetime(multi_tf_df['published_at']).dt.date
            macro_df['date'] = pd.to_datetime(macro_df['date']).dt.date
            multi_tf_df = pd.merge(multi_tf_df, macro_df, on='date', how='left')
        if not insider_df.empty:
            insider_df['date'] = pd.to_datetime(insider_df['date']).dt.date
            multi_tf_df = pd.merge(multi_tf_df, insider_df, on='date', how='left')
    except Exception as e:
        logger.error(f"[Stage2] ❌ Error merging macro/insider data: {e}")

    # Final validation and save
    logger.info("[Stage2] 🔍 Final validation and saving...")
    # ... (add validation logic from the original file)

    output_path = os.path.join(PROCESSED_DIR, "multi_tf_enriched_news.parquet")
    try:
        object_cols = multi_tf_df.select_dtypes(include=['object']).columns
        for col in object_cols:
            multi_tf_df[col] = multi_tf_df[col].fillna('').astype(str)
        multi_tf_df.to_parquet(output_path, index=False)
        logger.info(f"[Stage2] ✅ Results saved to: {output_path}")
    except Exception as e:
        logger.error(f"[Stage2] ❌ Error saving results: {e}")

    return raw_news, multi_tf_df
