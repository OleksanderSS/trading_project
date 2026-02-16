#!/usr/bin/env python3
"""
Stage 2: Data Factory (v7 - Context-Aware)

This version enhances the Data Factory by enriching the final dataset with two
new classes of features, as per the modular research plan:

1.  **Granular Features:** Simple, universal indicators calculated for each ticker,
    such as basic price momentum (`feature_close_momentum`).
2.  **Contextual Features:** High-level indicators describing the overall market
    state. These are prefixed with `context_` and are intended exclusively for
    the meta-learning stage (`run_context_analysis.py`).
"""

import logging
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
import pandas_ta as ta
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from typing import Optional, Dict, Tuple
from itertools import product

# --- Model Imports ---
from enrichment.roberta_sentiment import RobertaSentimentAnalyzer

# --- Local Imports ---
from utils.cache_utils import CacheManager
from config.config import TIME_FRAMES
from config.tickers_config import TICKERS
from config.adaptive_targets import AdaptiveTargetsSystem, TimeframeType
from config.technical_config import TECHNICAL_INDICATORS

# --- Setup & Configuration ---
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

@lru_cache(maxsize=1)
def get_enrichment_models() -> Tuple[RobertaSentimentAnalyzer, None]:
    logger.info("Initializing and loading enrichment models (Sentiment)...")
    sentiment_analyzer = RobertaSentimentAnalyzer()
    logger.info("Enrichment models loaded successfully.")
    return sentiment_analyzer, None

@lru_cache(maxsize=1)
def get_market_calendar(start_date, end_date):
    nyse = mcal.get_calendar('NYSE')
    return nyse.schedule(start_date=start_date.date(), end_date=end_date.date())

def _enrich_news_with_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty: return news_df
    sentiment_analyzer, _ = get_enrichment_models()
    texts = (news_df['title'].fillna('') + " - " + news_df.get('description', pd.Series(dtype=str)).fillna('')).tolist()
    with ThreadPoolExecutor() as executor:
        sentiment_results = list(executor.map(sentiment_analyzer.predict, texts))
    sentiment_df = pd.DataFrame(sentiment_results, index=news_df.index, columns=['label', 'score'])
    news_df['sentiment_label'] = sentiment_df['label']
    news_df['sentiment_score'] = sentiment_df['score']
    return news_df

def get_effective_event_time(publish_time: pd.Timestamp, market_schedule: pd.DataFrame) -> Optional[pd.Timestamp]:
    if publish_time.tzinfo is None: publish_time = publish_time.tz_localize('UTC')
    next_market_opens = market_schedule[market_schedule['market_open'] > publish_time]
    return next_market_opens.iloc[0]['market_open'] if not next_market_opens.empty else None

def _calculate_technical_indicators(price_data: pd.DataFrame) -> pd.DataFrame:
    """Calculates standard TA indicators and new granular features."""
    # Standard TA-Lib/Pandas-TA indicators
    all_indicators = TECHNICAL_INDICATORS['base'] + TECHNICAL_INDICATORS['short_term'] + TECHNICAL_INDICATORS['long_term']
    strategy = ta.Strategy(name="Research_Strategy", ta=all_indicators)
    price_data.ta.strategy(strategy)
    
    # New Granular Features
    price_data['feature_close_momentum'] = (price_data['close'] > price_data['close'].shift(1)).astype(int)
    
    return price_data

def _process_event_pair(
    pair_data: Dict,
    price_groups_with_ta: Dict,
    market_schedule: pd.DataFrame,
    target_system: AdaptiveTargetsSystem
) -> Optional[Dict]:
    news_item = pair_data['news']
    ticker = pair_data['ticker']

    effective_time = get_effective_event_time(news_item['published_at'], market_schedule)
    if not effective_time: return None

    feature_row = {
        'event_time': effective_time,
        'news_published_at': news_item['published_at'],
        'news_sentiment_label': news_item['sentiment_label'],
        'news_sentiment_score': news_item['sentiment_score'],
        'ticker': ticker
    }

    # Feature Aggregation from All Timeframes
    for interval in TIME_FRAMES.keys():
        price_group = price_groups_with_ta.get((ticker, interval))
        if price_group is None: continue
        pre_event_data = price_group[price_group['datetime'] < effective_time]
        if pre_event_data.empty: continue
        latest_indicators = pre_event_data.iloc[-1]
        
        # Add prefix to all columns except identifiers
        id_cols = ['open', 'high', 'low', 'close', 'volume', 'datetime', 'ticker', 'interval']
        feature_cols = {f"{col}_{interval}": val for col, val in latest_indicators.items() if col not in id_cols}
        feature_row.update(feature_cols)

    # Adaptive Target Calculation
    primary_timeframe_key = '1d'
    primary_price_group = price_groups_with_ta.get((ticker, primary_timeframe_key))
    if primary_price_group is None: return None

    try:
        target_matrix = target_system.generate_target_matrix(df=primary_price_group, timeframe=TimeframeType.DAILY)
        event_targets = target_matrix[target_matrix['datetime'] < effective_time]
        if event_targets.empty: return None
        latest_targets = event_targets.iloc[-1]
        feature_row.update({col: val for col, val in latest_targets.items() if col.startswith('target_')})
    except Exception: return None

    if not any(col.startswith('target_') for col in feature_row.keys()): return None
    return feature_row

def run_stage_2(stage1_data: dict) -> pd.DataFrame:
    logger.info("--- Starting Stage 2: Data Factory (v7 - Context-Aware) ---")
    prices_df = stage1_data.get('prices')
    news_df = stage1_data.get('news')
    macro_df = stage1_data.get('macro') # VIX, Bonds, etc.

    if news_df is None or prices_df is None or news_df.empty or prices_df.empty:
        logger.warning("Prices or news data is missing. Skipping.")
        return pd.DataFrame()

    enriched_news_df = _enrich_news_with_sentiment(news_df.copy())
    
    logger.info("Pre-calculating technical and granular features...")
    prices_df_with_ta = prices_df.groupby(['ticker', 'interval'], group_keys=False).apply(_calculate_technical_indicators)
    logger.info("Features calculated.")

    all_pairs = [{'news': news_item, 'ticker': ticker} for news_item, ticker in product(enriched_news_df.to_dict('records'), list(TICKERS.keys()))]
    target_system = AdaptiveTargetsSystem()

    prices_df_with_ta['datetime'] = pd.to_datetime(prices_df_with_ta['datetime'], utc=True)
    market_schedule = get_market_calendar(prices_df_with_ta['datetime'].min(), prices_df_with_ta['datetime'].max())
    price_groups_with_ta = {group: data.sort_values('datetime') for group, data in prices_df_with_ta.groupby(['ticker', 'interval'])}
    
    logger.info(f"Processing {len(all_pairs)} pairs to generate featureset...")
    with ThreadPoolExecutor() as executor:
        process_func = partial(_process_event_pair, price_groups_with_ta=price_groups_with_ta, market_schedule=market_schedule, target_system=target_system)
        results = list(executor.map(process_func, all_pairs))

    valid_events = [res for res in results if res is not None]
    if not valid_events: return pd.DataFrame()

    events_df = pd.DataFrame(valid_events)

    # Merge with Macro Data
    if macro_df is not None and not macro_df.empty:
        logger.info("Merging with macro data...")
        events_df['event_time'] = pd.to_datetime(events_df['event_time'], utc=True)
        macro_df.index = pd.to_datetime(macro_df.index, utc=True)
        events_df = pd.merge_asof(events_df.sort_values('event_time'), macro_df, left_on='event_time', right_index=True, direction='backward')

    # --- Add Contextual Features ---
    logger.info("Adding final contextual features...")
    events_df.set_index('event_time', inplace=True, drop=False)
    events_df.index = pd.to_datetime(events_df.index, utc=True)

    events_df['context_day_of_week'] = events_df.index.dayofweek
    events_df['context_month_of_year'] = events_df.index.month
    events_df['context_is_month_end'] = events_df.index.is_month_end.astype(int)

    # Safely add context features from macro data if they exist
    macro_context_map = {
        'vix_close': 'context_vix_close',
        'bond_10y_yield': 'context_bond_10y_yield_change'
    }
    for raw_col, ctx_col in macro_context_map.items():
        if raw_col in events_df.columns:
            if 'change' in ctx_col:
                events_df[ctx_col] = events_df[raw_col].diff()
            else:
                events_df[ctx_col] = events_df[raw_col]
        else:
            events_df[ctx_col] = np.nan # Fill with NaN if not available
            logger.warning(f"Macro column '{raw_col}' not found. '{ctx_col}' will be empty.")

    logger.info(f"--- Stage 2 Finished --- Created master featureset with shape {events_df.shape} ---")
    return events_df.reset_index(drop=True)
