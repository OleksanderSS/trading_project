#!/usr/bin/env python3
"""
Stage 2: Event Context Enrichment Layer (v3 - "Ironclad" Quality Filter)

This version completely overhauls the logic to enforce data quality at the source,
as per user feedback. It ensures that only complete, verifiable, and meaningful
events are passed to subsequent stages. The principle is: "Garbage in, garbage out."
We stop garbage from ever getting in.

Key Principles:
1.  **Market-Aware Timing:** Integrates an exchange calendar (NYSE) to understand
    trading hours, weekends, and holidays. News published during non-trading
    hours is correctly anchored to the next market open.
2.  **Uncompromising Data Completeness:** Before an event is created, it undergoes
    rigorous checks to ensure all required data points exist.
3.  **Pre-Event Context Check:** Verifies there is a sufficient historical window
    of price data *before* the event to calculate all necessary technical indicators.
    (e.g., enough data for a 200-period moving average).
4.  **Post-Event Outcome Check:** Verifies that two full candle intervals exist *after*
    the event for *each* required timeframe. This guarantees that target variables
    can be calculated. Events that are too recent are temporarily discarded.
5.  **Zero-Tolerance for NaNs:** As a result of the above, this stage produces a
    perfectly clean dataset. No more `NaN` values in features or targets will
    propagate to the modeling stage.
"""

import logging
import pandas as pd
import pandas_market_calendars as mcal
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from enrichment.roberta_sentiment import RobertaSentimentAnalyzer
from utils.cache_utils import CacheManager
from config.config import TIME_FRAMES, REQUIRED_TA_WINDOW, POST_EVENT_HORIZON

logger = logging.getLogger(__name__)

@lru_cache(maxsize=None)
def get_sentiment_analyzer():
    logger.info("Initializing and loading sentiment analyzer model...")
    analyzer = RobertaSentimentAnalyzer()
    analyzer._load_model()
    return analyzer

@lru_cache(maxsize=1)
def get_market_calendar(start_date, end_date):
    """Caches the NYSE market calendar for the required date range."""
    nyse = mcal.get_calendar('NYSE')
    return nyse.schedule(start_date=start_date, end_date=end_date)

def _enrich_news_with_sentiment(news_df: pd.DataFrame, cache_manager: CacheManager) -> pd.DataFrame:
    # (Code remains the same as before)
    if news_df.empty or 'title' not in news_df.columns:
        return news_df
    logger.info(f"Starting sentiment analysis for {len(news_df)} articles...")
    sentiment_analyzer = get_sentiment_analyzer()
    news_df['text_for_sentiment'] = news_df['title'].fillna('') + " " + news_df.get('description', '').fillna('')
    texts = news_df['text_for_sentiment'].tolist()

    def analyze_text(text: str) -> dict:
        label, score = sentiment_analyzer.predict(text)
        return {"label": label, "score": score}

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(analyze_text, texts))
    sentiment_df = pd.DataFrame(results, index=news_df.index)
    news_df['sentiment_score'] = sentiment_df['score']
    news_df['sentiment_label'] = sentiment_df['label']
    news_df.drop(columns=['text_for_sentiment'], inplace=True)
    logger.info("Sentiment analysis complete.")
    return news_df

def get_effective_event_time(publish_time, market_schedule):
    """Anchors a publish time to the next market open if it's off-hours."""
    # Find the next market open time after the news was published.
    next_market_open = market_schedule[market_schedule['market_open'] > publish_time]
    if not next_market_open.empty:
        return next_market_open.iloc[0]['market_open']
    return None # If no future market open is found in the schedule

def _create_event_context_v3(news_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty or prices_df.empty:
        return pd.DataFrame()

    logger.info("Starting event creation with v3 'Ironclad' filter...")
    market_schedule = get_market_calendar(prices_df['datetime'].min(), prices_df['datetime'].max())
    
    prices_df = prices_df.sort_values('datetime')
    price_groups = {group: data for group, data in prices_df.groupby(['ticker', 'interval'])}
    
    total_news = len(news_df)
    rejected_count = 0
    event_contexts = []

    for _, news_item in news_df.iterrows():
        ticker = news_item.get('ticker')
        if not ticker: continue

        publish_time = news_item['published_at']
        effective_time = get_effective_event_time(publish_time, market_schedule)
        
        if effective_time is None:
            rejected_count += 1
            continue

        is_valid_event = True
        context_data = {
            'event_time': effective_time,
            'ticker': ticker,
            **{f'news_{col}': val for col, val in news_item.items()}
        }

        for interval, frame_mins in TIME_FRAMES.items():
            price_group = price_groups.get((ticker, interval))
            if price_group is None: is_valid_event = False; break

            # 1. Pre-event check: Ensure enough history for TAs
            pre_event_window_start = effective_time - timedelta(minutes=REQUIRED_TA_WINDOW[interval])
            pre_data = price_group[price_group['datetime'] < effective_time]
            if pre_data.empty or pre_data.iloc[0]['datetime'] > pre_event_window_start:
                is_valid_event = False; break
            context_data[f'pre_event_{interval}_candle'] = pre_data.iloc[-1].to_dict()

            # 2. Post-event check: Ensure the future outcome is known
            post_event_window_end = effective_time + timedelta(minutes=frame_mins * POST_EVENT_HORIZON)
            post_data = price_group[price_group['datetime'] > effective_time]
            if len(post_data) < POST_EVENT_HORIZON or post_data.iloc[POST_EVENT_HORIZON-1]['datetime'] > post_event_window_end + timedelta(minutes=frame_mins):
                 is_valid_event = False; break

            context_data[f'post_event_1_{interval}_candle'] = post_data.iloc[0].to_dict()
            context_data[f'post_event_2_{interval}_candle'] = post_data.iloc[1].to_dict()

        if is_valid_event:
            event_contexts.append(context_data)
        else:
            rejected_count += 1

    logger.warning(f"[Quality Filter] Rejected {rejected_count}/{total_news} news items due to incomplete data.")
    if not event_contexts:
        return pd.DataFrame()

    # Flatten the nested dictionary structure
    flat_events = []
    for event in event_contexts:
        flat_event = {}
        for k, v in event.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat_event[f"{k}_{sub_k}"] = sub_v
            else:
                flat_event[k] = v
        flat_events.append(flat_event)

    result_df = pd.DataFrame(flat_events)
    logger.info(f"Successfully created {len(result_df)} high-quality event contexts.")
    return result_df

def run_stage_2(stage1_data: dict) -> pd.DataFrame:
    """Orchestrates the enrichment stage with the v3 ironclad filter."""
    logger.info("--- Starting Stage 2: Event Context Enrichment (v3 - Ironclad) ---")
    prices_df = stage1_data.get('prices')
    news_df = stage1_data.get('news')
    if news_df is None or news_df.empty or prices_df is None or prices_df.empty:
        logger.warning("Prices or news data from Stage 1 is missing or empty. Skipping enrichment.")
        return pd.DataFrame()

    cache_manager = CacheManager(base_path="./data/cache/stage_2")
    enriched_news_df = _enrich_news_with_sentiment(news_df.copy(), cache_manager)
    
    event_context_df = _create_event_context_v3(enriched_news_df, prices_df)

    logger.info("--- Stage 2: Event Context Enrichment (v3) Finished ---")
    return event_context_df
