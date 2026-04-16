#!/usr/bin/env python3
"""
Test script to diagnose why news collectors aren't producing data.

Usage:
    python scripts/test_news_collectors.py
"""

import asyncio
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.unified_config_manager import UnifiedConfigManager
from src.data.collectors.collector_factory import CollectorFactory
from src.data.management.data_manager import DataManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.logging.logger import ProjectLogger
from src.core.error_handling.error_handler import ErrorHandler


async def test_collectors():
    """Test news collectors and report results."""
    
    logger = ProjectLogger.get_logger(__name__)
    logger.info("=" * 80)
    logger.info("🧪 NEWS COLLECTORS DIAGNOSTIC TEST")
    logger.info("=" * 80)
    
    # Initialize components
    config_manager = UnifiedConfigManager()
    db_manager = DataManager(config_manager)
    error_handler = ErrorHandler()
    http_client_factory = HttpClientFactory(config_manager, error_handler)
    
    # Get collector configs
    collector_configs = config_manager.get_config('collectors')
    knowledge_base = config_manager.get_config('knowledge_base')
    
    # Create factory
    factory = CollectorFactory(
        configs=collector_configs,
        http_client_factory=http_client_factory,
        config_manager=config_manager,
        db_manager=db_manager,
    )
    
    # Get all collectors
    all_collectors = factory.get_all_collectors()
    logger.info(f"\n📊 Total collectors loaded: {len(all_collectors)}")
    
    # Filter to news collectors
    news_collectors = [
        c for c in all_collectors 
        if collector_configs.get(c.collector_type, {}).get('data_type') == 'news'
    ]
    
    logger.info(f"📰 News collectors: {len(news_collectors)}")
    for c in news_collectors:
        enabled = collector_configs.get(c.collector_type, {}).get('enabled', False)
        logger.info(f"   - {c.collector_type}: {'✅ ENABLED' if enabled else '❌ DISABLED'}")
    
    # Test parameters
    tickers = ['AMD', 'NVDA']
    keywords_raw = knowledge_base.get('keywords', {})
    if isinstance(keywords_raw, dict):
        from itertools import chain
        keywords = list(set(chain.from_iterable(keywords_raw.values())))[:10]  # First 10
    else:
        keywords = list(keywords_raw)[:10]
    
    logger.info(f"\n🔍 Test parameters:")
    logger.info(f"   Tickers: {tickers}")
    logger.info(f"   Keywords (first 10): {keywords}")
    
    # Test each news collector
    logger.info(f"\n" + "=" * 80)
    logger.info("TESTING NEWS COLLECTORS")
    logger.info("=" * 80)
    
    for collector in news_collectors:
        collector_type = collector.collector_type
        enabled = collector_configs.get(collector_type, {}).get('enabled', False)
        
        logger.info(f"\n📌 Testing: {collector_type}")
        logger.info(f"   Enabled: {'✅ YES' if enabled else '❌ NO'}")
        
        if not enabled:
            logger.info(f"   ⏭️  SKIPPED (disabled in config)")
            continue
        
        try:
            # Run collector
            logger.info(f"   Running collector.run()...")
            
            if collector_type == 'rss':
                rss_feeds = knowledge_base.get('rss_feeds', [])
                logger.info(f"   RSS feeds configured: {len(rss_feeds)}")
                result = await collector.run(
                    tickers=tickers,
                    keywords=keywords,
                    rss_feeds=rss_feeds,
                    config_manager=config_manager,
                )
            else:
                result = await collector.run(
                    tickers=tickers,
                    keywords=keywords,
                )
            
            # Analyze result
            if result is None:
                logger.warning(f"   ⚠️  Returned None (no new data or all filtered)")
            elif isinstance(result, pd.DataFrame):
                logger.info(f"   ✅ Returned DataFrame: {len(result)} rows")
                if len(result) > 0:
                    logger.info(f"      Columns: {list(result.columns)}")
                    logger.info(f"      Sample row:")
                    for col in result.columns[:5]:
                        val = result.iloc[0][col]
                        logger.info(f"        - {col}: {val}")
            elif isinstance(result, list):
                logger.info(f"   ✅ Returned list: {len(result)} items")
            else:
                logger.info(f"   ⚠️  Returned unexpected type: {type(result)}")
        
        except Exception as e:
            logger.error(f"   ❌ ERROR: {e}", exc_info=True)
    
    # Check database state
    logger.info(f"\n" + "=" * 80)
    logger.info("DATABASE STATE")
    logger.info("=" * 80)
    
    table_names = db_manager.get_all_table_names()
    news_tables = ['google_news', 'rss_news', 'sec_filings', 'newsapi_articles']
    
    for table in news_tables:
        if table in table_names:
            try:
                df = db_manager.fetch_data_from_table(table)
                if df is not None and not df.empty:
                    logger.info(f"\n✅ {table}: {len(df)} records")
                    logger.info(f"   Columns: {list(df.columns)}")
                else:
                    logger.warning(f"\n⚠️  {table}: EMPTY")
            except Exception as e:
                logger.error(f"\n❌ {table}: ERROR - {e}")
        else:
            logger.info(f"\n❌ {table}: NOT FOUND in database")
    
    logger.info(f"\n" + "=" * 80)
    logger.info("✅ DIAGNOSTIC TEST COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    asyncio.run(test_collectors())
