#!/usr/bin/env python
"""Тестує RSS колектор напряму."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.unified_config_manager import UnifiedConfigManager
from src.data.collectors.rss_collector import RSSCollector
from src.data.management.data_manager import DataManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.error_handling.error_handler import ErrorHandler
from src.core.logging.logger import ProjectLogger


async def test_rss():
    """Тестує RSS колектор."""
    
    logger = ProjectLogger.get_logger(__name__)
    logger.info("=" * 80)
    logger.info("🧪 RSS COLLECTOR DIRECT TEST")
    logger.info("=" * 80)
    
    # Initialize components
    config_manager = UnifiedConfigManager()
    db_manager = DataManager(config_manager)
    error_handler = ErrorHandler()
    http_client_factory = HttpClientFactory(config_manager, error_handler)
    
    # Get RSS config
    collector_configs = config_manager.get_config('collectors')
    rss_config = collector_configs.get('rss', {})
    knowledge_base = config_manager.get_config('knowledge_base')
    
    logger.info(f"\n📋 RSS Config:")
    logger.info(f"   Enabled: {rss_config.get('enabled')}")
    logger.info(f"   Timeout: {rss_config.get('timeout')}")
    logger.info(f"   Min quality: {rss_config.get('filter', {}).get('min_source_quality')}")
    
    # Create RSS collector
    rss_collector = RSSCollector(
        configs=rss_config,
        http_client_factory=http_client_factory,
        db_manager=db_manager,
    )
    
    # Get RSS feeds
    rss_feeds = knowledge_base.get('rss_feeds', [])
    logger.info(f"\n📰 RSS Feeds configured: {len(rss_feeds)}")
    for feed in rss_feeds[:3]:
        logger.info(f"   - {feed['name']}: {feed['url'][:60]}...")
    
    # Run collector
    logger.info(f"\n🚀 Running RSS collector...")
    result = await rss_collector.run(
        tickers=['AMD'],
        keywords=['AMD', 'stock', 'market'],
        rss_feeds=rss_feeds,
        config_manager=config_manager,
    )
    
    # Analyze result
    if result is None:
        logger.warning(f"⚠️  Returned None")
    else:
        logger.info(f"✅ Returned DataFrame: {len(result)} rows")
        logger.info(f"   Columns: {list(result.columns)}")
        logger.info(f"\n   First 3 rows:")
        for idx, row in result.head(3).iterrows():
            logger.info(f"   [{idx}] {row['title'][:60]}...")
            logger.info(f"        Source: {row['source']}, Date: {row['published_date']}")
    
    # Check database
    logger.info(f"\n📊 Checking database...")
    try:
        df = db_manager.fetch_data_from_table('rss_news')
        if df is not None and not df.empty:
            logger.info(f"✅ rss_news table: {len(df)} records")
        else:
            logger.warning(f"⚠️  rss_news table: EMPTY")
    except Exception as e:
        logger.error(f"❌ Error fetching rss_news: {e}")


if __name__ == '__main__':
    asyncio.run(test_rss())
