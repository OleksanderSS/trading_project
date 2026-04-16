#!/usr/bin/env python
"""Тестує NewsAPI, Insider та HuggingFace колектори."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.unified_config_manager import UnifiedConfigManager
from src.data.collectors.collector_factory import CollectorFactory
from src.data.management.data_manager import DataManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.error_handling.error_handler import ErrorHandler
from src.core.logging.logger import ProjectLogger


async def test_collectors():
    """Тестує NewsAPI, Insider та HuggingFace."""
    
    logger = ProjectLogger.get_logger(__name__)
    logger.info("=" * 80)
    logger.info("TESTING NEWSAPI, INSIDER, HUGGINGFACE")
    logger.info("=" * 80)
    
    # Initialize components
    config_manager = UnifiedConfigManager()
    db_manager = DataManager(config_manager)
    error_handler = ErrorHandler()
    http_client_factory = HttpClientFactory(config_manager, error_handler)
    
    # Get collector configs
    collector_configs = config_manager.get_config('collectors')
    
    # Create factory
    factory = CollectorFactory(
        configs=collector_configs,
        http_client_factory=http_client_factory,
        config_manager=config_manager,
        db_manager=db_manager,
    )
    
    # Test parameters
    tickers = ['AMD', 'NVDA']
    keywords = ['stock', 'market', 'trading']
    
    logger.info(f"\nTest parameters:")
    logger.info(f"  Tickers: {tickers}")
    logger.info(f"  Keywords: {keywords}")
    
    # Test NewsAPI
    logger.info("\n" + "=" * 80)
    logger.info("1. TESTING NEWSAPI")
    logger.info("=" * 80)
    
    newsapi_config = collector_configs.get('newsapi', {})
    logger.info(f"\nNewsAPI Config:")
    logger.info(f"  Enabled: {newsapi_config.get('enabled')}")
    logger.info(f"  Requires API Key: {newsapi_config.get('requires_api_key')}")
    logger.info(f"  API Key Env: {newsapi_config.get('api_key_env')}")
    
    # Check if API key exists
    import os
    api_key = os.getenv('NEWS_API_KEY')
    if api_key:
        logger.info(f"  API Key: [FOUND] (length: {len(api_key)})")
    else:
        logger.info(f"  API Key: [NOT FOUND]")
    
    # Try to get NewsAPI collector
    newsapi_collector = factory.get_collector('newsapi')
    if newsapi_collector:
        logger.info(f"\nNewsAPI Collector: [CREATED]")
        try:
            result = await newsapi_collector.run(tickers=tickers, keywords=keywords)
            if result is not None:
                logger.info(f"  Result: DataFrame with {len(result)} records")
                logger.info(f"  Columns: {list(result.columns)}")
            else:
                logger.info(f"  Result: None (no new data or all filtered)")
        except Exception as e:
            logger.error(f"  Error: {e}")
    else:
        logger.info(f"NewsAPI Collector: [NOT CREATED] (disabled or error)")
    
    # Test Insider
    logger.info("\n" + "=" * 80)
    logger.info("2. TESTING INSIDER")
    logger.info("=" * 80)
    
    insider_config = collector_configs.get('insider', {})
    logger.info(f"\nInsider Config:")
    logger.info(f"  Enabled: {insider_config.get('enabled')}")
    logger.info(f"  Table: {insider_config.get('table_name')}")
    
    # Try to get Insider collector
    insider_collector = factory.get_collector('insider')
    if insider_collector:
        logger.info(f"\nInsider Collector: [CREATED]")
        try:
            result = await insider_collector.run(tickers=tickers)
            if result is not None:
                logger.info(f"  Result: DataFrame with {len(result)} records")
                logger.info(f"  Columns: {list(result.columns)}")
            else:
                logger.info(f"  Result: None (no new data or all filtered)")
        except Exception as e:
            logger.error(f"  Error: {e}")
    else:
        logger.info(f"Insider Collector: [NOT CREATED] (disabled or error)")
    
    # Test HuggingFace
    logger.info("\n" + "=" * 80)
    logger.info("3. TESTING HUGGINGFACE")
    logger.info("=" * 80)
    
    huggingface_config = collector_configs.get('hugging_face', {})
    logger.info(f"\nHuggingFace Config:")
    logger.info(f"  Enabled: {huggingface_config.get('enabled')}")
    logger.info(f"  Dataset: {huggingface_config.get('dataset_name')}")
    logger.info(f"  Table: {huggingface_config.get('table_name')}")
    
    # Try to get HuggingFace collector
    huggingface_collector = factory.get_collector('hugging_face')
    if huggingface_collector:
        logger.info(f"\nHuggingFace Collector: [CREATED]")
        try:
            result = await huggingface_collector.run()
            if result is not None:
                logger.info(f"  Result: DataFrame with {len(result)} records")
                logger.info(f"  Columns: {list(result.columns)}")
            else:
                logger.info(f"  Result: None (no new data or all filtered)")
        except Exception as e:
            logger.error(f"  Error: {e}")
    else:
        logger.info(f"HuggingFace Collector: [NOT CREATED] (disabled or error)")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"\nCollector Status:")
    logger.info(f"  NewsAPI: {'ENABLED' if newsapi_config.get('enabled') else 'DISABLED'}")
    logger.info(f"  Insider: {'ENABLED' if insider_config.get('enabled') else 'DISABLED'}")
    logger.info(f"  HuggingFace: {'ENABLED' if huggingface_config.get('enabled') else 'DISABLED'}")
    
    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    asyncio.run(test_collectors())
