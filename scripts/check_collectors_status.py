#!/usr/bin/env python
"""Перевіряє статус всіх колекторів в пайплайні."""

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


async def check_collectors():
    """Перевіряє статус всіх колекторів."""
    
    logger = ProjectLogger.get_logger(__name__)
    logger.info("=" * 80)
    logger.info("COLLECTORS STATUS CHECK")
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
    
    # Get all collectors
    all_collectors = factory.get_all_collectors()
    
    logger.info(f"\nTotal collectors loaded: {len(all_collectors)}")
    logger.info("\n" + "=" * 80)
    logger.info("ENABLED COLLECTORS")
    logger.info("=" * 80)
    
    # Group by type
    by_type = {}
    for collector in all_collectors:
        collector_type = collector.collector_type
        if collector_type not in by_type:
            by_type[collector_type] = []
        by_type[collector_type].append(collector)
    
    # Display enabled collectors
    for collector_type in sorted(by_type.keys()):
        collectors = by_type[collector_type]
        logger.info(f"\n{collector_type.upper()}:")
        for collector in collectors:
            config = collector_configs.get(collector_type, {})
            enabled = config.get('enabled', False)
            critical = config.get('critical', False)
            data_type = config.get('data_type', 'market')
            table_name = config.get('table_name', 'N/A')
            
            status = "ENABLED" if enabled else "DISABLED"
            critical_str = " [CRITICAL]" if critical else ""
            
            logger.info(f"  [OK] {collector_type}")
            logger.info(f"     Status: {status}{critical_str}")
            logger.info(f"     Type: {data_type}")
            logger.info(f"     Table: {table_name}")
    
    # Check disabled collectors
    logger.info("\n" + "=" * 80)
    logger.info("DISABLED COLLECTORS")
    logger.info("=" * 80)
    
    disabled_count = 0
    for name, config in collector_configs.items():
        if not config.get('enabled', False):
            disabled_count += 1
            data_type = config.get('data_type', 'market')
            logger.info(f"  [DISABLED] {name} ({data_type})")
    
    if disabled_count == 0:
        logger.info("  (None)")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    enabled_count = len(all_collectors)
    total_count = len(collector_configs)
    
    logger.info(f"Enabled: {enabled_count}/{total_count}")
    logger.info(f"Disabled: {disabled_count}/{total_count}")
    
    # Breakdown by type
    logger.info("\nBreakdown by data type:")
    market_collectors = [c for c in all_collectors if collector_configs.get(c.collector_type, {}).get('data_type') == 'market']
    news_collectors = [c for c in all_collectors if collector_configs.get(c.collector_type, {}).get('data_type') == 'news']
    
    logger.info(f"  Market Data: {len(market_collectors)} collectors")
    for c in market_collectors:
        logger.info(f"    - {c.collector_type}")
    
    logger.info(f"  News Data: {len(news_collectors)} collectors")
    for c in news_collectors:
        logger.info(f"    - {c.collector_type}")
    
    # Critical collectors
    logger.info("\nCritical collectors:")
    critical_collectors = [c for c in all_collectors if collector_configs.get(c.collector_type, {}).get('critical', False)]
    if critical_collectors:
        for c in critical_collectors:
            logger.info(f"  [CRITICAL] {c.collector_type}")
    else:
        logger.info("  (None)")
    
    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    asyncio.run(check_collectors())
