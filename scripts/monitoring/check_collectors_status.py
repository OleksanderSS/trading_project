#!/usr/bin/env python
"""Перевіряє статус всіх колекторів в пайплайні."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.unified_config_manager import UnifiedConfigManager
from src.data.collectors.collector_factory import CollectorFactory
from src.data.management.data_manager import DataManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.error_handling.error_handler import ErrorHandler
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


def _initialize_components():
    """Initialize pipeline components."""
    config_manager = UnifiedConfigManager()
    db_manager = DataManager(config_manager)
    error_handler = ErrorHandler()
    http_client_factory = HttpClientFactory(config_manager, error_handler)
    
    return config_manager, db_manager, http_client_factory

def _create_collector_factory(config_manager, db_manager, http_client_factory):
    """Create collector factory."""
    collector_configs = config_manager.get_config('collectors')
    
    factory = CollectorFactory(
        configs=collector_configs,
        http_client_factory=http_client_factory,
        config_manager=config_manager,
        db_manager=db_manager,
    )
    
    return factory, collector_configs

def _group_collectors_by_type(all_collectors):
    """Group collectors by their type."""
    by_type = {}
    for collector in all_collectors:
        collector_type = collector.collector_type
        if collector_type not in by_type:
            by_type[collector_type] = []
        by_type[collector_type].append(collector)
    return by_type

def _log_enabled_collectors(by_type, collector_configs):
    """Log information about enabled collectors."""
    for collector_type in sorted(by_type.keys()):
        logger.info(f"\n{collector_type.upper()}:")
        
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

def _log_disabled_collectors(collector_configs):
    """Log information about disabled collectors."""
    disabled_count = 0
    for name, config in collector_configs.items():
        if not config.get('enabled', False):
            disabled_count += 1
            data_type = config.get('data_type', 'market')
            logger.info(f"  [DISABLED] {name} ({data_type})")
    
    if disabled_count == 0:
        logger.info("  (None)")
    
    return disabled_count

def _log_summary(all_collectors, collector_configs, disabled_count):
    """Log summary statistics."""
    enabled_count = len(all_collectors)
    total_count = len(collector_configs)
    
    logger.info(f"Enabled: {enabled_count}/{total_count}")
    logger.info(f"Disabled: {disabled_count}/{total_count}")

def _log_breakdown_by_type(all_collectors, collector_configs):
    """Log breakdown of collectors by data type."""
    logger.info("\nBreakdown by data type:")
    
    market_collectors = [c for c in all_collectors if collector_configs.get(c.collector_type, {}).get('data_type') == 'market']
    news_collectors = [c for c in all_collectors if collector_configs.get(c.collector_type, {}).get('data_type') == 'news']
    
    logger.info(f"  Market Data: {len(market_collectors)} collectors")
    for c in market_collectors:
        logger.info(f"    - {c.collector_type}")
    
    logger.info(f"  News Data: {len(news_collectors)} collectors")
    for c in news_collectors:
        logger.info(f"    - {c.collector_type}")

def _log_critical_collectors(all_collectors, collector_configs):
    """Log critical collectors."""
    logger.info("\nCritical collectors:")
    critical_collectors = [c for c in all_collectors if collector_configs.get(c.collector_type, {}).get('critical', False)]
    
    if critical_collectors:
        for c in critical_collectors:
            logger.info(f"  [CRITICAL] {c.collector_type}")
    else:
        logger.info("  (None)")

def check_collectors():
    """Перевіряє статус всіх колекторів."""
    
    logger = ProjectLogger.get_logger(__name__)
    logger.info("=" * 80)
    logger.info("COLLECTORS STATUS CHECK")
    logger.info("=" * 80)
    
    # Initialize components
    config_manager, db_manager, http_client_factory = _initialize_components()
    
    # Create factory and get configs
    factory, collector_configs = _create_collector_factory(config_manager, db_manager, http_client_factory)
    
    # Get all collectors
    all_collectors = factory.get_all_collectors()
    logger.info(f"\nTotal collectors loaded: {len(all_collectors)}")
    
    # Log enabled collectors
    logger.info("\n" + "=" * 80)
    logger.info("ENABLED COLLECTORS")
    logger.info("=" * 80)
    
    by_type = _group_collectors_by_type(all_collectors)
    _log_enabled_collectors(by_type, collector_configs)
    
    # Log disabled collectors
    logger.info("\n" + "=" * 80)
    logger.info("DISABLED COLLECTORS")
    logger.info("=" * 80)
    
    disabled_count = _log_disabled_collectors(collector_configs)
    
    # Log summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    _log_summary(all_collectors, collector_configs, disabled_count)
    
    # Log breakdown by type
    _log_breakdown_by_type(all_collectors, collector_configs)
    
    # Log critical collectors
    _log_critical_collectors(all_collectors, collector_configs)
    
    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    check_collectors()
