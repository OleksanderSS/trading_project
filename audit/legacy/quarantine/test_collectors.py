import asyncio
import os
import sys

# Add current dir to pythonpath
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config.unified_config_manager import UnifiedConfigManager
from src.data.management.data_manager import DataManager
from src.core.error_handling.error_handler import ErrorHandler
from src.pipeline.stages.stage_1_collection import CollectionStage
from src.core.logging.logger import ProjectLogger

async def main():
    logger = ProjectLogger.get_logger("test")
    logger.info("Starting test")
    config_manager = UnifiedConfigManager()
    error_handler = ErrorHandler(config_manager=config_manager)
    db_manager = DataManager(config_manager=config_manager, error_handler=error_handler)
    
    stage = CollectionStage(config_manager=config_manager, db_manager=db_manager, error_handler=error_handler)
    
    stage._tickers = ["TSLA", "NVDA"]
    # We only run a subset of collectors to test
    target_collectors = []
    for c in stage.collectors:
        if c.collector_type in ["fred", "google_news", "rss"]:
            target_collectors.append(c)
    stage.collectors = target_collectors
    
    logger.info(f"Running {len(target_collectors)} collectors")
    raw_data = await stage._fetch_data()
    logger.info("Test complete")

if __name__ == "__main__":
    asyncio.run(main())
