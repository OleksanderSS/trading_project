import asyncio
import logging
import pandas as pd
from src.config.unified_config_manager import UnifiedConfigManager
from src.data.collectors.collector_factory import CollectorFactory
from src.data.management.data_manager import DataManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.error_handling.error_handler import ErrorHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CollectorDiagnostic")

async def run_diagnostics():
    config_manager = UnifiedConfigManager()
    error_handler = ErrorHandler()
    db_manager = DataManager(config_manager, error_handler)
    http_factory = HttpClientFactory(config_manager, error_handler)
    
    factory = CollectorFactory(
        configs=config_manager.get_config('collectors'),
        http_client_factory=http_factory,
        config_manager=config_manager,
        db_manager=db_manager
    )
    
    collectors = factory.get_all_collectors()
    tickers = ['AAPL', 'TSLA'] # Small set for testing
    keywords = ['market']
    
    for collector in collectors:
        name = collector.__class__.__name__
        logger.info(f"--- Testing {name} ---")
        try:
            # Set a 10s timeout to detect blocking behavior
            task = asyncio.create_task(collector.run(tickers=tickers, keywords=keywords))
            result = await asyncio.wait_for(task, timeout=10.0)
            
            if result is not None:
                count = len(result) if hasattr(result, '__len__') else "unknown"
                logger.info(f"✅ {name} SUCCESS: Collected {count} items.")
            else:
                logger.info(f"⚠️ {name} RETURNED None.")
                
        except asyncio.TimeoutError:
            logger.error(f"❌ {name} TIMED OUT (blocked execution).")
        except Exception as e:
            logger.error(f"❌ {name} FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(run_diagnostics())
