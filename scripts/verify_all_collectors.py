
import asyncio

from src.config.unified_config_manager import get_current_config
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.error_handling.error_handler import get_error_handler
from src.core.logging.logger import ProjectLogger
from src.data.collectors.collector_factory import CollectorFactory

logger = ProjectLogger.get_logger("CollectorValidator")

async def validate_all_collectors():
    logger.info("Starting comprehensive collector validation...")
    
    config = get_current_config()
    eh = get_error_handler()
    hcf = HttpClientFactory(config, eh)
    factory = CollectorFactory(
        configs=config.get_config('collectors', {}),
        http_client_factory=hcf,
        config_manager=config
    )
    
    # Get all collectors (including disabled ones to test them)
    # We need to force enable them in the factory or bypass the enabled check if possible
    # For now, let's just attempt to instantiate them
    
    collector_names = config.get_config('collectors', {}).keys()
    
    for name in collector_names:
        logger.info(f"--- Validating collector: {name} ---")
        try:
            collector = factory.get_collector(name)
            if not collector:
                logger.warning(f"❌ Collector '{name}' could not be instantiated.")
                continue
            
            # Simple run test (if applicable)
            logger.info(f"✅ Instantiated {name}. Attempting to run test fetch...")
            # For test, we use minimal arguments
            result = await collector.run(tickers=["SPY"])
            if result is not None:
                logger.info(f"✅ {name} returned data.")
            else:
                logger.warning(f"⚠️ {name} returned no data.")
                
        except Exception as e:
            logger.error(f"❌ {name} failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(validate_all_collectors())
