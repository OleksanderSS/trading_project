import asyncio
import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.error_handling.error_handler import ErrorHandler
from src.data.collectors.fred_collector import FredCollector
from src.data.management.data_manager import DataManager


async def test_fred():
    print("Testing FredCollector...")
    config_manager = UnifiedConfigManager()
    error_handler = ErrorHandler()
    db_manager = DataManager(config_manager, error_handler)
    http_factory = HttpClientFactory(config_manager, error_handler)
    
    collectors = config_manager.get_config('collectors', {})
    collector_configs = collectors.get('fred')
    if not collector_configs:
        print("❌ Error: 'fred' collector config not found.")
        return
    collector = FredCollector(collector_configs, http_factory, db_manager)
    
    print(f"Running collector with {len(collector_configs.get('params', {}).get('series_ids', []))} series...")
    try:
        df = await collector.run()
        if df is not None:
            print(f"✅ Success! Collected {len(df)} rows.")
            print(df.head())
        else:
            print("⚠️ No data collected (all up to date or empty).")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(test_fred())
