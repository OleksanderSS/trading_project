
import asyncio

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.error_handling.error_handler import ErrorHandler
from src.data.management.data_manager import DataManager


async def test_collector(collector_class):
    print(f"Testing {collector_class.__name__}...")
    config = UnifiedConfigManager()
    error_handler = ErrorHandler()
    db = DataManager(config, error_handler=error_handler)
    client_factory = HttpClientFactory(config_manager=config, error_handler=error_handler)
    
    # Minimal config for test
    configs = {'table_name': f'test_{collector_class.__name__.lower()}'}
    
    try:
        collector = collector_class(configs, client_factory, db, error_handler=error_handler)
        # Try a generic run or specific call
        if hasattr(collector, 'run'):
            # Some collectors need tickers, try empty list or minimal mock
            # AIISentimentCollector doesn't need tickers
            result = await collector.run() if collector_class.__name__ == 'AIISentimentCollector' else await collector.run(tickers=['AAPL'])
            print(f"✅ Success: {len(result) if result is not None else 0} records")
        else:
            print("❌ No run method")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Failed: {e}")

# Imports
from src.data.collectors.aaii_sentiment_collector import AIISentimentCollector
from src.data.collectors.reddit_sentiment_collector import RedditSentimentCollector
from src.data.collectors.sec_filings_collector import SECFilingsCollector
from src.data.collectors.vix_collector import VIXCollector


async def main():
    await test_collector(AIISentimentCollector)
    await test_collector(RedditSentimentCollector)
    await test_collector(SECFilingsCollector)
    await test_collector(VIXCollector)

if __name__ == "__main__":
    asyncio.run(main())
