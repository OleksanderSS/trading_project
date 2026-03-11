import sys
import os
import asyncio
from datetime import datetime

# Ensure the source directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager
from src.core.logging.logger import ProjectLogger
from src.core.error_handling.error_handler import ErrorHandler
from src.core.cache.cache_manager import CacheManager
from src.data.collectors.yf_collector import YFCollector
from src.data.collectors.fred_collector import FredCollector
from src.data.collectors.google_news_collector import GoogleNewsCollector
from src.data.collectors.rss_collector import RSSCollector
from src.data.collectors.sec_filings_collector import SECFilingsCollector
from src.data.collectors.insider_collector import InsiderCollector
from src.data.collectors.economic_calendar_collector import EconomicCalendarCollector
from src.data.collectors.newsapi_collector import NewsAPICollector
from src.data.collectors.huggingface_collector import HuggingFaceCollector
from src.data.collectors.free_google_trends_collector import FreeGoogleTrendsCollector
from src.data.collectors.bigquery_collector import BigQueryCollector

async def run_and_verify():
    """
    A comprehensive integration test to run all enabled collectors twice,
    ensuring the caching mechanism works correctly across the entire data collection system.
    """
    ProjectLogger.setup_logging()
    logger = ProjectLogger.get_logger(__name__)
    
    config_manager = UnifiedConfigManager()
    
    # --- 1. Clean Slate: Ensure DB is deleted before the test ---
    db_path = config_manager.get('paths.raw_db')
    if db_path and os.path.exists(db_path):
        try:
            os.remove(db_path)
            logger.info(f"Clean slate: Removed existing database file at {db_path}")
        except OSError as e:
            logger.error(f"Error removing database file {db_path}: {e}")
            return

    try:
        # --- 2. Setup: Initialize all necessary components ---
        error_handler = ErrorHandler(config_manager=config_manager)
        http_client_factory = HttpClientFactory(config_manager=config_manager, error_handler=error_handler)
        db_manager = DataManager(config_manager=config_manager, error_handler=error_handler)
        cache_manager = CacheManager(data_manager=db_manager)

        # --- 3. Instantiate Collectors ---
        collectors = []
        collector_configs = config_manager.get_config('collectors')

        if collector_configs.get('yahoo_finance', {}).get('enabled'):
            collectors.append(YFCollector(collector_configs['yahoo_finance'], http_client_factory, db_manager=db_manager, cache_manager=cache_manager))
            logger.info("YFCollector initialized.")
        
        if collector_configs.get('fred', {}).get('enabled'):
            collectors.append(FredCollector(collector_configs['fred'], http_client_factory, db_manager=db_manager, cache_manager=cache_manager))
            logger.info("FredCollector initialized.")
        
        if collector_configs.get('google_news', {}).get('enabled'):
            collectors.append(GoogleNewsCollector(collector_configs['google_news'], http_client_factory, db_manager=db_manager, cache_manager=cache_manager))
            logger.info("GoogleNewsCollector initialized.")

        if collector_configs.get('rss', {}).get('enabled'):
            collectors.append(RSSCollector(collector_configs['rss'], http_client_factory, db_manager=db_manager, cache_manager=cache_manager))
            logger.info("RSSCollector initialized.")

        if collector_configs.get('sec_filings', {}).get('enabled'):
            collectors.append(SECFilingsCollector(collector_configs['sec_filings'], http_client_factory, db_manager=db_manager, config_manager=config_manager, cache_manager=cache_manager))
            logger.info("SECFilingsCollector initialized.")

        if collector_configs.get('insider_trading', {}).get('enabled'):
            collectors.append(InsiderCollector(collector_configs['insider_trading'], http_client_factory, db_manager=db_manager, cache_manager=cache_manager))
            logger.info("InsiderCollector initialized.")
        
        if collector_configs.get('economic_calendar', {}).get('enabled'):
            collectors.append(EconomicCalendarCollector(collector_configs['economic_calendar'], http_client_factory, db_manager=db_manager, cache_manager=cache_manager))
            logger.info("EconomicCalendarCollector initialized.")

        if collector_configs.get('newsapi', {}).get('enabled'):
            collectors.append(NewsAPICollector(collector_configs['newsapi'], http_client_factory, db_manager=db_manager, cache_manager=cache_manager))
            logger.info("NewsAPICollector initialized.")

        if collector_configs.get('huggingface', {}).get('enabled'):
            collectors.append(HuggingFaceCollector(collector_configs['huggingface'], http_client_factory, db_manager=db_manager, cache_manager=cache_manager))
            logger.info("HuggingFaceCollector initialized.")

        if collector_configs.get('google_trends', {}).get('enabled'):
            collectors.append(FreeGoogleTrendsCollector(collector_configs['google_trends'], http_client_factory, db_manager=db_manager, cache_manager=cache_manager))
            logger.info("FreeGoogleTrendsCollector initialized.")

        if collector_configs.get('bigquery', {}).get('enabled'):
            collectors.append(BigQueryCollector(collector_configs['bigquery'], http_client_factory, db_manager=db_manager, cache_manager=cache_manager))
            logger.info("BigQueryCollector initialized.")

        # --- 4. Get Assets and Keywords for Collectors ---
        asset_configs = config_manager.get_config('assets')
        tickers = asset_configs.get('presets', {}).get(asset_configs.get('active_preset'), {}).get('tickers')
        
        kb_configs = config_manager.get_config('knowledge_base')
        keywords = kb_configs.get('keywords', [])

        if not tickers:
            logger.error("Ticker list is empty. Collectors may not function correctly.")
            return

        # --- RUN 1: POPULATE CACHE ---
        logger.info("\n" + "="*20 + " STARTING FIRST RUN: POPULATING CACHE " + "="*20)
        
        run_1_results = {}
        fixed_end_date = datetime(2026, 3, 1, 12, 0, 0)
        run_date = datetime(2026, 3, 10, 12, 0, 0)

        async def execute_collector_run(collector):
            collector_name = collector.__class__.__name__
            if isinstance(collector, YFCollector):
                return await collector.run(tickers=tickers, end_date=fixed_end_date)
            elif isinstance(collector, (SECFilingsCollector, InsiderCollector)):
                return await collector.run(tickers=tickers, run_date=run_date)
            elif isinstance(collector, (GoogleNewsCollector, RSSCollector, NewsAPICollector)):
                return await collector.run(search_terms=keywords, tickers=tickers)
            elif isinstance(collector, FreeGoogleTrendsCollector):
                return await collector.run(keywords=keywords)
            elif isinstance(collector, HuggingFaceCollector):
                dataset_name = collector_configs.get('huggingface', {}).get('dataset_name')
                return await collector.run(dataset_name=dataset_name)
            elif isinstance(collector, (FredCollector, EconomicCalendarCollector, BigQueryCollector)):
                return await collector.run()
            else:
                logger.warning(f"No specific run condition for {collector_name}, trying a generic run().")
                return await collector.run()

        for collector in collectors:
            collector_name = collector.__class__.__name__
            logger.info(f"--- Running {collector_name} [RUN 1] ---")
            try:
                new_data = await execute_collector_run(collector)
                run_1_results[collector_name] = len(new_data) if new_data else 0
                logger.info(f"--- {collector_name} [RUN 1] finished, saved {run_1_results[collector_name]} records. ---")

            except Exception as e:
                logger.critical(f"An error occurred during RUN 1 with {collector_name}: {e}", exc_info=True)
                run_1_results[collector_name] = -1

        logger.info("\n" + "="*20 + " FIRST RUN COMPLETE " + "="*20)
        for name, count in run_1_results.items():
             print(f"    - {name}: {'ERROR' if count == -1 else f'{count} new records.'}")
        print("-"*60)


        # --- RUN 2: VERIFY CACHE ---
        logger.info("\n" + "="*20 + " STARTING SECOND RUN: VERIFYING CACHE " + "="*20)

        run_2_results = {}
        all_caches_worked = True

        for collector in collectors:
            collector_name = collector.__class__.__name__
            logger.info(f"--- Running {collector_name} [RUN 2] ---")
            try:
                new_data = await execute_collector_run(collector)
                count = len(new_data) if new_data else 0
                run_2_results[collector_name] = count
                
                if count > 0:
                    all_caches_worked = False
                    logger.error(f"CACHE FAILURE for {collector_name}: Expected 0 new records, but got {count}.")
                else:
                    logger.info(f"CACHE SUCCESS for {collector_name}: Correctly found 0 new records.")

            except Exception as e:
                logger.critical(f"An error occurred during RUN 2 with {collector_name}: {e}", exc_info=True)
                run_2_results[collector_name] = -1
                all_caches_worked = False

        logger.info("\n" + "="*20 + " SECOND RUN COMPLETE " + "="*20)
        for name, count in run_2_results.items():
             status = "SUCCESS" if count == 0 else "FAILURE"
             print(f"    - {name}: {count} new records. Status: {status}")
        
        print("\n" + "-"*60)
        if all_caches_worked:
            logger.info("FINAL RESULT: SUCCESS! All collectors correctly used the cache on the second run.")
        else:
            logger.error("FINAL RESULT: FAILURE! One or more collectors did not correctly use the cache.")
        print("-"*60)


    except Exception as e:
        logger.critical(f"A critical error occurred during the integration test: {e}", exc_info=True)
    finally:
        logger.info("Full collection test finished.")

if __name__ == "__main__":
    asyncio.run(run_and_verify())