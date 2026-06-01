import pandas as pd
import asyncio
from typing import Dict, List, Any, Optional
from src.data.collectors.collector_factory import CollectorFactory
from src.core.logging.logger import ProjectLogger
from src.core.exceptions import DataProcessingError

logger = ProjectLogger.get_logger(__name__)

class CollectionManager:
    """Manages execution and error handling of data collectors."""

    def __init__(self, factory: CollectorFactory):
        self.factory = factory
        self.collectors = self.factory.get_all_collectors()
        logger.info(f'Initialized CollectionManager with {len(self.collectors)} collectors.')

    async def fetch_all(self, tickers: List[str], keywords: List[str]) -> Dict[str, pd.DataFrame]:
        """Executes all collectors and aggregates results."""
        raw_data = {}
        for collector in self.collectors:
            try:
                result = await self._run_collector(collector, tickers, keywords)
                if result is None:
                    continue
                
                df = self._convert_to_dataframe(result)
                if df is not None and not df.empty:
                    table_name = f"{collector.__class__.__name__.lower().replace('collector', '')}_data"
                    raw_data[table_name] = df
                    logger.info(f'✅ Collected {len(df)} rows from {collector.__class__.__name__}')
            except Exception as e:
                logger.error(f'❌ Collector {collector.__class__.__name__} failed: {e}', exc_info=True)
                # Re-raising or handling depends on whether we want to fail the whole stage or just skip
                # Given current requirement to fix "silent" errors, we should at least log properly or raise if critical.
                # For now, let's keep it skipping but with better diagnostic log.
                continue
        return raw_data

    async def _run_collector(self, collector: Any, tickers: List[str], keywords: List[str]) -> Any:
        """Runs a collector with appropriate arguments."""
        # This mirrors the logic previously in CollectionStage._run_collector
        # Ideally, collectors should have a unified interface to avoid this type checking
        from src.data.collectors.yf_collector import YFCollector
        from src.data.collectors.fred_collector import FredCollector
        from src.data.collectors.google_news_collector import GoogleNewsCollector
        from src.data.collectors.rss_collector import RSSCollector
        from src.data.collectors.newsapi_collector import NewsAPICollector
        from src.data.collectors.sec_filings_collector import SECFilingsCollector
        from src.data.collectors.insider_collector import InsiderCollector
        from src.data.collectors.free_google_trends_collector import FreeGoogleTrendsCollector
        from src.data.collectors.huggingface_collector import HuggingfaceCollector
        from src.data.collectors.bigquery_collector import BigQueryCollector
        from src.data.collectors.economic_calendar_collector import EconomicCalendarCollector
        
        name = collector.__class__.__name__
        try:
            if isinstance(collector, YFCollector):
                return await collector.run(tickers=tickers)
            elif isinstance(collector, (SECFilingsCollector, InsiderCollector)):
                return await collector.run(tickers=tickers)
            elif isinstance(collector, (GoogleNewsCollector, NewsAPICollector)):
                return await collector.run(tickers=tickers, keywords=keywords)
            elif isinstance(collector, RSSCollector):
                # Assuming config_manager is accessible or passed
                return await collector.run(tickers=tickers, keywords=keywords)
            elif isinstance(collector, FreeGoogleTrendsCollector):
                return await collector.run(tickers=tickers, keywords=keywords)
            elif isinstance(collector, HuggingfaceCollector):
                return await collector.run()
            elif isinstance(collector, (FredCollector, EconomicCalendarCollector, BigQueryCollector)):
                return await collector.run()
            else:
                return await collector.run(tickers=tickers, keywords=keywords)
        except Exception as e:
            raise DataProcessingError(f"Collector {name} failed: {e}") from e

    def _convert_to_dataframe(self, res: Any) -> Optional[pd.DataFrame]:
        """Convert result to DataFrame if needed."""
        if isinstance(res, list) and len(res) > 0:
            return pd.DataFrame(res)
        elif isinstance(res, pd.DataFrame) and not res.empty:
            return res
        return None
