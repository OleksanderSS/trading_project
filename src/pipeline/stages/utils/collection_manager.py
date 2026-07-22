from typing import Any

import pandas as pd

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger
from src.data.collectors.collector_factory import CollectorFactory

logger = ProjectLogger.get_logger(__name__)

class CollectionManager:
    """Manages execution and error handling of data collectors."""

    def __init__(self, factory: CollectorFactory, config_manager=None):
        self.factory = factory
        self.config_manager = config_manager
        self.collectors = self.factory.get_all_collectors()
        logger.info(f'Initialized CollectionManager with {len(self.collectors)} collectors.')

    async def fetch_all(self, tickers: list[str], keywords: list[str]) -> dict[str, pd.DataFrame]:
        """Executes all collectors concurrently and aggregates results."""
        import asyncio
        raw_data = {}
        
        async def fetch_single(collector):
            try:
                result = await self._run_collector(collector, tickers, keywords)
                if result is None:
                    return None
                df = self._convert_to_dataframe(result)
                if df is not None and not df.empty:
                    table_name = f"{collector.__class__.__name__.lower().replace('collector', '')}_data"
                    logger.info(f'✅ Collected {len(df)} rows from {collector.__class__.__name__}')
                    return table_name, df
            except Exception as e:
                logger.exception(f'❌ Collector {collector.__class__.__name__} failed: {e}')
            return None

        # Запускаємо всі колектори паралельно
        tasks = [fetch_single(col) for col in self.collectors]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for res in results:
            if isinstance(res, tuple):
                table_name, df = res
                raw_data[table_name] = df
                
        return raw_data

    async def _run_collector(self, collector: Any, tickers: list[str], keywords: list[str]) -> Any:
        """Runs a collector with appropriate arguments."""
        # This mirrors the logic previously in CollectionStage._run_collector
        # Ideally, collectors should have a unified interface to avoid this type checking
        from src.data.collectors.bigquery_collector import BigQueryCollector
        from src.data.collectors.economic_calendar_collector import EconomicCalendarCollector
        from src.data.collectors.fred_collector import FredCollector
        from src.data.collectors.free_google_trends_collector import FreeGoogleTrendsCollector
        from src.data.collectors.google_news_collector import GoogleNewsCollector
        from src.data.collectors.huggingface_collector import HuggingfaceCollector
        from src.data.collectors.insider_collector import InsiderCollector
        from src.data.collectors.newsapi_collector import NewsAPICollector
        from src.data.collectors.rss_collector import RSSCollector
        from src.data.collectors.sec_filings_collector import SECFilingsCollector
        from src.data.collectors.yf_collector import YFCollector

        name = collector.__class__.__name__
        try:
            if isinstance(collector, YFCollector):
                return await collector.run(tickers=tickers)
            elif isinstance(collector, (SECFilingsCollector, InsiderCollector)):
                return await collector.run(tickers=tickers)
            elif isinstance(collector, (GoogleNewsCollector, NewsAPICollector)):
                return await collector.run(tickers=tickers, keywords=keywords)
            elif isinstance(collector, RSSCollector):
                # ✅ FIX: pass config_manager so RSS can load feeds from knowledge_base
                return await collector.run(
                    tickers=tickers,
                    keywords=keywords,
                    config_manager=self.config_manager,
                )
            elif isinstance(collector, FreeGoogleTrendsCollector):
                return await collector.run(tickers=tickers, keywords=keywords)
            elif isinstance(collector, HuggingfaceCollector):
                return await collector.run()
            elif isinstance(collector, (FredCollector, EconomicCalendarCollector, BigQueryCollector)):
                return await collector.run()
            else:
                return await collector.run(tickers=tickers, keywords=keywords)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            raise DataProcessingError(f"Collector {name} failed: {e}") from e

    def _convert_to_dataframe(self, res: Any) -> pd.DataFrame | None:
        """Convert result to DataFrame if needed."""
        if isinstance(res, list) and len(res) > 0:
            return pd.DataFrame(res)
        elif isinstance(res, pd.DataFrame) and not res.empty:
            return res
        return None
