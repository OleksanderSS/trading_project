# src/pipeline/stages/stage_1_collection.py

import asyncio
import pandas as pd
from itertools import chain
from typing import Dict, Optional, List, Any
from functools import lru_cache

from src.pipeline.stages.base_stage import BaseStage
from src.config.unified_config_manager import UnifiedConfigManager
from src.data.collectors.collector_factory import CollectorFactory
from src.data.management.data_manager import DataManager
from src.core.logging.logger import ProjectLogger
from src.core.error_handling.error_handler import ErrorHandler


class CollectionStage(BaseStage):
    """Stage for collecting data from various sources."""

    def __init__(
        self,
        config_manager: UnifiedConfigManager,
        db_manager: DataManager,
        error_handler: ErrorHandler,
        **kwargs,
    ):
        super().__init__(config_manager, error_handler, **kwargs)
        self.db_manager = db_manager
        self.logger = ProjectLogger.get_logger(__name__)

        collector_configs = self.config_manager.get_config('collectors')
        self.factory = CollectorFactory(
            configs=collector_configs,
            http_client_factory=self.http_client_factory,
            config_manager=self.config_manager,
            db_manager=self.db_manager,
        )
        self.collectors = self.factory.get_all_collectors()
        self.logger.info(f"Loaded {len(self.collectors)} collectors.")

    async def run(self, **kwargs) -> Dict:
        self.logger.info("Starting data collection stage...")
        
        # ✅ Отримати тікери з конфігурації або kwargs
        tickers_from_kwargs = kwargs.get('tickers')
        if tickers_from_kwargs:
            self._tickers = tickers_from_kwargs
        else:
            # Отримати з assets.yaml
            assets_config = self.config_manager.get_config('assets', {})
            active_preset = assets_config.get('active_preset', 'default_volatile')
            preset_config = assets_config.get('presets', {}).get(active_preset, {})
            self._tickers = preset_config.get('tickers', ['TSLA', 'NVDA', 'SPY', 'QQQ', 'AMD'])
        
        self.logger.info(f"📊 Collection stage using {len(self._tickers)} tickers: {self._tickers}")
        
        self._prepare_collection()
        raw_data = await self._fetch_data()
        # ✅ Load existing data from DuckDB
        db_data = self.fetch_all_data_from_db()
        # Merge with collected data
        raw_data.update(db_data)
        mapped_data = self._map_to_schema(raw_data)
        # ✅ Wrap in raw_data for Stage 2
        return {'raw_data': mapped_data}
    
    def _prepare_collection(self):
        """Prepare for data collection."""
        # TEMPORARY FIX: Force data collection for all tickers
        self.logger.info("🔄 FORCING data collection for all tickers (temporary fix)")
        
        if hasattr(self.fetch_all_data_from_db, 'cache_clear'):
            self.fetch_all_data_from_db.cache_clear()
            self.logger.info("Cleared fetch_all_data_from_db cache to ensure fresh data")
        
        self.logger.info("Collection stage finished.")
    
    async def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch data from collectors."""
        # TEMPORARY FIX: Force collector execution
        self.logger.info("🔄 EXECUTING collectors for fresh data")
        
        raw_data = {}
        # ✅ Використовувати тікери з self._tickers (встановлені в run())
        tickers = getattr(self, '_tickers', ['TSLA', 'NVDA', 'SPY', 'QQQ', 'AMD'])
        keywords = ['earnings', 'fed', 'inflation', 'market', 'trading']
        
        self.logger.info(f"📊 Fetching data for {len(tickers)} tickers: {tickers}")
        
        for collector in self.collectors:
            try:
                result = await self._run_collector(collector, tickers, keywords)
                
                # Handle different result types
                if result is None:
                    continue
                    
                # Convert list to DataFrame if needed
                if isinstance(result, list):
                    if len(result) == 0:
                        continue
                    # Convert list of dicts to DataFrame
                    import pandas as pd
                    result = pd.DataFrame(result)
                
                # Handle DataFrame validation safely
                if hasattr(result, 'empty'):
                    # It's a DataFrame-like object
                    try:
                        if not result.empty:
                            table_name = f"{collector.__class__.__name__.lower().replace('collector', '')}_data"
                            raw_data[table_name] = result
                            self.logger.info(f"✅ Collected {len(result)} rows from {collector.__class__.__name__}")
                    except Exception as df_error:
                        self.logger.warning(f"⚠️ DataFrame validation failed for {collector.__class__.__name__}: {df_error}")
                        continue
                elif hasattr(result, '__len__'):
                    # Handle other iterable types
                    if len(result) > 0:
                        table_name = f"{collector.__class__.__name__.lower().replace('collector', '')}_data"
                        raw_data[table_name] = result
                        self.logger.info(f"✅ Collected {len(result)} items from {collector.__class__.__name__}")
                else:
                    # Handle single item or other types
                    self.logger.info(f"✅ Collected result from {collector.__class__.__name__}: {type(result)}")
                    table_name = f"{collector.__class__.__name__.lower().replace('collector', '')}_data"
                    raw_data[table_name] = result
                    
            except Exception as e:
                self.logger.error(f"❌ Collector {collector.__class__.__name__} failed: {e}")
        
        return raw_data
    
    def _map_to_schema(self, raw_data: Dict[str, pd.DataFrame]) -> Dict:
        """Map table names to schema keys and wrap in raw_data."""
        result = {}
        
        for table_name, df in raw_data.items():
            schema_key = self._get_schema_key(table_name, df)
            if schema_key:
                result[schema_key] = df
        
        self.logger.info(f"Mapped {len(raw_data)} tables to {len(result)} schema keys")
        
        # ✅ Wrap in raw_data for Stage 2
        return {'raw_data': result}
    
    def _get_schema_key(self, table_name: str, df: pd.DataFrame) -> str:
        """Get schema key for table name."""
        mapping_rules = [
            ('news', 'news'),
            (('market', 'yahoo', 'yf'), 'market_data'),
            (('fred', 'macro'), 'macro_data'),
            (('sentiment', 'aai'), 'sentiment_data'),
            (('fear_greed', 'vix'), 'market_sentiment'),
            (('sec', 'insider'), 'institutional_data'),
            (('trends', 'google'), 'trends_data'),
            (('economic', 'calendar'), 'economic_data'),
            (('reddit', 'social'), 'social_sentiment'),
            (('huggingface', 'ml'), 'ml_features')
        ]
        
        for patterns, schema_key in mapping_rules:
            if isinstance(patterns, tuple):
                if any(pattern in table_name.lower() for pattern in patterns):
                    return schema_key
            elif patterns in table_name.lower():
                return schema_key
        
        # Handle legacy case
        if table_name == 'raw_data' and isinstance(df, pd.DataFrame):
            self.logger.warning("Detected legacy 'raw_data' table name, remapping to market_data.")
            return 'market_data'
        
        # Default to additional_data
        return self._handle_additional_data(table_name)
    
    def _handle_additional_data(self, table_name: str) -> str:
        """Handle additional data mapping."""
        self.logger.info(f"Mapping table '{table_name}' to additional_data")
        # This will be handled in the calling method
        return f"additional_{table_name}"

    async def _run_collector(self, collector, tickers: List[str], keywords: List[str]):
        """Runs a collector with appropriate arguments based on its type."""
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
                kb = self.config_manager.get_config("knowledge_base")
                return await collector.run(
                    tickers=tickers,
                    keywords=keywords,
                    rss_feeds=kb.get("rss_feeds", []),
                    config_manager=self.config_manager,
                )
            elif isinstance(collector, FreeGoogleTrendsCollector):
                return await collector.run(tickers=tickers, keywords=keywords)
            elif isinstance(collector, HuggingfaceCollector):
                return await collector.run()
            elif isinstance(collector, (FredCollector, EconomicCalendarCollector, BigQueryCollector)):
                return await collector.run()
            else:
                self.logger.warning(f"No specific run args for {name}, trying generic run().")
                return await collector.run(tickers=tickers, keywords=keywords)
        except Exception as e:
            self.logger.error(f"Collector {name} failed: {e}", exc_info=True)
            return None

    def process_and_save_results(self, results: List, collectors: List):
        """Processes and saves results from collectors."""
        successful = 0

        for i, res in enumerate(results):
            collector = collectors[i]
            collector_type = collector.collector_type

            if self._handle_collector_result(res, collector_type):
                successful += 1

        if successful > 0:
            self.logger.info(f"Successfully processed {successful} collectors.")
    
    def _handle_collector_result(self, res: Any, collector_type: str) -> bool:
        """Handle individual collector result and save to database."""
        if isinstance(res, Exception):
            self.logger.error(f"Error in '{collector_type}': {res}")
            return False

        if res is None:
            self.logger.info(f"Collector '{collector_type}' returned no new data.")
            return False

        # Convert to DataFrame and process
        df = self._convert_to_dataframe(res)
        if df is None or df.empty:
            return False

        self.logger.info(f"Received {len(df)} records from '{collector_type}'.")
        
        # Process and save data
        return self._save_collector_data(df, collector_type)
    
    def _convert_to_dataframe(self, res: Any) -> Optional[pd.DataFrame]:
        """Convert result to DataFrame if needed."""
        if isinstance(res, list) and len(res) > 0:
            return pd.DataFrame(res)
        elif isinstance(res, pd.DataFrame) and not res.empty:
            return res
        return None
    
    def _save_collector_data(self, df: pd.DataFrame, collector_type: str) -> bool:
        """Save collector data to database with proper processing."""
        # Convert dates
        df = self._convert_dates_in_dataframe(df)
        
        # Define unique keys
        unique_on = self._get_unique_keys(collector_type, df)
        
        if not unique_on:
            self.logger.warning(
                f"No unique keys for '{collector_type}'. Duplicates may occur."
            )
        
        # Save to database
        table_name = self.config_manager.get_config('collectors', {}).get(collector_type, {}).get('table_name', collector_type)
        return self._upsert_dataframe(table_name, df, unique_on)
    
    def _convert_dates_in_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert date columns to datetime."""
        date_col = self._find_date_column_in_df(df)
        if date_col:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')
        return df
    
    def _get_unique_keys(self, collector_type: str, df: pd.DataFrame) -> List[str]:
        """Get unique keys for collector."""
        unique_on = list(self.config_manager.get_config('collectors', {}).get(collector_type, {}).get('hash_keys', []))
        
        # Add hash key if present
        if 'hash' in df.columns and 'hash' not in unique_on:
            unique_on.append('hash')
        if 'link' in df.columns and 'link' not in unique_on:
            unique_on.append('link')
            
        return unique_on
    
    def _upsert_dataframe(self, table_name: str, df: pd.DataFrame, unique_on: List[str]) -> bool:
        """Upsert dataframe to database with filtering."""
        if not self.db_manager.table_exists(table_name):
            self.db_manager.upsert(
                table_name=table_name,
                df=df,
                unique_on=unique_on,
            )
            return True
        
        # Filter new records before upsert
        new_df = self.db_manager.filter_new_records(table_name, df)
        if not new_df.empty:
            self.db_manager.upsert(
                table_name=table_name,
                df=new_df,
                unique_on=unique_on,
            )
            self.logger.info(f"Saved {len(new_df)} new records to '{table_name}'.")
        else:
            self.logger.info(f"No new records for '{table_name}' after filtering.")
            return False

    @lru_cache(maxsize=1)
    def fetch_all_data_from_db(self) -> Dict[str, pd.DataFrame]:
        """Loads all data from the database for the next stage."""
        raw_data = {}
        all_news_dfs = []

        collector_configs = self.config_manager.get_config('collectors', {})
        table_names = self.db_manager.get_all_table_names()

        for table_name in table_names:
            if self._should_skip_table(table_name):
                continue
            
            df = self.db_manager.fetch_data_from_table(table_name)
            if df is None or df.empty:
                continue
            
            self._process_table_data(df, table_name, collector_configs, raw_data, all_news_dfs)

        self._combine_news_data(all_news_dfs, raw_data)
        self._log_summary(raw_data)
        
        return raw_data
    
    def _should_skip_table(self, table_name: str) -> bool:
        """Check if table should be skipped."""
        return table_name == 'cache_metadata'
    
    def _process_table_data(self, df: pd.DataFrame, table_name: str, 
                           collector_configs: dict, raw_data: dict, all_news_dfs: list):
        """Process data from a single table."""
        collector_info = self._find_collector_config(table_name, collector_configs)
        data_type = collector_info.get('data_type') if collector_info else None
        
        if data_type == 'news':
            all_news_dfs.append(df)
            self.logger.info(f"Fetched {len(df)} records from news table '{table_name}'.")
        else:
            raw_data[table_name] = df
            self.logger.info(f"Fetched {len(df)} records from '{table_name}'.")
    
    def _find_collector_config(self, table_name: str, collector_configs: dict) -> dict:
        """Find collector config by table name."""
        for config in collector_configs.values():
            if config.get('table_name') == table_name:
                return config
        
        # Try direct match
        return collector_configs.get(table_name, {})
    
    def _combine_news_data(self, all_news_dfs: list, raw_data: dict):
        """Combine all news data sources."""
        if not all_news_dfs:
            return
        
        news_df = pd.concat(all_news_dfs, ignore_index=True)
        news_df = self._remove_news_duplicates(news_df)
        raw_data['news'] = news_df
        
        self.logger.info(
            f"Combined {len(all_news_dfs)} news sources → {len(raw_data['news'])} records."
        )
    
    def _remove_news_duplicates(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates from news data."""
        hashable_cols = self._get_hashable_columns(news_df)
        if hashable_cols:
            return news_df.drop_duplicates(subset=hashable_cols)
        return news_df
    
    def _get_hashable_columns(self, df: pd.DataFrame) -> list:
        """Get columns that contain only hashable values."""
        return [col for col in df.columns 
                if df[col].apply(lambda x: isinstance(x, (str, int, float, bool, type(None)))).all()]
    
    def _log_summary(self, raw_data: dict):
        """Log summary of fetched data."""
        total = sum(len(df) for df in raw_data.values() if isinstance(df, pd.DataFrame))
        self.logger.info(f"Total {total} records fetched from DB for next stage.")

    def _find_date_column_in_df(self, df: pd.DataFrame) -> Optional[str]:
        for col in ['created_at', 'published_at', 'timestamp', 'date', 'updated_at']:
            if col in df.columns:
                return col
        return None