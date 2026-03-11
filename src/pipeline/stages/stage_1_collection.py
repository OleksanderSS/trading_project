# src/pipeline/stages/stage_1_collection.py

import asyncio
import pandas as pd
from itertools import chain
from typing import Dict, Optional, List
from functools import lru_cache

from src.pipeline.stages.base_stage import BaseStage
from src.config.unified_config_manager import UnifiedConfigManager
from src.data.collectors.collector_factory import CollectorFactory
from src.data.management.data_manager import DataManager
from src.core.logging.logger import ProjectLogger
from src.core.error_handling.error_handler import ErrorHandler

class CollectionStage(BaseStage):
    """
    A stage for collecting data from various sources.
    """

    def __init__(self, config_manager: UnifiedConfigManager, db_manager: DataManager, error_handler: ErrorHandler, **kwargs):
        super().__init__(config_manager, error_handler, **kwargs)
        self.db_manager = db_manager
        self.logger = ProjectLogger.get_logger(__name__)
        
        collector_configs = self.config_manager.get_config('collectors')
        self.factory = CollectorFactory(
            configs=collector_configs, 
            http_client_factory=self.http_client_factory, 
            config_manager=self.config_manager, 
            db_manager=self.db_manager
        )
        self.collectors = self.factory.get_all_collectors()
        self.logger.info(f"Loaded {len(self.collectors)} collectors.")

    async def run(self, **kwargs):
        """
        Runs the data collection process.
        """
        self.logger.info("Starting data collection stage...")

        assets_config = self.config_manager.get_config('assets')
        tickers = assets_config['presets'][assets_config['active_preset']]
        self.logger.info(f"Loaded {len(tickers)} tickers from preset '{assets_config['active_preset']}'.")
        
        knowledge_base = self.config_manager.get_config('knowledge_base')
        all_keywords = list(chain.from_iterable(knowledge_base['keywords'].values()))
        keywords = list(set(all_keywords + [t.lower() for t in tickers]))
        self.logger.info(f"Loaded {len(keywords)} unique keywords.")

        tasks_to_run = []
        for collector in self.collectors:
            # Directly create a task for the collector's run method
            task = asyncio.create_task(collector.run(tickers=tickers, keywords=keywords))
            tasks_to_run.append(task)

        if tasks_to_run:
            results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
            self.process_and_save_results(results, self.collectors)
        else:
            self.logger.info("No collectors were configured to run.")

        self.logger.info("Collection stage finished. Fetching collected data for next stage...")
        return {'raw_data': self.fetch_all_data_from_db()}

    def process_and_save_results(self, results: List, collectors: List):
        """Process and save the results from the collectors that were run."""
        successful_tasks_count = 0
        
        for i, res in enumerate(results):
            collector = collectors[i]
            collector_type = collector.collector_type

            if isinstance(res, Exception):
                self.logger.error(f"Error in collector task for '{collector_type}': {res}")
                continue

            # Convert List[Dict] to DataFrame if necessary
            df = None
            if isinstance(res, list) and len(res) > 0:
                df = pd.DataFrame(res)
            elif isinstance(res, pd.DataFrame):
                df = res

            if df is not None and not df.empty:
                self.logger.info(f"Received {len(df)} records from {collector_type}. Saving to database...")
                
                date_col = self._find_date_column_in_df(df)
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col], utc=True)

                unique_on = list(collector.configs.get('hash_keys', []))
                if 'link' in df.columns and 'link' not in unique_on:
                    unique_on.append('link')
                if 'hash' in df.columns and 'hash' not in unique_on:
                    unique_on.append('hash')
                
                # If no unique keys defined, we might get duplicates, but upsert needs them
                if not unique_on:
                    self.logger.warning(f"No unique keys (hash_keys) defined for {collector_type}. Duplicates may occur.")
                
                self.db_manager.upsert(table_name=collector_type, df=df, unique_on=unique_on if unique_on else None)
                successful_tasks_count += 1
            else:
                 self.logger.info(f"Collector {collector_type} returned no new data.")
        
        if successful_tasks_count > 0:
            self.logger.info(f"Successfully executed and saved data for {successful_tasks_count} collection tasks.")

    @lru_cache(maxsize=1)
    def fetch_all_data_from_db(self) -> Dict[str, pd.DataFrame]:
        """Fetches all data from all relevant tables in the database."""
        raw_data = {}
        all_news_dfs = []
        
        collector_configs = self.config_manager.get_config('collectors', {})
        table_names = self.db_manager.get_all_table_names()

        for table_name in table_names:
            df = self.db_manager.fetch_data_from_table(table_name)
            if df is not None and not df.empty:
                collector_info = collector_configs.get(table_name, {})
                data_type = collector_info.get('data_type')

                if data_type == 'news':
                    all_news_dfs.append(df)
                    self.logger.info(f"Fetched {len(df)} records from news table '{table_name}'.")
                else:
                    raw_data[table_name] = df
                    self.logger.info(f"Fetched {len(df)} records from '{table_name}'.")
        
        if all_news_dfs:
            raw_data['news'] = pd.concat(all_news_dfs, ignore_index=True).drop_duplicates()
            self.logger.info(f"Combined {len(all_news_dfs)} news sources into a single DataFrame with {len(raw_data['news'])} records.")

        total_records = sum(len(df) for df in raw_data.values() if isinstance(df, pd.DataFrame))
        self.logger.info(f"Fetched {total_records} total records from DB for the next stage.")
        return raw_data

    def _find_date_column_in_df(self, df: pd.DataFrame) -> Optional[str]:
        """Finds a date column in a DataFrame."""
        possible_date_columns = ['created_at', 'published_at', 'timestamp', 'date', 'updated_at']
        for col in possible_date_columns:
            if col in df.columns:
                return col
        return None