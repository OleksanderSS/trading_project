from typing import Any

import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler
from src.core.logging.logger import ProjectLogger
from src.data.collectors.collector_factory import CollectorFactory
from src.data.management.data_manager import DataManager
from src.pipeline.stages.base_stage import BaseStage
from src.pipeline.stages.utils.collection_manager import CollectionManager
from src.pipeline.stages.utils.data_schema_mapper import DataSchemaMapper


class CollectionStage(BaseStage):
    """Stage for collecting data from various sources."""

    def __init__(self, config_manager: UnifiedConfigManager, db_manager:
        DataManager, error_handler: ErrorHandler, **kwargs):
        super().__init__(config_manager, error_handler, **kwargs)
        self.db_manager = db_manager
        self.logger = ProjectLogger.get_logger(__name__)
        self.schema_mapper = DataSchemaMapper()
        collector_configs = self.config_manager.get_config('collectors')
        factory = CollectorFactory(configs=collector_configs,
            http_client_factory=self.http_client_factory, config_manager=
            self.config_manager, db_manager=self.db_manager)
        self.collection_manager = CollectionManager(factory, config_manager=self.config_manager)
        self.logger.info('CollectionStage initialized.')

    async def run(self, **kwargs) ->dict:
        self.logger.info('Starting data collection stage...')
        tickers_from_kwargs = kwargs.get('tickers')
        if tickers_from_kwargs:
            self._tickers = tickers_from_kwargs
        else:
            assets_config = self.config_manager.get_config('assets', {})
            active_preset = assets_config.get('active_preset',
                'default_volatile')
            preset_config = assets_config.get('presets', {}).get(active_preset,
                {})
            self._tickers = preset_config.get('tickers', ['TSLA', 'NVDA',
                'SPY', 'QQQ', 'AMD'])
        self.logger.info(
            f'📊 Collection stage using {len(self._tickers)} tickers: {self._tickers}'
            )
        self._prepare_collection()
        keywords = ['earnings', 'fed', 'inflation', 'market', 'trading']
        raw_data = await self.collection_manager.fetch_all(self._tickers, keywords)
        db_data = self.fetch_all_data_from_db()
        raw_data.update(db_data)
        mapped_data = self.schema_mapper.map_to_schema(raw_data)
        return mapped_data

    def _prepare_collection(self):
        """Prepare for data collection."""
        self.logger.info('Collection stage finished.')

    def process_and_save_results(self, results: list, collectors: list):
        """Processes and saves results from collectors."""
        successful = 0
        for i, res in enumerate(results):
            collector = collectors[i]
            collector_type = collector.collector_type
            if self._handle_collector_result(res, collector_type):
                successful += 1
        if successful > 0:
            self.logger.info(f'Successfully processed {successful} collectors.'
                )

    def _handle_collector_result(self, res: Any, collector_type: str) ->bool:
        """Handle individual collector result and save to database."""
        if isinstance(res, Exception):
            self.logger.error(f"Error in '{collector_type}': {res}")
            return False
        if res is None:
            self.logger.info(
                f"Collector '{collector_type}' returned no new data.")
            return False
        df = self._convert_to_dataframe(res)
        if df is None or df.empty:
            return False
        self.logger.info(f"Received {len(df)} records from '{collector_type}'."
            )
        return self._save_collector_data(df, collector_type)

    def _convert_to_dataframe(self, res: Any) ->pd.DataFrame | None:
        """Convert result to DataFrame if needed."""
        if isinstance(res, list) and len(res) > 0:
            return pd.DataFrame(res)
        elif isinstance(res, pd.DataFrame) and not res.empty:
            return res
        return None

    def _save_collector_data(self, df: pd.DataFrame, collector_type: str
        ) ->bool:
        """Save collector data to database with proper processing."""
        df = self._convert_dates_in_dataframe(df)
        unique_on = self._get_unique_keys(collector_type, df)
        if not unique_on:
            self.logger.warning(
                f"No unique keys for '{collector_type}'. Duplicates may occur."
                )
        table_name = self.config_manager.get_config('collectors', {}).get(
            collector_type, {}).get('table_name', collector_type)
        return self._upsert_dataframe(table_name, df, unique_on)

    def _convert_dates_in_dataframe(self, df: pd.DataFrame) ->pd.DataFrame:
        """Convert date columns to datetime."""
        date_col = self._find_date_column_in_df(df)
        if date_col:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], utc=True, errors=
                'coerce')
        return df

    def _get_unique_keys(self, collector_type: str, df: pd.DataFrame) ->list[
        str]:
        """Get unique keys for collector."""
        unique_on = list(self.config_manager.get_config('collectors', {}).
            get(collector_type, {}).get('hash_keys', []))
        if 'hash' in df.columns and 'hash' not in unique_on:
            unique_on.append('hash')
        if 'link' in df.columns and 'link' not in unique_on:
            unique_on.append('link')
        return unique_on

    def _upsert_dataframe(self, table_name: str, df: pd.DataFrame,
        unique_on: list[str]) ->bool:
        """Upsert dataframe to database with filtering."""
        if not self.db_manager.table_exists(table_name):
            self.db_manager.upsert(table_name=table_name, df=df, unique_on=
                unique_on)
            return True
        new_df = self.db_manager.filter_new_records(table_name, df)
        if not new_df.empty:
            self.db_manager.upsert(table_name=table_name, df=new_df,
                unique_on=unique_on)
            self.logger.info(
                f"Saved {len(new_df)} new records to '{table_name}'.")
            return True
        else:
            self.logger.info(
                f"No new records for '{table_name}' after filtering.")
            return False

    def fetch_all_data_from_db(self) ->dict[str, pd.DataFrame]:
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
            self._process_table_data(df, table_name, collector_configs,
                raw_data, all_news_dfs)
        self._combine_news_data(all_news_dfs, raw_data)
        self._log_summary(raw_data)
        return raw_data

    def _should_skip_table(self, table_name: str) ->bool:
        """Check if table should be skipped."""
        return table_name == 'cache_metadata'

    def _process_table_data(self, df: pd.DataFrame, table_name: str,
        collector_configs: dict, raw_data: dict, all_news_dfs: list):
        """Process data from a single table."""
        collector_info = self._find_collector_config(table_name,
            collector_configs)
        data_type = collector_info.get('data_type') if collector_info else None
        if data_type == 'news':
            all_news_dfs.append(df)
            self.logger.info(
                f"Fetched {len(df)} records from news table '{table_name}'.")
        else:
            raw_data[table_name] = df
            self.logger.info(f"Fetched {len(df)} records from '{table_name}'.")

    def _find_collector_config(self, table_name: str, collector_configs: dict
        ) ->dict:
        """Find collector config by table name."""
        for config in collector_configs.values():
            if config.get('table_name') == table_name:
                return config
        return collector_configs.get(table_name, {})

    def _combine_news_data(self, all_news_dfs: list, raw_data: dict):
        """Combine all news data sources."""
        if not all_news_dfs:
            return
        news_df = pd.concat(all_news_dfs, ignore_index=True)
        news_df = self._remove_news_duplicates(news_df)

        # ✅ Integrated: temporal alignment check to prevent news leakage
        news_df = self._check_news_temporal_alignment(news_df, raw_data)

        raw_data['news'] = news_df
        self.logger.info(
            f"Combined {len(all_news_dfs)} news sources → {len(raw_data['news'])} records."
            )

    def _check_news_temporal_alignment(self, news_df: pd.DataFrame, raw_data: dict) -> pd.DataFrame:
        """Run TemporalAlignmentChecker to filter future-dated news."""
        try:
            from src.data.quality.temporal_alignment_checker import TemporalAlignmentChecker
            market_df = raw_data.get('market_data', raw_data.get('prices'))
            if market_df is None or not isinstance(market_df, pd.DataFrame) or market_df.empty:
                return news_df
            checker = TemporalAlignmentChecker()
            # Find the right timestamp columns
            news_ts_col = next(
                (c for c in ['published_date', 'published_at', 'publishedAt', 'datetime', 'timestamp']
                 if c in news_df.columns), None
            )
            market_ts_col = next(
                (c for c in ['datetime', 'timestamp', 'date'] if c in market_df.columns), None
            )
            if news_ts_col and market_ts_col:
                result = checker.check_news_alignment(
                    market_df, news_df,
                    market_timestamp_col=market_ts_col,
                    news_timestamp_col=news_ts_col
                )
                if result.get('future_news_count', 0) > 0:
                    self.logger.warning(
                        f"[TemporalAlignment] {result['future_news_count']} future-dated news records filtered"
                    )
                    # Filter out future news
                    if 'future_indices' in result:
                        news_df = news_df.drop(index=result['future_indices'], errors='ignore').reset_index(drop=True)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.debug(f"[TemporalAlignment] Check skipped: {e}")
        return news_df

    def _remove_news_duplicates(self, news_df: pd.DataFrame) ->pd.DataFrame:
        """Remove duplicates from news data."""
        hashable_cols = self._get_hashable_columns(news_df)
        if hashable_cols:
            return news_df.drop_duplicates(subset=hashable_cols)
        return news_df

    def _get_hashable_columns(self, df: pd.DataFrame) ->list:
        """Get columns that contain only hashable values."""
        return [col for col in df.columns if df[col].apply(lambda x:
            isinstance(x, (str, int, float, bool, type(None)))).all()]

    def _log_summary(self, raw_data: dict):
        """Log summary of fetched data."""
        total = sum(len(df) for df in raw_data.values() if isinstance(df,
            pd.DataFrame))
        self.logger.info(
            f'Total {total} records fetched from DB for next stage.')

    def _find_date_column_in_df(self, df: pd.DataFrame) ->str | None:
        for col in ['created_at', 'published_at', 'timestamp', 'date',
            'updated_at']:
            if col in df.columns:
                return col
        return None
