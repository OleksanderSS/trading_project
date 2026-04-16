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

        # --- Тікери ---
        # ✅ ПРІОРИТЕТ: якщо тікери передані явно (наприклад, через CLI), використовуємо їх
        tickers = kwargs.get('tickers')
        
        if tickers:
            self.logger.info(f"🧪 Використання переданих тікерів: {tickers}")
        else:
            assets_config = self.config_manager.get_config('assets')
            active_preset = assets_config.get('active_preset')
            # Завантажуємо з пресету лише якщо нічого не передано явно
            tickers = (
                assets_config
                .get('presets', {})
                .get(active_preset, {})
                .get('tickers', [])
            )
            self.logger.info(f"Loaded {len(tickers)} tickers from preset '{active_preset}'.")

        if not tickers:
            self.logger.error("No tickers available. Aborting collection.")
            return {'raw_data': {}}

        # --- Keywords: flatten словника категорій ---
        knowledge_base = self.config_manager.get_config('knowledge_base')
        keywords_raw = knowledge_base.get('keywords', {})
        if isinstance(keywords_raw, dict):
            all_keywords = list(set(chain.from_iterable(keywords_raw.values())))
        else:
            all_keywords = list(keywords_raw)

        # Тікери в lowercase додаємо до keywords для новинних колекторів
        keywords = list(set(all_keywords + [t.lower() for t in tickers]))
        self.logger.info(f"Loaded {len(keywords)} unique keywords.")

        # --- Запуск колекторів ---
        # Розділяємо на важкі (новини) і легкі колектори
        # Важкі запускаємо з семафором щоб не перевантажити мережу
        tasks_to_run = []
        for collector in self.collectors:
            task = asyncio.create_task(
                self._run_collector(collector, tickers=tickers, keywords=keywords)
            )
            tasks_to_run.append(task)

        if tasks_to_run:
            try:
                # Add timeout to prevent hanging
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks_to_run, return_exceptions=True),
                    timeout=300  # 5 minutes timeout
                )
                self.process_and_save_results(results, self.collectors)
            except asyncio.TimeoutError:
                self.logger.warning("Collection timeout after 5 minutes, processing partial results")
                # Cancel remaining tasks
                for task in tasks_to_run:
                    if not task.done():
                        task.cancel()
                # Wait for tasks to finish cancellation
                await asyncio.gather(*tasks_to_run, return_exceptions=True)
            except Exception as e:
                self.logger.error(f"Collection failed: {e}")
                # Cancel remaining tasks
                for task in tasks_to_run:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks_to_run, return_exceptions=True)
        else:
            self.logger.info("No collectors were configured to run.")

        # CRITICAL FIX: Clear cache before fetching to ensure fresh data
        if hasattr(self.fetch_all_data_from_db, 'cache_clear'):
            self.fetch_all_data_from_db.cache_clear()
            self.logger.info("Cleared fetch_all_data_from_db cache to ensure fresh data")

        self.logger.info("Collection stage finished.")
        return {'raw_data': self.fetch_all_data_from_db()}

    async def _run_collector(self, collector, tickers: List[str], keywords: List[str]):
        """Запускає колектор з правильними аргументами залежно від його типу."""
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
        from datetime import datetime

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
        """Обробляє та зберігає результати колекторів."""
        successful = 0

        for i, res in enumerate(results):
            collector = collectors[i]
            collector_type = collector.collector_type

            if isinstance(res, Exception):
                self.logger.error(f"Error in '{collector_type}': {res}")
                continue

            if res is None:
                self.logger.info(f"Collector '{collector_type}' returned no new data.")
                continue

            # Конвертуємо в DataFrame якщо потрібно
            df = None
            if isinstance(res, list) and len(res) > 0:
                df = pd.DataFrame(res)
            elif isinstance(res, pd.DataFrame) and not res.empty:
                df = res

            if df is not None and not df.empty:
                self.logger.info(f"Received {len(df)} records from '{collector_type}'.")

                # Конвертуємо дати
                date_col = self._find_date_column_in_df(df)
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')

                # Визначаємо унікальні ключі
                unique_on = list(collector.configs.get('hash_keys', []))
                if 'hash' in df.columns and 'hash' not in unique_on:
                    unique_on.append('hash')
                if 'link' in df.columns and 'link' not in unique_on:
                    unique_on.append('link')

                if not unique_on:
                    self.logger.warning(
                        f"No unique keys for '{collector_type}'. Duplicates may occur."
                    )

                # Колектори що вже зберегли дані самостійно (мають run() з upsert)
                # не потребують повторного збереження — перевіряємо по наявності hash
                table_name = collector.configs.get('table_name', collector_type)
                if not self.db_manager.table_exists(table_name):
                    self.db_manager.upsert(
                        table_name=table_name,
                        df=df,
                        unique_on=unique_on if unique_on else ['hash'],
                    )
                else:
                    # Фільтруємо нові перед upsert
                    new_df = self.db_manager.filter_new_records(table_name, df)
                    if not new_df.empty:
                        self.db_manager.upsert(
                            table_name=table_name,
                            df=new_df,
                            unique_on=unique_on if unique_on else ['hash'],
                        )
                        self.logger.info(f"Saved {len(new_df)} new records to '{table_name}'.")
                    else:
                        self.logger.info(f"No new records for '{table_name}' after filtering.")

                successful += 1

        if successful > 0:
            self.logger.info(f"Successfully processed {successful} collectors.")

    @lru_cache(maxsize=1)
    def fetch_all_data_from_db(self) -> Dict[str, pd.DataFrame]:
        """Завантажує всі дані з БД для наступного етапу."""
        raw_data = {}
        all_news_dfs = []

        collector_configs = self.config_manager.get_config('collectors', {})
        table_names = self.db_manager.get_all_table_names()

        for table_name in table_names:
            # Пропускаємо службову таблицю кешу
            if table_name == 'cache_metadata':
                continue

            df = self.db_manager.fetch_data_from_table(table_name)
            if df is None or df.empty:
                continue

            collector_info = collector_configs.get(table_name, {})
            data_type = collector_info.get('data_type')

            if data_type == 'news':
                all_news_dfs.append(df)
                self.logger.info(f"Fetched {len(df)} records from news table '{table_name}'.")
            else:
                raw_data[table_name] = df
                self.logger.info(f"Fetched {len(df)} records from '{table_name}'.")

        if all_news_dfs:
            raw_data['news'] = (
                pd.concat(all_news_dfs, ignore_index=True)
                .drop_duplicates()
            )
            self.logger.info(
                f"Combined {len(all_news_dfs)} news sources → "
                f"{len(raw_data['news'])} records."
            )

        total = sum(len(df) for df in raw_data.values() if isinstance(df, pd.DataFrame))
        self.logger.info(f"Total {total} records fetched from DB for next stage.")
        return raw_data

    def _find_date_column_in_df(self, df: pd.DataFrame) -> Optional[str]:
        for col in ['created_at', 'published_at', 'timestamp', 'date', 'updated_at']:
            if col in df.columns:
                return col
        return None