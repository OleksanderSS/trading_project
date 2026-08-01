# src/pipeline/stages/stage_1_collection.py

import asyncio
import hashlib
import json
from itertools import chain

import pandas as pd
from datetime import datetime

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler
from src.core.logging.logger import ProjectLogger
from src.data.collectors.aaii_sentiment_collector import AIISentimentCollector
from src.data.collectors.bigquery_collector import BigQueryCollector
from src.data.collectors.cftc_collector import CFTCCollector
from src.data.collectors.collector_factory import CollectorFactory
from src.data.collectors.economic_calendar_collector import EconomicCalendarCollector
from src.data.collectors.fear_greed_collector import FearGreedCollector
from src.data.collectors.fred_collector import FredCollector
from src.data.collectors.free_google_trends_collector import FreeGoogleTrendsCollector
from src.data.collectors.google_news_collector import GoogleNewsCollector
from src.data.collectors.huggingface_collector import HuggingfaceCollector
from src.data.collectors.insider_collector import InsiderCollector
from src.data.collectors.newsapi_collector import NewsAPICollector
from src.data.collectors.put_call_ratio_collector import PutCallRatioCollector
from src.data.collectors.reddit_sentiment_collector import RedditSentimentCollector
from src.data.collectors.rss_collector import RSSCollector
from src.data.collectors.sec_filings_collector import SECFilingsCollector
from src.data.collectors.vix_collector import VIXCollector
from src.data.collectors.yf_collector import YFCollector
from src.data.management.data_manager import DataManager
from src.pipeline.stages.base_stage import BaseStage


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

    async def run(self, **kwargs) -> dict:
        self.logger.info("Starting data collection stage...")

        # --- Тікери ---
        assets_config = self.config_manager.get_config('assets') or {}
        active_preset = assets_config.get('active_preset')
        # ФІКС: правильно дістаємо список тікерів з пресету
        tickers = (
            assets_config
            .get('presets', {})
            .get(active_preset, {})
            .get('tickers', [])
        )
        tickers_override = kwargs.get('tickers')
        if tickers_override:
            if isinstance(tickers_override, dict):
                tickers = list(tickers_override.keys())
            else:
                tickers = list(tickers_override)
            self.logger.info(f"Using {len(tickers)} tickers from pipeline inputs override.")
        if not tickers:
            self.logger.error(f"No tickers found for preset '{active_preset}'. Aborting collection.")
            # Return the same SHAPE as the success path (fetch_all_data_from_db
            # returns a flat {data_type: DataFrame} map, which the orchestrator
            # assigns straight to stage_outputs['raw_data']). The old
            # {'raw_data': {}} nested one level deeper AND was truthy, so the
            # abort survived every emptiness check and only surfaced later as a
            # confusing failure in ProcessingStage.
            return {}
        self.logger.info(f"Loaded {len(tickers)} tickers from preset '{active_preset}'.")

        # --- Keywords: flatten словника категорій ---
        knowledge_base = self.config_manager.get_config('knowledge_base') or {}
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
        force_collectors = kwargs.get('force_collectors', [])

        for collector in self.collectors:
            freq = collector.configs.get('frequency', 'daily')
            collector_name = collector.collector_type

            # Пропуск On-Demand колекторів (погода, GDELT), якщо вони не викликані явно
            if freq == 'on_demand' and collector_name not in force_collectors:
                self.logger.debug(f"Skipping {collector_name} (on_demand, not forced).")
                continue

            # Пропуск Weekly колекторів, якщо сьогодні не понеділок (і не forced)
            if freq == 'weekly' and datetime.now().weekday() != 0 and collector_name not in force_collectors:
                self.logger.debug(f"Skipping {collector_name} (weekly, runs on Monday).")
                continue

            task = asyncio.create_task(
                self._run_collector(collector, tickers=tickers, keywords=keywords)
            )
            tasks_to_run.append(task)

        if tasks_to_run:
            results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
            self.process_and_save_results(results, self.collectors)
        else:
            self.logger.info("No collectors were configured to run.")

        self.logger.info("Collection stage finished.")
        return self.fetch_all_data_from_db(tickers=tickers)

    async def _run_collector(self, collector, tickers: list[str], keywords: list[str]):
        """Запускає колектор з правильними аргументами залежно від його типу."""

        name = collector.__class__.__name__
        collector_name = getattr(collector, 'collector_name', name)

        try:
            # Додаємо таймаут для кожного колектора
            timeout = 300  # 5 хвилин максимум

            if isinstance(collector, YFCollector):
                return await asyncio.wait_for(collector.run(tickers=tickers), timeout=timeout)
            elif isinstance(collector, (SECFilingsCollector, InsiderCollector)):
                return await asyncio.wait_for(collector.run(tickers=tickers), timeout=timeout)
            elif isinstance(collector, (GoogleNewsCollector, NewsAPICollector)):
                return await asyncio.wait_for(collector.run(tickers=tickers, keywords=keywords), timeout=timeout)
            elif isinstance(collector, RSSCollector):
                kb = self.config_manager.get_config("knowledge_base")
                return await asyncio.wait_for(collector.run(
                    tickers=tickers,
                    keywords=keywords,
                    rss_feeds=kb.get("rss_feeds", []),
                ), timeout=timeout)
            elif isinstance(collector, FreeGoogleTrendsCollector):
                # Для Google Trends збільшуємо таймаут через retry
                return await asyncio.wait_for(collector.run(tickers=tickers, keywords=keywords), timeout=600)
            elif isinstance(collector, HuggingfaceCollector):
                return await asyncio.wait_for(collector.run(), timeout=timeout)
            elif isinstance(collector, (FredCollector, EconomicCalendarCollector, BigQueryCollector,
                                        VIXCollector, PutCallRatioCollector, FearGreedCollector,
                                        AIISentimentCollector, CFTCCollector, RedditSentimentCollector)):
                return await asyncio.wait_for(collector.run(), timeout=timeout)
            else:
                self.logger.warning(f"No specific run args for {name}, trying generic run().")
                return await asyncio.wait_for(collector.run(tickers=tickers, keywords=keywords), timeout=timeout)

        except TimeoutError as e:
            self.logger.error(f"Collector {name} перевищив таймаут {timeout} секунд")
            # Re-raise (not return None) so this reaches
            # process_and_save_results as an Exception via
            # asyncio.gather(..., return_exceptions=True) - that method
            # already has a dedicated `isinstance(res, Exception)` branch
            # for real failures, but it was dead code as long as this
            # swallowed every failure into None, which is treated
            # identically to "collector ran fine, nothing new to report"
            # and counted as a success.
            raise
        except Exception as e:
            self.logger.error(f"Collector {name} failed: {e}", exc_info=True)
            raise

    def process_and_save_results(self, results: list, collectors: list):
        """
        УНІФІКОВАНА обробка та збереження результатів колекторів.
        Всі колектори обробляються однаково - без SELF_SAVING.
        """
        successful = 0

        for i, res in enumerate(results):
            collector = collectors[i]
            collector_type = collector.collector_type

            if isinstance(res, Exception):
                self.logger.error(f"Error in '{collector_type}': {res}")
                continue

            if res is None:
                self.logger.info(f"Collector '{collector_type}' returned no new data.")
                successful += 1
                continue

            # Конвертуємо в DataFrame якщо потрібно
            df = self._normalize_dataframe(res)

            if df is None or df.empty:
                self.logger.info(f"Collector '{collector_type}' returned empty data.")
                successful += 1
                continue

            self.logger.info(f"Received {len(df)} records from '{collector_type}'.")

            # ЄДИНА логіка обробки для всіх колекторів
            try:
                # 1. Нормалізація даних
                df = self._normalize_data(df, collector)

                # 2. Збереження в базу даних
                self._save_collector_data(collector, df)

                # 3. Кешування результатів
                self._cache_collector_data(collector, df)

                successful += 1

            except Exception as e:
                self.logger.error(f"Failed to process data from '{collector_type}': {e}")
                continue

        if successful > 0:
            self.logger.info(f"Successfully processed {successful} collectors.")

    def _normalize_dataframe(self, result) -> pd.DataFrame | None:
        """
        Нормалізує результат в DataFrame.
        """
        if isinstance(result, pd.DataFrame):
            return result
        elif isinstance(result, list) and len(result) > 0:
            return pd.DataFrame(result)
        else:
            return None

    def _normalize_data(self, df: pd.DataFrame, collector) -> pd.DataFrame:
        """
        Нормалізує дані: конвертація дат, додавання хешів.
        """
        # Конвертуємо дати
        date_col = self._find_date_column_in_df(df)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')

        # Додаємо хеші якщо потрібно.
        # collector.generate_hash() doesn't exist on BaseCollector or most
        # subclasses (some define a private _generate_hash, some define a
        # public generate_hash, some define neither) - calling it here
        # unconditionally raised AttributeError for every collector that
        # doesn't happen to have a public generate_hash, which was silently
        # swallowed by the broad except in process_and_save_results,
        # discarding real collected data before it ever reached the DB.
        # Every collector that does define a hash method uses the exact
        # same formula (see cftc/fear_greed/put_call_ratio/vix/aaii/
        # fred_collector.py), so compute it directly here instead of
        # depending on a per-collector method existing at all.
        if 'hash' not in df.columns and collector.get_hash_keys():
            hash_keys = collector.get_hash_keys()
            df['hash'] = df.apply(
                lambda row: hashlib.sha256(
                    '|'.join(str(row.get(key, '')) for key in hash_keys).encode()
                ).hexdigest(),
                axis=1,
            )

        return df

    def _save_collector_data(self, collector, df: pd.DataFrame):
        """
        Єдине збереження даних для всіх колекторів.
        """
        table_name = collector.get_table_name()
        hash_keys = collector.get_hash_keys()

        # Фільтруємо нові записи
        new_df = self.db_manager.filter_new_records(table_name, df)

        if not new_df.empty:
            self.db_manager.upsert(table_name, new_df, unique_on=hash_keys)
            self.logger.info(f"Saved {len(new_df)} new records to '{table_name}'.")
        else:
            self.logger.info(f"No new records for '{table_name}' after filtering.")

    def _cache_collector_data(self, collector, df: pd.DataFrame):
        """
        Єдине кешування для всіх колекторів.
        """
        if not collector.cache_manager:
            return

        cache_key = f"{collector.collector_type}_data"
        cache_params = {
            "tickers": collector.get_active_tickers(),
            "data_type": collector.get_data_type(),
        }

        try:
            collector.cache_manager.set(
                cache_key,
                df.to_dict("records"),
                cache_params,
                ttl=collector.get_cache_ttl(),
                namespace="collectors"
            )
            self.logger.debug(f"Cached {len(df)} records for '{collector.collector_type}'.")
        except Exception as e:
            self.logger.warning(f"Failed to cache data for '{collector.collector_type}': {e}")


    def fetch_all_data_from_db(self, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
            """Завантажує всі дані з БД для наступного етапу."""
            raw_data = {}
            all_news_dfs = []

            collector_configs = self.config_manager.get_config('collectors', {}) or {}
            table_name_to_info = {}
            for name, info in collector_configs.items():
                if isinstance(info, dict):
                    table = info.get('table_name')
                    if table:
                        table_name_to_info[table] = {**info, "_name": name}

            table_names = self.db_manager.get_all_table_names()

            news_types = {'google_news', 'rss', 'newsapi', 'sec_filings', 'huggingface', 'hugging_face'}
            macro_types = {'fred', 'economic_calendar', 'custom_csv'}
            market_types = {'yahoo_finance'}
            trends_types = {'free_google_trends'}
            reddit_types = {'reddit', 'reddit_sentiment'}

            # Завантажуємо активні тікери та ключові слова для фільтрації
            if tickers:
                active_tickers = tickers
            else:
                assets_config = self.config_manager.get_config('assets') or {}
                active_preset = assets_config.get('active_preset')
                active_tickers = (
                    assets_config
                    .get('presets', {})
                    .get(active_preset, {})
                    .get('tickers', [])
                )
            active_tickers_lower = [t.lower() for t in active_tickers]

            knowledge_base = self.config_manager.get_config('knowledge_base') or {}
            keywords_raw = knowledge_base.get('keywords', {})
            if isinstance(keywords_raw, dict):
                all_keywords = list(set(chain.from_iterable(keywords_raw.values())))
            else:
                all_keywords = list(keywords_raw)
            all_keywords_lower = [k.lower() for k in all_keywords]

            for table_name in table_names:
                # Пропускаємо службову таблицю кешу
                if table_name == 'cache_metadata':
                    continue

                df = self.db_manager.fetch_data_from_table(table_name)
                if df is None or df.empty:
                    continue

                collector_info = table_name_to_info.get(table_name, {})
                data_type = collector_info.get('data_type')
                collector_type = collector_info.get('type', '')

                if not data_type:
                    if collector_type in news_types or 'news' in table_name:
                        data_type = 'news'
                    elif collector_type in macro_types or table_name in {'fred_data', 'economic_calendar', 'news_patterns'}:
                        data_type = 'macro_data'
                    elif collector_type in market_types or table_name in {'market_data_raw', 'market_data'}:
                        data_type = 'market_data'
                    elif collector_type in trends_types or 'trends' in table_name:
                        data_type = 'google_trends'
                    elif collector_type in reddit_types or 'reddit' in table_name:
                        data_type = 'reddit_sentiment'

                if data_type == 'news':
                    df_filtered = self._filter_news_by_keywords_and_tickers(
                        df, all_keywords_lower, active_tickers_lower
                    )
                    if not df_filtered.empty:
                        all_news_dfs.append(df_filtered)
                        self.logger.info(
                            f"Fetched {len(df_filtered)}/{len(df)} records from news table '{table_name}' "
                            f"(filtered by keywords/tickers)."
                        )
                    else:
                        self.logger.info(f"No matching records in news table '{table_name}' after filtering.")
                elif data_type in ('macro_data', 'macro', 'macro_context'):
                    if 'macro_data' in raw_data:
                        raw_data['macro_data'] = pd.concat([raw_data['macro_data'], df], ignore_index=True)
                    else:
                        raw_data['macro_data'] = df
                    self.logger.info(f"Fetched {len(df)} records from macro table '{table_name}'.")
                elif data_type in ('market_data', 'market'):
                    df_filtered = self._filter_market_data_by_tickers(df, active_tickers_lower)
                    if not df_filtered.empty:
                        if 'market_data' in raw_data:
                            raw_data['market_data'] = pd.concat([raw_data['market_data'], df_filtered], ignore_index=True)
                        else:
                            raw_data['market_data'] = df_filtered
                        self.logger.info(
                            f"Fetched {len(df_filtered)}/{len(df)} records from market table '{table_name}' "
                            f"(filtered by tickers)."
                        )
                    else:
                        self.logger.info(f"No matching records in market table '{table_name}' after filtering.")
                elif data_type == 'google_trends':
                    df_filtered = self._filter_trends_by_keywords_and_tickers(
                        df, all_keywords_lower, active_tickers_lower
                    )
                    if not df_filtered.empty:
                        raw_data['google_trends'] = df_filtered
                        self.logger.info(
                            f"Fetched {len(df_filtered)}/{len(df)} records from trends table '{table_name}' "
                            f"(filtered by keywords/tickers)."
                        )
                elif data_type == 'reddit_sentiment':
                    raw_data['reddit_sentiment'] = df
                    self.logger.info(f"Fetched {len(df)} records from reddit table '{table_name}'.")
                else:
                    raw_data[table_name] = df
                    self.logger.info(f"Fetched {len(df)} records from '{table_name}'.")

            if all_news_dfs:
                # Спочатку нормалізуємо назви колонок з датами в усіх датафреймах
                normalized_news_dfs = []
                for df in all_news_dfs:
                    df = df.copy()
                    # Знаходимо колонку, яка містить дату публікації (різні варіанти)
                    date_col = None
                    for possible_name in ['published_at', 'publishedAt', 'published_date', 'filing_date', 'date', 'timestamp']:
                        if possible_name in df.columns:
                            date_col = possible_name
                            break
                    # Якщо знайшли і вона не є стандартною 'published_at' – перейменовуємо
                    if date_col and date_col != 'published_at':
                        df = df.rename(columns={date_col: 'published_at'})
                        self.logger.debug(f"Renamed date column '{date_col}' to 'published_at' for a news source")
                    normalized_news_dfs.append(df)

                # Об'єднуємо всі нормалізовані датафрейми
                news_df = pd.concat(normalized_news_dfs, ignore_index=True)

                # --- ПОЧАТОК ВИПРАВЛЕНОГО БЛОКУ ---
                # Примусово серіалізуємо всі колонки, які містять хоча б одне значення-словник
                self.logger.info("Checking for dictionary columns in news DataFrame...")
                for col in news_df.select_dtypes(include=['object']).columns:
                    # Перевіряємо ВСІ непорожні значення на наявність dict
                    non_null = news_df[col].dropna()
                    if non_null.empty:
                        continue
                    has_dict = non_null.apply(lambda x: isinstance(x, dict)).any()
                    if has_dict:
                        self.logger.warning(f"Column '{col}' contains dictionaries. Converting to JSON strings.")
                        # Додатково: покажемо приклад
                        example = non_null[non_null.apply(lambda x: isinstance(x, dict))].iloc[0]
                        self.logger.info(f"Example dict in column '{col}': {example}")
                        news_df[col] = news_df[col].apply(
                            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x
                        )
                # --- КІНЕЦЬ ВИПРАВЛЕНОГО БЛОКУ ---

                # Визначаємо, які з потрібних колонок реально присутні
                required_cols = ['title', 'published_at', 'source']
                existing_cols = [col for col in required_cols if col in news_df.columns]

                if existing_cols:
                    try:
                        news_df = news_df.drop_duplicates(subset=existing_cols)
                        self.logger.info(f"Deduplicated news by {existing_cols}: {len(news_df)} records")
                    except TypeError as e:
                        self.logger.error(f"TypeError in drop_duplicates: {e}")
                        # Діагностика проблемної колонки
                        for col in existing_cols:
                            try:
                                news_df.duplicated(subset=[col])
                            except TypeError as e2:
                                self.logger.error(f"Problematic column: '{col}' - {e2}")
                                sample = news_df[col].dropna().head(5).tolist()
                                self.logger.error(f"Sample values in column '{col}': {sample}")
                                # Примусово конвертуємо в рядок
                                news_df[col] = news_df[col].astype(str)
                        # Повторюємо спробу
                        news_df = news_df.drop_duplicates(subset=existing_cols)
                        self.logger.info(f"Deduplicated after forced conversion: {len(news_df)} records")
                else:
                    self.logger.warning("No common columns for deduplication found, skipping duplicate removal")

                raw_data['news'] = news_df
                self.logger.info(
                    f"Combined {len(all_news_dfs)} news sources → "
                    f"{len(raw_data['news'])} records."
                )
            # Backwards-compatible aliases for legacy entrypoints
            if 'market_data' in raw_data and 'market_data_raw' not in raw_data:
                raw_data['market_data_raw'] = raw_data['market_data']
            if 'macro_data' in raw_data and 'fred_data' not in raw_data:
                raw_data['fred_data'] = raw_data['macro_data']

            total = sum(len(df) for df in raw_data.values() if isinstance(df, pd.DataFrame))
            self.logger.info(f"Total {total} records fetched from DB for next stage.")
            return raw_data

    def _find_date_column_in_df(self, df: pd.DataFrame) -> str | None:
        for col in ['created_at', 'published_at', 'timestamp', 'date', 'updated_at']:
            if col in df.columns:
                return col
        return None

    def _filter_news_by_keywords_and_tickers(
        self, df: pd.DataFrame, keywords: list[str], tickers: list[str]
    ) -> pd.DataFrame:
        """Фільтрує новини за ключовими словами та тікерами."""
        if df.empty:
            return df

        text_cols = [c for c in ['title', 'description', 'content', 'text'] if c in df.columns]
        if not text_cols:
            return df

        # Об'єднуємо текстові колонки
        df['_combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1).str.lower()

        # Фільтруємо за ключовими словами або тікерами
        mask = df['_combined_text'].str.contains(
            '|'.join(keywords + tickers), case=False, na=False, regex=True
        )
        result = df[mask].drop(columns=['_combined_text'])
        return result

    def _filter_market_data_by_tickers(
        self, df: pd.DataFrame, tickers: list[str]
    ) -> pd.DataFrame:
        """Фільтрує ринкові дані за тікерами."""
        if df.empty or 'ticker' not in df.columns:
            return df

        mask = df['ticker'].str.lower().isin(tickers)
        return df[mask]

    def _filter_trends_by_keywords_and_tickers(
        self, df: pd.DataFrame, keywords: list[str], tickers: list[str]
    ) -> pd.DataFrame:
        """Фільтрує тренди за ключовими словами та тікерами."""
        if df.empty:
            return df

        text_cols = [c for c in ['keyword', 'term', 'query', 'search_term'] if c in df.columns]
        if not text_cols:
            return df

        df['_combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1).str.lower()
        mask = df['_combined_text'].str.contains(
            '|'.join(keywords + tickers), case=False, na=False, regex=True
        )
        result = df[mask].drop(columns=['_combined_text'])
        return result

