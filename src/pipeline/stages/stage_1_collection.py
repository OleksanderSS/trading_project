# src/pipeline/stages/stage_1_collection.py

import asyncio
from functools import lru_cache
from typing import Any

import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler
from src.core.logging.logger import ProjectLogger
from src.data.collectors.collector_factory import CollectorFactory
from src.data.management.data_manager import DataManager
from src.pipeline.stages.base_stage import BaseStage
from src.processing.deduplication_utils import deduplicate_dataframe

# Hard cap on how long the entire RSS collector may run inside this stage.
# Individual feeds are capped inside RSSCollector itself (_FEED_TIMEOUT).
_RSS_STAGE_TIMEOUT = 300.0  # seconds


class CollectionStage(BaseStage):
    """Stage 1 – collect data from all configured sources."""

    def __init__(
        self,
        config_manager: UnifiedConfigManager,
        error_handler: ErrorHandler,
        db_manager: DataManager,
        **kwargs,
    ) -> None:
        super().__init__(config_manager, error_handler, **kwargs)
        self.db_manager = db_manager
        self.logger = ProjectLogger.get_logger(__name__)

        collector_configs = self.config_manager.get_config("collectors")
        self.factory = CollectorFactory(
            configs=collector_configs,
            http_client_factory=self.http_client_factory,
            config_manager=self.config_manager,
            db_manager=self.db_manager,
        )
        self.collectors = self.factory.get_all_collectors()
        self.logger.info(f"Loaded {len(self.collectors)} collectors.")

    # ── public interface ──────────────────────────────────────────────────────

    def execute(self, **kwargs) -> dict:
        """Synchronous entry-point (legacy compatibility)."""
        return asyncio.run(self.run(**kwargs))

    async def run(self, **kwargs) -> dict:
        self.logger.info("Starting data collection stage…")

        # Resolve tickers
        tickers_from_kwargs = kwargs.get("tickers")
        if tickers_from_kwargs:
            self._tickers = tickers_from_kwargs
        else:
            assets_config  = self.config_manager.get_config("assets", {})
            active_preset  = assets_config.get("active_preset", "default_volatile")
            preset_config  = assets_config.get("presets", {}).get(active_preset, {})
            self._tickers  = preset_config.get("tickers", ["TSLA", "NVDA", "SPY", "QQQ", "AMD"])

        self.logger.info(f"Collection stage: {len(self._tickers)} tickers → {self._tickers}")
        self._prepare_collection()

        try:
            collected_data = await self._fetch_data()
            db_data        = self.fetch_all_data_from_db()

            # Merge fresh + historical (never overwrite, always concat)
            all_data = db_data.copy()
            for key, new_df in collected_data.items():
                if (
                    key in all_data
                    and isinstance(all_data[key], pd.DataFrame)
                    and isinstance(new_df, pd.DataFrame)
                ):
                    self.logger.info(
                        f"Merging {len(new_df)} fresh + {len(all_data[key])} historical rows for '{key}'"
                    )
                    combined = pd.concat([all_data[key], new_df], ignore_index=True)
                    all_data[key] = self._safe_drop_duplicates(combined)
                else:
                    all_data[key] = new_df if not isinstance(new_df, pd.DataFrame) else self._normalize_unhashable_columns(new_df)

        except Exception as exc:
            self.logger.error(f"Data collection failed: {exc}", exc_info=True)
            all_data = {}

        mapped_data = self._map_to_schema(all_data)

        return {
            "market_data":      mapped_data.get("market_data",      pd.DataFrame()),
            "news":             mapped_data.get("news",             pd.DataFrame()),
            "macro_data":       mapped_data.get("macro_data",       pd.DataFrame()),
            "market_sentiment": mapped_data.get("market_sentiment", pd.DataFrame()),
            "models_metadata":  {},
        }

    # ── fetch helpers ─────────────────────────────────────────────────────────

    def _prepare_collection(self) -> None:
        if hasattr(self.fetch_all_data_from_db, "cache_clear"):
            self.fetch_all_data_from_db.cache_clear()
            self.logger.info("Cleared fetch_all_data_from_db cache.")

    async def _fetch_data(self) -> dict[str, pd.DataFrame]:
        tickers  = getattr(self, "_tickers", ["TSLA", "NVDA", "SPY", "QQQ", "AMD"])
        keywords = ["earnings", "fed", "inflation", "market", "trading"]

        self.logger.info(f"Fetching data for {len(tickers)} tickers…")

        raw_data: dict[str, Any] = {}

        for collector in self.collectors:
            name = collector.__class__.__name__
            self.logger.info(f"STARTING collector: {name}…")

            try:
                result = await self._run_collector(collector, tickers, keywords)
            except Exception as exc:
                self.logger.error(f"❌ Collector {name} raised unexpected exception: {exc}", exc_info=True)
                continue

            if result is None:
                continue

            # Normalise to DataFrame
            if isinstance(result, list):
                if not result:
                    continue
                result = pd.DataFrame(result)

            if isinstance(result, pd.DataFrame):
                if result.empty:
                    continue
                table_name = f"{name.lower().replace('collector', '')}_data"
                raw_data[table_name] = result
                self.logger.info(f"✅ {len(result)} rows from {name}")
            else:
                table_name = f"{name.lower().replace('collector', '')}_data"
                raw_data[table_name] = result
                self.logger.info(f"✅ result ({type(result).__name__}) from {name}")

        return raw_data

    # ── collector dispatch ────────────────────────────────────────────────────

    async def _run_collector(
        self,
        collector,
        tickers:  list[str],
        keywords: list[str],
    ) -> Any:
        """
        Dispatch a collector with the correct arguments.

        RSSCollector gets a hard total timeout (_RSS_STAGE_TIMEOUT) here.
        Individual feeds are already capped inside RSSCollector._fetch_feed().
        Any other collector that hangs will propagate as a normal exception
        (caught in _fetch_data).
        """
        # Late imports keep startup fast and avoid circular deps
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
            # ── YFinance ──────────────────────────────────────────────────
            if isinstance(collector, YFCollector):
                if not tickers:
                    self.logger.warning(f"No tickers for {name}, skipping.")
                    return None
                return await collector.run(tickers=tickers)

            # ── SEC / Insider ─────────────────────────────────────────────
            if isinstance(collector, (SECFilingsCollector, InsiderCollector)):
                return await collector.run(tickers=tickers)

            # ── News (text keywords) ──────────────────────────────────────
            if isinstance(collector, (GoogleNewsCollector, NewsAPICollector)):
                return await collector.run(tickers=tickers, keywords=keywords)

            # ── RSS — total stage timeout wraps the whole call ────────────
            if isinstance(collector, RSSCollector):
                kb        = self.config_manager.get_config("knowledge_base")
                rss_feeds = (kb or {}).get("rss_feeds", [])
                self.logger.info(f"Found {len(rss_feeds)} RSS feeds in KB")
                try:
                    return await asyncio.wait_for(
                        collector.run(
                            tickers=tickers,
                            keywords=keywords,
                            rss_feeds=rss_feeds,
                        ),
                        timeout=_RSS_STAGE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"[RSS] Total timeout ({_RSS_STAGE_TIMEOUT}s) reached – skipping RSS collector."
                    )
                    return None

            # ── Google Trends ─────────────────────────────────────────────
            if isinstance(collector, FreeGoogleTrendsCollector):
                return await collector.run(tickers=tickers, keywords=keywords)

            # ── HuggingFace ───────────────────────────────────────────────
            if isinstance(collector, HuggingfaceCollector):
                return await collector.run()

            # ── FRED / EconCal / BigQuery ─────────────────────────────────
            if isinstance(collector, (FredCollector, EconomicCalendarCollector, BigQueryCollector)):
                return await collector.run(tickers=tickers)

            # ── fallback ──────────────────────────────────────────────────
            self.logger.warning(f"No specific dispatch for {name}, using generic run().")
            return await collector.run(tickers=tickers, keywords=keywords)

        except SystemExit:
            # Never swallow a shutdown signal
            raise
        except Exception as exc:
            self.logger.error(f"Collector {name} failed: {exc}", exc_info=True)
            return None

    # ── schema mapping ────────────────────────────────────────────────────────

    def _map_to_schema(self, raw_data: dict[str, pd.DataFrame]) -> dict:
        result: dict[str, Any] = {}

        for table_name, df in raw_data.items():
            schema_key = self._get_schema_key(table_name, df)
            if schema_key in result and isinstance(result[schema_key], pd.DataFrame) and isinstance(df, pd.DataFrame):
                combined = pd.concat([result[schema_key], df], ignore_index=True)
                result[schema_key] = self._safe_drop_duplicates(combined)
            else:
                result[schema_key] = df

        self.logger.info(f"Mapped {len(raw_data)} tables → {len(result)} schema keys")
        return result

    def _normalize_unhashable_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert dict/list columns to strings so they are hashable and serializable."""
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna()
                if not sample.empty and isinstance(sample.iloc[0], (dict, list)):
                    df = df.copy()
                    df[col] = df[col].apply(
                        lambda x: x.get('name', str(x)) if isinstance(x, dict)
                        else (', '.join(str(i) for i in x) if isinstance(x, list)
                              else x)
                    )
        return df

    def _safe_drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicates safely, handling unhashable types (dict, list) in columns."""
        df_work = df.copy()

        # Serialize ALL object columns that contain dicts or lists
        for col in df_work.columns:
            if df_work[col].dtype == object:
                sample = df_work[col].dropna()
                if not sample.empty and isinstance(sample.iloc[0], (dict, list)):
                    df_work[col] = df_work[col].apply(
                        lambda x: x.get('name', str(x)) if isinstance(x, dict)
                        else (str(x) if isinstance(x, list) else x)
                    )

        try:
            deduped = df_work.drop_duplicates()
            # Return rows from original df at the deduplicated indices
            return df.loc[deduped.index].reset_index(drop=True)
        except TypeError:
            # Absolute fallback: keep all rows, no dedup
            self.logger.warning("Could not deduplicate DataFrame — keeping all rows")
            return df.reset_index(drop=True)

    def _get_schema_key(self, table_name: str, df: pd.DataFrame) -> str:
        mapping_rules = [
            ("news",                                         "news"),
            (("market", "yahoo", "yf", "market_data_raw"),  "market_data"),
            (("fred", "macro"),                             "macro_data"),
            (("sentiment", "aai"),                          "sentiment_data"),
            (("fear_greed", "vix"),                         "market_sentiment"),
            (("sec", "insider"),                            "institutional_data"),
            (("trends", "google"),                          "trends_data"),
            (("economic", "calendar"),                      "economic_data"),
            (("reddit", "social"),                          "social_sentiment"),
            (("huggingface", "ml"),                         "ml_features"),
        ]

        t = table_name.lower()
        for patterns, schema_key in mapping_rules:
            if isinstance(patterns, tuple):
                if any(p in t for p in patterns):
                    return schema_key
            elif patterns in t:
                return schema_key

        if table_name == "raw_data":
            self.logger.warning("Legacy 'raw_data' table → remapping to market_data.")
            return "market_data"

        return f"additional_{table_name}"

    # ── DB fetch (cached) ────────────────────────────────────────────────────

    @lru_cache(maxsize=1)
    def fetch_all_data_from_db(self) -> dict[str, pd.DataFrame]:
        """Load all relevant tables from DB for downstream stages."""
        raw_data: dict[str, Any]       = {}
        all_news_dfs: list[pd.DataFrame] = []

        collector_configs = self.config_manager.get_config("collectors", {})
        table_names       = self.db_manager.get_all_table_names()

        for table_name in table_names:
            if self._should_skip_table(table_name):
                continue
            self.logger.info(f"Fetching DB table '{table_name}'…")
            df = self.db_manager.fetch_data_from_table(table_name)
            if df is None or df.empty:
                continue
            # ✅ FIX: Normalize dict/list columns before any processing
            df = self._normalize_unhashable_columns(df)
            self._process_table_data(df, table_name, collector_configs, raw_data, all_news_dfs)

        self._combine_news_data(all_news_dfs, raw_data)
        self._log_db_summary(raw_data)
        return raw_data

    def _should_skip_table(self, table_name: str) -> bool:
        skip = {
            "cache_metadata",
            "huggingface_data",   # too large
            "enriched_features",  # Stage 3 output
            "experience_diary",
        }
        return table_name in skip

    def _process_table_data(
        self,
        df:               pd.DataFrame,
        table_name:       str,
        collector_configs: dict,
        raw_data:         dict,
        all_news_dfs:     list,
    ) -> None:
        collector_info = self._find_collector_config(table_name, collector_configs)
        data_type      = (collector_info or {}).get("data_type")

        if data_type == "news":
            all_news_dfs.append(df)
        else:
            raw_data[table_name] = df
        self.logger.info(f"Fetched {len(df)} rows from '{table_name}' (type={data_type or 'unknown'}).")

    def _find_collector_config(self, table_name: str, collector_configs: dict) -> dict:
        for config in collector_configs.values():
            if config.get("table_name") == table_name:
                return config
        return collector_configs.get(table_name, {})

    def _combine_news_data(self, all_news_dfs: list, raw_data: dict) -> None:
        if not all_news_dfs:
            return
        news_df = pd.concat(all_news_dfs, ignore_index=True)
        hashable_cols = [
            col for col in news_df.columns
            if news_df[col].apply(lambda x: isinstance(x, (str, int, float, bool, type(None)))).all()
        ]
        news_df, _ = deduplicate_dataframe(news_df, hashable_cols)
        raw_data["news"] = news_df
        self.logger.info(f"Combined {len(all_news_dfs)} news sources → {len(news_df)} rows.")

    def _log_db_summary(self, raw_data: dict) -> None:
        total = sum(len(df) for df in raw_data.values() if isinstance(df, pd.DataFrame))
        self.logger.info(f"Total {total} rows fetched from DB.")

    # ── legacy helpers kept for backward compat ──────────────────────────────

    def process_and_save_results(self, results: list, collectors: list) -> None:
        successful = 0
        for i, res in enumerate(results):
            if self._handle_collector_result(res, collectors[i].collector_type):
                successful += 1
        if successful:
            self.logger.info(f"Processed {successful} collectors successfully.")

    def _handle_collector_result(self, res: Any, collector_type: str) -> bool:
        if isinstance(res, Exception):
            self.logger.error(f"Error in '{collector_type}': {res}")
            return False
        if res is None:
            return False
        df = self._convert_to_dataframe(res)
        if df is None or df.empty:
            return False
        self.logger.info(f"Received {len(df)} records from '{collector_type}'.")
        return self._save_collector_data(df, collector_type)

    def _convert_to_dataframe(self, res: Any) -> pd.DataFrame | None:
        if isinstance(res, list) and res:
            return pd.DataFrame(res)
        if isinstance(res, pd.DataFrame) and not res.empty:
            return res
        return None

    def _save_collector_data(self, df: pd.DataFrame, collector_type: str) -> bool:
        df        = self._convert_dates_in_dataframe(df)
        unique_on = self._get_unique_keys(collector_type, df)
        if not unique_on:
            self.logger.warning(f"No unique keys for '{collector_type}'. Duplicates may occur.")
        table_name = (
            self.config_manager
            .get_config("collectors", {})
            .get(collector_type, {})
            .get("table_name", collector_type)
        )
        return self._upsert_dataframe(table_name, df, unique_on)

    def _convert_dates_in_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        date_col = self._find_date_column_in_df(df)
        if date_col:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
        return df

    def _get_unique_keys(self, collector_type: str, df: pd.DataFrame) -> list[str]:
        unique_on = list(
            self.config_manager
            .get_config("collectors", {})
            .get(collector_type, {})
            .get("hash_keys", [])
        )
        for col in ("hash", "link"):
            if col in df.columns and col not in unique_on:
                unique_on.append(col)
        return unique_on

    def _upsert_dataframe(
        self,
        table_name: str,
        df:         pd.DataFrame,
        unique_on:  list[str],
    ) -> bool:
        if not self.db_manager.table_exists(table_name):
            self.db_manager.upsert(table_name=table_name, df=df, unique_on=unique_on)
            return True
        new_df = self.db_manager.filter_new_records(table_name, df)
        if not new_df.empty:
            self.db_manager.upsert(table_name=table_name, df=new_df, unique_on=unique_on)
            self.logger.info(f"Saved {len(new_df)} new rows to '{table_name}'.")
        else:
            self.logger.info(f"No new rows for '{table_name}'.")
        return False

    def _find_date_column_in_df(self, df: pd.DataFrame) -> str | None:
        for col in ("created_at", "published_at", "timestamp", "date", "updated_at"):
            if col in df.columns:
                return col
        return None