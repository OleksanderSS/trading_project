# src/data/collectors/economic_calendar_collector.py

import asyncio
import hashlib
import pandas as pd
from io import StringIO
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .base_collector import BaseCollector
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager
from src.core.cache.cache_manager import CacheManager


class EconomicCalendarCollector(BaseCollector):
    """Збирає дані економічного календаря з Investing.com API."""
    collector_type = "economic_calendar"
    data_type = "economic"

    def __init__(
        self,
        configs: Dict[str, Any],
        http_client_factory: HttpClientFactory,
        db_manager: DataManager,
        cache_manager: Optional[CacheManager] = None,
        **kwargs,
    ):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)

    async def run(self, **kwargs) -> Optional[pd.DataFrame]:
        table_name = self.configs.get("table_name", "economic_calendar")
        cache_key = f"{self.__class__.__name__}_run"

        start_date, end_date = self._get_date_range()
        cache_params = {
            "start": str(start_date.date()),
            "end": str(end_date.date()),
        }

        # 1. Кеш
        if self.cache_manager:
            cached = self.cache_manager.get(cache_key, cache_params, namespace="collectors")
            if cached is not None:
                df_cached = pd.DataFrame(cached) if isinstance(cached, list) else cached
                if "hash" in df_cached.columns:
                    new_from_cache = self.db_manager.filter_new_records(table_name, df_cached)
                    if new_from_cache.empty:
                        self.logger.info("[EconCalendar] Cache hit — нових подій немає.")
                        return None
                    return new_from_cache

        raw = await self.fetch_raw_data()
        if not raw:
            return None

        df = pd.DataFrame(raw)
        df["hash"] = df.apply(
            lambda row: hashlib.sha256(
                f"{row.get('timestamp','')}|{row.get('event','')}|{row.get('country','')}".encode()
            ).hexdigest(),
            axis=1,
        )

        new_df = self.db_manager.filter_new_records(table_name, df)
        if new_df.empty:
            self.logger.info("[EconCalendar] Нових подій не знайдено.")
            if self.cache_manager:
                self.cache_manager.set(
                    cache_key, df.to_dict("records"), cache_params, namespace="collectors"
                )
            return None

        self.db_manager.upsert(table_name, new_df, unique_on=["hash"])

        if self.cache_manager:
            self.cache_manager.set(
                cache_key, df.to_dict("records"), cache_params, namespace="collectors"
            )

        self.logger.info(f"[EconCalendar] Збережено {len(new_df)} нових подій.")
        return new_df

    async def fetch_raw_data(self, **kwargs) -> List[Dict[str, Any]]:
        api_url = self.configs.get("api_url")
        if not api_url:
            self.logger.error("Відсутній 'api_url' в конфігурації.")
            return []

        headers = self.configs.get("headers")
        if not headers:
            self.logger.error("Відсутні 'headers' в конфігурації.")
            return []

        start_date, end_date = self._get_date_range()
        payload = self._build_payload(start_date, end_date)

        try:
            client = self.http_client_factory.get_http_client()
            response = await client.post(api_url, data=payload, headers=headers)
            response.raise_for_status()

            html_data = response.json().get("data")
            if not html_data:
                self.logger.warning("[EconCalendar] Відповідь не містить 'data'.")
                return []

            records = await asyncio.to_thread(self._parse_html, html_data)
            self.logger.info(f"[EconCalendar] Розпарсено {len(records)} подій.")
            return records

        except Exception as e:
            self.handle_error(e, {"url": api_url})
            return []

    def _get_date_range(self):
        days_back = self.configs.get("days_back", 7)
        days_ahead = self.configs.get("days_ahead", 30)
        start = datetime.now() - timedelta(days=days_back)
        end = datetime.now() + timedelta(days=days_ahead)
        return start, end

    def _build_payload(self, start_date: datetime, end_date: datetime) -> Dict:
        countries = self.configs.get("countries", [])
        importance = self.configs.get("importance", [])
        api_mappings = self.configs.get("api_mappings", {})

        country_map = api_mappings.get("country", {})
        impact_map = api_mappings.get("impact", {})

        payload = self.configs.get("request_payload", {}).copy()
        payload.update({
            "country[]": [country_map[c] for c in countries if c in country_map],
            "importance[]": [impact_map[i] for i in importance if i in impact_map],
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
        })
        return payload

    def _parse_html(self, html_data: str) -> List[Dict[str, Any]]:
        column_names = self.configs.get("column_names")
        if not column_names:
            self.logger.error("Відсутні 'column_names' в конфігурації.")
            return []
        try:
            df = pd.read_html(StringIO(f"<table>{html_data}</table>"))[0]
            df.columns = column_names

            if "_del" in df.columns:
                df = df.drop(columns=["_del"])

            df["date"] = pd.to_datetime(df.get("time", ""), errors="coerce").dt.strftime("%Y-%m-%d")
            df["date"] = df["date"].ffill()
            df = df.dropna(subset=["event"]).copy()

            if df.empty:
                return []

            df["timestamp"] = pd.to_datetime(
                df["date"] + " " + df.get("time", "").astype(str),
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
                utc=True,
            )
            df = df.dropna(subset=["timestamp"])
            df["impact"] = df["impact"].astype(str).str.strip()

            if "currency" in df.columns:
                df = df.rename(columns={"currency": "country"})

            final_cols = ["timestamp", "country", "impact", "event", "actual", "forecast", "previous"]
            df = df.reindex(columns=final_cols)
            return df.to_dict("records")

        except Exception as e:
            self.logger.error(f"[EconCalendar] Помилка парсингу HTML: {e}", exc_info=True)
            return []