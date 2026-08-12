# src/data/collectors/wikimedia_attention_collector.py

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector


class WikimediaAttentionCollector(BaseCollector):
    """
    Collects Wikipedia pageview statistics as a proxy for public attention/demand.
    Uses the free Wikimedia REST API.
    """
    collector_type = "wikimedia_attention"
    data_type = "alternative"

    def __init__(
        self,
        configs: dict[str, Any],
        http_client_factory: HttpClientFactory,
        db_manager: DataManager,
        cache_manager: CacheManager | None = None,
        **kwargs,
    ):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.enabled = self.configs.get("enabled", True)
        self.table_name = self.configs.get("table_name", "wikipedia_attention_data")
        self.days_back = self.configs.get("days_back", 30)
        self.project = self.configs.get("project", "en.wikipedia")
        
        self.logger.info(f"WikimediaAttentionCollector initialized. Target: {self.table_name}")

    def _generate_hash(self, row: pd.Series) -> str:
        hash_string = f"{row.get('date', '')}|{row.get('article', '')}"
        return hashlib.sha256(hash_string.encode()).hexdigest()

    async def run(self, tickers: list[str], keywords: list[str] | None = None, **kwargs) -> pd.DataFrame | None:
        if not self.enabled:
            return None

        # Оскільки Вікіпедія чутлива до реєстру, робимо Title Case
        search_terms = []
        if keywords:
            search_terms.extend([k.title().replace(" ", "_") for k in keywords])
        # Тікери можуть не бути гарними статтями (напр. "AAPL"), але можна спробувати
        if tickers:
            search_terms.extend([t for t in tickers])
            
        search_terms = list(set(search_terms))
        
        if not search_terms:
            self.logger.warning("No search terms for Wikimedia API.")
            return None

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)
        
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        self.logger.info(f"Fetching Wikipedia pageviews for {len(search_terms)} terms from {start_str} to {end_str}")

        client = await self.http_client_factory.get_http_client()
        # "DEAN_OS_Agent research@example.com" was refused with HTTP 403 and
        # the body "Please set a user-agent and respect our robot policy" --
        # Wikimedia requires a real client name, a version and a contact, and
        # example.com is not one. Every request returned 403, so this collector
        # produced nothing while reporting a bare exception with no message.
        # Measured 2026-08-12: the string below returns HTTP 200.
        # Set collectors.wikimedia_attention.user_agent to a real contact
        # address; it is what Wikimedia asks for and what keeps access.
        headers = {
            "User-Agent": self.configs.get(
                "user_agent",
                "trading-research-pipeline/1.0 "
                "(+https://github.com/OleksanderSS; automated research)",
            ),
            "Accept": "application/json",
        }

        all_views = []
        
        async with client:
            for term in search_terms:
                url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{self.project}/all-access/all-agents/{term}/daily/{start_str}/{end_str}"
                try:
                    response = await client.get(url, headers=headers)
                    if response.status_code == 404:
                        # Статті не існує, пропускаємо
                        continue
                    response.raise_for_status()
                    
                    data = response.json()
                    if "items" in data:
                        for item in data["items"]:
                            # timestamp is YYYYMMDD00
                            date_str = item["timestamp"][:8]
                            dt = datetime.strptime(date_str, "%Y%m%d")
                            all_views.append({
                                "date": dt.strftime("%Y-%m-%d"),
                                "article": term,
                                "views": item["views"],
                                "project": item["project"]
                            })
                except Exception as e:
                    self.logger.debug(f"Could not fetch wiki for {term}: {e}")
                
                # Rate limiting 100 requests / sec is the limit, but we sleep slightly
                await asyncio.sleep(0.1)

        if not all_views:
            self.logger.warning("No Wikimedia data retrieved.")
            return None

        df = pd.DataFrame(all_views)
        df["record_hash"] = df.apply(self._generate_hash, axis=1)

        # Deduplicate WITHIN the batch before comparing against the database.
        # filter_new_records only asks "is this hash already stored", so two
        # identical rows in one fetch -- which happens when two tickers map to
        # the same Wikipedia article -- both pass, and the second insert
        # violates the unique constraint:
        #   Constraint Error: Duplicate key "record_hash: 7d6719894d..."
        # The database stayed consistent (9,135 rows, 9,135 distinct hashes),
        # but the exception was raised, reported through the notifier, and
        # showed up as a failure at the end of an otherwise successful run.
        duplicates = int(df.duplicated(subset=["record_hash"]).sum())
        if duplicates:
            self.logger.info(
                "Wikimedia batch carried %d row(s) duplicated within itself "
                "(the same article requested more than once); keeping one of "
                "each.", duplicates,
            )
            df = df.drop_duplicates(subset=["record_hash"], keep="first")

        new_df = self.db_manager.filter_new_records(self.table_name, df, unique_cols=["record_hash"])
        if new_df.empty:
            self.logger.info("No new Wikimedia data found.")
            return None

        self.db_manager.upsert(self.table_name, new_df, unique_on=["record_hash"])
        self.logger.info(f"Successfully saved {len(new_df)} Wikipedia pageview records.")
        return new_df
