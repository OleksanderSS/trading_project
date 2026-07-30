import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector


class FredCollector(BaseCollector):
    """Collector for fetching economic data from the Federal Reserve Economic Data (FRED)."""
    collector_type = "fred"
    data_type = "macro_data"
    runtime_request_contract = {
        "contract": "fred_bounded_runtime_request_v1",
        "runtime_series_ids_supported": True,
        "timezone_aware_as_of_required": True,
        "fred_vintage_dates_supported": True,
        "observation_end_cutoff_supported": True,
        "point_in_time_availability_field": "realtime_start",
        "maximum_runs_enforced_by_external_gate": True,
    }

    def __init__(self, configs: dict[str, Any], http_client_factory: HttpClientFactory, db_manager: DataManager, cache_manager: CacheManager | None = None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.timeout = self.configs.get('timeout', 20.0)
        period_str = self.configs.get('params', {}).get('period', '1y')
        self.start_date = self._calculate_start_date(period_str)
        self.hash_keys = self.configs.get(
            'hash_keys',
            ["series_id", "date", "realtime_start", "value"],
        )
        self.logger.info(f"FredCollector configured to fetch data from {self.start_date} onwards.")

    def _calculate_start_date(self, period: str) -> str:
        if 'y' in period:
            years = int(period.replace('y', ''))
            return (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
        elif 'd' in period:
            days = int(period.replace('d', ''))
            return (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        else:
            self.logger.warning(f"Unsupported period format for FRED: {period}. Defaulting to 1 year.")
            return (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    def _generate_hash(self, row: pd.Series) -> str:
        """Generates a stable hash for a record."""
        hash_string = "|".join(str(row.get(key, "")) for key in self.hash_keys)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    def _validate_config(self, **kwargs) -> tuple[str | None, list[str] | None]:
        """Validate FRED configuration and return API key and series IDs."""
        import os
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            self.logger.error("FRED_API_KEY environment variable not set.")
            return None, None

        runtime_series_ids = kwargs.get("series_ids")
        if runtime_series_ids is not None and not isinstance(
            runtime_series_ids, (list, tuple, set)
        ):
            raise ValueError("FRED runtime series_ids must be a list-like collection")
        configured_series_ids = self.configs.get('params', {}).get('series_ids', [])
        raw_series_ids = (
            runtime_series_ids
            if runtime_series_ids is not None
            else configured_series_ids
        )
        series_ids = list(
            dict.fromkeys(
                str(series_id).strip()
                for series_id in raw_series_ids
                if str(series_id).strip()
            )
        )
        if not series_ids:
            self.logger.warning("No series_ids specified for FRED. Skipping collection.")
            return api_key, None

        return api_key, series_ids

    # NOTE: this collector used to keep a per-record marker in CacheManager
    # (one `cache_manager.get(hash)` per row on the read side, one
    # `cache_manager.set(hash, True)` per row on the write side) as a
    # "fast path" ahead of the database dedup below. That was strictly
    # slower than the thing it was meant to avoid: every `get` is its own
    # `SELECT ... WHERE key_hash = ?` against DuckDB and every `set` is a
    # pickle file write plus a single-row DuckDB upsert, so a normal run
    # (~7.9k new observations) issued ~7.9k selects and ~7.9k single-row
    # upserts at roughly 4/sec. Measured: it pushed a 40-second HTTP job
    # past a 15-minute wall clock and starved every collector scheduled
    # after FRED in the same stage.
    #
    # `db_manager.filter_new_records()` already performs exactly this
    # dedup, on the same `hash` column (its default), in ONE query -- and
    # `upsert(unique_on=['hash'])` enforces the same uniqueness at write
    # time regardless. So the cache layer was pure duplicated work, not a
    # safety net. CacheManager is still the right tool for caching whole
    # payloads (see yf_collector / rss_collector), just not for per-row
    # dedup markers.

    async def run(self, **kwargs) -> pd.DataFrame | None:
        """Fetches data from FRED and filters for new records using cache and DB."""
        api_key, series_ids = self._validate_config(**kwargs)
        if not api_key or not series_ids:
            return None

        observation_start = str(kwargs.get("observation_start") or self.start_date)
        observation_end = kwargs.get("observation_end")
        vintage_date = None
        if kwargs.get("as_of"):
            parsed_as_of = datetime.fromisoformat(
                str(kwargs["as_of"]).replace("Z", "+00:00")
            )
            if parsed_as_of.tzinfo is None or parsed_as_of.utcoffset() is None:
                raise ValueError("FRED runtime as_of must be timezone-aware")
            vintage_date = parsed_as_of.date().isoformat()
            observation_end = observation_end or vintage_date

        client = await self.http_client_factory.get_http_client()
        async with client:
            tasks = [
                self._fetch_series(
                    series_id,
                    client,
                    api_key,
                    observation_start=observation_start,
                    observation_end=str(observation_end) if observation_end else None,
                    vintage_date=vintage_date,
                )
                for series_id in series_ids
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        all_series_data = []
        for res in results:
            if isinstance(res, list):
                all_series_data.extend(res)
            elif isinstance(res, Exception):
                self.logger.error(f"Error fetching FRED series: {res}")

        if not all_series_data:
            return None

        df = pd.DataFrame(all_series_data)
        df['hash'] = df.apply(self._generate_hash, axis=1)

        # Dedup against what is already stored -- one bulk query on `hash`.
        table_name = self.configs.get('table_name', 'fred_data')
        new_records_df = self.db_manager.filter_new_records(table_name, df)

        if new_records_df.empty:
            self.logger.info("No new FRED records after DB filtering.")
            return None

        self.logger.info(f"Found {len(new_records_df)} new FRED records.")

        # Save to DB so macro data accumulates and is available for Stage 3.
        try:
            self.db_manager.upsert(table_name, new_records_df, unique_on=['hash'])
            self.logger.info(f"Saved {len(new_records_df)} FRED records to '{table_name}'")
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.warning(f"Could not save FRED records to DB: {e}")

        return new_records_df

    async def _fetch_series(
        self,
        series_id: str,
        client,
        api_key: str,
        *,
        observation_start: str | None = None,
        observation_end: str | None = None,
        vintage_date: str | None = None,
    ) -> list[dict[str, Any]]:
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": observation_start or self.start_date,
        }
        if observation_end:
            params["observation_end"] = observation_end
        if vintage_date:
            params["vintage_dates"] = vintage_date
        url = "https://api.stlouisfed.org/fred/series/observations?" + urlencode(params)
        try:
            response = await client.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            observations = data.get('observations', [])
            for obs in observations:
                missing = [
                    field
                    for field in ("date", "value", "realtime_start")
                    if not str(obs.get(field) or "").strip()
                ]
                if missing:
                    raise ValueError(
                        "FRED observation missing point-in-time fields: "
                        + ", ".join(missing)
                    )
                obs['series_id'] = series_id
                obs['source_locator'] = (
                    f"https://fred.stlouisfed.org/series/{series_id}"
                )
            return observations
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Failed to fetch FRED series {series_id}: {e}")
            raise RuntimeError(f"Failed to fetch FRED series {series_id}") from e
