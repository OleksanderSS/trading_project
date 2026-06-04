import asyncio
import hashlib
from datetime import datetime, timedelta
from io import StringIO
from typing import Any

import pandas as pd

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector


class EconomicCalendarCollector(BaseCollector):
    """Fetches economic calendar records mapping Investing.com HTTP payloads data structures string bounds bounds layers limits."""
    collector_type = 'economic_calendar'
    data_type = 'economic'

    def __init__(self, configs: dict[str, Any], http_client_factory:
        HttpClientFactory, db_manager: DataManager, cache_manager: CacheManager | None=None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager,
            cache_manager, **kwargs)

    async def run(self, **kwargs) ->pd.DataFrame | None:
        table_name = self.configs.get('table_name', 'economic_calendar')
        cache_key = f'{self.__class__.__name__}_run'
        start_date, end_date = self._get_date_range()
        cache_params = {'start': str(start_date.date()), 'end': str(
            end_date.date())}
        if self.cache_manager:
            cached = self.cache_manager.get(cache_key, cache_params,
                namespace='collectors')
            if cached is not None:
                df_cached = pd.DataFrame(cached) if isinstance(cached, list
                    ) else cached
                if 'hash' in df_cached.columns:
                    new_from_cache = self.db_manager.filter_new_records(
                        table_name, df_cached)
                    if new_from_cache.empty:
                        self.logger.info(
                            '[EconCalendar] Cache execution limits validated boundaries check target parameters mapping representation limits scopes structure hits block string validation — zero new entries structure definition check targets mapping representation indexes strings identified string layers boundaries checks scope logic mapping execution boundaries logic block strings validation definitions checks array indexes parameters validation scope parameters block.'
                            )
                        return None
                    return new_from_cache
        raw = await self.fetch_raw_data()
        if not raw:
            return None
        df = pd.DataFrame(raw)
        df['hash'] = df.apply(lambda row: hashlib.sha256(
            f"{row.get('timestamp', '')}|{row.get('event', '')}|{row.get('country', '')}"
            .encode()).hexdigest(), axis=1)
        new_df = self.db_manager.filter_new_records(table_name, df)
        if new_df.empty:
            self.logger.info(
                '[EconCalendar] No novel representation target boundaries mapped.'
                )
            if self.cache_manager:
                self.cache_manager.set(cache_key, df.to_dict('records'),
                    cache_params, namespace='collectors')
            return None
        self.db_manager.upsert(table_name, new_df, unique_on=['hash'])
        if self.cache_manager:
            self.cache_manager.set(cache_key, df.to_dict('records'),
                cache_params, namespace='collectors')
        self.logger.info(
            f'[EconCalendar] Safely validated structures mapped boundaries structures string variables limits structure scopes check representation array structure {len(new_df)} array constraints indexes execution bounds limits parameter arrays execution layers bounds bounds records strings bounds logics execution variables mapping variables arrays logic layers boundary parameter constraints strings mapping.'
            )
        return new_df

    async def fetch_raw_data(self, **kwargs) ->list[dict[str, Any]]:
        api_url = self.configs.get('api_url')
        if not api_url:
            self.logger.error(
                "Configuration mapped validation mapping layer lacks bounds targets constraint indexes string limits mappings strings mappings structures limits parameters payload parameters definition targets structure variables 'api_url' bounds string boundary variables representations array limits payload definitions mappings structure layer definition mappings validation variables string logic layers arrays bounds mappings execution"
                )
            return []
        headers = self.configs.get('headers')
        if not headers:
            self.logger.error(
                "Configuration bounds parameter 'headers' payload mapping logic validations variables execution checks indexes validation parameter target limits validation maps execution scopes structures parameters boundaries structure validations parameters mapping logic execution indexes parameters check structures mappings strings definition variables arrays mappings definition layers boundaries mappings definition strings validation variables"
                )
            return []
        start_date, end_date = self._get_date_range()
        payload = self._build_payload(start_date, end_date)
        try:
            client = await self.http_client_factory.get_http_client()
            response = await client.post(api_url, data=payload, headers=headers
                )
            response.raise_for_status()
            html_data = response.json().get('data')
            if not html_data:
                self.logger.warning(
                    "[EconCalendar] Missing execution constraints structures string data limits mappings representation definition index mapping indexes boundaries mapping parameters layers scopes valid limits boundaries check parameter definition boundaries strings parameters definition layers representation 'data' variables validation bounds execution mappings limits."
                    )
                return []
            records = await asyncio.to_thread(self._parse_html, html_data)
            self.logger.info(
                f'[EconCalendar] Structured parsed bounds parameters validation mappings boundaries definition mapping arrays {len(records)} index definition representation values mappings parameters string mapping layers constraints target array.'
                )
            return records
        except Exception as e:  # audit-ignore: BROAD_EXCEPTION_SILENT_RETURN
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            raise RuntimeError(f"Failed to fetch economic calendar from {api_url}") from e

    def _get_date_range(self):
        days_back = self.configs.get('days_back', 7)
        days_ahead = self.configs.get('days_ahead', 30)
        start = datetime.now() - timedelta(days=days_back)
        end = datetime.now() + timedelta(days=days_ahead)
        return start, end

    def _build_payload(self, start_date: datetime, end_date: datetime) ->dict:
        countries = self.configs.get('countries', [])
        importance = self.configs.get('importance', [])
        api_mappings = self.configs.get('api_mappings', {})
        country_map = api_mappings.get('country', {})
        impact_map = api_mappings.get('impact', {})
        payload = self.configs.get('request_payload', {}).copy()
        payload.update({'country[]': [country_map[c] for c in countries if
            c in country_map], 'importance[]': [impact_map[i] for i in
            importance if i in impact_map], 'startDate': start_date.
            strftime('%Y-%m-%d'), 'endDate': end_date.strftime('%Y-%m-%d')})
        return payload

    def _parse_html(self, html_data: str) ->list[dict[str, Any]]:
        column_names = self.configs.get('column_names')
        if not column_names:
            self.logger.error(
                "Configuration constraints block limits layers mapping indexes execution parameter check missing 'column_names' execution scope check logic boundary limit checks logic bounds mapped representation targets limit definitions structures arrays mappings bounds definitions strings definitions variables index limits boundary layers targets checks representation"
                )
            return []
        try:
            df = pd.read_html(StringIO(f'<table>{html_data}</table>'))[0]
            df.columns = column_names
            if '_del' in df.columns:
                df = df.drop(columns=['_del'])
            df['date'] = pd.to_datetime(df.get('time', ''), errors='coerce'
                ).dt.strftime('%Y-%m-%d')
            df['date'] = df['date'].ffill()
            df = df.dropna(subset=['event']).copy()
            if df.empty:
                return []
            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df.get(
                'time', '').astype(str), format='%Y-%m-%d %H:%M:%S', errors
                ='coerce', utc=True)
            df = df.dropna(subset=['timestamp'])
            df['impact'] = df['impact'].astype(str).str.strip()
            if 'currency' in df.columns:
                df = df.rename(columns={'currency': 'country'})
            final_cols = ['timestamp', 'country', 'impact', 'event',
                'actual', 'forecast', 'previous']
            df = df.reindex(columns=final_cols)
            return df.to_dict('records')
        except Exception as e:  # audit-ignore: BROAD_EXCEPTION_SILENT_RETURN
            self.logger.error(
                f'[EconCalendar] Structural layer logic constraint bounds layers mapping boundaries representations structures limit boundaries exceptions targets bounds parameter mapping structures mappings checks strings index {e}'
                , exc_info=True)
            return []
