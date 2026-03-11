
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from io import StringIO

import pandas as pd

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

class EconomicCalendarCollector(BaseCollector):
    """
    Асинхронно збирає дані економічного календаря з внутрішнього API Investing.com.
    """
    collector_type = "economic_calendar"
    data_type = "economic"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Конструктор максимально спрощено

    async def fetch_raw_data(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Асинхронно завантажує та парсить дані економічних подій.
        """
        api_url = self.config.get('api_url')
        if not api_url:
            self.logger.error("В конфігурації відсутній 'api_url'.")
            return []

        start_date, end_date = self._get_date_range()
        self.logger.info(f"Збираємо події з {start_date.date()} по {end_date.date()}.")

        form_data = self._build_request_payload(start_date, end_date)
        headers = self.config.get('headers')
        if not headers:
            self.logger.error("В конфігурації відсутні 'headers'.")
            return []

        try:
            async with self.http_client_factory.get_http_client() as client:
                response = await client.post(api_url, data=form_data, headers=headers)
                response.raise_for_status()
            
            api_response = response.json()
            html_data = api_response.get('data')
            if not html_data:
                self.logger.warning("Відповідь API не містить поля 'data'.")
                return []

            records = await asyncio.to_thread(self._parse_html, html_data)
            self.logger.info(f"Успішно розпарсено {len(records)} подій.")
            return records

        except Exception as e:
            self.handle_error(e, {"url": api_url, "action": "fetch_raw_data"})
            return []

    def _get_date_range(self) -> (datetime, datetime):
        days_back = self.config.get('days_back', 2)
        days_ahead = self.config.get('days_ahead', 7)
        start_date = datetime.now() - timedelta(days=days_back)
        end_date = datetime.now() + timedelta(days=days_ahead)
        return start_date, end_date

    def _build_request_payload(self, start_date: datetime, end_date: datetime) -> Dict:
        countries = self.config.get('countries', [])
        impact_levels = self.config.get('importance', [])
        api_mappings = self.config.get('api_mappings', {})
        base_payload = self.config.get('request_payload', {})

        country_map = api_mappings.get('country', {})
        impact_map = api_mappings.get('impact', {})
        
        country_codes = [country_map.get(c.lower()) for c in countries if c.lower() in country_map]
        impact_codes = [impact_map.get(i.lower()) for i in impact_levels if i.lower() in impact_map]

        payload = base_payload.copy()
        payload.update({
            "country[]": country_codes,
            "importance[]": impact_codes,
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
        })
        return payload

    def _parse_html(self, html_data: str) -> List[Dict[str, Any]]:
        """Синхронна функція для парсингу HTML за допомогою pandas."""
        column_names = self.config.get('column_names')
        if not column_names:
            self.logger.error("В конфігурації відсутні 'column_names'.")
            return []

        try:
            df = pd.read_html(StringIO(f"<table>{html_data}</table>"))[0]
            df.columns = column_names
            df = df.drop(columns=['_del'])
            
            df['date'] = pd.to_datetime(df['time'], errors='coerce').dt.strftime('%Y-%m-%d')
            df['date'] = df['date'].ffill()
            df = df.dropna(subset=['event']).copy()
            if df.empty:
                return []

            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce', utc=True)
            
            df = df.dropna(subset=['timestamp'])
            df['impact'] = df['impact'].astype(str).str.strip()
            df = df.rename(columns={"currency": "country"})
            
            df['event_id'] = df['event'].str.extract(r'event_\d+_(\d+)').astype(str)

            final_cols = ['timestamp', 'country', 'impact', 'event', 'actual', 'forecast', 'previous', 'event_id']
            df = df.reindex(columns=final_cols)

            return df.to_dict('records')

        except Exception as e:
            self.logger.error(f"Помилка під час обробки DataFrame: {e}", exc_info=True)
            return []
