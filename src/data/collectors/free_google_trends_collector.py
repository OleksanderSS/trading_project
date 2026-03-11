
# src/data/collectors/free_google_trends_collector.py

import asyncio
import logging
from typing import List, Dict, Any, Optional

import pandas as pd
from pytrends.request import TrendReq

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

class FreeGoogleTrendsCollector(BaseCollector):
    """
    Асинхронно збирає дані Google Trends. 
    Ключові слова та тікери передаються динамічно.
    """
    collector_type = "free_google_trends"
    data_type = "alternative"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Завантажуємо статичні параметри конфігурації
        self.geo = self.config.get('geo', 'US')
        self.timeframe = self.config.get('timeframe', 'today 5-y')
        self.language = self.config.get('language', 'en-US')
        self.timezone = self.config.get('timezone', 360)
        self.batch_size = self.config.get('batch_size', 4) # Google обробляє до 5 ключових слів одночасно
        self.request_delay = self.config.get('request_delay_seconds', 5)
        self.cat = self.config.get('cat', 0)
        self.gprop = self.config.get('gprop', '')
        self.pytrends: Optional[TrendReq] = None

    def _initialize_pytrends(self):
        """Відкладена ініціалізація клієнта TrendReq."""
        if self.pytrends is None:
            try:
                self.pytrends = TrendReq(hl=self.language, tz=self.timezone)
            except Exception as e:
                raise ConnectionError(f"Не вдалося ініціалізувати TrendReq: {e}")

    async def fetch_raw_data(self, tickers: List[str], keywords: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Асинхронно завантажує дані Google Trends для наданих тікерів та ключових слів.
        """
        search_terms = list(set(tickers + keywords))
        if not search_terms:
            self.logger.warning("Немає термінів для пошуку в Google Trends. Пропускаємо.")
            return []

        try:
            self._initialize_pytrends()
        except ConnectionError as e:
            self.handle_error(e, context={})
            return []

        self.logger.info(f"Починаємо збір для {len(search_terms)} термінів в Google Trends (батчі по {self.batch_size}).")
        keyword_batches = [search_terms[i:i + self.batch_size] for i in range(0, len(search_terms), self.batch_size)]
        all_trends_data: List[Dict[str, Any]] = []

        for i, batch in enumerate(keyword_batches):
            try:
                if i > 0:
                    await asyncio.sleep(self.request_delay)
                
                batch_data = await self._fetch_trends_for_batch(batch)
                if batch_data:
                    all_trends_data.extend(batch_data)

            except Exception as e:
                self.handle_error(e, context={"keyword_batch": batch})
        
        self.logger.info(f"Всього отримано {len(all_trends_data)} точок даних з Google Trends.")
        return all_trends_data

    async def _fetch_trends_for_batch(self, keyword_batch: List[str]) -> List[Dict[str, Any]]:
        """
        Завантажує дані для одного батчу, виконуючи синхронний виклик pytrends в окремому потоці.
        """
        self.logger.debug(f"Завантажуємо Google Trends для батчу: {keyword_batch}")
        try:
            interest_df = await asyncio.to_thread(
                self._run_pytrends_request, keyword_batch
            )

            if interest_df is None or interest_df.empty:
                self.logger.warning(f"Pytrends не повернув даних для ключових слів: {keyword_batch}")
                return []

            # Перетворюємо DataFrame в потрібний формат
            if 'isPartial' in interest_df.columns:
                interest_df = interest_df.drop(columns=['isPartial'])
            
            long_df = interest_df.reset_index().melt(id_vars='date', var_name='keyword', value_name='interest')
            long_df['geo'] = self.geo
            long_df['date'] = long_df['date'].astype(str)

            return long_df.to_dict('records')

        except Exception as e:
            self.logger.error(f"Помилка при завантаженні батчу {keyword_batch}: {e}", exc_info=True)
            raise

    def _run_pytrends_request(self, keyword_batch: List[str]) -> Optional[pd.DataFrame]:
        """
        Синхронна функція, що виконує блокуючий мережевий запит pytrends.
        """
        if not self.pytrends:
            # Ця перевірка є запобіжником, хоча помилка має бути перехоплена раніше.
            raise RuntimeError("Клієнт Pytrends не був ініціалізований.")
            
        self.pytrends.build_payload(
            kw_list=keyword_batch,
            cat=self.cat,
            timeframe=self.timeframe,
            geo=self.geo,
            gprop=self.gprop
        )
        return self.pytrends.interest_over_time()
