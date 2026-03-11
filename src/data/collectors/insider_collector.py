
# src/data/collectors/insider_collector.py

import asyncio
import httpx
import logging
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

class InsiderCollector(BaseCollector):
    """
    Асинхронно збирає дані про інсайдерські угоди шляхом скрейпінгу OpenInsider.
    """
    collector_type = "insider"
    data_type = "alternative"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Конструктор максимально спрощено

    async def fetch_raw_data(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Асинхронно завантажує та парсить дані з усіх URL, вказаних у конфігурації.
        """
        urls_to_scrape = self.config.get("urls")
        if not urls_to_scrape:
            self.logger.warning(f"В конфігурації для '{self.collector_name}' не вказано 'urls'. Пропускаємо.")
            return []

        self.logger.info(f"Починаємо асинхронний скрейпінг для {len(urls_to_scrape)} URL.")
        all_trades: List[Dict[str, Any]] = []
        
        async with self._get_async_http_client() as client:
            tasks = [self._scrape_url(url, client) for url in urls_to_scrape]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.handle_error(result, context={"url": urls_to_scrape[i]})
                elif result:
                    all_trades.extend(result)

        self.logger.info(f"Всього зібрано {len(all_trades)} сирих записів про угоди.")
        return all_trades

    async def _scrape_url(self, url: str, client: httpx.AsyncClient) -> Optional[List[Dict[str, Any]]]:
        """
        Виконує один асинхронний запит і парсить HTML-таблицю.
        """
        user_agent = self.config.get("user_agent", "Mozilla/5.0")
        self.logger.debug(f"Скрейпінг URL: {url}")
        headers = {"User-Agent": user_agent}
        
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

            return self._parse_html(response.text, url)

        except Exception as e:
            # Помилка буде оброблена в `gather`, тут її просто перевикидаємо
            self.logger.error(f"Помилка під час скрейпінгу {url}")
            raise e

    def _parse_html(self, html: str, url: str) -> List[Dict[str, Any]]:
        """Парсить HTML-контент і витягує дані з таблиці."""
        column_mapping = self.config.get("column_mapping")
        if not column_mapping:
            raise ValueError(f"В конфігурації '{self.collector_name}' відсутній 'column_mapping'.")

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", class_="tinytable")
        
        if not table:
            self.logger.warning(f"Не знайдено таблицю з класом 'tinytamle' за адресою {url}.")
            return []

        rows = table.find_all("tr")[1:]  # Пропускаємо заголовок
        parsed_trades = []
        expected_col_count = len(column_mapping)

        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < expected_col_count:
                continue

            trade_data = {column_mapping[i]: cells[i] for i in range(expected_col_count)}
            parsed_trades.append(trade_data)

        return parsed_trades
