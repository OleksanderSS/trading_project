# src/data/collectors/newsapi_collector.py

import asyncio
import os
from typing import List, Dict, Any, Optional

from src.config.unified_config_manager import UnifiedConfigManager
from src.data.collectors.base_collector import BaseCollector

class NewsAPICollector(BaseCollector):
    """Колектор для збору новин з NewsAPI.
    Робить окремі запити для кожного тікера чи ключового слова.
    """
    collector_type = "newsapi"
    data_type = "news"

    def __init__(self, config_manager: UnifiedConfigManager, **kwargs):
        super().__init__(config_manager=config_manager, **kwargs)
        self.base_url = self.config.get("base_url", "https://newsapi.org/v2/everything")
        self.language = self.config.get("language", "en")
        self.page_size = self.config.get("page_size", 100)
        self.api_key_env = self.config.get("api_key_env")
        self._api_key = None

    def _get_api_key(self) -> Optional[str]:
        """Отримує API ключ з конфігураційного менеджера."""
        if self._api_key is None:
            if not self.api_key_env:
                self.logger.error("'api_key_env' не вказано в конфігурації для NewsAPI.")
                return None
            self._api_key = self.config_manager.get_secret(self.api_key_env)
            if not self._api_key:
                self.logger.error(f"Не вдалося знайти секрет для NewsAPI в змінній оточення: {self.api_key_env}")
        return self._api_key

    async def fetch_raw_data(self, search_terms: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Збирає дані, роблячи паралельні запити для кожного пошукового терміну."""
        api_key = self._get_api_key()
        if not api_key:
            return [] # Помилка вже залогована

        if not search_terms:
            self.logger.warning("Не надано пошукових термінів. Пропускаємо збір новин з NewsAPI.")
            return []

        self.logger.info(f"Починаємо збір новин з NewsAPI для {len(search_terms)} термінів.")

        tasks = [self._fetch_articles_for_term(term, api_key) for term in search_terms]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                term = search_terms[i]
                self.handle_error(result, context={"search_term": term, "source": "NewsAPI"})
            elif result:
                all_articles.extend(result)
        
        self.logger.info(f"Всього отримано {len(all_articles)} сирих статей з NewsAPI.")
        return all_articles

    async def _fetch_articles_for_term(self, term: str, api_key: str) -> List[Dict[str, Any]]:
        """Робить один запит до NewsAPI для заданого терміну."""
        params = {
            "q": f'\"{term}\"', # Шукаємо точну фразу для релевантності
            "language": self.language,
            "pageSize": self.page_size,
            "apiKey": api_key
        }

        async with self.http_client_factory.get_http_client() as client:
            try:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                articles = data.get('articles', [])
                
                # Додаємо пошуковий термін до кожної статті для подальшого аналізу
                for article in articles:
                    article['search_term'] = term
                
                return articles
            except Exception as e:
                self.logger.error(f"Помилка при запиті до NewsAPI для терміну '{term}': {e}", exc_info=True)
                raise # Дозволяємо gather обробити це як виняток
