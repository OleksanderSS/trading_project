
# src/data/collectors/bigquery_collector.py

import asyncio
import logging
from typing import List, Dict, Any, Optional

from .base_collector import BaseCollector
from ..management.connectors.bigquery_connector import BigQueryConnector

logger = logging.getLogger(__name__)

class BigQueryCollector(BaseCollector):
    """
    Збирач даних, що виконує SQL-запити до Google BigQuery.
    """
    collector_type = "bigquery"
    data_type = "generic"

    def __init__(self, configs: Dict[str, Any], http_client_factory, db_manager, cache_manager=None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        # Initialization simplified. Client created on demand in fetch_raw_data.

    async def fetch_raw_data(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Асинхронно виконує запит до BigQuery.
        """
        query = self.configs.get("query")
        if not query:
            self.logger.error(f"Конфігурація для '{self.collector_type}' повинна містити 'query'.")
            return []

        project_id = self.configs.get("project_id")

        try:
            # Виконуємо блокуючі операції в окремому потоці
            df = await asyncio.to_thread(self._execute_query, project_id, query)
            
            if df is None or df.empty:
                self.logger.warning(f"Збирач '{self.collector_name}' не отримав даних з BigQuery.")
                return []

            self.logger.info(f"Успішно отримано {len(df)} записів з BigQuery для '{self.collector_name}'.")
            return df.to_dict('records')

        except Exception as e:
            self.handle_error(e, context={"query": query, "project_id": project_id})
            return []

    def _execute_query(self, project_id: str, query: str):
        """
        Синхронна функція для ініціалізації конектора та виконання запиту.
        """
        try:
            connector = BigQueryConnector(project_id=project_id)
            
            # Валідація запиту
            validation_result = connector.validate_query(query)
            if not validation_result['valid']:
                raise ValueError(f"Запит не пройшов валідацію: {validation_result['errors']}")

            # Виконання запиту
            return connector.execute_query(query)
        except Exception as e:
            logger.error(f"Помилка під час виконання запиту до BigQuery: {e}", exc_info=True)
            raise
