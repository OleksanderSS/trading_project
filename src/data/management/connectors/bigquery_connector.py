# src/data/management/connectors/bigquery_connector.py

from typing import Dict, Any, Optional
import pandas as pd

from src.integrations.data.bigquery_client import BigQueryClient
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("BigQueryConnector")

class BigQueryConnector:
    """
    Адаптер, що надає доступ до клієнта BigQuery для системи.

    Цей клас використовує централізований `BigQueryClient` для виконання всіх операцій,
    забезпечуючи єдину точку конфігурації та взаємодії з BigQuery API.
    """

    def __init__(self, project_id: Optional[str] = None, location: str = "US"):
        """
        Ініціалізує коннектор, створюючи екземпляр BigQueryClient.

        Args:
            project_id: GCP Project ID. Якщо None, буде використано стандартний проект з середовища.
            location: Регіон для обробки даних (за замовчуванням: "US").
        """
        try:
            self.client = BigQueryClient(project_id=project_id, location=location)
            logger.info(
                f"BigQueryConnector успішно ініціалізовано. "
                f"Режим симулятора: {self.client.use_simulator}. Проект: {self.client.project_id}"
            )
        except Exception as e:
            logger.error(f"Не вдалося ініціалізувати BigQueryClient у коннекторі: {e}", exc_info=True)
            raise

    def execute_query(self, query: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Делегує виконання SQL-запиту до BigQueryClient.

        Args:
            query: SQL-запит для виконання.
            use_cache: Чи використовувати кешовані результати.

        Returns:
            DataFrame з результатами запиту або None у разі помилки.
        """
        logger.info("Делегування запиту до BigQueryClient...")
        try:
            return self.client.execute_query(query, use_cache=use_cache)
        except Exception as e:
            logger.error(f"Помилка під час делегування запиту: {e}", exc_info=True)
            return None

    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Делегує валідацію SQL-запиту до BigQueryClient.

        Args:
            query: SQL-запит для валідації.

        Returns:
            Словник з результатами валідації.
        """
        return self.client.validate_query(query)

    def get_status(self) -> Dict[str, Any]:
        """
        Отримує статус підключення від BigQueryClient.

        Returns:
            Словник зі статусом доступності та конфігурацією.
        """
        return self.client.get_status()

# Приклад використання (для локальної перевірки)
if __name__ == "__main__":
    print("--- Запуск тесту BigQueryConnector ---")
    try:
        # Для реального тестування встановіть змінну середовища GOOGLE_APPLICATION_CREDENTIALS
        # та запустіть `gcloud auth application-default login`
        connector = BigQueryConnector()

        print("\n--- Перевірка статусу ---")
        status = connector.get_status()
        print(status)

        if status.get('reachable'):
            # Приклад запиту до публічного датасету GDELT
            test_query = (
                "SELECT GKGRECORDID, DocumentIdentifier "
                "FROM `gdelt-bq.gdeltv2.gkg` "
                "WHERE DATE(_PARTITIONTIME) = '2024-01-01' "
                "AND DocumentIdentifier LIKE '%kyivpost.com%' "
                "LIMIT 5"
            )

            print(f"\n--- Валідація запиту ---\n{test_query}")
            validation = connector.validate_query(test_query)
            print(validation)

            if validation['valid']:
                print("\n--- Виконання запиту ---")
                results_df = connector.execute_query(test_query)
                if results_df is not None:
                    print("Результат:")
                    print(results_df.head())
                else:
                    print("Не вдалося отримати результати.")
        
    except Exception as e:
        print(f"\nВиникла помилка під час тестування коннектора: {e}")

    print("\n--- Тестування завершено ---")
