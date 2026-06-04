from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.integrations.data.bigquery_client import BigQueryClient

logger = ProjectLogger.get_logger('BigQueryConnector')


class BigQueryConnector:
    """
    Adapter that provides access to BigQuery client for the system.

    This class uses the centralized `BigQueryClient` to perform all operations,
    ensuring a single point of configuration and interaction with BigQuery API.
    """

    def __init__(self, project_id: str | None=None, location: str='US'):
        """
        Initializes the connector by creating a BigQueryClient instance.

        Args:
            project_id: GCP Project ID. If None, the default project from environment will be used.
            location: Region for data processing (default: "US").
        """
        try:
            self.client = BigQueryClient(project_id=project_id, location=
                location)
            logger.info(
                f'BigQueryConnector successfully initialized. Simulator mode: {self.client.use_simulator}. Project: {self.client.project_id}'
                )
        except Exception as e:
            logger.error(
                f'Failed to initialize BigQueryClient in connector: {e}',
                exc_info=True)
            raise

    def execute_query(self, query: str, use_cache: bool=True) ->pd.DataFrame | None:
        """
        Delegates SQL query execution to BigQueryClient.

        Args:
            query: SQL query to execute.
            use_cache: Whether to use cached results.

        Returns:
            DataFrame with query results or None in case of error.
        """
        logger.info('Delegating query to BigQueryClient...')
        try:
            return self.client.execute_query(query, use_cache=use_cache)
        except Exception as e:
            logger.error(f'Помилка під час делегування запиту: {e}',
                exc_info=True)
            raise RuntimeError("BigQuery query execution failed") from e

    def validate_query(self, query: str) ->dict[str, Any]:
        """
        Делегує валідацію SQL-запиту до BigQueryClient.

        Args:
            query: SQL-запит для валідації.

        Returns:
            Словник з результатами валідації.
        """
        return self.client.validate_query(query)

    def get_status(self) ->dict[str, Any]:
        """
        Отримує статус підключення від BigQueryClient.

        Returns:
            Словник зі статусом доступності та конфігурацією.
        """
        return self.client.get_status()


if __name__ == '__main__':
    logger.info('--- Запуск тесту BigQueryConnector ---')
    try:
        connector = BigQueryConnector()
        logger.info('\n--- Перевірка статусу ---')
        status = connector.get_status()
        logger.info(status)
        if status.get('reachable'):
            test_query = (
                "SELECT GKGRECORDID, DocumentIdentifier FROM `gdelt-bq.gdeltv2.gkg` WHERE DATE(_PARTITIONTIME) = '2024-01-01' AND DocumentIdentifier LIKE '%kyivpost.com%' LIMIT 5"
                )
            logger.info(f'\n--- Валідація запиту ---\n{test_query}')
            validation = connector.validate_query(test_query)
            logger.info(validation)
            if validation['valid']:
                logger.info('\n--- Виконання запиту ---')
                results_df = connector.execute_query(test_query)
                if results_df is not None:
                    logger.info('Результат:')
                    logger.info(results_df.head())
                else:
                    logger.warning('Не вдалося отримати результати.')
    except Exception as e:
        logger.error(f'Виникла помилка: {e}', exc_info=True)
        raise
    logger.info('\n--- Тестування завершено ---')
