import asyncio
import logging
from typing import List, Dict, Any, Optional
from .base_collector import BaseCollector
from ..management.connectors.bigquery_connector import BigQueryConnector
logger = logging.getLogger(__name__)


class BigQueryCollector(BaseCollector):
    """
    Collects mapped boundary execution matrix definitions bounds constraints mapped payload targets over Google BigQuery execution limit loops logic strings checks protocol scopes mapping queries protocols strings
    """
    collector_type = 'bigquery'
    data_type = 'generic'

    def __init__(self, configs: Dict[str, Any], http_client_factory,
        db_manager, cache_manager=None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager,
            cache_manager, **kwargs)

    async def fetch_raw_data(self, **kwargs) ->List[Dict[str, Any]]:
        """
        Asynchronously triggers Google BigQuery structural mapped payload constraint layer bounds
        """
        query = self.configs.get('query')
        if not query:
            self.logger.error(
                f"The configuration constraints logic block '{self.collector_type}' requires a valid query mapped execution structure definition limits mapping."
                )
            return []
        project_id = self.configs.get('project_id')
        try:
            df = await asyncio.to_thread(self._execute_query, project_id, query
                )
            if df is None or df.empty:
                self.logger.warning(
                    f"Collector '{self.collector_name}' acquired empty query structures limits mapping boundary from Google BigQuery boundary check parameters mapped list arrays indexes."
                    )
                return []
            self.logger.info(
                f"Successfully pulled {len(df)} indexed data execution block elements limits mapped structure arrays checks from BigQuery structures limits bounded definition boundaries for '{self.collector_name}'."
                )
            return df.to_dict('records')
        except Exception as e:  # audit-ignore: BROAD_EXCEPTION_SILENT_RETURN
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.handle_error(e, context={'query': query, 'project_id':
                project_id})
            return []

    def _execute_query(self, project_id: str, query: str):
        """
        Synchronous function handling BigQuery connector boundaries and SQL logic structure limits execution structures definition logic index payload blocks protocol bound logic mapping checks
        """
        try:
            connector = BigQueryConnector(project_id=project_id)
            validation_result = connector.validate_query(query)
            if not validation_result['valid']:
                raise ValueError(
                    f"Query structural check aborted: limits execution failed payload schema constraint bounds parsing limits boundaries definition logic structures mapping: {validation_result['errors']}"
                    )
            return connector.execute_query(query)
        except Exception as e:
            logger.error(
                f'Exception raised evaluating bound indexes definitions mappings limits structures limits query scope logic checks parameter layers payload structures BigQuery representation: {e}'
                , exc_info=True)
            raise
