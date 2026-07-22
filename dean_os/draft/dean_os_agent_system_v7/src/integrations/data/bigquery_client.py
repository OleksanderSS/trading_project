import os
import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery
from pandas_gbq.gbq import GenericGBQException

from src.core.base_integration import BaseIntegration
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger('BigQueryClient')


class BigQueryClient(BaseIntegration):
    """
    Client for interacting with Google BigQuery API.
    Provides methods for executing queries, validation, and cost estimation.
    Supports a simulation mode for local development.
    """

    def __init__(self, project_id: str | None=None, location: str='US'):
        """
        Initializes the BigQuery client.

        Authentication is handled automatically via Google Cloud's standard
        authentication methods (e.g., `GOOGLE_APPLICATION_CREDENTIALS` env var).

        Args:
            project_id: GCP Project ID. If None, it will try to infer from the environment.
            location: Data processing location (default: "US").
        """
        super().__init__()
        self.location = location
        self.client: bigquery.Client | None = None
        self.project_id: str | None = project_id
        self.use_simulator = os.environ.get('BIGQUERY_SIMULATOR_MODE', 'false'
            ).lower() == 'true'
        if self.use_simulator:
            logger.warning(
                'BigQueryClient is running in SIMULATOR MODE. No real queries will be executed.'
                )
            self.project_id = 'simulated-project'
        else:
            try:
                self.client = bigquery.Client(project=self.project_id,
                    location=self.location)
                if not self.project_id:
                    self.project_id = self.client.project
                logger.info(
                    f'BigQueryClient initialized for project: {self.project_id} (Location: {self.location})'
                    )
            except DefaultCredentialsError:
                logger.error(
                    'GCP authentication failed. Please configure your environment with valid credentials. You can run in simulator mode by setting BIGQUERY_SIMULATOR_MODE=true.'
                    )
                self.use_simulator = True
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.exception(f'Failed to initialize BigQuery client: {e}')
                self.use_simulator = True

    @property
    def name(self) ->str:
        """Returns the unique name of the integration."""
        return 'bigquery'

    def ping(self) ->bool:
        """
        Checks if the BigQuery service is reachable.
        In real mode, executes a simple query. In simulated mode, returns True.
        """
        if self.use_simulator or not self.client:
            logger.info('Pinging BigQueryClient (Simulated)...')
            return True
        try:
            self.client.query('SELECT 1').result()
            logger.info('BigQuery ping successful.')
            return True
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f'BigQuery ping failed: {e}')
            return False

    def execute_query(self, query: str, use_cache: bool=True) ->pd.DataFrame:
        """
        Executes an SQL query against BigQuery and returns results as a DataFrame.
        If in simulator mode, returns mock data.
        """
        logger.info(
            f'Executing BigQuery query (length: {len(query)} characters)')
        if self.use_simulator:
            logger.warning('Using simulated data for BigQuery response.')
            if 'gdelt' in query.lower():
                df = self._generate_gdelt_mock_data()
            else:
                df = self._generate_generic_mock_data()
            logger.info(
                f'Query simulation successful. Returned {len(df)} rows.')
            return df
        try:
            df = pd.read_gbq(query, project_id=self.project_id, location=self.location)
            logger.info(
                f'Query executed successfully. Returned {len(df)} rows.')
            return df
        except (GenericGBQException, DefaultCredentialsError) as e:
            logger.exception(f'Error executing BigQuery query: {e}')
            return pd.DataFrame()

    def validate_query(self, query: str) ->dict[str, Any]:
        """Performs basic and specific validation on the SQL query."""
        result = {'valid': True, 'errors': [], 'warnings': [],
            'suggestions': []}
        query_lower = query.lower()
        if 'select' not in query_lower:
            result['errors'].append("Query must contain 'SELECT' statement.")
            result['valid'] = False
        if 'from' not in query_lower:
            result['errors'].append("Query must contain 'FROM' clause.")
            result['valid'] = False
        if 'limit' not in query_lower:
            result['suggestions'].append(
                "Add 'LIMIT' to prevent pulling excessive amounts of data.")
        if 'gdelt' in query_lower:
            if '`' not in query:
                result['errors'].append(
                    'BigQuery table names should be enclosed in backticks (``).'
                    )
                result['valid'] = False
        if not result['valid']:
            logger.error(f"Query validation failed: {result['errors']}")
        return result

    def get_query_cost_estimate(self, query: str) ->dict[str, Any]:
        """
        Estimates the processing cost of the query.
        In real mode, uses BigQuery's dry run feature. In simulated mode, uses heuristics.
        """
        if self.use_simulator or not self.client:
            return self._get_heuristic_cost_estimate(query)
        try:
            job_config = bigquery.QueryJobConfig(dry_run=True,
                use_query_cache=False)
            query_job = self.client.query(query, job_config=job_config)
            bytes_processed = query_job.total_bytes_processed
            gb_processed = bytes_processed / 1024 ** 3
            cost_usd = gb_processed * (6.25 / 1024)
            estimate = {'estimated_gb': round(gb_processed, 4),
                'estimated_cost_usd': round(cost_usd, 4),
                'optimization_suggestions': []}
            logger.info(
                f'BigQuery dry run successful. Estimated GB to be processed: {gb_processed:.4f}'
                )
            return estimate
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            logger.warning(
                f'Could not perform BigQuery dry run: {e}. Falling back to heuristic estimation.'
                )
            return self._get_heuristic_cost_estimate(query)

    def _get_heuristic_cost_estimate(self, query: str) ->dict[str, Any]:
        """Provides a rough, heuristic-based cost estimate for simulator mode."""
        estimate = {'estimated_gb': 0.1, 'estimated_cost_usd': 0.05,
            'complexity': 'low', 'optimization_suggestions': []}
        query_upper = query.upper()
        if any(keyword in query_upper for keyword in ['JOIN', 'GROUP BY']):
            estimate.update({'complexity': 'medium', 'estimated_gb': 0.5,
                'estimated_cost_usd': 0.25})
        if any(keyword in query_upper for keyword in ['ARRAY_AGG', 'OVER(',
            'PARTITION BY']):
            estimate.update({'complexity': 'high', 'estimated_gb': 2.0,
                'estimated_cost_usd': 1.0})
        logger.info(
            f"Heuristic query cost estimate: {estimate['estimated_gb']} GB (Complexity: {estimate['complexity']})"
            )
        return estimate

    def _generate_gdelt_mock_data(self) ->pd.DataFrame:
        """Generates mock GDELT event data for testing."""
        from src.config.unified_config_manager import get_current_config
        config = get_current_config()
        seed = config.get('performance.random_seed', 42)
        random.seed(seed)
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y%m%d') for
            i in range(30)]
        return pd.DataFrame({'SQLDATE': dates, 'daily_tone': [round(random.
            uniform(-2, 2), 2) for _ in range(30)], 'daily_stability': [
            round(random.uniform(-1, 1), 2) for _ in range(30)],
            'event_volume': [random.randint(100, 1000) for _ in range(30)]})

    def _generate_generic_mock_data(self) ->pd.DataFrame:
        """Generates generic mock data for testing."""
        from src.config.unified_config_manager import get_current_config
        config = get_current_config()
        seed = config.get('performance.random_seed', 42)
        np.random.seed(seed)
        return pd.DataFrame({'timestamp': pd.to_datetime(pd.date_range(end=
            datetime.now(), periods=5, freq='D')), 'value': np.random.randn
            (5), 'label': ['A', 'B', 'C', 'D', 'E']})


if __name__ == '__main__':
    logger.info('--- Running BigQuery Client Verification ---')
    gcp_project_id = os.environ.get('GCP_PROJECT_ID')
    logger.info('--- Initializing Client ---')
    client = BigQueryClient(project_id=gcp_project_id)
    logger.info(f'Client running in simulator mode: {client.use_simulator}')
    logger.info('--- Checking Status ---')
    status = client.get_status()
    logger.info(f'{status}')
    if status['reachable']:
        test_q = (
            "SELECT SourceCommonName, COUNT(*) as ArticleCount FROM `gdelt-bq.gdeltv2.gkg` WHERE DATE(_PARTITIONTIME) = '2023-10-01' AND SourceCommonName LIKE '%yahoo.com%' GROUP BY 1 ORDER BY ArticleCount DESC LIMIT 10"
            )
        print(f'\n--- Validating Query ---\n{test_q}')
        validation = client.validate_query(test_q)
        print(validation)
        if validation['valid']:
            print('\n--- Estimating Query Cost ---')
            cost = client.get_query_cost_estimate(test_q)
            print(cost)
            print('\n--- Executing Query ---')
            df_result = client.execute_query(test_q)
            print('Result (first 5 rows):')
            print(df_result.head())
    print('\n--- Verification Complete ---')
