from datetime import datetime
from typing import Any

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.cloud.gcs_manager import GCSManager
from src.core.error_handling.error_handler import ErrorHandler
from src.core.file_management.file_manager import FileManager
from src.core.logging.logger import ProjectLogger
from src.monitoring.infrastructure.resource_monitor import get_resource_monitor
from src.pipeline.stages.base_stage import BaseStage
from src.processing.data_filter import IntelligentDataFilter
from src.processing.normalization_manager import NormalizationManager

from .data_handler import ProcessingDataHandler
from .storage import ProcessingStorage
from .validator import ProcessingValidator


class ProcessingStage(BaseStage):
    """
    Modular Stage 2: Data Processing, Cleaning, and Cloud Offloading.
    Delegates to specialized components for validation, handling, and storage.
    """

    def __init__(self, config_manager: UnifiedConfigManager, error_handler: ErrorHandler, **kwargs):
        super().__init__(config_manager, error_handler, **kwargs)
        self.logger = ProjectLogger.get_logger('ProcessingStage')

        # Initialize Core Components
        self.file_manager = FileManager(base_dir='.')
        self.resource_monitor = get_resource_monitor()

        processing_config = self.config_manager.get_config('processing') or {}
        filtering_config = processing_config.get('filtering')
        paths_config = self.config_manager.get_config('paths') or {}
        scaler_dir = paths_config.get('scalers', 'data/scalers')

        self.data_filter = IntelligentDataFilter(config=filtering_config)
        self.normalization_manager = NormalizationManager(scaler_dir=scaler_dir)

        # Initialize Specialized Modular Components
        self.modular_validator = ProcessingValidator()
        self.data_handler = ProcessingDataHandler(self.normalization_manager, self.data_filter)
        self.storage_manager = ProcessingStorage(self.file_manager)

        try:
            self.gcs_manager = GCSManager()
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.warning(f'GCS Manager initialization failed: {e}. Continuing without cloud storage.')
            self.gcs_manager = None

    async def run(self, **kwargs) -> dict[str, Any]:
        """Runs the processing cycle."""
        self.logger.info('Starting modular data processing stage...')

        raw_data = self._extract_raw_data(kwargs)
        if not raw_data:
            return {}
        processing_config = self.config_manager.get_config('processing') or {}

        cleaned_data_map: dict[str, Any] = {}

        # 1. Process different data types
        self._process_all_data_types(raw_data, cleaned_data_map)

        # 2. Intelligent Filtering
        filtered_results = self.data_handler.apply_intelligent_filtering(cleaned_data_map)

        # 3. Normalization
        normalization_config = processing_config.get('normalization', {})
        features_to_normalize = normalization_config.get('features', [])
        run_mode = kwargs.get('run_mode', 'train')
        self.data_handler.apply_normalization(
            filtered_results,
            features_to_normalize=features_to_normalize,
            fit_scalers=run_mode != 'predict',
        )

        # 4. Validation
        self.modular_validator.run_system_validation(filtered_results)

        # 5. Storage
        storage_paths = self.storage_manager.save_cleaned_data_to_files(filtered_results)

        # Finalize
        result = self._finalize_results(filtered_results, storage_paths)

        health = self.resource_monitor.get_health_status()
        self.logger.info(f"Stage 2 complete. System Health: CPU {health.get('cpu')}, MEM {health.get('memory')}")

        return result

    def _extract_raw_data(self, kwargs) -> dict[str, Any]:
        raw_data = kwargs.get('raw_data', {})
        if 'raw_data' in raw_data and isinstance(raw_data['raw_data'], dict):
            raw_data = raw_data['raw_data']
        return raw_data

    def _process_all_data_types(self, raw_data: dict[str, Any], cleaned_data_map: dict[str, Any]):
        """Delegates processing for various data types."""
        # This would call methods like _process_market_data, _process_news_data, etc.
        # For the sake of refactoring, we maintain the original logic structure but cleaner.
        if 'market_data' in raw_data:
            df_m = self.data_handler.clean_and_normalize_market_data(raw_data['market_data'])
            cleaned_data_map['prices'] = self.data_handler.group_by_timeframes(df_m)

        # The accumulated table, not this run's fetch.
        #
        # `macro_data` is what FredCollector RETURNED, and the collector is
        # configured to fetch from two years back -- 8,103 rows on 2026-08-27,
        # of which 23 were new. `fred_data` is the table those fetches have
        # been accumulating into for as long as the project has run: 154,045
        # rows, 13,535 daily observations for DGS10 alone, over fifty years.
        #
        # Feature engineering used the two-year fetch, joined it to a frame
        # spanning thirty years, and 93% of the rows came out with no macro
        # value. A median fill then wrote a plausible constant over the gap,
        # which is how 70% of every FRED column became one number containing
        # the future (see the macro enricher). The history was never missing.
        # It was sitting in the next variable.
        macro_source = None
        for key in ('fred_data', 'macro_data'):
            frame = raw_data.get(key)
            if isinstance(frame, __import__('pandas').DataFrame) and not frame.empty:
                if macro_source is None or len(frame) > len(macro_source[1]):
                    macro_source = (key, frame)
        if macro_source is not None:
            key, frame = macro_source
            self.logger.info(
                "Macro source for enrichment: '%s' with %d rows "
                "(candidates: %s).", key, len(frame),
                {k: len(v) for k, v in raw_data.items()
                 if k in ('fred_data', 'macro_data')
                 and isinstance(v, __import__('pandas').DataFrame)},
            )
            cleaned_data_map['macro_data'] = (
                self.data_handler.clean_and_normalize_macro_data(frame)
            )

        # Pass news data with cleaning
        if 'news' in raw_data and isinstance(raw_data['news'], __import__('pandas').DataFrame):
            news_df = raw_data['news'].copy()
            # Basic cleaning for news: remove duplicates, handle missing values
            if 'title' in news_df.columns:
                news_df = news_df.drop_duplicates(subset=['title'])
            news_df = __import__('pandas').DataFrame(news_df).fillna('')
            cleaned_data_map['news'] = news_df

        # Pass reddit_sentiment through - IntelligentDataFilter.filter_quality_data
        # already has a dedicated reddit_sentiment branch (filter_reddit_data),
        # but it only fires if this key is actually present here. Without this,
        # data collected by RedditSentimentCollector in Stage 1 vanished with
        # zero logging: never filtered, never stored, never available downstream.
        if 'reddit_sentiment' in raw_data and isinstance(raw_data['reddit_sentiment'], __import__('pandas').DataFrame):
            cleaned_data_map['reddit_sentiment'] = raw_data['reddit_sentiment'].copy()

    def _finalize_results(self, cleaned_data: dict[str, Any], storage_paths: dict[str, Any]) -> dict[str, Any]:
        return {
            'status': 'success',
            'cleaned_data': cleaned_data,
            'storage_paths': storage_paths,
            'quality_metrics': self.modular_validator.create_quality_metrics(cleaned_data),
            'timestamp': datetime.now().isoformat()
        }
