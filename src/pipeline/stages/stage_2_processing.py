# src/pipeline/stages/stage_2_processing.py

import pandas as pd
import io
import time
from datetime import datetime

from src.pipeline.stages.base_stage import BaseStage
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler
from src.processing.price_preprocessor import normalize_price_df
from src.processing.cleaners import DataCleaner
from src.processing.data_filter import IntelligentDataFilter
from src.processing.normalization_manager import NormalizationManager
from src.validation.validators import UnifiedValidator
from src.utils.trading_calendar import TradingCalendar
from src.monitoring.infrastructure.resource_monitor import get_resource_monitor
from src.core.logging.logger import ProjectLogger
from src.core.cloud.gcs_manager import GCSManager
from src.core.file_management.file_manager import FileManager

class ProcessingStage(BaseStage):
    """
    Stage 2: Data Processing, Cleaning, and Cloud Offloading.
    - Normalizes and cleans market and macro data locally.
    - Offloads raw news data to cloud storage for heavy NLP processing.
    """
    def __init__(self, config_manager: UnifiedConfigManager, error_handler: ErrorHandler, **kwargs):
        super().__init__(config_manager, error_handler, **kwargs)
        self.logger = ProjectLogger.get_logger("ProcessingStage")
        self.validator = UnifiedValidator()
        self.calendar = TradingCalendar()
        self.resource_monitor = get_resource_monitor()
        self.gcs_manager = GCSManager()
        self.file_manager = FileManager(base_dir='.')

        # Safely get nested configuration
        processing_config = self.config_manager.get_config('processing') or {}
        filtering_config = processing_config.get('filtering')
        
        paths_config = self.config_manager.get_config('paths') or {}
        scaler_path = paths_config.get('scalers')

        if not scaler_path:
            self.logger.critical("Scaler path ('paths.scalers') not found in configuration. Cannot proceed.")
            raise ValueError("Scaler path configuration is missing.")

        self.data_filter = IntelligentDataFilter(config=filtering_config)
        self.normalization_manager = NormalizationManager(scaler_dir=scaler_path)

    async def run(self, **kwargs) -> dict:
        """
        Runs the processing cycle. Offloads news data to GCS.
        """
        raw_data = kwargs.get('raw_data')
        if not raw_data:
            self.logger.error("No 'raw_data' found for processing. Skipping Stage 2.")
            return {}

        self.logger.info("Starting data processing stage...")
        cleaned_data_map = {}

        # 1. Process Market Data (Prices) - Local
        if 'market_data' in raw_data:
            self.logger.info("Normalizing and cleaning market data...")
            df_m = raw_data['market_data']
            df_m = normalize_price_df(df_m)
            df_m = DataCleaner.remove_outliers_zscore(df_m, columns=['close'], threshold=3.0)
            df_m = DataCleaner.handle_missing_values(df_m, method='ffill')
            
            price_data_dict = {}
            if 'interval' in df_m.columns:
                for interval, group in df_m.groupby('interval'):
                    price_data_dict[interval] = group
            else:
                self.logger.warning("'interval' column not found in market data. Assuming single timeframe '1d'.")
                price_data_dict['1d'] = df_m
            
            cleaned_data_map['prices'] = price_data_dict

        # 2. Process News Data
        if 'news_data' in raw_data and not raw_data['news_data'].empty:
            self.logger.info("Processing news data...")
            df_n = raw_data['news_data'].copy().drop_duplicates(subset=['title', 'link'])
            
            # DEBUG: Temporarily reduce dataset size to test cloud function
            self.logger.warning(f"Original news articles: {len(df_n)}. Truncating to 100 for debugging cloud function.")
            df_n = df_n.head(100)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_file_name = f"raw_news_{timestamp}.parquet"
            gcs_path_prefix_raw = self.config_manager.get_config('cloud_storage')['paths']['raw_data']
            gcs_full_path_raw = f"{gcs_path_prefix_raw}/{raw_file_name}"

            try:
                buffer = io.BytesIO()
                df_n.to_parquet(buffer, index=False)
                buffer.seek(0)
                self.gcs_manager.upload_blob_from_memory(buffer, gcs_full_path_raw)
                self.logger.info(f"Successfully uploaded news data to: gs://{self.gcs_manager.bucket_name}/{gcs_full_path_raw}")

                # Wait for the processed file
                gcs_path_prefix_processed = self.config_manager.get_config('cloud_storage')['paths']['processed_data']
                processed_file_name = f"{gcs_path_prefix_processed}/{raw_file_name}"
                
                self.logger.info(f"Waiting for processed file at: {processed_file_name}")
                processed_blob = self.gcs_manager.wait_for_blob(processed_file_name, timeout=300)

                if processed_blob:
                    self.logger.info("Processed file found. Downloading...")
                    processed_buffer = io.BytesIO()
                    processed_blob.download_to_file(processed_buffer)
                    processed_buffer.seek(0)
                    processed_df = pd.read_parquet(processed_buffer)
                    cleaned_data_map['news'] = processed_df
                    self.logger.info("Processed news data successfully downloaded and loaded.")
                else:
                    self.logger.error("Timed out waiting for processed news file. Using raw data for now.")
                    cleaned_data_map['news'] = df_n

            except Exception as e:
                self.logger.error(f"An error occurred during news processing pipeline: {e}", exc_info=True)
                cleaned_data_map['news'] = df_n
        
        # 3. Process Macro Data - Local
        if 'macro_data' in raw_data:
            self.logger.info("Processing macro data...")
            df_macro = raw_data['macro_data'].copy().drop_duplicates()
            df_macro = df_macro.ffill().bfill()
            cleaned_data_map['macro_data'] = df_macro

        # 4. Intelligent Filtering
        self.logger.info("Applying intelligent data filtering...")
        filtered_results = self.data_filter.filter_quality_data(cleaned_data_map)
        
        # 5. Normalization
        self.logger.info("Fitting normalization scalers...")
        features_to_normalize = self.config_manager.get_config('processing.normalization.features')
        
        if 'prices' in filtered_results.get('filtered_data', {}) and filtered_results['filtered_data']['prices']:
            first_timeframe = next(iter(filtered_results['filtered_data']['prices']))
            fittable_data = filtered_results['filtered_data']['prices'][first_timeframe]['data'].copy()
            self.normalization_manager.fit_scalers(fittable_data, features_to_normalize)

        # 6. System Validation
        self.logger.info("Running unified data validation...")
        validation_results = self.validator.validate_cleaned_data(filtered_results.get('filtered_data', {}))
        
        if not validation_results.get('is_valid', False):
            self.logger.warning(f"Validation issues detected: {validation_results.get('issues', [])}")

        health = self.resource_monitor.get_health_status()
        self.logger.info(f"Stage 2 complete. System Health: CPU {health.get('cpu')}, MEM {health.get('memory')}")

        return {"cleaned_data": filtered_results.get('filtered_data', {})}
