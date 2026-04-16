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
from src.features.utils.datetime_utils import ensure_datetime_column, normalize_metadata_columns

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
        
        # Try to initialize GCS manager, but don't fail if credentials are missing
        try:
            self.gcs_manager = GCSManager()
        except Exception as e:
            self.logger.warning(f"GCS Manager initialization failed: {e}. Continuing without cloud storage.")
            self.gcs_manager = None
        
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
        self.logger.info(f"Raw data keys: {raw_data.keys()}")
        self.logger.info(f"Raw data type: {type(raw_data)}")
        
        cleaned_data_map = {}

        # 1. Process Market Data (Prices) - Local
        market_data_key = None
        if 'market_data_raw' in raw_data:
            market_data_key = 'market_data_raw'
        elif 'market_data' in raw_data:
            market_data_key = 'market_data'
        
        if market_data_key:
            self.logger.info(f"Normalizing and cleaning market data from key: {market_data_key}...")
            df_m = raw_data[market_data_key]
            self.logger.info(f"Market data shape before processing: {df_m.shape}")
            df_m = normalize_price_df(df_m)
            df_m = DataCleaner.remove_outliers_zscore(df_m, columns=['close'], threshold=3.0)
            df_m = DataCleaner.handle_missing_values(df_m, method='ffill')
            
            price_data_dict = {}
            if 'interval' in df_m.columns:
                for interval, group in df_m.groupby('interval'):
                    price_data_dict[interval] = group
                    self.logger.info(f"  Interval {interval}: {len(group)} rows")
            else:
                self.logger.error("CRITICAL: 'interval' column not found in market data!")
                self.logger.error("This will cause mixing of 15m/1h/1d candles.")
                self.logger.error("Check Stage 1 - yf_collector should add 'interval' column.")
                # For now, log error but continue - will be fixed in next run
                self.logger.warning("Continuing with mixed timeframes (will be fixed after Stage 1 cache clear)")
                price_data_dict['mixed'] = df_m
            
            cleaned_data_map['prices'] = price_data_dict
            self.logger.info(f"Added prices to cleaned_data_map with {len(price_data_dict)} timeframes")

        # 2. Process News Data
        news_key = None
        for key in ['news', 'news_data', 'google_news', 'newsapi_articles']:
            if key in raw_data and not raw_data[key].empty:
                news_key = key
                break
        
        if news_key:
            self.logger.info(f"Processing news data from key: {news_key}...")
            df_n = raw_data[news_key].copy()
            
            # Check which columns exist for deduplication
            dedup_cols = []
            if 'title' in df_n.columns:
                dedup_cols.append('title')
            if 'link' in df_n.columns:
                dedup_cols.append('link')
            elif 'url' in df_n.columns:
                dedup_cols.append('url')
            
            if dedup_cols:
                df_n = df_n.drop_duplicates(subset=dedup_cols)
                self.logger.info(f"Deduplicated news using columns: {dedup_cols}")
            else:
                self.logger.warning("No deduplication columns found in news data")
            
            # Try GCS processing first, fallback to local if it fails
            gcs_success = False
            if self.gcs_manager:
                # DEBUG: Temporarily reduce dataset size to test cloud function
                self.logger.warning(f"Original news articles: {len(df_n)}. Truncating to 100 for debugging cloud function.")
                df_n_gcs = df_n.head(100)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                raw_file_name = f"raw_news_{timestamp}.parquet"
                gcs_path_prefix_raw = self.config_manager.get_config('cloud_storage')['paths']['raw_data']
                gcs_full_path_raw = f"{gcs_path_prefix_raw}/{raw_file_name}"

                try:
                    buffer = io.BytesIO()
                    df_n_gcs.to_parquet(buffer, index=False)
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
                        gcs_success = True
                    else:
                        self.logger.warning("Timed out waiting for processed news file. Falling back to local processing.")

                except Exception as e:
                    self.logger.warning(f"GCS processing failed: {e}. Falling back to local processing.")
            
            # If GCS failed or not available, process locally
            if not gcs_success:
                self.logger.info("Processing news locally with FinBERT sentiment analysis...")
                
                # Add basic sentiment column if not present
                if 'sentiment' not in df_n.columns:
                    try:
                        from transformers import pipeline
                        self.logger.info("Loading FinBERT sentiment model locally...")
                        sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
                        
                        # Process in batches to avoid memory issues
                        batch_size = 32
                        sentiments = []
                        
                        text_col = 'title' if 'title' in df_n.columns else 'description' if 'description' in df_n.columns else None
                        if text_col:
                            texts = df_n[text_col].fillna('').astype(str).tolist()
                            for i in range(0, len(texts), batch_size):
                                batch = texts[i:i+batch_size]
                                batch_results = sentiment_pipeline(batch, truncation=True, max_length=512)
                                sentiments.extend([r['score'] if r['label'] == 'positive' else -r['score'] if r['label'] == 'negative' else 0 for r in batch_results])
                            
                            df_n['sentiment'] = sentiments
                            self.logger.info(f"✅ Added sentiment scores to {len(df_n)} news articles")
                        else:
                            self.logger.warning("No text column found for sentiment analysis")
                            df_n['sentiment'] = 0.0
                    except Exception as e:
                        self.logger.warning(f"Local sentiment analysis failed: {e}. Using neutral sentiment.")
                        df_n['sentiment'] = 0.0
                
                cleaned_data_map['news'] = df_n
                self.logger.info(f"✅ Added news with {len(df_n)} articles (local processing with sentiment)")
        
        # 3. Process Macro Data - Local
        macro_key = None
        for key in ['macro_data', 'fred_data', 'fred', 'macro']:
            if key in raw_data and raw_data[key] is not None:
                if isinstance(raw_data[key], pd.DataFrame) and not raw_data[key].empty:
                    macro_key = key
                    break
        
        if macro_key:
            self.logger.info(f"Processing macro data from key: {macro_key}...")
            df_macro = raw_data[macro_key].copy().drop_duplicates()
            df_macro = df_macro.ffill().bfill()
            cleaned_data_map['macro_data'] = df_macro
            self.logger.info(f"Added macro_data with shape: {df_macro.shape}")
        else:
            self.logger.warning(f"No macro data found in raw_data. Available keys: {list(raw_data.keys())}")

        # 4. Intelligent Filtering
        self.logger.info("Applying intelligent data filtering...")
        self.logger.info(f"Data before filtering - keys: {cleaned_data_map.keys()}")
        for key, value in cleaned_data_map.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}: dict with keys {value.keys()}")
            elif isinstance(value, pd.DataFrame):
                self.logger.info(f"  {key}: DataFrame with shape {value.shape}")
            else:
                self.logger.info(f"  {key}: {type(value)}")
        
        filtered_results = self.data_filter.filter_quality_data(cleaned_data_map)
        self.logger.info(f"Filtered results keys: {filtered_results.keys()}")
        self.logger.info(f"Filtered data keys: {filtered_results.get('filtered_data', {}).keys()}")
        
        # 5. Normalization
        self.logger.info("Fitting normalization scalers...")
        features_to_normalize = self.config_manager.get_config('processing.normalization.features')
        
        if features_to_normalize and 'prices' in filtered_results.get('filtered_data', {}) and filtered_results['filtered_data']['prices']:
            try:
                first_timeframe = next(iter(filtered_results['filtered_data']['prices']))
                # У декого структура може бути іншою, перевіряємо
                prices_data = filtered_results['filtered_data']['prices'][first_timeframe]
                fittable_data = prices_data['data'].copy() if isinstance(prices_data, dict) and 'data' in prices_data else prices_data.copy()
                
                self.normalization_manager.fit_scalers(fittable_data, features_to_normalize)
            except Exception as e:
                self.logger.warning(f"Normalization fitting failed: {e}. Continuing without normalization updates.")
        else:
            self.logger.info("Skipping normalization (features_to_normalize is missing or no data)")

        # 6. System Validation
        self.logger.info("Running unified data validation...")
        validation_results = self.validator.validate_cleaned_data(filtered_results.get('filtered_data', {}))
        
        if not validation_results.get('is_valid', False):
            self.logger.warning(f"Validation issues detected: {validation_results.get('issues', [])}")

        # 7. Save cleaned data to separate files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = 'data/colab/accumulated'
        
        cleaned_data = filtered_results.get('filtered_data', {})
        
        # Save prices by timeframe
        if 'prices' in cleaned_data and isinstance(cleaned_data['prices'], dict):
            for timeframe, df in cleaned_data['prices'].items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    price_file = f"{output_dir}/stage2_prices_{timeframe}_{timestamp}.parquet"
                    df.to_parquet(price_file, index=False)
                    self.logger.info(f"Saved {timeframe} prices to {price_file} ({len(df)} rows)")
        
        # Save news data separately
        if 'news' in cleaned_data and isinstance(cleaned_data['news'], pd.DataFrame):
            news_file = f"{output_dir}/stage2_news_{timestamp}.parquet"
            cleaned_data['news'].to_parquet(news_file, index=False)
            self.logger.info(f"Saved news data to {news_file} ({len(cleaned_data['news'])} rows)")
        
        # Save macro data if present
        if 'macro_data' in cleaned_data and isinstance(cleaned_data['macro_data'], pd.DataFrame):
            macro_file = f"{output_dir}/stage2_macro_{timestamp}.parquet"
            cleaned_data['macro_data'].to_parquet(macro_file, index=False)
            self.logger.info(f"Saved macro data to {macro_file} ({len(cleaned_data['macro_data'])} rows)")

        health = self.resource_monitor.get_health_status()
        self.logger.info(f"Stage 2 complete. System Health: CPU {health.get('cpu')}, MEM {health.get('memory')}")

        # ✅ CRITICAL FIX: Normalize all data with datetime/ticker columns
        # This ensures Stage 3 always gets consistent column structure
        for key in ['prices', 'news', 'macro_data']:
            if key in cleaned_data:
                if isinstance(cleaned_data[key], dict):
                    # For nested dicts (like prices by timeframe)
                    for sub_key, sub_df in cleaned_data[key].items():
                        if isinstance(sub_df, pd.DataFrame):
                            cleaned_data[key][sub_key] = normalize_metadata_columns(sub_df)
                            self.logger.info(f"✅ Normalized {key}[{sub_key}]: has datetime={('datetime' in cleaned_data[key][sub_key].columns)}, ticker={('ticker' in cleaned_data[key][sub_key].columns)}")
                elif isinstance(cleaned_data[key], pd.DataFrame):
                    cleaned_data[key] = normalize_metadata_columns(cleaned_data[key])
                    self.logger.info(f"✅ Normalized {key}: has datetime={('datetime' in cleaned_data[key].columns)}, ticker={('ticker' in cleaned_data[key].columns)}")

        result = {"cleaned_data": cleaned_data}
        self.logger.info(f"Stage 2 returning result with keys: {result.keys()}")
        if 'cleaned_data' in result:
            self.logger.info(f"cleaned_data keys: {result['cleaned_data'].keys()}")
        
        return result
