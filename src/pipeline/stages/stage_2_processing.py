# src/pipeline/stages/stage_2_processing.py

import pandas as pd
import io
import time
import os
from datetime import datetime
from typing import Optional, Dict, Any

from src.pipeline.stages.base_stage import BaseStage
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler
from src.processing.price_preprocessor import PricePreprocessor
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
            err = ValueError("Scaler path configuration is missing.")
            self.handle_stage_error(err, context="ScalerPath-Config", severity="critical", should_raise=True)

        self.data_filter = IntelligentDataFilter(config=filtering_config)
        self.normalization_manager = NormalizationManager(scaler_dir=scaler_path)

    def _extract_raw_data(self, kwargs) -> dict:
        """Extract and validate raw data from kwargs."""
        raw_data = kwargs
        if 'raw_data' in raw_data and isinstance(raw_data['raw_data'], dict):
            self.logger.info("Unwrapping nested raw_data payload from previous stage.")
            raw_data = raw_data['raw_data']
        if not raw_data:
            self.logger.error("No data found for processing. Skipping Stage 2.")
            return {}
        return raw_data

    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        Runs the processing cycle. Offloads raw data with GCS offloading and local fallback.
        """
        self.logger.info("Starting data processing stage...")
        
        raw_data = kwargs.get('raw_data', {})
        
        # ✅ Unwrap nested raw_data if present
        if 'raw_data' in raw_data and isinstance(raw_data['raw_data'], dict):
            self.logger.info("Unwrapping nested raw_data from Stage 1")
            raw_data = raw_data['raw_data']
        
        # Debug: Log raw data structure
        self.logger.info(f"Raw data keys: {list(raw_data.keys())}")
        for key, value in raw_data.items():
            if hasattr(value, 'shape'):
                self.logger.info(f"  {key}: {value.shape}")
            elif hasattr(value, '__len__'):
                self.logger.info(f"  {key}: {len(value)} items")
            else:
                self.logger.info(f"  {key}: {type(value)}")
        self.logger.info(f"Raw data type: {type(raw_data)}")
        
        cleaned_data_map = {}

        # 1. Process Market Data (Prices) - Local
        self._process_market_data(raw_data, cleaned_data_map)

        # 2. Process News Data
        self._process_news_data(raw_data, cleaned_data_map)
        
        # 3. Process Macro Data - Local
        self._process_macro_data(raw_data, cleaned_data_map)
        
        # 4. Process Sentiment Data
        self._process_sentiment_data(raw_data, cleaned_data_map)
        
        # 5. Process Market Sentiment (Fear/Greed, VIX)
        self._process_market_sentiment_data(raw_data, cleaned_data_map)
        
        # 6. Process Institutional Data (SEC, Insider)
        self._process_institutional_data(raw_data, cleaned_data_map)
        
        # 7. Process Trends Data
        self._process_trends_data(raw_data, cleaned_data_map)
        
        # 8. Process Economic Data
        self._process_economic_data(raw_data, cleaned_data_map)
        
        # 9. Process Social Sentiment
        self._process_social_sentiment_data(raw_data, cleaned_data_map)
        
        # 10. Process ML Features
        self._process_ml_features_data(raw_data, cleaned_data_map)
        
        # 11. Process Additional Data
        self._process_additional_data(raw_data, cleaned_data_map)

        # 4. Intelligent Filtering
        filtered_results = self._apply_intelligent_filtering(cleaned_data_map)
        
        # 5. Normalization
        self._apply_normalization(filtered_results)

        # 6. System Validation
        self._run_system_validation(filtered_results)

        # 7. Save cleaned data to separate files
        cleaned_data = self._save_cleaned_data_to_files(filtered_results)

        # 8. Finalize and normalize data
        result = self._finalize_results(cleaned_data)
        
        health = self.resource_monitor.get_health_status()
        self.logger.info(f"Stage 2 complete. System Health: CPU {health.get('cpu')}, MEM {health.get('memory')}")
        
        return result

    def _process_market_data(self, raw_data: dict, cleaned_data_map: dict):
        """Process market data (prices) locally."""
        market_data_key = self._find_market_data_key(raw_data)
        
        if market_data_key:
            self.logger.info(f"Normalizing and cleaning market data from key: {market_data_key}...")
            df_m = raw_data[market_data_key]
            self.logger.info(f"Market data shape before processing: {df_m.shape}")
            df_m = self._clean_and_normalize_market_data(df_m)
            price_data_dict = self._group_by_timeframes(df_m)
            cleaned_data_map['prices'] = price_data_dict
            self._log_prices_summary(price_data_dict)

    def _find_market_data_key(self, raw_data: dict) -> Optional[str]:
        """Find the market data key in raw_data."""
        if 'market_data' in raw_data:
            return 'market_data'
        elif 'market_data_raw' in raw_data:
            return 'market_data_raw'
        return None

    def _clean_and_normalize_market_data(self, df_m: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize market data."""
        df_m = PricePreprocessor().normalize_price_df(df_m)
        df_m = DataCleaner.remove_outliers_zscore(df_m, columns=['close'], threshold=3.0)
        df_m = DataCleaner.handle_missing_values(df_m, method='ffill')
        return df_m

    def _group_by_timeframes(self, df_m: pd.DataFrame) -> dict:
        """Group market data by timeframes."""
        price_data_dict = {}
        if 'interval' in df_m.columns:
            for interval, group in df_m.groupby('interval'):
                self._process_timeframe_group(interval, group, price_data_dict)
        else:
            self._handle_missing_interval_column(df_m, price_data_dict)
        return price_data_dict

    def _process_timeframe_group(self, interval: str, group: pd.DataFrame, price_data_dict: dict):
        """Process a single timeframe group."""
        self.logger.info(f"  Processing interval {interval}: {len(group)} rows, type: {type(group)}")
        if hasattr(group, 'empty') and not group.empty:
            price_data_dict[interval] = group
            self.logger.info(f"  ✅ Added {len(group)} rows for interval {interval}")
        else:
            self.logger.warning(f"  ⚠️ Skipping empty group for interval {interval}")

    def _handle_missing_interval_column(self, df_m: pd.DataFrame, price_data_dict: dict):
        """Handle missing interval column in market data."""
        self.logger.error("CRITICAL: 'interval' column not found in market data!")
        self.logger.error("This will cause mixing of 15m/1h/1d candles.")
        self.logger.error("Check Stage 1 - yf_collector should add 'interval' column.")
        self.logger.warning("Continuing with mixed timeframes (will be fixed after Stage 1 cache clear)")
        price_data_dict['mixed'] = df_m

    def _log_prices_summary(self, price_data_dict: dict):
        """Log summary of processed prices."""
        self.logger.info(f"Added prices to cleaned_data_map with {len(price_data_dict)} timeframes")
        for tf, df in price_data_dict.items():
            self.logger.info(f"  prices['{tf}']: type={type(df)}, shape={df.shape if hasattr(df, 'shape') else 'no shape'}")

    def _process_news_data(self, raw_data: dict, cleaned_data_map: dict):
        """Process news data with GCS offloading or local sentiment analysis."""
        news_key = self._find_news_data_key(raw_data)
        
        if news_key:
            self.logger.info(f"Processing news data from key: {news_key}...")
            df_n = raw_data[news_key].copy()
            
            df_n = self._deduplicate_news_data(df_n)
            
            gcs_success = self._try_gcs_processing(df_n, cleaned_data_map)
            
            if not gcs_success:
                self._process_news_locally(df_n, cleaned_data_map)
    
    def _find_news_data_key(self, raw_data: dict) -> Optional[str]:
        """Find the news data key in raw_data."""
        for key in ['news', 'news_data', 'google_news', 'newsapi_articles']:
            if key in raw_data and raw_data[key] is not None:
                # Check if it has data (DataFrame or list)
                if self._has_data(raw_data[key]):
                    return key
        return None
    
    def _has_data(self, obj) -> bool:
        """Check if object has data (not empty)."""
        if hasattr(obj, 'empty'):
            return not obj.empty
        elif hasattr(obj, '__len__'):
            return len(obj) > 0
        return False
    
    def _deduplicate_news_data(self, df_n: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate news data based on available columns."""
        dedup_cols = self._get_deduplication_columns(df_n)
        
        if dedup_cols:
            df_n = df_n.drop_duplicates(subset=dedup_cols)
            self.logger.info(f"Deduplicated news using columns: {dedup_cols}")
        else:
            self.logger.warning("No deduplication columns found in news data")
        
        return df_n
    
    def _get_deduplication_columns(self, df_n: pd.DataFrame) -> list:
        """Get columns available for deduplication."""
        dedup_cols = []
        if 'title' in df_n.columns:
            dedup_cols.append('title')
        if 'link' in df_n.columns:
            dedup_cols.append('link')
        elif 'url' in df_n.columns:
            dedup_cols.append('url')
        return dedup_cols

    def _try_gcs_processing(self, df_n: pd.DataFrame, cleaned_data_map: dict) -> bool:
        """Try to process news data via GCS cloud function."""
        if not self.gcs_manager:
            return False
            
        try:
            # Process all news articles for full pipeline
            self.logger.info(f"Processing {len(df_n)} news articles for full pipeline.")
            df_n_gcs = df_n

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_file_name = f"raw_news_{timestamp}.parquet"
            gcs_path_prefix_raw = self.config_manager.get_config('cloud_storage')['paths']['raw_data']
            gcs_full_path_raw = f"{gcs_path_prefix_raw}/{raw_file_name}"

            buffer = io.BytesIO()
            df_n_gcs.to_parquet(buffer, index=False)
            buffer.seek(0)
            self.gcs_manager.upload_blob_from_memory(buffer, gcs_full_path_raw)
            self.logger.info(f"Successfully uploaded news data to: gs://{self.gcs_manager.bucket_name}/{gcs_full_path_raw}")

            # Wait for processed file
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
                return True
            else:
                self.logger.warning("Timed out waiting for processed news file. Falling back to local processing.")
                return False

        except Exception as e:
            self.logger.warning(f"GCS processing failed: {e}. Falling back to local processing.")
            return False

    def _get_text_column(self, df_n: pd.DataFrame) -> Optional[str]:
        """Get the text column for sentiment analysis."""
        if 'title' in df_n.columns:
            return 'title'
        elif 'description' in df_n.columns:
            return 'description'
        else:
            return None
    
    def _perform_sentiment_analysis(self, df_n: pd.DataFrame, text_col: str) -> list:
        """Perform sentiment analysis on the specified text column."""
        # Use global cached pipeline to avoid reloading
        if not hasattr(self, '_sentiment_pipeline'):
            from transformers import pipeline
            self.logger.info("Loading FinBERT sentiment model locally (cached)...")
            self._sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device="cpu")
        else:
            self.logger.info("Using cached FinBERT sentiment model...")
        
        sentiment_pipeline = self._sentiment_pipeline
        
        # Process in optimized batches
        batch_size = 64  # Increased for better performance
        sentiments = []
        texts = df_n[text_col].fillna('').astype(str).tolist()
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        self.logger.info(f"Processing {len(texts)} texts in {total_batches} batches...")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            if batch_num % 10 == 0 or batch_num == total_batches:
                self.logger.info(f"Processing batch {batch_num}/{total_batches}...")
            
            batch_results = sentiment_pipeline(batch, truncation=True, max_length=512, padding=True)
            sentiments.extend([r['score'] if r['label'] == 'positive' else -r['score'] if r['label'] == 'negative' else 0 for r in batch_results])
        
        return sentiments
    
    def _handle_sentiment_failure(self, df_n: pd.DataFrame, error: Exception):
        """Handle sentiment analysis failure by setting neutral sentiment."""
        self.logger.warning(f"Local sentiment analysis failed: {error}. Using neutral sentiment.")
        df_n['sentiment'] = 0.0
    
    def _process_news_locally(self, df_n: pd.DataFrame, cleaned_data_map: dict):
        """Process news data locally with cached FinBERT sentiment analysis."""
        self.logger.info("Processing news locally with FinBERT sentiment analysis...")
        
        # Check if sentiment already exists from cache
        if 'sentiment' in df_n.columns:
            self.logger.info(f"Sentiment already exists in {len(df_n)} news articles (from cache)")
            cleaned_data_map['news'] = df_n
            return
        
        # Try to load cached sentiment results
        cached_sentiments = self._load_cached_sentiments(df_n)
        if cached_sentiments is not None:
            df_n['sentiment'] = cached_sentiments
            self.logger.info(f"Loaded cached sentiment for {len(df_n)} news articles")
            cleaned_data_map['news'] = df_n
            return
        
        # ✅ FIX: Filter news by date limit (60 days for intraday)
        from datetime import datetime, timedelta
        
        # Find date column
        date_col = None
        for col in ['published_date', 'published_at', 'datetime', 'timestamp']:
            if col in df_n.columns:
                date_col = col
                break
        
        if date_col:
            # Convert to datetime
            df_n[date_col] = pd.to_datetime(df_n[date_col], errors='coerce')
            
            # Calculate cutoff date (60 days ago from latest data)
            # Get first available price DataFrame from prices dict
            price_df = None
            if 'prices' in cleaned_data_map and cleaned_data_map['prices']:
                for tf, price_data in cleaned_data_map['prices'].items():
                    if isinstance(price_data, pd.DataFrame) and not price_data.empty:
                        price_df = price_data
                        break
            
            if price_df is not None and 'datetime' in price_df.columns:
                latest_price_date = price_df['datetime'].max()
                cutoff_date = latest_price_date - timedelta(days=60)
            else:
                cutoff_date = datetime.now() - timedelta(days=60)
            
            # Filter news
            before_filter = len(df_n)
            df_n = df_n[df_n[date_col] >= cutoff_date]
            after_filter = len(df_n)
            
            self.logger.info(f"Filtered news: {before_filter} → {after_filter} (60-day limit from {cutoff_date})")
        
        # ✅ FIX: Filter news by tickers from prices
        if 'prices' in cleaned_data_map:
            # Get unique tickers from prices
            price_tickers = set()
            for tf, price_data in cleaned_data_map['prices'].items():
                if isinstance(price_data, pd.DataFrame) and 'ticker' in price_data.columns:
                    price_tickers.update(price_data['ticker'].unique())
            
            # Filter news by tickers
            if 'ticker' in df_n.columns and price_tickers:
                before_ticker_filter = len(df_n)
                df_n = df_n[df_n['ticker'].isin(price_tickers)]
                after_ticker_filter = len(df_n)
                self.logger.info(f"Filtered news by ticker: {before_ticker_filter} → {after_ticker_filter}")
        
        # For full pipeline, process filtered texts
        self.logger.info(f"Processing {len(df_n)} news articles for full pipeline (60-day limit)")
        
        # Add sentiment column with FinBERT
        try:
            text_col = self._get_text_column(df_n)
            
            if text_col:
                self.logger.info(f"Running FinBERT analysis on {len(df_n)} texts (will be cached)...")
                sentiments = self._perform_sentiment_analysis(df_n, text_col)
                df_n['sentiment'] = sentiments
                self.logger.info(f" Added FinBERT sentiment scores to {len(df_n)} news articles (cached for future runs)")
                
                # Cache the results for future runs
                self._cache_sentiments(df_n, text_col, sentiments)
                self.logger.info(f" Cached sentiment results for {len(df_n)} news articles")
            else:
                self.logger.warning("No text column found for sentiment analysis")
                self._handle_sentiment_failure(df_n, ValueError("No text column found for sentiment analysis"))
        except Exception as e:
            self.logger.error(f"FinBERT sentiment analysis failed: {e}")
            self._handle_sentiment_failure(df_n, e)
        
        cleaned_data_map['news'] = df_n
    
    def _load_cached_sentiments(self, df_n: pd.DataFrame) -> Optional[list]:
        """Load cached sentiment results if available."""
        try:
            cache_file = 'data/cache/sentiment_cache.parquet'
            if not os.path.exists(cache_file):
                return None
            
            cache_df = pd.read_parquet(cache_file)
            text_col = self._get_text_column(df_n)
            
            if not text_col or text_col not in cache_df.columns:
                return None
            
            # Find matching texts in cache
            cached_sentiments = []
            for text in df_n[text_col].fillna('').astype(str):
                match = cache_df[cache_df[text_col] == text]
                if not match.empty:
                    cached_sentiments.append(match['sentiment'].iloc[0])
                else:
                    cached_sentiments.append(None)
            
            # Check if we have cached results for all texts
            if all(s is not None for s in cached_sentiments):
                return cached_sentiments
            else:
                self.logger.info(f"Partial cache hit: {sum(1 for s in cached_sentiments if s is not None)}/{len(cached_sentiments)} texts cached")
                return None
        
        except Exception as e:
            self.logger.warning(f"Failed to load cached sentiments: {e}")
            return None
    
    def _cache_sentiments(self, df_n: pd.DataFrame, text_col: str, sentiments: list):
        """Cache sentiment results for future use."""
        try:
            os.makedirs('data/cache', exist_ok=True)
            cache_file = 'data/cache/sentiment_cache.parquet'
            
            # Create cache DataFrame
            cache_data = {
                text_col: df_n[text_col].fillna('').astype(str),
                'sentiment': sentiments,
                'timestamp': pd.Timestamp.now()
            }
            cache_df = pd.DataFrame(cache_data)
            
            # Append to existing cache or create new
            if os.path.exists(cache_file):
                existing_cache = pd.read_parquet(cache_file)
                # Remove duplicates and append new data
                combined_cache = pd.concat([existing_cache, cache_df], ignore_index=True)
                combined_cache = combined_cache.drop_duplicates(subset=[text_col], keep='last')
                combined_cache.to_parquet(cache_file, index=False)
            else:
                cache_df.to_parquet(cache_file, index=False)
            
        except Exception as e:
            self.logger.warning(f"Failed to cache sentiments: {e}")

    def _process_macro_data(self, raw_data: dict, cleaned_data_map: dict):
        """Process macro data from FRED and other sources."""
        macro_key = None
        for key in ['macro_data', 'fred_data']:
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

    def _apply_intelligent_filtering(self, cleaned_data_map: dict) -> dict:
        """Apply intelligent data filtering."""
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
        return filtered_results
    

    def _apply_normalization(self, filtered_results: dict):
        """Apply normalization to filtered data."""
        self.logger.info("Fitting normalization scalers...")
        features_to_normalize = self.config_manager.get_config('processing.normalization.features')
        
        if self._should_apply_normalization(features_to_normalize, filtered_results):
            self._apply_normalization_fitting(filtered_results, features_to_normalize)
        else:
            self.logger.info("Skipping normalization (features_to_normalize is missing or no data)")
    
    def _should_apply_normalization(self, features_to_normalize: list, filtered_results: dict) -> bool:
        """Check if normalization should be applied."""
        return (features_to_normalize and 
                'prices' in filtered_results.get('filtered_data', {}) and 
                filtered_results['filtered_data']['prices'])
    
    def _apply_normalization_fitting(self, filtered_results: dict, features_to_normalize: list):
        """Apply normalization fitting to the data."""
        try:
            first_timeframe = next(iter(filtered_results['filtered_data']['prices']))
            prices_data = filtered_results['filtered_data']['prices'][first_timeframe]
            fittable_data = prices_data['data'].copy() if isinstance(prices_data, dict) and 'data' in prices_data else prices_data.copy()
            
            self.normalization_manager.fit_scalers(fittable_data, features_to_normalize)
        except Exception as e:
            self.logger.warning(f"Normalization fitting failed: {e}. Continuing without normalization updates.")

    def _run_system_validation(self, filtered_results: dict):
        """Run system validation on filtered data."""
        self.logger.info("Running unified data validation...")
        validation_results = self.validator.validate_cleaned_data(filtered_results.get('filtered_data', {}))
        
        if not validation_results.get('is_valid', False):
            self.logger.warning(f"Validation issues detected: {validation_results.get('issues', [])}")

    def _save_cleaned_data_to_files(self, filtered_results: dict) -> dict:
        """Save cleaned data to separate files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = 'data/colab/accumulated'
        
        cleaned_data = filtered_results.get('filtered_data', {})
        
        # Process and save each data type
        self._save_prices_data(cleaned_data, timestamp, output_dir)
        self._save_news_data(cleaned_data, timestamp, output_dir)
        self._save_macro_data(cleaned_data, timestamp, output_dir)
        
        return cleaned_data
    
    def _process_sentiment_data(self, raw_data: dict, cleaned_data_map: dict) -> None:
        """Process sentiment data from various sources."""
        sentiment_data = raw_data.get('sentiment_data')
        if sentiment_data is not None and not sentiment_data.empty:
            cleaned_data_map['sentiment_data'] = sentiment_data
            self.logger.info(f"Processed sentiment data with shape: {sentiment_data.shape}")
    
    def _process_market_sentiment_data(self, raw_data: dict, cleaned_data_map: dict) -> None:
        """Process market sentiment data (Fear/Greed, VIX)."""
        market_sentiment = raw_data.get('market_sentiment')
        if market_sentiment is not None and not market_sentiment.empty:
            cleaned_data_map['market_sentiment'] = market_sentiment
            self.logger.info(f"Processed market sentiment data with shape: {market_sentiment.shape}")
    
    def _process_institutional_data(self, raw_data: dict, cleaned_data_map: dict) -> None:
        """Process institutional data (SEC filings, insider trading)."""
        institutional_data = raw_data.get('institutional_data')
        if institutional_data is not None and not institutional_data.empty:
            cleaned_data_map['institutional_data'] = institutional_data
            self.logger.info(f"Processed institutional data with shape: {institutional_data.shape}")
    
    def _process_trends_data(self, raw_data: dict, cleaned_data_map: dict) -> None:
        """Process trends data (Google Trends, etc.)."""
        trends_data = raw_data.get('trends_data')
        if trends_data is not None and not trends_data.empty:
            cleaned_data_map['trends_data'] = trends_data
            self.logger.info(f"Processed trends data with shape: {trends_data.shape}")
    
    def _process_economic_data(self, raw_data: dict, cleaned_data_map: dict) -> None:
        """Process economic calendar data."""
        economic_data = raw_data.get('economic_data')
        if economic_data is not None and not economic_data.empty:
            cleaned_data_map['economic_data'] = economic_data
            self.logger.info(f"Processed economic data with shape: {economic_data.shape}")
    
    def _process_social_sentiment_data(self, raw_data: dict, cleaned_data_map: dict) -> None:
        """Process social sentiment data (Reddit, etc.)."""
        social_sentiment = raw_data.get('social_sentiment')
        if social_sentiment is not None and not social_sentiment.empty:
            cleaned_data_map['social_sentiment'] = social_sentiment
            self.logger.info(f"Processed social sentiment data with shape: {social_sentiment.shape}")
    
    def _process_ml_features_data(self, raw_data: dict, cleaned_data_map: dict) -> None:
        """Process ML features data (HuggingFace, etc.)."""
        ml_features = raw_data.get('ml_features')
        if ml_features is not None and not ml_features.empty:
            cleaned_data_map['ml_features'] = ml_features
            self.logger.info(f"Processed ML features data with shape: {ml_features.shape}")
    
    def _process_additional_data(self, raw_data: dict, cleaned_data_map: dict) -> None:
        """Process additional uncategorized data."""
        additional_data = raw_data.get('additional_data', {})
        if additional_data:
            for table_name, df in additional_data.items():
                if df is not None and not df.empty:
                    cleaned_data_map[f'additional_{table_name}'] = df
                    self.logger.info(f"Processed additional data {table_name} with shape: {df.shape}")
    
    def _save_prices_data(self, cleaned_data: dict, timestamp: str, output_dir: str) -> None:
        """Save prices data to separate files."""
        if 'prices' not in cleaned_data:
            return
        
        prices_data = cleaned_data['prices']
        if not isinstance(prices_data, dict):
            self.logger.warning("No prices data to save")
            return
        
        # Save prices by timeframe
        for timeframe, df in prices_data.items():
            df = self._prepare_dataframe_for_saving(df)
            price_file = f"{output_dir}/stage2_prices_{timeframe}_{timestamp}.parquet"
            df.to_parquet(price_file, index=False)
            self.logger.info(f"Saved {timeframe} prices to {price_file} ({len(df)} rows)")
    
    def _save_news_data(self, cleaned_data: dict, timestamp: str, output_dir: str) -> None:
        """Save news data to file."""
        if 'news' not in cleaned_data:
            return
        
        news_data = cleaned_data['news']
        if not isinstance(news_data, pd.DataFrame) or news_data.empty:
            self.logger.warning("No news data to save")
            return
        
        news_file = f"{output_dir}/stage2_news_{timestamp}.parquet"
        news_data.to_parquet(news_file, index=False)
        self.logger.info(f"Saved news data to {news_file} ({len(news_data)} rows)")
    
    def _save_macro_data(self, cleaned_data: dict, timestamp: str, output_dir: str) -> None:
        """Save macro data to file."""
        if 'macro_data' not in cleaned_data:
            return
        
        macro_data = cleaned_data['macro_data']
        if not isinstance(macro_data, pd.DataFrame) or macro_data.empty:
            self.logger.warning("No macro data to save")
            return
        
        macro_file = f"{output_dir}/stage2_macro_{timestamp}.parquet"
        macro_data.to_parquet(macro_file, index=False)
        self.logger.info(f"Saved macro data to {macro_file} ({len(macro_data)} rows)")
    
    def _prepare_dataframe_for_saving(self, df) -> pd.DataFrame:
        """Prepare DataFrame for saving by ensuring required columns."""
        df = self._convert_to_dataframe(df)
        df = self._ensure_required_columns(df)
        return df.copy()
    
    def _convert_to_dataframe(self, df) -> pd.DataFrame:
        """Convert input to DataFrame if needed."""
        if isinstance(df, pd.DataFrame):
            return df
        
        if not isinstance(df, dict):
            self.logger.error(f"Unexpected input type: {type(df)}")
            return pd.DataFrame(columns=['datetime', 'ticker'])
        
        return self._handle_dict_input(df)
    
    def _handle_dict_input(self, df_dict: dict) -> pd.DataFrame:
        """Handle dictionary input (DataFilter output)."""
        # Reduce warning frequency - this is expected behavior for DataFilter output
        if not hasattr(self, '_dict_warning_logged'):
            self.logger.warning(f"Expected DataFrame but got dict with keys: {list(df_dict.keys())}. This is normal for DataFilter output.")
            self._dict_warning_logged = True
        
        # Try to extract 'data' key if it's DataFilter output
        if 'data' in df_dict and isinstance(df_dict['data'], pd.DataFrame):
            df = df_dict['data']
            self.logger.info(f"Extracted DataFrame from 'data' key: {df.shape}")
            return df
        
        # Try to convert dict to DataFrame
        try:
            return pd.DataFrame(df_dict)
        except Exception as e:
            self.logger.error(f"Could not convert dict to DataFrame: {e}")
            return pd.DataFrame(columns=['datetime', 'ticker'])
    
    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure datetime and ticker columns exist."""
        df = self._add_datetime_column(df)
        df = self._add_ticker_column(df)
        return df
    
    def _add_datetime_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add datetime column if missing."""
        if 'datetime' in df.columns:
            return df
        
        if hasattr(df.index, 'to_datetime'):
            df['datetime'] = pd.to_datetime(df.index, utc=True)
        else:
            df['datetime'] = pd.Timestamp.now(tz='UTC')
        
        return df
    
    def _add_ticker_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ticker column if missing."""
        if 'ticker' in df.columns:
            return df
        
        if (hasattr(df.index, 'get_level_values') and 
            'ticker' in df.index.names):
            df['ticker'] = df.index.get_level_values(0)
        else:
            df['ticker'] = 'unknown'
        
        return df

    def _normalize_nested_data(self, cleaned_data: dict, key: str):
        """Normalize nested dictionary data structures."""
        for sub_key, sub_df in cleaned_data[key].items():
            if isinstance(sub_df, pd.DataFrame):
                cleaned_data[key][sub_key] = normalize_metadata_columns(sub_df)
                self.logger.info(f"✅ Normalized {key}[{sub_key}]: has datetime={('datetime' in cleaned_data[key][sub_key].columns)}, ticker={('ticker' in cleaned_data[key][sub_key].columns)}")
    
    def _normalize_dataframe(self, cleaned_data: dict, key: str):
        """Normalize single DataFrame structures."""
        cleaned_data[key] = normalize_metadata_columns(cleaned_data[key])
        self.logger.info(f"✅ Normalized {key}: has datetime={('datetime' in cleaned_data[key].columns)}, ticker={('ticker' in cleaned_data[key].columns)}")
    
    def _create_quality_metrics(self, cleaned_data: dict) -> dict:
        """Create quality metrics for the processed data."""
        news_data = cleaned_data.get('news', [])
        price_data = cleaned_data.get('prices', {})
        
        return {
            "news_count": len(news_data) if isinstance(news_data, pd.DataFrame) else 0,
            "price_timeframes": len(price_data) if isinstance(price_data, dict) else 0,
        }
    
    def _finalize_results(self, cleaned_data: dict) -> dict:
        """Finalize results and normalize metadata columns."""
        # ✅ CRITICAL FIX: Normalize all data with datetime/ticker columns
        # This ensures Stage 3 always gets consistent column structure
        for key in ['prices', 'news', 'macro_data']:
            if key in cleaned_data:
                if isinstance(cleaned_data[key], dict):
                    self._normalize_nested_data(cleaned_data, key)
                elif isinstance(cleaned_data[key], pd.DataFrame):
                    self._normalize_dataframe(cleaned_data, key)

        result = {
            "cleaned_data": cleaned_data,
            "market_data": cleaned_data.get('prices', {}),  # For compatibility with Stage 3
            "news": cleaned_data.get('news', pd.DataFrame()),
            "macro_data": cleaned_data.get('macro_data', pd.DataFrame()),
            "normalization_params": {},
            "quality_metrics": self._create_quality_metrics(cleaned_data),
            "models_metadata": {  # Required by pipeline validation
                "processing_models": {
                    "data_filter": "DataFilter",
                    "normalizer": "NormalizationManager",
                    "validator": "UnifiedValidator"
                },
                "version": "1.0",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        self.logger.info(f"Stage 2 returning result with keys: {result.keys()}")
        if 'cleaned_data' in result:
            self.logger.info(f"cleaned_data keys: {result['cleaned_data'].keys()}")
        
        return result
