# src/pipeline/stages/stage_2_processing.py

from typing import Any

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.cloud.gcs_manager import GCSManager
from src.core.error_handling.error_handler import ErrorHandler
from src.core.file_management.file_manager import FileManager
from src.core.logging.logger import ProjectLogger
from src.monitoring.infrastructure.resource_monitor import get_resource_monitor
from src.patterns.pattern_analyzer import PatternAnalyzer
from src.pipeline.stages.base_stage import BaseStage
from src.processing.data_filter import IntelligentDataFilter
from src.processing.normalization_manager import NormalizationManager
from src.processing.price_preprocessor import PricePreprocessor
from src.utils.trading_calendar import TradingCalendar
from src.validation.validators import UnifiedValidator


class ProcessingStage(BaseStage):
    """
    Stage 2: Data Processing, Cleaning, and Cloud Offloading.
    """
    def __init__(self, config_manager: UnifiedConfigManager, error_handler: ErrorHandler, **kwargs):
        super().__init__(config_manager, error_handler, **kwargs)
        self.logger = ProjectLogger.get_logger("ProcessingStage")
        self.analysis_history: list[dict[str, Any]] = []
        self.validator = UnifiedValidator()
        self.calendar = TradingCalendar()
        self.resource_monitor = get_resource_monitor()
        self.pattern_analyzer = PatternAnalyzer(enable_debug=True)
        self.file_manager: FileManager = FileManager(base_dir='.')

        try:
            self.gcs_manager: GCSManager | None = GCSManager()
        except Exception as e:
            self.logger.warning(f"GCS Manager initialization failed: {e}. Continuing without cloud storage.")
            self.gcs_manager = None

        paths_config = self.config_manager.get_config('paths') or {}
        scaler_dir = paths_config.get('scalers')
        self.scaler_dir: str | None = scaler_dir
        self.data_filter: IntelligentDataFilter = IntelligentDataFilter(config_manager=self.config_manager)
        self.normalization_manager = NormalizationManager(scaler_dir=scaler_dir or '')

    async def run(self, **kwargs) -> dict[str, Any]:
        """Runs the processing cycle."""
        self.logger.info("Starting data processing stage...")
        raw_data = {**kwargs}
        if 'raw_data' in raw_data and isinstance(raw_data['raw_data'], dict):
            raw_data.update(raw_data.pop('raw_data'))

        self.cleaned_data_map: dict[str, Any] = {}
        self._process_market_data(raw_data, self.cleaned_data_map)
        self._process_news_data(raw_data, self.cleaned_data_map)

        # Застосування фільтрації та нормалізації
        filtered_results = self._apply_intelligent_filtering(self.cleaned_data_map)
        self._apply_normalization(filtered_results)

        return self._finalize_results(filtered_results)

    def _process_market_data(self, raw_data: dict, cleaned_data_map: dict):
        """Process market data (prices) and split by interval."""
        if 'market_data' in raw_data:
            df = raw_data['market_data']
            df = PricePreprocessor().normalize_price_df(df)

            # ✅ SPLIT BY INTERVAL: Instead of hardcoding '1d', split the data
            if 'interval' in df.columns:
                timeframes = {}
                for interval, group in df.groupby('interval'):
                    timeframes[str(interval)] = group
                cleaned_data_map['prices'] = timeframes
                self.logger.info(f"✅ Split market data into timeframes: {list(timeframes.keys())}")
            else:
                self.logger.warning("⚠️ No 'interval' column in market data, defaulting to '1d'")
                cleaned_data_map['prices'] = {'1d': df}

    def _process_news_data(self, raw_data: dict, cleaned_data_map: dict):
        """Process news data."""
        if 'news' in raw_data:
            cleaned_data_map['news'] = raw_data['news']

    def _apply_intelligent_filtering(self, cleaned_data_map: dict) -> dict:
        filtered = self.data_filter.filter_quality_data(cleaned_data_map)

        # Simple NaN% quality check on numeric price columns (informational only).
        import pandas as pd
        prices = filtered.get('filtered_data', {}).get('prices', {})
        for tf, tf_entry in prices.items():
            df = tf_entry.get('data') if isinstance(tf_entry, dict) else tf_entry

            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                self.logger.warning(f"⚠️ No valid DataFrame for timeframe '{tf}', skipping quality check.")
                continue

            numeric_df = df.select_dtypes(include='number')
            if numeric_df.empty:
                continue

            total_cells = numeric_df.size
            nan_cells = numeric_df.isna().sum().sum()
            nan_pct = (nan_cells / total_cells * 100) if total_cells > 0 else 0

            if nan_pct > 20:
                self.logger.warning(
                    f"⚠️ High NaN rate in '{tf}': {nan_pct:.1f}% missing values. "
                    f"Continuing with forward-fill."
                )
            else:
                self.logger.info(f"✅ Data quality for '{tf}': {nan_pct:.1f}% NaN (rows={len(df)})")

        return filtered

    def _unwrap_price_entries(self, filtered_data: dict) -> dict:
        """
        Unwrap PriceFilter's {'data': df, 'quality': ..., ...} entries into plain DataFrames.
        ProcessedDataSchema._validate_price_dataframe expects plain DataFrames, not dicts.
        """
        import pandas as pd
        prices = filtered_data.get('prices', {})
        unwrapped: dict[str, pd.DataFrame] = {}
        for tf, entry in prices.items():
            if isinstance(entry, dict) and 'data' in entry:
                df = entry['data']
            else:
                df = entry
            if isinstance(df, pd.DataFrame) and not df.empty:
                unwrapped[tf] = df
        if unwrapped:
            filtered_data = {**filtered_data, 'prices': unwrapped}
        return filtered_data

    def _apply_normalization(self, filtered_results: dict):
        self.logger.info("Fitting normalization scalers...")
        features_to_normalize = self.config_manager.get_config('processing.normalization.features') or []

        # Unwrap PriceFilter entries to get plain DataFrames
        prices_data = self._unwrap_price_entries(
            filtered_results.get('filtered_data', {})
        ).get('prices', {})
        first_tf = next(iter(prices_data.values()), None)

        if first_tf is not None and not first_tf.empty:
            self.normalization_manager.fit_scalers(first_tf, features_to_normalize)
        else:
            self.logger.info("Skipping normalization (no data for fitting)")

    def _finalize_results(self, filtered_results: dict) -> dict:
        # Unwrap {'data': df, ...} entries so downstream stages and the schema
        # validator receive plain DataFrames under 'prices'.
        cleaned = self._unwrap_price_entries(filtered_results.get('filtered_data', {}))
        return {"cleaned_data": cleaned}
