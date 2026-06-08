from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.processing.cleaners import DataCleaner
from src.processing.price_preprocessor import PricePreprocessor

logger = ProjectLogger.get_logger('ProcessingDataHandler')

class ProcessingDataHandler:
    """Handles data cleaning, normalization, and grouping for the processing stage."""

    def __init__(self, normalization_manager: Any, data_filter: Any):
        self.logger = logger
        self.normalization_manager = normalization_manager
        self.data_filter = data_filter

    def clean_and_normalize_market_data(self, df_m: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize market data."""
        self.logger.info(f"Cleaning market data, shape: {df_m.shape}")
        df_m = PricePreprocessor().normalize_price_df(df_m)
        df_m = DataCleaner.remove_outliers_zscore(df_m, columns=['close'], threshold=3.0)
        df_m = DataCleaner.handle_missing_values(df_m, method='ffill')
        return df_m

    def group_by_timeframes(self, df_m: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Group data by timeframe if an interval column exists."""
        if 'interval' not in df_m.columns:
            return {'daily': df_m} # Default

        groups = {}
        for interval, group in df_m.groupby('interval'):
            groups[str(interval)] = group
        return groups

    def apply_intelligent_filtering(self, cleaned_data_map: dict[str, Any]) -> dict[str, Any]:
        """Apply filters to reduce noise in data."""
        self.logger.info("Applying intelligent filtering to data...")
        filter_result = self.data_filter.filter_quality_data(cleaned_data_map)
        return self._extract_filtered_data(filter_result)

    def apply_normalization(
        self,
        filtered_results: dict[str, Any],
        features_to_normalize: list[dict[str, Any]] | None = None,
        fit_scalers: bool = True,
    ):
        """Apply scaling/normalization to features."""
        if not features_to_normalize:
            self.logger.info("No normalization features configured; skipping normalization.")
            return
        self.logger.info("Applying normalization...")
        frames = self._collect_dataframes(filtered_results)
        if not frames:
            self.logger.warning("No DataFrames found for normalization.")
            return

        if fit_scalers:
            combined = pd.concat(frames, ignore_index=True, sort=False)
            self.normalization_manager.fit_scalers(combined, features_to_normalize)
        else:
            self.normalization_manager.load_scalers(
                [cfg["feature"] for cfg in features_to_normalize if "feature" in cfg]
            )

        self._transform_nested_dataframes(filtered_results)

    def _extract_filtered_data(self, filter_result: dict[str, Any]) -> dict[str, Any]:
        """Keep accepted data while preserving quality metadata separately."""
        filtered_data = filter_result.get('filtered_data', filter_result)
        result: dict[str, Any] = {}

        if isinstance(filtered_data, dict):
            self._extract_prices(filtered_data, result)
            for key, value in filtered_data.items():
                if key != 'prices':
                    result[key] = value

        self._extract_metadata(filter_result, result)
        return result

    def _extract_prices(self, filtered_data: dict[str, Any], result: dict[str, Any]) -> None:
        """Extract prices from filtered data."""
        prices = filtered_data.get('prices')
        if isinstance(prices, dict):
            result['prices'] = {}
            for timeframe, payload in prices.items():
                data = payload.get('data') if isinstance(payload, dict) else payload
                if isinstance(data, pd.DataFrame):
                    result['prices'][timeframe] = data

    def _extract_metadata(self, filter_result: dict[str, Any], result: dict[str, Any]) -> None:
        """Extract quality metadata."""
        for meta_key in ('quality_report', 'patterns', 'filtering_summary'):
            if meta_key in filter_result:
                result[meta_key] = filter_result[meta_key]

    def _collect_dataframes(self, data: Any) -> list[pd.DataFrame]:
        """Collect all DataFrame leaves from nested dictionaries."""
        if isinstance(data, pd.DataFrame):
            return [data]
        if isinstance(data, dict):
            frames: list[pd.DataFrame] = []
            for value in data.values():
                frames.extend(self._collect_dataframes(value))
            return frames
        return []

    def _transform_nested_dataframes(self, data: dict[str, Any]) -> None:
        """Apply fitted scalers to every DataFrame leaf in place."""
        for key, value in list(data.items()):
            if isinstance(value, pd.DataFrame):
                data[key] = self.normalization_manager.transform_data(value)
            elif isinstance(value, dict):
                self._transform_nested_dataframes(value)
