from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.pipeline.timeframe_lineage import (
    partition_market_frame_by_timeframe,
)
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
        if "datetime" in df_m.columns:
            df_m["datetime"] = pd.to_datetime(
                df_m["datetime"],
                errors="coerce",
                utc=True,
            )
            df_m.attrs["datetime_timezone"] = "UTC"
            df_m.attrs["datetime_timezone_source"] = (
                "market_data_normalized_to_utc"
            )
        return df_m

    def clean_and_normalize_macro_data(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize long-form macro observations without applying OHLCV rules."""
        if not isinstance(macro_df, pd.DataFrame):
            raise TypeError("Macro data must be a pandas DataFrame.")

        result = macro_df.copy()
        rename_map = {}
        if "series_id" not in result.columns and "series" in result.columns:
            rename_map["series"] = "series_id"
        if "datetime" not in result.columns:
            date_column = next(
                (column for column in ("date", "timestamp") if column in result.columns),
                None,
            )
            if date_column:
                rename_map[date_column] = "datetime"
        if rename_map:
            result = result.rename(columns=rename_map)

        canonical_columns = ["datetime", "series_id", "value"]
        if result.empty:
            for column in canonical_columns:
                if column not in result.columns:
                    result[column] = pd.Series(dtype="object")
            return result

        missing = [column for column in canonical_columns if column not in result.columns]
        if missing:
            raise ValueError(
                f"Macro data is missing canonical columns: {', '.join(missing)}."
            )

        availability_column = next(
            (
                column
                for column in (
                    "available_at",
                    "released_at",
                    "realtime_start",
                )
                if column in result.columns
            ),
            None,
        )
        if availability_column is None:
            raise ValueError(
                "Macro data is missing an authoritative point-in-time "
                "availability column (available_at, released_at, or "
                "realtime_start)."
            )
        invalid_availability = pd.to_datetime(
            result[availability_column], errors="coerce", utc=True
        ).isna()
        if invalid_availability.any():
            raise ValueError(
                "Macro data contains missing or invalid point-in-time "
                f"values in {availability_column}."
            )

        result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce", utc=True)
        result["series_id"] = result["series_id"].astype("string").str.strip()
        result["value"] = pd.to_numeric(result["value"], errors="coerce")
        result = result.dropna(subset=canonical_columns)
        result = result.loc[result["series_id"].ne("")]
        # A vintage is part of a row's identity, not a duplicate of it.
        #
        # Deduplicating on (datetime, series_id) alone collapsed the stored
        # table from 314,062 rows to 97,090 -- every revision of every figure
        # discarded, keeping one row whose survival was decided by row order
        # rather than by publication date. The macro enricher is built to use
        # exactly what this threw away: it keys the pivot on `available_at`,
        # sorts by `realtime_start` so `aggfunc='last'` means most recently
        # published, and forward-fills along the publication axis.
        #
        # The harm is loss, not look-ahead: a surviving late revision carries
        # its own 2026 stamp, so an old bar does not see it -- it sees nothing
        # at all where the original 1996 print used to be. Keeping the vintage
        # in the key preserves both the first release and every restatement,
        # and still removes rows that are genuinely identical.
        dedup_key = ["datetime", "series_id"]
        if availability_column not in dedup_key:
            dedup_key.append(availability_column)
        result = result.sort_values(dedup_key).drop_duplicates(
            dedup_key,
            keep="last",
        )
        return result.reset_index(drop=True)

    def group_by_timeframes(self, df_m: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Partition by declared timeframe or infer cadence fail-closed."""
        groups = partition_market_frame_by_timeframe(df_m)
        for timeframe, group in groups.items():
            source = group.attrs.get("timeframe_source")
            self.logger.info(
                "Prepared %s market rows for %s from %s",
                len(group),
                timeframe,
                source,
            )
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
