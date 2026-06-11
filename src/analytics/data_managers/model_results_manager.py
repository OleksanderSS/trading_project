# src/analytics/data_managers/model_results_manager.py
"""
Model Results Manager
Manages the persistence, retrieval, and caching of model performance results and analysis.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

class ModelResultsManager:
    """
    Handles all file I/O and in-memory management for model results.
    Support for saving/loading results from different model categories (light/heavy).
    Acts as a centralized data access layer for analytical performance artifacts.
    """

    # Standardized filenames and model categorizations
    LIGHT_MODELS_FILENAME = "light_models_results.parquet"
    HEAVY_MODELS_FILENAME = "heavy_models_results.parquet"
    COMBINED_FILENAME = "combined_analysis.parquet"
    LIGHT_MODEL_TYPES = ["lgbm", "rf", "linear", "mlp", "ensemble"]

    def __init__(self, base_path: str = "data/models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.light_results_path = self.base_path / self.LIGHT_MODELS_FILENAME
        self.heavy_results_path = self.base_path / self.HEAVY_MODELS_FILENAME
        self.combined_path = self.base_path / self.COMBINED_FILENAME

        # Primary LRU-like cache for session-level performance optimization
        self._cache: dict[str, pd.DataFrame] = {}
        logger.info(f"ModelResultsManager synchronized with base path: {self.base_path}")

    def save_results(self, results_df: pd.DataFrame, is_heavy: bool = False):
        """
        Persists model results to the appropriate Parquet structure.

        Args:
            results_df: DataFrame containing the fresh model performance metrics.
            is_heavy: Boolean flag to switch between light and heavy storage buckets.
        """
        if not isinstance(results_df, pd.DataFrame) or results_df.empty:
            logger.warning("Attempted to persist an empty or invalid results DataFrame. Operation aborted.")
            return

        target_path = self.heavy_results_path if is_heavy else self.light_results_path
        model_category = 'heavy' if is_heavy else 'light'

        # Augment with temporal and categorical metadata
        df_to_save = results_df.copy()
        df_to_save['model_category'] = model_category
        df_to_save['ingestion_timestamp'] = datetime.now()

        try:
            if target_path.exists():
                existing_df = pd.read_parquet(target_path)
                combined_df = pd.concat([existing_df, df_to_save], ignore_index=True)
                # Deduplication logic to ensure unique model-asset-timeframe mapping
                combined_df = combined_df.drop_duplicates(
                    subset=['model', 'ticker', 'timeframe', 'ingestion_timestamp'],
                    keep='last'
                )
            else:
                combined_df = df_to_save

            combined_df.to_parquet(target_path)
            logger.info(f"Persisted {len(df_to_save)} new results to {target_path}")

            # Invalidate localized and combined caches
            self._cache.pop(str(target_path), None)
            self._cache.pop('combined', None)

        except Exception as e:
            logger.error(f"Failed to persist results to {target_path}: {e}", exc_info=True)

    def get_results(self, model_type: str = 'all') -> pd.DataFrame:
        """
        Retrieves historical performance records from Parquet storage.

        Args:
            model_type: Result category filter ('light', 'heavy', or 'all').

        Returns:
            pd.DataFrame: Merged or filtered historical performance dataset.
        """
        if model_type == 'light':
            return self._load_file(self.light_results_path)
        elif model_type == 'heavy':
            return self._load_file(self.heavy_results_path)
        elif model_type == 'all':
            if 'combined' in self._cache:
                logger.debug("Retrieving combined analysis from session cache.")
                return self._cache['combined']

            light_df = self.get_results('light')
            heavy_df = self.get_results('heavy')

            if not light_df.empty and not heavy_df.empty:
                combined = pd.concat([light_df, heavy_df], ignore_index=True)
            elif not light_df.empty:
                combined = light_df
            else:
                combined = heavy_df

            self._cache['combined'] = combined
            return combined
        else:
            logger.warning(f"Requested unknown model result category: '{model_type}'. Returning empty set.")
            return pd.DataFrame()

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """Internal helper for Parquet ingestion with session caching."""
        if str(file_path) in self._cache:
            logger.debug(f"Cache hit for persistence layer: {file_path.name}")
            return self._cache[str(file_path)]

        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                self._cache[str(file_path)] = df
                logger.info(f"Loaded {len(df)} performance records from {file_path.name}")
                return df
            except Exception as e:
                logger.error(f"Data corruption or I/O error reading {file_path}: {e}", exc_info=True)
                return pd.DataFrame()

        logger.debug(f"Persistence artifact not found: {file_path.name}")
        return pd.DataFrame()

    def get_confidence_score(self, model_id: str, context_fingerprint: str, diary_path: str = "logs/experience_diary.csv") -> float:
        """
        Calculates a confidence metric from historical Experience Diary records.
        Used for dynamic signal weighting and anomaly detection.

        Args:
            model_id: Identifier of the trained model instance.
            context_fingerprint: Unique identifier for the market context (e.g., asset_timeframe).
            diary_path: File path to the experience diary ledger.

        Returns:
            float: Calculated confidence score (0.0 - 1.0), defaults to 0.5 (neutral).
        """
        diary_p = Path(diary_path)
        if not diary_p.exists():
            return 0.5

        try:
            diary = pd.read_csv(diary_p)
            # Match specific model within specific context
            mask = (diary['model_name'] == model_id) & (diary['context_fingerprint'] == context_fingerprint)
            relevant_history = diary[mask]

            if relevant_history.empty:
                # Context fallback: Aggregate model performance across all contexts
                relevant_history = diary[diary['model_name'] == model_id]

            if not relevant_history.empty:
                # Recency-weighted mean of the last 10 historical events
                return float(relevant_history['metric_value'].tail(10).mean())

            return 0.5
        except Exception as e:
            logger.warning(f"Experience Diary lookup failed for {model_id}: {e}")
            return 0.5

    def get_cached_analysis(self, data_hash: str) -> dict[str, Any] | None:
        """
        Retrieves cached analytical results based on input data fingerprint.

        Args:
            data_hash: Fingerprint of the input datasets.

        Returns:
            Optional[Dict[str, Any]]: Cached analytical report or None if miss.
        """
        cache_key = f"analysis_cache_{data_hash}"
        if cache_key in self._cache:
            logger.debug(f"Retrieving cached analysis results for hash: {data_hash}")
            cached_result = self._cache[cache_key]
            # Ensure the cached result is properly typed
            if isinstance(cached_result, dict):
                return cached_result
            else:
                logger.warning(f"Cache contains invalid type for key {cache_key}: {type(cached_result)}")
                return None
        return None

    def cache_analysis(self, data_hash: str, results: dict[str, Any]) -> None:
        """
        Stores analytical results in the session cache using data fingerprint.

        Args:
            data_hash: Fingerprint of the input datasets.
            results: The analytical payload to be cached.
        """
        cache_key = f"analysis_cache_{data_hash}"
        self._cache[cache_key] = results
        logger.debug(f"Analytical payload cached for hash: {data_hash}")

    def save_json_result(self, data: dict[str, Any], filename: str) -> None:
        """
        Saves a JSON report to the base path.

        Args:
            data: The JSON serializable dictionary to save.
            filename: The name of the file.
        """
        import json
        target_path = self.base_path / filename
        try:
            with open(target_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            logger.info(f"Saved JSON report to {target_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON report to {target_path}: {e}")
