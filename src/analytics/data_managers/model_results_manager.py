"""
Manages the persistence and retrieval of model performance results.
"""
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ModelResultsManager:
    """
    Handles all file I/O for model results, including saving, loading, 
    and combining results from different model types (light/heavy).

    This class acts as a data access layer for model performance data.
    """
    
    # Constants for filenames and model types
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
        
        # In-memory cache to avoid redundant file reads
        self._cache: Dict[str, pd.DataFrame] = {}
        logger.info(f"ModelResultsManager initialized with base path: {self.base_path}")

    def save_results(self, results_df: pd.DataFrame, is_heavy: bool = False):
        """
        Saves a DataFrame of model results to the appropriate Parquet file.
        
        Args:
            results_df (pd.DataFrame): The DataFrame containing model results.
            is_heavy (bool): Flag indicating if the results are from heavy models.
        """
        if not isinstance(results_df, pd.DataFrame) or results_df.empty:
            logger.warning("Attempted to save an empty or invalid DataFrame. Aborting.")
            return

        target_path = self.heavy_results_path if is_heavy else self.light_results_path
        model_type = 'heavy' if is_heavy else 'light'

        # Standardize the DataFrame
        df_to_save = results_df.copy()
        df_to_save['model_type'] = model_type
        df_to_save['timestamp'] = datetime.now()

        try:
            if target_path.exists():
                existing_df = pd.read_parquet(target_path)
                combined_df = pd.concat([existing_df, df_to_save], ignore_index=True)
                # Deduplicate to keep the latest results for a given run
                combined_df = combined_df.drop_duplicates(
                    subset=['model', 'ticker', 'timeframe', 'timestamp'], 
                    keep='last'
                )
            else:
                combined_df = df_to_save

            combined_df.to_parquet(target_path)
            logger.info(f"Successfully saved {len(df_to_save)} new results to {target_path}")
            
            # Invalidate cache for the saved file and combined results
            self._cache.pop(str(target_path), None)
            self._cache.pop('combined', None)

        except Exception as e:
            logger.error(f"Error saving results to {target_path}: {e}", exc_info=True)

    def get_results(self, model_type: str = 'all') -> pd.DataFrame:
        """
        Loads model results from Parquet files.

        Args:
            model_type (str): Type of results to load ('light', 'heavy', or 'all').

        Returns:
            pd.DataFrame: A DataFrame with the requested model results.
        """
        if model_type == 'light':
            return self._load_file(self.light_results_path)
        elif model_type == 'heavy':
            return self._load_file(self.heavy_results_path)
        elif model_type == 'all':
            if 'combined' in self._cache:
                logger.debug("Returning combined results from cache.")
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
            logger.warning(f"Unknown model_type '{model_type}' requested.")
            return pd.DataFrame()

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """
        Internal method to load a single Parquet file with caching.
        """
        if str(file_path) in self._cache:
            logger.debug(f"Returning cached DataFrame for {file_path.name}")
            return self._cache[str(file_path)]
        
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                self._cache[str(file_path)] = df
                logger.info(f"Loaded {len(df)} records from {file_path.name}")
                return df
            except Exception as e:
                logger.error(f"Failed to load Parquet file {file_path}: {e}", exc_info=True)
                return pd.DataFrame()
        
        logger.info(f"File not found: {file_path.name}")
        return pd.DataFrame()

    def get_confidence_score(self, model_id: str, context_fingerprint: str, diary_path: str = "logs/experience_diary.csv") -> float:
        """
        Retrieves historical reliability from the Experience Diary for a specific model/context.
        This can be used to dynamically weight signals.

        Args:
            model_id (str): The name or ID of the model.
            context_fingerprint (str): A unique string representing the context (e.g., ticker + timeframe).
            diary_path (str): The path to the experience diary CSV.

        Returns:
            float: The confidence score (typically between 0 and 1), defaulting to 0.5.
        """
        diary_p = Path(diary_path)
        if not diary_p.exists():
            return 0.5  # Default neutral confidence

        try:
            diary = pd.read_csv(diary_p)
            # Filter by model and context
            mask = (diary['model_name'] == model_id) & (diary['context_fingerprint'] == context_fingerprint)
            relevant_history = diary[mask]

            if relevant_history.empty:
                # Fallback to model-only average if context is new
                relevant_history = diary[diary['model_name'] == model_id]

            if not relevant_history.empty:
                # Simple recency-weighted average (last 10 results)
                return float(relevant_history['metric_value'].tail(10).mean())
            
            return 0.5 # Default if no history found
        except Exception as e:
            logger.warning(f"Error reading Experience Diary for {model_id}: {e}")
            return 0.5
